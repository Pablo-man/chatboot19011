from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from websockets.exceptions import ConnectionClosed

import chromadb
import json
import fitz
import uvicorn

ENDPOINT = "http://127.0.0.1:39281/v1"
#MODEL = "phi-3.5:3b-gguf-q4-km"
#MODEL = "deepseek-r1-distill-qwen-14b:14b-gguf-q4-km"
MODEL = "llama3.2:3b"

####
def load_pdf_content(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

####
client = chromadb.Client()
####
collection = client.create_collection("all-my-documents")

# --- NUEVO: Cargar el PDF ---
pdf_text = load_pdf_content("iso19011.pdf")  # o la ruta que prefieras

# Puedes dividir el texto en partes si es muy largo
chunks = [pdf_text[i:i+500] for i in range(0, len(pdf_text), 1000)]

# Ahora lo agregas al collection
collection.add(
    documents=chunks,
    ids=[f"id{i}" for i in range(len(chunks))]
)
####

system_prompt = """
Eres un especialista en la normativa ISO19011, te encargas de dar solucion a distintos casos utilizando esta metodologia. Sigue estas instrucciones:
- Ofrece respuestas cortas y concisas de no mas de 25 palabras.
- Explica al cliente cosas relacionadas con en la siguiente lista JSON: """


client = AsyncOpenAI(
    base_url=ENDPOINT,
    api_key="not-needed"
)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
 
@app.get("/", response_class=HTMLResponse)
async def root( request: Request ):
    return RedirectResponse("/static/index.html")

@app.websocket("/init")
async def init(websocket: WebSocket):
    print("Aceptando nueva conexión WebSocket...")
    await websocket.accept()
    print("Conexión WebSocket aceptada")
    try:
        print("Esperando mensaje JSON del cliente...")
        while True:
            data = await websocket.receive_json()
            print(f"Mensaje recibido: {json.dumps(data)}")
            await websocket.send_json({"action": "init_system_response"})
            print("Enviado init_system_response")
            response = await process_messages(data, websocket)
            print("Proceso de mensajes completado")
            await websocket.send_json({"action": "finish_system_response"})
            print("Enviado finish_system_response")
    except WebSocketDisconnect:
        print("Desconexión normal de WebSocket")
    except ConnectionClosed as e:
        print(f"Conexión cerrada con código: {e.code}, razón: {e.reason}")
    except json.JSONDecodeError:
        print("Error decodificando JSON del cliente")
        await websocket.close(code=1003)  # Código de cierre: datos no aceptables
    except Exception as e:
        print(f"Error inesperado: {str(e)}")
        await websocket.close(code=1011)  # Error interno
    finally:
        print("Conexión WebSocket finalizada")

async def process_messages( messages, websocket ):

    results = collection.query(
        query_texts=[ messages[ -1 ]["content"] ], 
        n_results=2 
    )

    pmsg = [ { "role": "system", "content": system_prompt + str( results["documents"][0] ) } ]
    print( json.dumps( pmsg + messages, indent=4) )
    completion_payload = {
        "messages": pmsg + messages
    }

    response = await client.chat.completions.create(
        top_p=0.9,
        temperature=0.6,
        model=MODEL,
        messages=completion_payload["messages"],
        stream=True
    )

    respStr = ""
    async for chunk in response:
        if (not chunk.choices[0] or
            not chunk.choices[0].delta or
            not chunk.choices[0].delta.content):
          continue

        await websocket.send_json( { "action": "append_system_response", "content": chunk.choices[0].delta.content } )

    return respStr


uvicorn.run(app, host="0.0.0.0", port=8000)