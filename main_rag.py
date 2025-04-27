from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from websockets.exceptions import ConnectionClosed

import chromadb
import json
import uvicorn

ENDPOINT = "http://127.0.0.1:39281/v1"
#MODEL = "phi-3.5:3b-gguf-q4-km"
#MODEL = "deepseek-r1-distill-qwen-14b:14b-gguf-q4-km"
MODEL = "llama3.2:3b"

client = chromadb.Client()

collection = client.create_collection("all-my-documents")

collection.add(
    documents=[
        "La empresa Lostsys se dedica a ofrecer servícios y productos a empresas sobre informática corporativa como software de gestión, CRMs, ERPs, portales corporativos, eCommerce, formación, DevOps, etc.",
        "En Lostsys podemos ayudarte ha mejorar tus procesos de CI/CD con nuestros productos y servícios de DevOps.",
        "En Lostsys podemos ayudarte a digitalizarte con nuestros servícios de desarrollo de aplicaciones corporativas.",
        "En Lostsys te podemos entrenar y formar a múltiples áreas de la informática corporativa como desarrollo, Data, IA o DevOps.",
        "En Lostsys te podemos desarrollar una tienda online para vender por todo el mundo y mas allà.",
        "En Lostsys te podemos desarrollar un eCommerce para vender por todo el mundo y mas allà",
    ],
    ids=["id1", "id2","id3", "id4","id5", "id6"]
)

system_prompt = """
Eres un asistente de la empresa Lostsys que ayuda a sus clientes a encontrar el servicio o producto que les interesa. Sigue estas instrucciones:
- Ofrece respuestas cortas y concisas de no mas de 25 palabras. 
- No ofrezcas consejos, productos o servícios de terceros.
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