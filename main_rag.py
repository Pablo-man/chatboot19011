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
MODEL = "llama3.2:3b"

def load_pdf_content(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

client = chromadb.Client()
try:
    collection = client.get_collection("all-my-documents")
    print("Colección existente cargada")
except:
    collection = client.create_collection("all-my-documents")
    print("Nueva colección creada")
    # Cargar el PDF
    pdf_text = load_pdf_content("iso19011.pdf")
    # Dividir el texto en partes
    chunks = [pdf_text[i:i+500] for i in range(0, len(pdf_text), 1000)]
    # Agregar al collection
    collection.add(
        documents=chunks,
        ids=[f"id{i}" for i in range(len(chunks))]
    )

system_prompt = """
Eres un especialista en la normativa ISO19011, te encargas de dar solución a distintos casos utilizando esta metodología. 

Tu tarea es:
1. Analizar el caso de estudio proporcionado.
2. Generar una respuesta basada en ISO19011.
3. Cuando el usuario proporcione su propia respuesta, deberás:
   - Comparar ambas respuestas (la tuya y la del usuario)
   - Evaluar los puntos fuertes de cada una
   - Proporcionar recomendaciones para mejorar

Utiliza la siguiente información de la ISO19011: """

comparison_prompt = """
Ahora debes realizar una comparación entre tu respuesta inicial (que el usuario no ha visto) y la respuesta proporcionada por el usuario.

Tu respuesta inicial fue:
"{ai_response}"

La respuesta del usuario es:
"{user_response}"

Por favor, realiza un análisis comparativo siguiendo estos puntos:
1. Compara ambos enfoques (similitudes y diferencias)
2. Identifica los puntos fuertes y áreas de mejora de cada respuesta
3. Proporciona 2-3 recomendaciones concretas para mejorar el abordaje del caso
4. Evalúa cuál de las dos respuestas se alinea mejor con ISO19011 y por qué

Mantén un tono constructivo y educativo, centrándote en el aprendizaje.
"""

client = AsyncOpenAI(
    base_url=ENDPOINT,
    api_key="not-needed"
)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
 
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return RedirectResponse("/static/index.html")

@app.websocket("/init")
async def init(websocket: WebSocket):
    print("Aceptando nueva conexión WebSocket...")
    await websocket.accept()
    print("Conexión WebSocket aceptada")
    
    # Para guardar las respuestas del AI
    stored_ai_responses = {}
    current_case_id = None
    
    try:
        print("Esperando mensaje JSON del cliente...")
        while True:
            data = await websocket.receive_json()
            print(f"Mensaje recibido: {json.dumps(data)}")
            
            # Identificar el tipo de acción
            action = data.get("action", "message")
            
            if action == "new_case":
                # Nuevo caso de estudio
                case_text = data.get("content", "")
                case_id = data.get("case_id", "default")
                current_case_id = case_id
                
                # Generamos la respuesta pero no la enviamos
                ai_response = await generate_ai_response(case_text)
                stored_ai_responses[case_id] = ai_response
                
                # Informar que el caso fue procesado
                await websocket.send_json({
                    "action": "case_processed",
                    "case_id": case_id
                })
                
            elif action == "submit_user_response":
                # Usuario envía su respuesta
                case_id = data.get("case_id", "default")
                user_response = data.get("content", "")
                
                if case_id in stored_ai_responses:
                    # Recuperamos la respuesta AI
                    ai_response = stored_ai_responses[case_id]
                    
                    # Enviar la respuesta AI guardada
                    await websocket.send_json({
                        "action": "ai_response",
                        "content": ai_response,
                        "case_id": case_id
                    })
                    
                    # Generar y enviar comparación
                    comparison = await generate_comparison(ai_response, user_response)
                    await websocket.send_json({
                        "action": "comparison",
                        "content": comparison,
                        "case_id": case_id
                    })
                else:
                    await websocket.send_json({
                        "action": "error",
                        "message": "No se encontró respuesta AI para este caso"
                    })
            
            elif action == "message":
                # Mensaje normal (para compatibilidad)
                messages = data
                await websocket.send_json({"action": "init_system_response"})
                response = await process_messages(messages, websocket)
                await websocket.send_json({"action": "finish_system_response"})
                
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

async def generate_ai_response(case_text):
    """Genera respuesta AI pero no la envía al usuario"""
    
    results = collection.query(
        query_texts=[case_text], 
        n_results=2 
    )
    
    messages = [
        {"role": "system", "content": system_prompt + str(results["documents"][0])},
        {"role": "user", "content": case_text}
    ]
    
    completion = await client.chat.completions.create(
        top_p=0.9,
        temperature=0.6,
        model=MODEL,
        messages=messages
    )
    
    return completion.choices[0].message.content

async def generate_comparison(ai_response, user_response):
    """Genera una comparación entre la respuesta AI y la del usuario"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": comparison_prompt.format(
            ai_response=ai_response, 
            user_response=user_response
        )}
    ]
    
    completion = await client.chat.completions.create(
        top_p=0.9,
        temperature=0.6,
        model=MODEL,
        messages=messages
    )
    
    return completion.choices[0].message.content

async def process_messages(messages, websocket):
    """Procesa mensajes en el modo normal (compatibilidad)"""
    
    results = collection.query(
        query_texts=[messages[-1]["content"]], 
        n_results=2 
    )

    pmsg = [{"role": "system", "content": system_prompt + str(results["documents"][0])}]
    print(json.dumps(pmsg + messages, indent=4))
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

        await websocket.send_json({"action": "append_system_response", "content": chunk.choices[0].delta.content})

    return respStr

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)