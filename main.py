from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from websockets.exceptions import ConnectionClosed

import uvicorn

ENDPOINT = "http://127.0.0.1:39281/v1"
#MODEL = "phi-3.5:3b-gguf-q4-km"
#MODEL = "deepseek-r1-distill-qwen-14b:14b-gguf-q4-km"
MODEL = "llama3.2:3b"

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
async def init( websocket: WebSocket ):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()

            await websocket.send_json( { "action": "init_system_response" } )
            response = await process_messages( data, websocket )
            await websocket.send_json( { "action": "finish_system_response" } )
    except (WebSocketDisconnect, ConnectionClosed):
        print( "Conexión cerrada" )

async def process_messages( messages, websocket ):
    completion_payload = {
        "messages": messages
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