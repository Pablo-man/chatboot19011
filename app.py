from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from websockets.exceptions import ConnectionClosed

import os
import json
import uvicorn
import fitz  # PyMuPDF
from dotenv import load_dotenv

import os

load_dotenv()

GEMINI_API_KEY= os.getenv("GEMINI_API_KEY")

# LangChain and ChromaDB imports
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings

# Gemini integration
from langchain_google_genai import ChatGoogleGenerativeAI

# Set up environment variables (you can also use .env file with python-dotenv)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
COLLECTION_NAME = "iso19011-collection"
PDF_PATH = "iso19011.pdf"

# Initialize FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize embeddings
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize ChromaDB
chroma_client = chromadb.Client()

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GEMINI_API_KEY,
    temperature=0.6,
    top_p=0.9
)

def load_pdf_content(file_path):
    """Load and return raw text from PDF"""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def setup_vector_store():
    """Initialize or load existing vector store with document chunks"""
    try:
        # Try to get existing collection
        db = Chroma(
            client=chroma_client,
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_function
        )
        print(f"Existing collection '{COLLECTION_NAME}' loaded")
        return db
    except:
        print(f"Creating new collection '{COLLECTION_NAME}'")
        # Load PDF document
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        # Create new vector store
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            collection_name=COLLECTION_NAME,
            client=chroma_client
        )
        print(f"Added {len(chunks)} chunks to ChromaDB")
        return db

# Setup vector store
vector_store = setup_vector_store()

# Define system prompts
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

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return RedirectResponse("/static/index.html")

@app.websocket("/init")
async def init(websocket: WebSocket):
    print("Accepting new WebSocket connection...")
    await websocket.accept()
    print("WebSocket connection accepted")
    
    # For storing AI responses
    stored_ai_responses = {}
    current_case_id = None
    
    try:
        print("Waiting for JSON message from client...")
        while True:
            data = await websocket.receive_json()
            print(f"Message received: {json.dumps(data)}")
            
            # Identify action type
            action = data.get("action", "message")
            
            if action == "new_case":
                # New case study
                case_text = data.get("content", "")
                case_id = data.get("case_id", "default")
                current_case_id = case_id
                
                # Generate response but don't send it
                ai_response = await generate_ai_response(case_text)
                stored_ai_responses[case_id] = ai_response
                
                # Inform that the case was processed
                await websocket.send_json({
                    "action": "case_processed",
                    "case_id": case_id
                })
                
            elif action == "submit_user_response":
                # User submits their response
                case_id = data.get("case_id", "default")
                user_response = data.get("content", "")
                
                if case_id in stored_ai_responses:
                    # Retrieve stored AI response
                    ai_response = stored_ai_responses[case_id]
                    
                    # Send stored AI response
                    await websocket.send_json({
                        "action": "ai_response",
                        "content": ai_response,
                        "case_id": case_id
                    })
                    
                    # Generate and send comparison
                    comparison = await generate_comparison(ai_response, user_response)
                    await websocket.send_json({
                        "action": "comparison",
                        "content": comparison,
                        "case_id": case_id
                    })
                else:
                    await websocket.send_json({
                        "action": "error",
                        "message": "No AI response found for this case"
                    })
            
            elif action == "message":
                # Normal message (for compatibility)
                messages = data
                await websocket.send_json({"action": "init_system_response"})
                response = await process_messages(messages, websocket)
                await websocket.send_json({"action": "finish_system_response"})
                
    except WebSocketDisconnect:
        print("Normal WebSocket disconnection")
    except ConnectionClosed as e:
        print(f"Connection closed with code: {e.code}, reason: {e.reason}")
    except json.JSONDecodeError:
        print("Error decoding JSON from client")
        await websocket.close(code=1003)  # Close code: Unacceptable data
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        await websocket.close(code=1011)  # Internal error
    finally:
        print("WebSocket connection terminated")

async def generate_ai_response(case_text):
    """Generate AI response but don't send it to the user"""
    
    # Query vector store for relevant information
    docs = vector_store.similarity_search(case_text, k=2)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create prompt
    full_prompt = system_prompt + context
    
    # Generate response with Gemini
    response = await llm.ainvoke([
        {"role": "system", "content": full_prompt},
        {"role": "user", "content": case_text}
    ])
    
    return response.content

async def generate_comparison(ai_response, user_response):
    """Generate a comparison between the AI response and user response"""
    
    # Create comparison prompt
    prompt_text = comparison_prompt.format(
        ai_response=ai_response, 
        user_response=user_response
    )
    
    # Generate comparison with Gemini
    response = await llm.ainvoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text}
    ])
    
    return response.content

async def process_messages(messages, websocket):
    """Process messages in normal mode (compatibility)"""
    
    # Query vector store for relevant information
    user_message = messages[-1]["content"]
    docs = vector_store.similarity_search(user_message, k=2)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create system message with context
    system_message = {"role": "system", "content": system_prompt + context}
    
    # Print messages for debugging
    print(json.dumps([system_message] + messages, indent=4))
    
    # Stream responses using Gemini (note: streaming implementation depends on specific SDK version)
    # This is a simplified version that doesn't actually stream but simulates it with chunks
    response = await llm.ainvoke([system_message] + messages)
    
    # Split response into chunks to simulate streaming
    content = response.content
    chunk_size = 20  # Characters per chunk
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
    
    for chunk in chunks:
        await websocket.send_json({"action": "append_system_response", "content": chunk})
    
    return content