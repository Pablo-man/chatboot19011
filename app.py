from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from websockets.exceptions import ConnectionClosed

import os
import json
import uvicorn
import fitz  # PyMuPDF
from dotenv import load_dotenv
import gc  # For garbage collection
from typing import Dict, Any
import tempfile

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# LangChain and ChromaDB imports
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings

# Gemini integration
from langchain_google_genai import ChatGoogleGenerativeAI

# Constants
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
COLLECTION_NAME = "iso19011-collection"
PDF_PATH = "iso19011.pdf"
PERSISTENCE_DIRECTORY = "./chroma_db"
MAX_STORED_RESPONSES = 50  # Limit cached responses

# Initialize FastAPI app
app = FastAPI()

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Create persistent directory for ChromaDB
os.makedirs(PERSISTENCE_DIRECTORY, exist_ok=True)

# Initialize embeddings - load only when needed
embedding_function = None

# Initialize ChromaDB client with persistence
chroma_client = chromadb.PersistentClient(path=PERSISTENCE_DIRECTORY)

# Initialize LLM as None, load only when needed
llm = None

# LRU cache for AI responses
class LRUCache:
    def __init__(self, capacity: int):
        self.cache: Dict[str, str] = {}
        self.capacity = capacity
        self.order = []

    def get(self, key: str) -> str:
        if key in self.cache:
            # Move to the end to show it was recently used
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: str) -> None:
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            # Remove the least recently used item
            oldest = self.order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = value
        self.order.append(key)

# Initialize response cache
response_cache = LRUCache(MAX_STORED_RESPONSES)

def get_embedding_function():
    """Lazy load embedding function"""
    global embedding_function
    if embedding_function is None:
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embedding_function

def get_llm():
    """Lazy load LLM"""
    global llm
    if llm is None:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.6,
            top_p=0.9
        )
    return llm

def chunk_pdf(file_path, chunk_size=1000, chunk_overlap=100):
    """Process PDF in chunks to avoid loading entire content into memory"""
    doc = fitz.open(file_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    all_chunks = []
    # Process pages in batches to reduce memory usage
    batch_size = 5
    
    for i in range(0, len(doc), batch_size):
        batch_text = ""
        for page_num in range(i, min(i + batch_size, len(doc))):
            page = doc[page_num]
            batch_text += page.get_text()
            # Release page resources
            page = None
        
        # Split the batch text into chunks
        chunks = text_splitter.create_documents([batch_text])
        all_chunks.extend(chunks)
        
        # Explicitly clear batch text to help garbage collector
        batch_text = None
        gc.collect()
    
    doc.close()
    return all_chunks

def setup_vector_store():
    """Initialize or load existing vector store with document chunks"""
    try:
        # Try to get existing collection
        db = Chroma(
            client=chroma_client,
            collection_name=COLLECTION_NAME,
            embedding_function=get_embedding_function()
        )
        print(f"Existing collection '{COLLECTION_NAME}' loaded")
        return db
    except Exception as e:
        print(f"Creating new collection '{COLLECTION_NAME}': {e}")
        
        # Process PDF in chunks
        chunks = chunk_pdf(PDF_PATH)
        
        # Create new vector store
        db = Chroma.from_documents(
            documents=chunks,
            embedding=get_embedding_function(),
            collection_name=COLLECTION_NAME,
            client=chroma_client
        )
        print(f"Added {len(chunks)} chunks to ChromaDB")
        
        # Help garbage collector
        chunks = None
        gc.collect()
        
        return db

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

# Lazy-loaded vector store
vector_store = None

def get_vector_store():
    """Lazy load vector store"""
    global vector_store
    if vector_store is None:
        vector_store = setup_vector_store()
    return vector_store

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return RedirectResponse("/static/index.html")

@app.websocket("/init")
async def init(websocket: WebSocket):
    print("Accepting new WebSocket connection...")
    await websocket.accept()
    print("WebSocket connection accepted")
    
    current_case_id = None
    
    try:
        print("Waiting for JSON message from client...")
        while True:
            data = await websocket.receive_json()
            action = data.get("action", "message")
            
            if action == "new_case":
                # New case study
                case_text = data.get("content", "")
                case_id = data.get("case_id", "default")
                current_case_id = case_id
                
                # Generate response but don't send it
                ai_response = await generate_ai_response(case_text)
                response_cache.put(case_id, ai_response)
                
                # Inform that the case was processed
                await websocket.send_json({
                    "action": "case_processed",
                    "case_id": case_id
                })
                
                # Help garbage collector
                case_text = None
                gc.collect()
                
            elif action == "submit_user_response":
                # User submits their response
                case_id = data.get("case_id", "default")
                user_response = data.get("content", "")
                
                ai_response = response_cache.get(case_id)
                if ai_response:
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
                
                # Help garbage collector
                user_response = None
                gc.collect()
            
            elif action == "message":
                # Normal message (for compatibility)
                messages = data
                await websocket.send_json({"action": "init_system_response"})
                response = await process_messages(messages, websocket)
                await websocket.send_json({"action": "finish_system_response"})
                
                # Help garbage collector
                messages = None
                response = None
                gc.collect()
                
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
    docs = get_vector_store().similarity_search(case_text, k=2)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create prompt
    full_prompt = system_prompt + context
    
    # Generate response with Gemini
    response = await get_llm().ainvoke([
        {"role": "system", "content": full_prompt},
        {"role": "user", "content": case_text}
    ])
    
    # Help garbage collector
    docs = None
    context = None
    full_prompt = None
    gc.collect()
    
    return response.content

async def generate_comparison(ai_response, user_response):
    """Generate a comparison between the AI response and user response"""
    
    # Create comparison prompt
    prompt_text = comparison_prompt.format(
        ai_response=ai_response, 
        user_response=user_response
    )
    
    # Generate comparison with Gemini
    response = await get_llm().ainvoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text}
    ])
    
    # Help garbage collector
    prompt_text = None
    gc.collect()
    
    return response.content

async def process_messages(messages, websocket):
    """Process messages in normal mode (compatibility)"""
    
    # Query vector store for relevant information
    user_message = messages[-1]["content"]
    docs = get_vector_store().similarity_search(user_message, k=2)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create system message with context
    system_message = {"role": "system", "content": system_prompt + context}
    
    # Generate response with Gemini
    response = await get_llm().ainvoke([system_message] + messages)
    content = response.content
    
    # Stream in smaller chunks to avoid memory spikes
    chunk_size = 20  # Characters per chunk
    for i in range(0, len(content), chunk_size):
        chunk = content[i:i+chunk_size]
        await websocket.send_json({"action": "append_system_response", "content": chunk})
    
    # Help garbage collector
    docs = None
    context = None
    system_message = None
    gc.collect()
    
    return content

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when shutting down"""
    global vector_store, embedding_function, llm
    
    # Clear caches and references
    vector_store = None
    embedding_function = None
    llm = None
    response_cache.cache.clear()
    response_cache.order.clear()
    
    # Force garbage collection
    gc.collect()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
