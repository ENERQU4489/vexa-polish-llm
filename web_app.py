"""
Vexa Polish LLM Web Interface
FastAPI-based web application for interactive chat with the model
"""

import os
import json
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Import Vexa components
from src.core.graph import AntGraph
from src.core.engine import VexaEngine
from src.utils.tokenizer import VexaTokenizer
from src.integration.llm_interface import VexaLLM
import yaml

app = FastAPI(
    title="Vexa Polish LLM",
    description="Web interface for Vexa Polish LLM using ACO algorithm",
    version="1.0.0"
)

# Templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for model components
vexa_llm = None
config = None

class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.8
    max_length: int = 200

class FeedbackRequest(BaseModel):
    rating: float

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = "config/hyperparams.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def initialize_model():
    """Initialize the Vexa LLM model."""
    global vexa_llm, config

    try:
        config = load_config()

        # Check if model data exists
        vocab_path = "data/vocab.json"
        graph_path = "data/graph.npy"
        engine_checkpoint = "data/checkpoints"

        if not os.path.exists(vocab_path):
            raise FileNotFoundError("Vocabulary file not found. Please train the model first.")

        if not os.path.exists(graph_path):
            raise FileNotFoundError("Graph file not found. Please train the model first.")

        # Load components
        tokenizer = VexaTokenizer(vocab_path=vocab_path)
        graph = AntGraph.load_from_file(graph_path)
        engine = VexaEngine(graph=graph, config=config)

        # Initialize LLM interface
        vexa_llm = VexaLLM(
            graph=graph,
            tokenizer=tokenizer,
            engine=engine,
            config=config
        )

        print("✓ Model loaded successfully")

    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        vexa_llm = None

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    initialize_model()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main chat interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/status")
async def get_status():
    """Get model status."""
    if vexa_llm is None:
        return {
            "status": "not_loaded",
            "message": "Model not loaded. Please train the model first."
        }

    stats = vexa_llm.get_stats()
    return {
        "status": "ready",
        "stats": stats
    }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat with the model."""
    if vexa_llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        response = vexa_llm.chat(request.message)

        return {
            "response": response,
            "stats": vexa_llm.get_stats()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/api/feedback")
async def provide_feedback(request: FeedbackRequest):
    """Provide feedback on last response."""
    if vexa_llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        vexa_llm.provide_feedback(request.rating)
        return {"message": f"Feedback saved: {request.rating:.2f}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback failed: {str(e)}")

@app.post("/api/clear")
async def clear_history():
    """Clear conversation history."""
    if vexa_llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        vexa_llm.clear_history()
        return {"message": "Conversation history cleared"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")

@app.get("/api/stats")
async def get_stats():
    """Get detailed statistics."""
    if vexa_llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        return vexa_llm.get_stats()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    print("🐜 Starting Vexa Polish LLM Web Interface")
    print("📱 Open http://localhost:8000 in your browser")

    uvicorn.run(
        "web_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
