"""
Multilingual Indian Language Chatbot — Main FastAPI Application
Powered by Ollama (Llama 3.2) + deep-translator + FastText

Run: uvicorn app:app --reload --port 8000
"""
import os
import sys
import uuid
from datetime import datetime
from contextlib import asynccontextmanager

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
import json
import sqlite3
from services.nlp_service.semantic_cache import SemanticCache

# --- Global pipeline instance ---
pipeline = None
semantic_cache = None

# Ensure data directory exists
os.makedirs("data", exist_ok=True)
db_path = "data/feedback.db"

def init_db():
    """Initialize SQLite database for continuous learning (RLHF)."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_message TEXT,
            bot_response TEXT,
            feedback_type INTEGER,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize NLP pipeline on startup."""
    global pipeline, semantic_cache
    print("🚀 Starting Multilingual Chatbot Server...")
    from services.nlp_service.pipeline.main_pipeline import NLPPipeline
    pipeline = NLPPipeline(model_name="llama3.2:3b")
    semantic_cache = SemanticCache(ollama_model="llama3.2:3b", similarity_threshold=0.5)
    print("✅ Server ready!")
    yield
    print("👋 Shutting down server...")


app = FastAPI(
    title="Multilingual Indian Language Chatbot",
    description="An intelligent chatbot supporting Hindi, Bengali, Marathi, Tamil, Telugu, Kannada, and English. Powered by Ollama (Llama 3.2).",
    version="1.0.0",
    lifespan=lifespan
)

# CORS — allow chat widget from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message text")
    session_id: Optional[str] = Field(None, description="Session ID for multi-turn")
    user_id: Optional[str] = Field("guest", description="User identifier")

class ChatResponse(BaseModel):
    response: str
    intent: str
    confidence: float
    language: str
    session_id: str
    entities: dict = {}
    timestamp: str

class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source: str = Field("auto", description="Source language code")
    target: str = Field("en", description="Target language code")

class KBEntryRequest(BaseModel):
    intent: str
    description: str = ""
    responses: dict = {}
    examples: dict = {}
    dynamic: bool = False

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = Field("hi", description="Language code")

class FeedbackRequest(BaseModel):
    session_id: str
    user_message: str
    bot_response: str
    feedback_type: int  # 1 for up, -1 for down


# ============================================================
# CHAT ENDPOINTS
# ============================================================

@app.post("/chat/message", response_model=ChatResponse)
async def chat_message(req: ChatRequest):
    """Process a chat message and return bot response."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="NLP Pipeline not ready")

    # Create session if needed
    session_id = req.session_id or str(uuid.uuid4())

    # Process message through pipeline
    result = pipeline.process(req.message, session_id)

    return ChatResponse(
        response=result["response"],
        intent=result["intent"],
        confidence=result["confidence"],
        language=result["language"],
        session_id=session_id,
        entities=result.get("entities", {}),
        timestamp=datetime.now().isoformat()
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Stream chat response in real-time using Server-Sent Events."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="NLP Pipeline not ready")

    session_id = req.session_id or str(uuid.uuid4())
    language, _ = pipeline.detect_language(req.message)

    # Check Semantic Cache First
    if semantic_cache:
        cached_response = semantic_cache.get(req.message)
        if cached_response:
            pipeline._update_session(session_id, req.message, cached_response["response"],
                                     cached_response["intent"], language)
            async def instant_stream():
                data = json.dumps({
                    "token": cached_response["response"],
                    "intent": cached_response["intent"],
                    "language": language,
                    "done": True
                })
                yield f"data: {data}\n\n"
            return StreamingResponse(instant_stream(), media_type="text/event-stream")

    # Check quick match first (instant response)
    quick = pipeline._quick_intent_match(req.message, language)
    if quick:
        # Save quick match to session history for multi-turn memory
        pipeline._update_session(session_id, req.message, quick["response"],
                                 quick["intent"], language)
        async def instant_stream():
            data = json.dumps({
                "token": quick["response"],
                "intent": quick["intent"],
                "language": language,
                "done": True
            })
            yield f"data: {data}\n\n"
        return StreamingResponse(instant_stream(), media_type="text/event-stream")

    # Get session history for multi-turn context
    history = pipeline._get_session_history(session_id)

    # Stream from Ollama with conversation history
    def generate():
        full_text = ""
        
        # Inject RAG Context if available
        message_to_send = req.message
        if hasattr(pipeline, "rag") and pipeline.rag and pipeline.rag.has_document(session_id):
            context = pipeline.rag.search(req.message, session_id)
            if context:
                message_to_send = f"Answer the User Question based ONLY on the following Document Context. If the answer is not in the context, say 'I don't know based on the document'.\n\nDocument Context:\n{context}\n\nUser Question:\n{req.message}"

        for token in pipeline.ollama.stream_response(
            message_to_send, language, history=history
        ):
            full_text += token
            data = json.dumps({"token": token, "done": False})
            yield f"data: {data}\n\n"

        # After streaming completes, save the full exchange to session memory
        # Note: We save the original req.message to history, not the massive context chunk!
        pipeline._update_session(session_id, req.message, full_text,
                                 "chat", language)

        # Cache the response for future identical/similar queries
        if semantic_cache:
            semantic_cache.set(req.message, {
                "response": full_text,
                "intent": "chat"
            })

        # Send final event with metadata
        yield f"data: {json.dumps({'token': '', 'intent': 'chat', 'language': language, 'done': True})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/chat/session/start")
async def start_session():
    """Create a new chat session."""
    session_id = str(uuid.uuid4())
    return {"session_id": session_id, "created_at": datetime.now().isoformat()}


@app.post("/chat/session/end")
async def end_session(session_id: str):
    """End a chat session."""
    if pipeline:
        pipeline.clear_session(session_id)
    return {"status": "session_ended", "session_id": session_id}


@app.post("/upload-document")
async def upload_document(session_id: str, file: UploadFile = File(...)):
    """Upload a PDF document for RAG context."""
    if not pipeline or not hasattr(pipeline, "rag") or not pipeline.rag:
        raise HTTPException(status_code=503, detail="RAG Service not available. Check dependencies.")
        
    import shutil
    os.makedirs("data/uploads", exist_ok=True)
    file_path = f"data/uploads/{session_id}_{file.filename}"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        success = pipeline.rag.process_document(file_path, session_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process document")
            
        return {"status": "success", "message": f"Document {file.filename} processed and indexed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Save user feedback for Continuous Learning (RLHF)."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO user_feedback (session_id, user_message, bot_response, feedback_type, timestamp) VALUES (?, ?, ?, ?, ?)",
            (req.session_id, req.user_message, req.bot_response, req.feedback_type, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
        return {"status": "success", "message": "Feedback recorded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# TRANSLATION ENDPOINT
# ============================================================

@app.post("/translate")
async def translate(req: TranslateRequest):
    """Translate text between Indian languages."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")

    # Auto-detect source language if set to "auto"
    source = req.source
    if source == "auto":
        source, _ = pipeline.detect_language(req.text)

    result = pipeline.translator.translate(req.text, source, req.target)
    return {
        "original": req.text,
        "translated": result,
        "source": source,
        "target": req.target
    }


# ============================================================
# LANGUAGE DETECTION ENDPOINT
# ============================================================

@app.post("/detect-language")
async def detect_language(text: str):
    """Detect the language of input text."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")

    lang, confidence = pipeline.detect_language(text)
    lang_names = {
        "hi": "Hindi", "bn": "Bengali", "mr": "Marathi",
        "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "en": "English"
    }
    return {
        "language_code": lang,
        "language_name": lang_names.get(lang, lang),
        "confidence": confidence
    }


# ============================================================
# VOICE ENDPOINTS (TTS)
# ============================================================

@app.post("/voice/synthesize")
async def synthesize_speech(req: TTSRequest):
    """Convert text to speech using gTTS."""
    try:
        from gtts import gTTS
        import io

        tts = gTTS(text=req.text, lang=req.language)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        return StreamingResponse(
            audio_buffer,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=response.mp3"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")


# ============================================================
# ADMIN / KNOWLEDGE BASE ENDPOINTS
# ============================================================

@app.get("/admin/knowledge-base")
async def list_kb_entries():
    """List all knowledge base entries."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")

    entries = []
    for intent_name, data in pipeline.kb.intents.items():
        entries.append({
            "intent": intent_name,
            "description": data.get("description", ""),
            "dynamic": data.get("dynamic", False),
            "languages": list(data.get("responses", {}).keys()),
            "example_count": sum(len(v) for v in data.get("examples", {}).values())
        })
    return {"total": len(entries), "entries": entries}


@app.get("/admin/knowledge-base/{intent_name}")
async def get_kb_entry(intent_name: str):
    """Get a specific knowledge base entry."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")

    entry = pipeline.kb.get_intent_details(intent_name)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Intent '{intent_name}' not found")
    return entry


@app.post("/admin/knowledge-base")
async def add_kb_entry(entry: KBEntryRequest):
    """Add or update a knowledge base entry."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")

    pipeline.kb.add_intent(entry.dict())
    return {"status": "success", "intent": entry.intent}


@app.delete("/admin/knowledge-base/{intent_name}")
async def delete_kb_entry(intent_name: str):
    """Delete a knowledge base entry."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")

    if pipeline.kb.delete_intent(intent_name):
        return {"status": "deleted", "intent": intent_name}
    raise HTTPException(status_code=404, detail=f"Intent '{intent_name}' not found")


# ============================================================
# ADMIN / ANALYTICS ENDPOINTS
# ============================================================

@app.get("/admin/analytics")
async def get_analytics():
    """Get chatbot analytics overview."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")

    sessions = pipeline.get_all_sessions()
    intent_counts = {}
    lang_counts = {}
    total_messages = 0

    for sid, info in sessions.items():
        lang = info.get("language", "unknown")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
        intent = info.get("last_intent", "unknown")
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
        total_messages += info.get("turns", 0)

    return {
        "total_sessions": len(sessions),
        "total_messages": total_messages,
        "intent_distribution": intent_counts,
        "language_distribution": lang_counts
    }


@app.get("/admin/sessions")
async def get_sessions():
    """Get all active sessions."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"sessions": pipeline.get_all_sessions()}


# ============================================================
# SUPPORTED LANGUAGES
# ============================================================

@app.get("/languages")
async def get_languages():
    """List supported languages."""
    return {
        "supported": [
            {"code": "hi", "name": "Hindi", "script": "Devanagari", "native": "हिंदी"},
            {"code": "bn", "name": "Bengali", "script": "Bengali", "native": "বাংলা"},
            {"code": "mr", "name": "Marathi", "script": "Devanagari", "native": "मराठी"},
            {"code": "ta", "name": "Tamil", "script": "Tamil", "native": "தமிழ்"},
            {"code": "te", "name": "Telugu", "script": "Telugu", "native": "తెలుగు"},
            {"code": "kn", "name": "Kannada", "script": "Kannada", "native": "ಕನ್ನಡ"},
            {"code": "en", "name": "English", "script": "Latin", "native": "English"}
        ]
    }


# ============================================================
# HEALTH & INFO
# ============================================================

@app.get("/health")
async def health_check():
    """System health check."""
    backend = pipeline.ollama.active_backend() if pipeline else "none"
    return {
        "status": "healthy" if backend != "none" else "degraded",
        "llm_backend": backend,  # kept for internal frontend logic
        "model": "BhashaMitra AI v1.0",
        "model_available": backend != "none",
        "kb_intents": len(pipeline.kb.intents) if pipeline else 0,
        "active_sessions": len(pipeline.sessions) if pipeline else 0,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================
# FRONTEND SERVING
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def serve_chat_widget():
    """Serve the chat widget frontend."""
    widget_path = os.path.join(frontend_dir, "chat-widget", "index.html")
    if os.path.exists(widget_path):
        return FileResponse(widget_path)
    return HTMLResponse("<h1>Multilingual Chatbot API</h1><p>Visit <a href='/docs'>/docs</a> for API documentation.</p>")


@app.get("/admin", response_class=HTMLResponse)
async def serve_admin_dashboard():
    """Serve the admin dashboard."""
    admin_path = os.path.join(frontend_dir, "admin-dashboard", "index.html")
    if os.path.exists(admin_path):
        return FileResponse(admin_path)
    return HTMLResponse("<h1>Admin Dashboard</h1><p>Coming soon.</p>")
