from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..services.ai import chat

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


# Simple in-memory session storage (use Redis in production)
sessions: dict[str, list[dict]] = {}


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Process a chat message and return AI response."""
    import uuid

    # Get or create session
    session_id = request.session_id or str(uuid.uuid4())
    messages = sessions.get(session_id, [])

    try:
        response_text, updated_messages = await chat(messages, request.message)
        sessions[session_id] = updated_messages

        return ChatResponse(
            response=response_text,
            session_id=session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/chat/{session_id}")
async def clear_session(session_id: str):
    """Clear a chat session."""
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "cleared"}
