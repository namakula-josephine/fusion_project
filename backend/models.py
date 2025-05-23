from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import uuid

class MessageResult(BaseModel):
    predicted_class: Optional[str] = None
    confidence: Optional[str] = None
    explanation: Optional[str] = None
    treatment_plans: Optional[str] = None

class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    result: Optional[MessageResult] = None

class Chat(BaseModel):
    chat_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user: str
    title: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    messages: List[Message] = []

class ChatCreate(BaseModel):
    title: Optional[str] = "New Chat"

class ChatUpdate(BaseModel):
    title: str

class ChatResponse(BaseModel):
    chat_id: str
    title: str
    created_at: datetime
    last_message: Optional[str] = None
    message_count: int
