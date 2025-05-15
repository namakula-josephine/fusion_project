from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, status, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from retriever import search_rag
import os
import json
import numpy as np
from dotenv import load_dotenv
import openai
import shutil
from datetime import datetime
from typing import List, Optional, Dict
import uuid

# Load environment variables
load_dotenv('secrets.env')

# Session management
active_sessions = {}  # Store active sessions

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# User models
class User(BaseModel):
    username: str
    password: str
    email: Optional[str] = None
    full_name: Optional[str] = None

# Login model
class LoginData(BaseModel):
    username: str
    password: str

# Chat models
class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: str
    result: Optional[Dict] = None

class ChatSession(BaseModel):
    id: str
    title: str
    messages: List[ChatMessage]
    created_at: str
    user_id: str

# In-memory storage
users_db = {}
chat_sessions = {}

# Load the vision model
try:
    vision_model = load_model('model/potato_classification_model.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    vision_model = None

# Define class names for the vision model
class_names = ['Early Blight', 'Healthy', 'Late Blight']

# Authentication functions
def authenticate_user(username: str, password: str):
    user = users_db.get(username)
    if not user or user.password != password:
        return None
    return user

async def get_current_user(request: Request):
    session_id = request.headers.get('authorization')
    if not session_id or session_id not in active_sessions:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    username = active_sessions[session_id]
    return users_db[username]

# Helper functions
def preprocess_image(image_path, img_size=(224, 224)):
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            image = img.convert('RGB').resize(img_size)
            image = np.array(image) / 255.0
            return np.expand_dims(image, axis=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def predict_image(image_path):
    if vision_model is None:
        raise HTTPException(status_code=500, detail="Vision model not loaded")
    try:
        image = preprocess_image(image_path)
        prediction = vision_model.predict(image, verbose=0)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        return predicted_class, confidence
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Auth endpoints
@app.post("/api/register")
async def register_user(
    username: str = Form(...),
    password: str = Form(...),
    email: str = Form(...)
):
    if username in users_db:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    users_db[username] = User(
        username=username,
        password=password,
        email=email
    )
    return {"message": "Registration successful"}

@app.post("/api/login")
async def login(username: str = Form(...), password: str = Form(...)):
    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    session_id = str(uuid.uuid4())
    active_sessions[session_id] = user.username
    return {
        "session_id": session_id,
        "username": user.username,
        "message": "Login successful"
    }

@app.post("/api/logout")
async def logout(request: Request):
    session_id = request.headers.get('authorization')
    if session_id and session_id in active_sessions:
        del active_sessions[session_id]
    return {"message": "Logout successful"}

# Chat endpoints
@app.post("/chat-sessions")
async def create_chat_session(
    title: str = Form(...),
    session_id: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    if session_id not in active_sessions:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    chat_session_id = str(uuid.uuid4())
    chat_sessions[chat_session_id] = ChatSession(
        id=chat_session_id,
        title=title,
        messages=[],
        created_at=datetime.now().isoformat(),
        user_id=current_user.username
    )
    return {"session_id": chat_session_id, "title": title}

@app.get("/chat-sessions")
async def get_chat_sessions(current_user: User = Depends(get_current_user)):
    user_sessions = [
        {
            "id": session.id,
            "title": session.title,
            "created_at": session.created_at,
            "message_count": len(session.messages)
        }
        for session in chat_sessions.values()
        if session.user_id == current_user.username
    ]
    return {"sessions": sorted(user_sessions, key=lambda x: x["created_at"], reverse=True)}

@app.post("/query")
async def query(
    current_user: User = Depends(get_current_user),
    image: Optional[UploadFile] = File(None),
    question: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None)
):
    try:
        if not session_id:
            session_id = str(uuid.uuid4())
            chat_sessions[session_id] = ChatSession(
                id=session_id,
                title="New Chat",
                messages=[],
                created_at=datetime.now().isoformat(),
                user_id=current_user.username
            )
        elif session_id not in chat_sessions:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        session = chat_sessions[session_id]
        
        if session.user_id != current_user.username:
            raise HTTPException(status_code=403, detail="Not authorized to access this chat session")

        if image:
            # Handle image analysis
            os.makedirs('temp', exist_ok=True)
            image_path = f"temp/{image.filename}"
            try:
                with open(image_path, "wb") as buffer:
                    shutil.copyfileobj(image.file, buffer)
                
                predicted_class, confidence = predict_image(image_path)
                
                explanation_query = f"The model predicts the plant has {predicted_class} with {confidence:.2%} confidence. Can you explain what this means?"
                explanation_response = await openai.ChatCompletion.acreate(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that provides information about potato plant diseases and treatments."},
                        {"role": "user", "content": explanation_query}
                    ]
                )
                explanation = explanation_response.choices[0].message.content.strip()

                treatment_query = f"{predicted_class} treatment potato plant"
                relevant_docs = search_rag(treatment_query)
                augmented_treatment_query = f"Based on the following documents: {relevant_docs}, provide detailed treatment plans for {predicted_class} in potato plants."
                
                treatment_response = await openai.ChatCompletion.acreate(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that provides information about potato plant diseases and treatments."},
                        {"role": "user", "content": augmented_treatment_query}
                    ]
                )
                treatment_plans = treatment_response.choices[0].message.content.strip()

                session.messages.extend([
                    ChatMessage(
                        role="user",
                        content=f"Uploaded image: {image.filename}",
                        timestamp=datetime.now().isoformat()
                    ),
                    ChatMessage(
                        role="assistant",
                        content="Here's my analysis:",
                        timestamp=datetime.now().isoformat(),
                        result={
                            "predicted_class": predicted_class,
                            "confidence": f"{confidence:.2%}",
                            "explanation": explanation,
                            "treatment_plans": treatment_plans
                        }
                    )
                ])

                return {
                    "predicted_class": predicted_class,
                    "confidence": f"{confidence:.2%}",
                    "explanation": explanation,
                    "treatment_plans": treatment_plans,
                    "session_id": session_id,
                    "messages": [msg.dict() for msg in session.messages]
                }
            finally:
                if os.path.exists(image_path):
                    os.remove(image_path)

        elif question:
            session.messages.append(ChatMessage(
                role="user",
                content=question,
                timestamp=datetime.now().isoformat()
            ))

            relevant_docs = search_rag(question)
            augmented_query = f"Based on the following documents: {relevant_docs}, answer the question: {question}"
            
            answer_response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides information about potato plant diseases and treatments."},
                    {"role": "user", "content": augmented_query}
                ]
            )
            answer = answer_response.choices[0].message.content.strip()

            session.messages.append(ChatMessage(
                role="assistant",
                content=answer,
                timestamp=datetime.now().isoformat()
            ))

            return {
                "answer": answer,
                "session_id": session_id,
                "messages": [msg.dict() for msg in session.messages]
            }

        raise HTTPException(status_code=400, detail="No valid input provided")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat-history")
async def get_chat_history(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    try:
        if session_id not in chat_sessions:
            raise HTTPException(status_code=404, detail="Chat session not found")
            
        session = chat_sessions[session_id]
        
        if session.user_id != current_user.username:
            raise HTTPException(status_code=403, detail="Not authorized to access this chat history")
        
        return {
            "messages": [msg.dict() for msg in session.messages],
            "session_info": {
                "id": session.id,
                "title": session.title,
                "created_at": session.created_at
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Replace the existing message endpoints with these corrected versions

@app.post("/{session_id}/messages")
async def send_message(
    session_id: str,
    message: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    try:
        if session_id not in chat_sessions:
            raise HTTPException(status_code=404, detail="Chat session not found")
            
        session = chat_sessions[session_id]
        
        if session.user_id != current_user.username:
            raise HTTPException(status_code=403, detail="Not authorized to access this chat session")

        # Add user message
        session.messages.append(ChatMessage(
            role="user",
            content=message,
            timestamp=datetime.now().isoformat()
        ))

        # Get response using RAG
        relevant_docs = search_rag(message)
        augmented_query = f"Based on the following documents: {relevant_docs}, answer the question: {message}"
        
        answer_response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides information about potato plant diseases and treatments."},
                {"role": "user", "content": augmented_query}
            ]
        )
        answer = answer_response.choices[0].message.content.strip()

        # Add assistant response
        session.messages.append(ChatMessage(
            role="assistant",
            content=answer,
            timestamp=datetime.now().isoformat()
        ))

        return {
            "session_id": session_id,
            "messages": [msg.dict() for msg in session.messages]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/{session_id}/messages")
async def get_messages(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    try:
        if session_id not in chat_sessions:
            raise HTTPException(status_code=404, detail="Chat session not found")
            
        session = chat_sessions[session_id]
        
        if session.user_id != current_user.username:
            raise HTTPException(status_code=403, detail="Not authorized to access this chat session")

        return {
            "session_id": session_id,
            "messages": [msg.dict() for msg in session.messages]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)