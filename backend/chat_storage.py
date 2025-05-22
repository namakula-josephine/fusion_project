import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from backend.models import Chat, Message

class ChatStorage:
    def __init__(self, data_dir: str = "data/chats"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def _get_user_file_path(self, username: str) -> str:
        return os.path.join(self.data_dir, f"{username}_chats.json")
    
    def _load_user_chats(self, username: str) -> Dict[str, Chat]:
        file_path = self._get_user_file_path(username)
        if not os.path.exists(file_path):
            return {}
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            return {
                chat_id: Chat(**chat_data)
                for chat_id, chat_data in data.items()
            }
    
    def _save_user_chats(self, username: str, chats: Dict[str, Chat]):
        file_path = self._get_user_file_path(username)
        with open(file_path, 'w') as f:
            json.dump(
                {chat_id: chat.model_dump() for chat_id, chat in chats.items()},
                f,
                default=str
            )
    
    def create_chat(self, username: str, title: str = "New Chat") -> Chat:
        chats = self._load_user_chats(username)
        new_chat = Chat(user=username, title=title)
        chats[new_chat.chat_id] = new_chat
        self._save_user_chats(username, chats)
        return new_chat
    
    def get_user_chats(self, username: str) -> List[Chat]:
        chats = self._load_user_chats(username)
        return list(chats.values())
    
    def get_chat(self, username: str, chat_id: str) -> Optional[Chat]:
        chats = self._load_user_chats(username)
        return chats.get(chat_id)
    
    def update_chat(self, username: str, chat_id: str, title: str) -> Optional[Chat]:
        chats = self._load_user_chats(username)
        if chat_id not in chats:
            return None
            
        chats[chat_id].title = title
        self._save_user_chats(username, chats)
        return chats[chat_id]
    
    def delete_chat(self, username: str, chat_id: str) -> bool:
        chats = self._load_user_chats(username)
        if chat_id not in chats:
            return False
            
        del chats[chat_id]
        self._save_user_chats(username, chats)
        return True
    
    def add_message(self, username: str, chat_id: str, role: str, content: str) -> Optional[Chat]:
        chats = self._load_user_chats(username)
        if chat_id not in chats:
            return None
            
        message = Message(role=role, content=content)
        chats[chat_id].messages.append(message)
        self._save_user_chats(username, chats)
        return chats[chat_id]
