import sqlite3
import json
import os
import datetime

class ChatDatabase:
    def __init__(self, db_path="chat_history.db"):
        """
        Khởi tạo module cơ sở dữ liệu để lưu trữ lịch sử chat
        
        Args:
            db_path (str): Đường dẫn đến file cơ sở dữ liệu SQLite
        """
        self.db_path = db_path
        self.initialize_db()
    
    def initialize_db(self):
        """Khởi tạo cơ sở dữ liệu và các bảng cần thiết"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tạo bảng chứa các phiên hội thoại
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            title TEXT NOT NULL,
            last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Tạo bảng chứa các tin nhắn trong hội thoại
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            audio_path TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_session(self, title="Phiên hội thoại mới"):
        """
        Tạo một phiên hội thoại mới
        
        Args:
            title (str): Tiêu đề cho phiên hội thoại
            
        Returns:
            int: ID của phiên hội thoại mới
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO sessions (title, created_at, last_updated) VALUES (?, datetime('now'), datetime('now'))",
            (title,)
        )
        
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return session_id
    
    def add_message(self, session_id, role, content, audio_path=None):
        """
        Thêm tin nhắn vào phiên hội thoại
        
        Args:
            session_id (int): ID của phiên hội thoại
            role (str): Vai trò ('user' hoặc 'assistant')
            content (str): Nội dung tin nhắn
            audio_path (str, optional): Đường dẫn đến file audio (nếu có)
            
        Returns:
            int: ID của tin nhắn mới
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Thêm tin nhắn mới
        cursor.execute(
            "INSERT INTO messages (session_id, role, content, audio_path) VALUES (?, ?, ?, ?)",
            (session_id, role, content, audio_path)
        )
        
        # Cập nhật thời gian last_updated của phiên
        cursor.execute(
            "UPDATE sessions SET last_updated = datetime('now') WHERE session_id = ?",
            (session_id,)
        )
        
        message_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return message_id
    
    def get_session_history(self, session_id):
        """
        Lấy lịch sử hội thoại của một phiên
        
        Args:
            session_id (int): ID của phiên hội thoại
            
        Returns:
            list: Danh sách các tin nhắn trong phiên
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,)
        )
        
        messages = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return messages
    
    def get_all_sessions(self):
        """
        Lấy tất cả các phiên hội thoại
        
        Returns:
            list: Danh sách các phiên hội thoại
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM sessions ORDER BY last_updated DESC"
        )
        
        sessions = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return sessions
    
    def format_conversation_history(self, session_id):
        """
        Định dạng lịch sử hội thoại để sử dụng với LLM
        
        Args:
            session_id (int): ID của phiên hội thoại
            
        Returns:
            str: Lịch sử hội thoại được định dạng
        """
        messages = self.get_session_history(session_id)
        formatted_history = ""
        
        for msg in messages:
            role_name = "Học sinh" if msg["role"] == "user" else "Trợ lý"
            formatted_history += f"{role_name}: {msg['content']}\n"
        
        return formatted_history
    
    def delete_session(self, session_id):
        """
        Xóa một phiên hội thoại và các tin nhắn của nó
        
        Args:
            session_id (int): ID của phiên hội thoại
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Xóa tất cả tin nhắn thuộc phiên
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        
        # Xóa phiên
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        
        conn.commit()
        conn.close()