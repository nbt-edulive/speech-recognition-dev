from flask import Flask, render_template, request, jsonify, send_file
import os
import tempfile
import uuid
import time
from dotenv import load_dotenv
from datetime import datetime
import json

# Import các module
from stt_v1 import SpeechToText
# from stt import SpeechToText
from tts import TextToSpeech
from llm import GeminiLLM
from database import ChatDatabase

# Load biến môi trường
load_dotenv()

# Khởi tạo Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['CSS_FOLDER'] = 'static/css'
app.config['JS_FOLDER'] = 'static/js'

# Đảm bảo các thư mục cần thiết tồn tại
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['CSS_FOLDER'], exist_ok=True)
os.makedirs(app.config['JS_FOLDER'], exist_ok=True)

# Khởi tạo các module
stt_module = SpeechToText(language="vi")
# stt_module = SpeechToText(model_size="small", device="cpu") #model_size = tiny base small medium large
tts_module = TextToSpeech(api_key=os.environ.get("ELEVEN_API_KEY"))
llm_module = GeminiLLM(api_key=os.environ.get("GEMINI_API_KEY_1"))
db = ChatDatabase()

# Trang chủ
@app.route('/')
def index():
    # Lấy danh sách các phiên chat
    sessions = db.get_all_sessions()
    # Lấy danh sách các giọng nói có sẵn
    voices = tts_module.get_available_voices()
    
    return render_template('index.html', 
                          sessions=sessions, 
                          voices=voices)

# Flask sẽ tự động phục vụ các file tĩnh từ thư mục static

# API endpoint để xử lý audio được gửi lên
@app.route('/api/process-audio', methods=['POST'])
def process_audio():
    try:
        # Lấy session_id từ form
        session_id = request.form.get('session_id', None)
        if not session_id or session_id == 'new':
            # Tạo phiên mới nếu không có hoặc yêu cầu tạo mới
            session_title = f"Hội thoại {datetime.now().strftime('%d/%m/%Y %H:%M')}"
            session_id = db.create_session(session_title)
        else:
            session_id = int(session_id)
        
        # Kiểm tra xem có file audio được gửi không
        if 'audio' not in request.files:
            return jsonify({'error': 'Không tìm thấy file audio'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'Không có file được chọn'}), 400
        
        # Lưu file audio
        audio_filename = f"{uuid.uuid4()}.wav"
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        audio_file.save(audio_path)
        
        # Chuyển đổi audio thành text
        user_text = stt_module.transcribe(audio_path)
        
        # Lưu tin nhắn của người dùng vào database
        db.add_message(session_id, "user", user_text, audio_path)
        
        # Lấy lịch sử hội thoại
        conversation_history = db.format_conversation_history(session_id)
        
        # Truy vấn Gemini để lấy phản hồi
        assistant_response = llm_module.get_response(user_text, conversation_history)
        
        # Lưu tin nhắn của assistant vào database
        db.add_message(session_id, "assistant", assistant_response)
        
        # Tạo giọng nói từ phản hồi của assistant
        voice_name = request.form.get('voice', 'elli')
        output_filename = f"{uuid.uuid4()}.mp3"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        tts_module.text_to_speech(
            text=assistant_response,
            voice_name=voice_name,
            output_path=output_path
        )
        
        # Trả về kết quả
        return jsonify({
            'success': True,
            'session_id': session_id,
            'user_text': user_text,
            'assistant_response': assistant_response,
            'audio_url': f"/static/outputs/{output_filename}"
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API endpoint để xử lý câu hỏi text
@app.route('/api/process-text', methods=['POST'])
def process_text():
    try:
        # Lấy dữ liệu từ request
        data = request.json
        user_text = data.get('text')
        session_id = data.get('session_id')
        voice_name = data.get('voice', 'elli')
        
        if not user_text:
            return jsonify({'error': 'Không có nội dung text'}), 400
        
        if not session_id or session_id == 'new':
            # Tạo phiên mới nếu không có hoặc yêu cầu tạo mới
            session_title = f"Hội thoại {datetime.now().strftime('%d/%m/%Y %H:%M')}"
            session_id = db.create_session(session_title)
        else:
            session_id = int(session_id)
        
        # Lưu tin nhắn của người dùng vào database
        db.add_message(session_id, "user", user_text)
        
        # Lấy lịch sử hội thoại
        conversation_history = db.format_conversation_history(session_id)
        
        # Truy vấn Gemini để lấy phản hồi
        assistant_response = llm_module.get_response(user_text, conversation_history)
        
        # Lưu tin nhắn của assistant vào database
        db.add_message(session_id, "assistant", assistant_response)
        
        # Tạo giọng nói từ phản hồi của assistant
        output_filename = f"{uuid.uuid4()}.mp3"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        tts_module.text_to_speech(
            text=assistant_response,
            voice_name=voice_name,
            output_path=output_path
        )
        
        # Trả về kết quả
        return jsonify({
            'success': True,
            'session_id': session_id,
            'user_text': user_text,
            'assistant_response': assistant_response,
            'audio_url': f"/static/outputs/{output_filename}"
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API endpoint để lấy lịch sử chat
@app.route('/api/session/<int:session_id>', methods=['GET'])
def get_session(session_id):
    try:
        messages = db.get_session_history(session_id)
        return jsonify({
            'success': True,
            'session_id': session_id,
            'messages': messages
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API endpoint để xóa một phiên chat
@app.route('/api/session/<int:session_id>', methods=['DELETE'])
def delete_session(session_id):
    try:
        db.delete_session(session_id)
        return jsonify({
            'success': True,
            'message': f'Đã xóa phiên chat {session_id}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API endpoint để tạo phiên chat mới
@app.route('/api/session', methods=['POST'])
def create_session():
    try:
        title = request.json.get('title', f"Hội thoại {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        session_id = db.create_session(title)
        return jsonify({
            'success': True,
            'session_id': session_id,
            'title': title
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)