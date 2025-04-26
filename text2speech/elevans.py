import requests
import os
from pathlib import Path
from dotenv import load_dotenv
import time
load_dotenv()

def text_to_speech_elevenlabs(text, voice_id, api_key, output_path="output.mp3"):
    """
    Chuyển đổi văn bản thành giọng nói sử dụng ElevenLabs API
    
    Parameters:
        text (str): Văn bản cần chuyển thành giọng nói
        voice_id (str): ID của giọng nói trong ElevenLabs
        api_key (str): API key của ElevenLabs
        output_path (str): Đường dẫn tới file output
    
    Returns:
        str: Đường dẫn tới file audio đã tạo
    """
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    
    data = {
        "text": text,
        "model_id": "eleven_flash_v2_5",  # Có thể thay đổi model
        "voice_settings": {
            "speed" : 1,
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True
        }
    }
    
    print(f"Đang gửi yêu cầu text-to-speech cho văn bản: {text[:50]}...")
    
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 200:
        # Đảm bảo thư mục tồn tại
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Lưu file audio
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"Đã tạo thành công file audio: {output_path}")
        return output_path
    else:
        print(f"Lỗi: {response.status_code}")
        print(response.text)
        return None

# Sử dụng hàm
if __name__ == "__main__":
    # Thay thế bằng thông tin thực của bạn
    API_KEY = os.getenv("api_text2speech")  # API key của bạn
    print(API_KEY)
    # IDs giọng trẻ em từ ElevenLabs 
    # (Dưới đây là một số ID ví dụ, bạn cần thay bằng ID thực từ tài khoản của bạn)
    CHILD_VOICE_IDS = {
        "callum": "N2lVS1w4EtoT3dr4eOWO",  # ID ví dụ cho giọng bé trai
        "alice": "Xb7hH8MSUJpSbSDYk0k2",
        "aria":" 9BWtsMINqrJLrRacOk9x",
        "rachel": "21m00Tcm4TlvDq8ikWAM",
        "Bill": " pqHfZKP75CvOlQylNhV4",
        "Brian" : "nPczCjzI2devNBz1zQrb",
        "Domi": "AZnzlk1XvdvUeBnXmlld",
        "Elli" : "MF3mGyEYCl7XYWbV9V6O",
        "Nicole" : "MF3mGyEYCl7XYWbV9V6O",
        "Harry" : "SOYHLrjzK2X1ezoPC6cr",
        "Ethan" : "g5CIjZEefAph4nQFvHAz"  # ID ví dụ cho giọng bé gái
    }
    
    text = "Con xin chào tất cả mọi người, con tên là Nguyễn Bá Tiến, hiện tại con đang học lớp một, môn học yêu thích nhất của con là môn toán, môn thể thao mà con yêu thích nhất là môn cầu lông"
    
    # Chọn giọng trẻ em
    selected_voice = CHILD_VOICE_IDS["Elli"]  # hoặc "boy" tùy vào nhu cầu
    
    # Tạo file audio
    output_file = text_to_speech_elevenlabs(
        text=text,
        voice_id=selected_voice,
        api_key=API_KEY,
        output_path="audio_output/child_voice.mp3"
    )
    
    print(f"File audio đã được tạo tại: {output_file}")