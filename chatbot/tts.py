import requests
import os
import json
import time

class TextToSpeech:
    def __init__(self, api_key=None):
        """
        Khởi tạo module Text to Speech với ElevenLabs
        
        Args:
            api_key (str, optional): API key cho ElevenLabs
        """
        self.api_key = api_key or os.environ.get("ELEVEN_API_KEY")
        if not self.api_key:
            raise ValueError("Cần cung cấp ElevenLabs API key")
        
        # IDs giọng từ ElevenLabs
        self.voices = {
            "callum": "N2lVS1w4EtoT3dr4eOWO",
            "alice": "Xb7hH8MSUJpSbSDYk0k2",
            "aria": "9BWtsMINqrJLrRacOk9x",
            "rachel": "21m00Tcm4TlvDq8ikWAM",
            "bill": "pqHfZKP75CvOlQylNhV4",
            "brian": "nPczCjzI2devNBz1zQrb",
            "domi": "AZnzlk1XvdvUeBnXmlld",
            "elli": "MF3mGyEYCl7XYWbV9V6O",
            "nicole": "MF3mGyEYCl7XYWbV9V6O",
            "harry": "SOYHLrjzK2X1ezoPC6cr",
            "ethan": "g5CIjZEefAph4nQFvHAz"
        }
    
    def get_available_voices(self):
        """
        Lấy danh sách giọng nói có sẵn
        
        Returns:
            dict: Dictionary của tên giọng và ID
        """
        return self.voices
    
    def text_to_speech(self, text, voice_name="elli", output_path="output.mp3", 
                       model_id="eleven_flash_v2_5", speed=1.0, stability=0.5, 
                       similarity_boost=0.75):
        """
        Chuyển đổi văn bản thành giọng nói sử dụng ElevenLabs API
        
        Args:
            text (str): Văn bản cần chuyển đổi
            voice_name (str): Tên giọng (key trong dict voices)
            output_path (str): Đường dẫn lưu file audio
            model_id (str): ID model ElevenLabs
            speed (float): Tốc độ nói (0.5-2.0)
            stability (float): Độ ổn định (0.0-1.0)
            similarity_boost (float): Tăng độ tương đồng (0.0-1.0)
            
        Returns:
            str: Đường dẫn đến file audio hoặc None nếu lỗi
        """
        if voice_name.lower() not in self.voices:
            raise ValueError(f"Không tìm thấy giọng: {voice_name}. Giọng có sẵn: {list(self.voices.keys())}")
        
        voice_id = self.voices[voice_name.lower()]
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        data = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "speed": speed,
                "stability": stability,
                "similarity_boost": similarity_boost,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
        
        try:
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                # Đảm bảo thư mục tồn tại
                output_dir = os.path.dirname(output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                # Lưu file audio
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                return output_path
            else:
                print(f"Lỗi: {response.status_code}")
                print(response.text)
                return None
        except Exception as e:
            print(f"Lỗi khi chuyển đổi text to speech: {str(e)}")
            return None