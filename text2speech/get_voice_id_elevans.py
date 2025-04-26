from elevenlabs import ElevenLabs
import os
from dotenv import load_dotenv

load_dotenv()
client = ElevenLabs(
    api_key= os.getenv("api_text2speech"),
)

try:
    voices = client.voices.search(include_total_count=True)
    print("Kết quả tìm kiếm:", voices)
    
    # Nếu voices là một đối tượng phức tạp, có thể cần kiểm tra thuộc tính của nó
    if hasattr(voices, 'voices'):
        print("Danh sách voices:", voices.voices)
    
    # Nếu là dictionary
    if isinstance(voices, dict) and 'voices' in voices:
        print("Danh sách voices:", voices['voices'])
        
except Exception as e:
    print(f"Có lỗi xảy ra: {e}")