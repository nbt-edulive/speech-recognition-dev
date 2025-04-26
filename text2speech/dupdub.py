import requests
import os
from dotenv import load_dotenv

load_dotenv()

# API endpoint
url = "https://moyin-gateway.dupdub.com/tts/v1/playDemo/dubForSpeaker"

# API key
api_key = os.getenv("api_dupdub")

headers = {
    "dupdub_token": f"{api_key}",
    "Content-Type": "application/json"
}

payload = {
    # 'speaker': 'uranus||||c2d38855d8f15bedd8d3881fd6d85647', #spoony
    'speaker': "uranus||||b4f0a08396ed164c2a7a9abfd1e4b02b", #Luke
    "speed": 0.85,
    "pitch": 0,
    "textList": ["Con xin chào tất cả mọi người, con tên là Nguyễn Bá Tiến, hiện tại con đang học lớp một, môn học yêu thích nhất của con là môn toán, môn thể thao mà con yêu thích nhất là môn cầu lông "],
    "source": "web",
    'language': ''
}

# Gọi API để lấy JSON với link download
response = requests.post(url, json=payload, headers=headers)

# Parse response và download file
if response.status_code == 200:
    result = response.json()
    print("API response:", result)
    
    # Lấy URL từ kết quả JSON
    if result['code'] == 200 and 'data' in result and 'resList' in result['data']:
        audio_url = result['data']['resList'][0]['result']['ossFile']
        
        # Download file từ URL
        audio_response = requests.get(audio_url)
        
        # Kiểm tra download thành công
        if audio_response.status_code == 200:
            # Tạo tên file từ URL hoặc sử dụng tên tùy chỉnh
            filename = "output.wav"  # URL trả về file WAV, không phải MP3
            
            # Lưu file vào thư mục local
            with open(filename, 'wb') as f:
                f.write(audio_response.content)
                
            print(f"File âm thanh đã được lưu thành công: {filename}")
        else:
            print(f"Không thể tải file âm thanh. Status code: {audio_response.status_code}")
    else:
        print("Không tìm thấy URL âm thanh trong phản hồi")
else:
    print(f"Gọi API thất bại. Status code: {response.status_code}")
    print(response.text)