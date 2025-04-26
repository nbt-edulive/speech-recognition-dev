import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()

group_id = os.getenv("group_id")
api_key = os.getenv("api_minimax")

url = f"https://api.minimaxi.chat/v1/t2a_v2?GroupId={group_id}"
print(f"GroupId: {group_id}")
print(f"API Key: {'*' * 5 + api_key[-4:] if api_key else 'Not found'}")  # Hiển thị bảo mật hơn

payload = json.dumps({
  "model":"speech-02-hd",
  "text":"The real danger is not that computers start thinking like people, but that people start thinking like computers. Computers can only help us with simple tasks.",
  "stream":False,
  "subtitle_enable":False,
  "voice_setting":{
    "voice_id":"Grinch",
    "speed":1,
    "vol":1,
    "pitch":0
  },
  "audio_setting":{
    "sample_rate":32000,
    "bitrate":128000,
    "format":"mp3",
    "channel":1
  }
})
headers = {
  'Authorization': f'Bearer {api_key}',
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, stream=True, headers=headers, data=payload)
print(f"Status code: {response.status_code}")

try:
    parsed_json = json.loads(response.text)
    print("Response structure:", json.dumps(parsed_json, indent=2))
    
    # Kiểm tra xem 'data' có tồn tại không
    if 'data' in parsed_json and 'audio' in parsed_json['data']:
        audio_value = bytes.fromhex(parsed_json['data']['audio'])
        with open('output.mp3', 'wb') as f:
            f.write(audio_value)
        print("File output.mp3 created successfully!")
    else:
        print("Response does not contain expected data structure.")
        print("Available keys in response:", list(parsed_json.keys()))
        if 'base_resp' in parsed_json and 'status_msg' in parsed_json['base_resp']:
            print("Error message:", parsed_json['base_resp']['status_msg'])
except json.JSONDecodeError:
    print("Failed to parse JSON response. Raw response:")
    print(response.text[:500])  # Print only first 500 chars to avoid overwhelming output