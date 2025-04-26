import requests
import os
from dotenv import load_dotenv

load_dotenv()

# API key
api_key = os.getenv("api_dupdub")
# API endpoint for query the speaker id
url = "https://moyin-gateway.dupdub.com/tts/v1/storeSpeakerV2/searchSpeakerList?language={language}"

# Your API key

headers = {
    "dupdub_token": f"{api_key}",
    "Content-Type": "application/json"
}

response = requests.get(url.format(language="Vietnamese"), headers=headers)

# Parse the response
if response.status_code == 200:
    result = response.json()
    print(result)
else:
    print("Failed to query voiceover actor information. Status Code:", response.status_code)