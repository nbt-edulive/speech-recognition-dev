# 🗣️ Speech Recognition Project

This project includes two main components:

- `chatbot`: Voice/text-based conversational assistant.
- `clone_voice`: Voice cloning using ViXTTS.

---

## 🧩 System Requirements

- Python 3.10  
- Git  
- Mini Anaconda (recommended)  
- CUDA Toolkit + cuDNN (for GPU support)

---

## 🛠️ Environment Setup

### Step 1: Install Mini Anaconda

Download Miniconda at:  
[https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

Install the version suitable for your operating system.

### Step 2: Create and activate a virtual environment

```bash
conda create -n speech_env python=3.10 -y
conda activate speech_env

git clone git@github.com:edulive-ai/speech-recognition.git
cd speech-recognition
pip install -r requirements.txt

```

### Run Chatbot
add .env 
```bash

ELEVEN_API_KEY = "YOUR API KEY" 
GEMINI_API_KEY_1 = "YOUR API KEY"
```

```bash
cd chatbot
python app.py
```
### Run clone_voice
add .env
```bash
ELEVEN_API_KEY = "YOUR API KEY" 
GEMINI_API_KEY_1 = "YOUR API KEY"
```


```bash
cd clone_voice
python setup_vixtts.py
python app.py
```
