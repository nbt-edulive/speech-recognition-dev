import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
from faster_whisper import WhisperModel
import requests
import os
from pathlib import Path
from dotenv import load_dotenv
import time
import google.generativeai as genai
import pygame
from rich.console import Console
from rich.panel import Panel

# Khởi tạo console để hiển thị đẹp hơn
console = Console()

# Load biến môi trường
load_dotenv()

# Cấu hình Gemini API
GEMINI_API_KEY = os.getenv("api_gemini")
if not GEMINI_API_KEY:
    raise ValueError("Bạn cần cung cấp GEMINI_API_KEY trong file .env")

genai.configure(api_key=GEMINI_API_KEY)

# Cấu hình API Eleven Labs
ELEVEN_API_KEY = os.getenv("api_text2speech")
if not ELEVEN_API_KEY:
    raise ValueError("Bạn cần cung cấp api_text2speech trong file .env")

# IDs giọng từ ElevenLabs
VOICE_IDS = {
    "callum": "N2lVS1w4EtoT3dr4eOWO",
    "alice": "Xb7hH8MSUJpSbSDYk0k2",
    "aria": "9BWtsMINqrJLrRacOk9x",
    "rachel": "21m00Tcm4TlvDq8ikWAM",
    "Bill": "pqHfZKP75CvOlQylNhV4",
    "Brian": "nPczCjzI2devNBz1zQrb",
    "Domi": "AZnzlk1XvdvUeBnXmlld",
    "Elli": "MF3mGyEYCl7XYWbV9V6O",
    "Nicole": "MF3mGyEYCl7XYWbV9V6O",
    "Harry": "SOYHLrjzK2X1ezoPC6cr",
    "Ethan": "g5CIjZEefAph4nQFvHAz"
}

# Cấu hình model Whisper và Gemini
WHISPER_MODEL = "base"  # Có thể thay đổi thành "base", "small", "medium", "large" tùy nguồn lực
GEMINI_MODEL = "gemini-1.5-flash"  # Hoặc "gemini-1.5-flash" cho tốc độ nhanh hơn

# Hàm thu âm
def record_audio(filename, duration=5, sample_rate=16000):
    console.print(Panel("[bold yellow]Đang thu âm...[/bold yellow] (Hãy nói câu hỏi của bạn)", title="Thu âm"))
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    wavfile.write(filename, sample_rate, recording)
    console.print(f"[green]Đã lưu file:[/green] {filename}")

# Hàm chuyển đổi audio sang text sử dụng Whisper
def transcribe_audio(audio_file, language="vi"):
    console.print(Panel("[bold blue]Đang chuyển đổi âm thanh thành văn bản...[/bold blue]", title="Speech to Text"))
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_file, language=language)
    
    full_text = ""
    for segment in segments:
        full_text += segment.text + " "
    
    console.print(f"[green]Văn bản nhận được:[/green] {full_text.strip()}")
    return full_text.strip()

# Hàm truy vấn Gemini API
def query_gemini(prompt, context=""):
    console.print(Panel("[bold magenta]Đang tạo câu trả lời từ Gemini...[/bold magenta]", title="AI Processing"))
    
    # Cấu hình model
    model = genai.GenerativeModel(GEMINI_MODEL)
    
    # Tạo nội dung cho trợ lý gia sư
    system_prompt = """
    Bạn là một trợ lý gia sư thân thiện, chuyên môn trong việc giúp học sinh các cấp học từ tiểu học đến trung học phổ thông.
    Hãy trả lời câu hỏi của học sinh một cách rõ ràng, dễ hiểu và phù hợp với độ tuổi.
    Hãy sử dụng ngôn ngữ đơn giản, thân thiện và khuyến khích học sinh tư duy.
    Khi giải thích các khái niệm khó, hãy sử dụng ví dụ thực tế và liên hệ với cuộc sống hàng ngày.
    Khi trả lời, hãy giữ câu trả lời ngắn gọn, dễ hiểu và súc tích (tối đa 3-4 câu).
    """
    
    # Tạo nội dung cho prompt
    conversation_context = f"""
    {system_prompt}
    
    Lịch sử trò chuyện:
    {context}
    
    Câu hỏi của học sinh: {prompt}
    """
    
    try:
        response = model.generate_content(conversation_context)
        answer = response.text
        console.print(f"[green]Câu trả lời từ Gemini:[/green] {answer}")
        return answer
    except Exception as e:
        console.print(f"[bold red]Lỗi khi truy vấn Gemini API:[/bold red] {str(e)}")
        return "Xin lỗi, tôi đang gặp vấn đề kỹ thuật. Hãy thử lại sau nhé."

# Hàm chuyển văn bản thành giọng nói
def text_to_speech_elevenlabs(text, voice_id, api_key, output_path="output.mp3"):
    console.print(Panel("[bold cyan]Đang chuyển văn bản thành giọng nói...[/bold cyan]", title="Text to Speech"))
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    
    data = {
        "text": text,
        "model_id": "eleven_flash_v2_5",
        "voice_settings": {
            "speed": 1,
            "stability": 0.5,
            "similarity_boost": 0.75,
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
            
            console.print(f"[green]Đã tạo thành công file audio:[/green] {output_path}")
            return output_path
        else:
            console.print(f"[bold red]Lỗi: {response.status_code}[/bold red]")
            console.print(response.text)
            return None
    except Exception as e:
        console.print(f"[bold red]Lỗi khi chuyển đổi text to speech:[/bold red] {str(e)}")
        return None

# Hàm phát file audio
def play_audio(file_path):
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        console.print(f"[bold red]Lỗi khi phát audio:[/bold red] {str(e)}")

# Quy trình chính
def main():
    # Thiết lập các file path
    input_audio_file = "input.wav"
    output_audio_file = "audio_output/response.mp3"
    
    # Đảm bảo thư mục tồn tại
    os.makedirs("audio_output", exist_ok=True)
    
    # Chọn giọng
    selected_voice = VOICE_IDS["Elli"]  # Thay đổi thành giọng mong muốn
    
    # Khởi tạo để lưu lịch sử trò chuyện
    conversation_history = ""
    
    console.print(Panel("[bold green]Chào mừng đến với Trợ lý Gia sư![/bold green]\n"
                       "Hãy nói câu hỏi của bạn sau khi nhấn Enter.", 
                       title="Trợ lý Gia sư"))
    
    while True:
        try:
            # Chờ người dùng nhấn Enter để bắt đầu thu âm
            input("Nhấn Enter để đặt câu hỏi (hoặc gõ 'q' để thoát): ")
            
            # Thu âm
            record_audio(input_audio_file, duration=5)  # Thu âm 5 giây
            
            # Chuyển đổi sang text
            user_text = transcribe_audio(input_audio_file)
            
            if user_text.lower() in ["tạm biệt", "bye", "kết thúc", "q", "quit"]:
                console.print("[bold yellow]Kết thúc phiên trò chuyện. Tạm biệt![/bold yellow]")
                break
            
            # Cập nhật lịch sử trò chuyện
            conversation_history += f"Học sinh: {user_text}\n"
            
            # Truy vấn Gemini
            response = query_gemini(user_text, conversation_history)
            
            # Cập nhật lịch sử trò chuyện
            conversation_history += f"Trợ lý: {response}\n"
            
            # Chuyển đổi văn bản sang giọng nói
            audio_file = text_to_speech_elevenlabs(
                text=response,
                voice_id=selected_voice,
                api_key=ELEVEN_API_KEY,
                output_path=output_audio_file
            )
            
            # Phát âm thanh
            if audio_file:
                play_audio(audio_file)
            
        except KeyboardInterrupt:
            console.print("[bold yellow]Chương trình đã bị dừng bởi người dùng.[/bold yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]Lỗi không mong muốn:[/bold red] {str(e)}")

if __name__ == "__main__":
    main()