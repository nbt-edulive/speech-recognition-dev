import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
from faster_whisper import WhisperModel

# Hàm thu âm
def record_audio(filename, duration=5, sample_rate=16000):
    print("Đang thu âm...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    wavfile.write(filename, sample_rate, recording)
    print(f"Đã lưu file: {filename}")

# Hàm chuyển đổi audio sang text
def transcribe_audio(audio_file):
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_file, language="vi")
    print("Ngôn ngữ được phát hiện:", info.language)
    print("Nội dung:")
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

# Quy trình chính
def main():
    audio_file = "output.wav"
    record_audio(audio_file, duration=3)  # Thu âm 5 giây
    transcribe_audio(audio_file)  # Chuyển đổi sang text

if __name__ == "__main__":
    main()