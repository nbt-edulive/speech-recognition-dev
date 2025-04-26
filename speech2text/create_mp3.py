from gtts import gTTS
from pydub import AudioSegment

# Văn bản cần chuyển thành giọng nói
text = "Xin chào tất cả mọi người, chúc mọi người có một ngày mới vui vẻ"

# Ngôn ngữ (ví dụ: 'vi' cho tiếng Việt, 'en' cho tiếng Anh)
language = 'vi'

# Tạo đối tượng gTTS
tts = gTTS(text=text, lang=language, slow=False)

# Lưu file MP3 tạm thời
mp3_file = "output_temp.mp3"
tts.save(mp3_file)

# Chuyển đổi từ MP3 sang WAV
sound = AudioSegment.from_mp3(mp3_file)
wav_file = "output.wav"
sound.export(wav_file, format="wav")

# Xóa file MP3 tạm thời nếu muốn
import os
os.remove(mp3_file)

print(f"Đã tạo file {wav_file}")