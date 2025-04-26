import os
import torch
import torchaudio
import numpy as np
import time
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Đường dẫn đến thư mục chứa model
MODEL_PATH = "model"

# Khởi tạo model
start_load_time = time.time()
config = XttsConfig()
config.load_json(os.path.join(MODEL_PATH, "config.json"))
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=MODEL_PATH)
end_load_time = time.time()
time_load_model = end_load_time - start_load_time
print(f"Thời gian tải mô hình: {end_load_time - start_load_time:.2f} giây")

# Dùng GPU nếu có
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Đang sử dụng thiết bị: {device}")
model.to(device)

def text_to_speech(text, speaker_wav, output_path, language="vi"):
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Bắt đầu đo thời gian cho việc tạo conditioning latents
    start_cond_time = time.time()
    # Tạo conditioning latents từ file giọng mẫu
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=speaker_wav,
        gpt_cond_len=model.config.gpt_cond_len,
        max_ref_length=model.config.max_ref_len,
        sound_norm_refs=model.config.sound_norm_refs,
    )
    end_cond_time = time.time()
    cond_time = end_cond_time - start_cond_time
    print(f"Thời gian tạo conditioning latents: {cond_time:.2f} giây")
    
    # Bắt đầu đo thời gian cho việc inference
    start_inference_time = time.time()
    # Tạo wav từ văn bản
    outputs = model.inference(
        text=text,
        language=language,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.3,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=30,
        top_p=0.85,
    )
    end_inference_time = time.time()
    inference_time = end_inference_time - start_inference_time
    print(f"Thời gian inference: {inference_time:.2f} giây")
    
    # Bắt đầu đo thời gian cho việc lưu file
    start_save_time = time.time()
    # Lưu file WAV
    wav_path = output_path
    if output_path.endswith('.mp3'):
        wav_path = output_path.replace('.mp3', '.wav')
    
    # Chuyển numpy array sang tensor torch
    wav_tensor = torch.tensor(outputs["wav"]).unsqueeze(0)
    torchaudio.save(wav_path, wav_tensor, 24000)
    
    # Chuyển sang MP3 nếu cần
    if output_path.endswith('.mp3'):
        try:
            os.system(f"ffmpeg -i {wav_path} -vn -ar 44100 -ac 2 -b:a 192k {output_path} -y -loglevel quiet")
            if os.path.exists(output_path):
                os.remove(wav_path)  # Xóa file wav tạm nếu chuyển đổi thành công
                end_save_time = time.time()
                save_time = end_save_time - start_save_time
                print(f"Thời gian lưu file: {save_time:.2f} giây")
                return output_path, cond_time, inference_time, save_time
        except Exception as e:
            end_save_time = time.time()
            save_time = end_save_time - start_save_time
            print(f"Lỗi khi chuyển sang MP3: {e}")
            print(f"Thời gian lưu file: {save_time:.2f} giây")
            return wav_path, cond_time, inference_time, save_time
    
    end_save_time = time.time()
    save_time = end_save_time - start_save_time
    print(f"Thời gian lưu file: {save_time:.2f} giây")
    return wav_path, cond_time, inference_time, save_time

if __name__ == "__main__":
    # Thông số đầu vào
    input_text = "Xin chào, tôi là trợ lý ảo tiếng Việt, tôi có thể làm bất cứ điều gì cho bạn, bạn có cần gì không?"
    speaker_file = os.path.join("../text2speech", "user_sample.wav")
    output_file = "./output/output.mp3"
    
    # Bắt đầu đo thời gian tổng thể
    start_total_time = time.time()
    
    # Chạy hàm text-to-speech
    result_file, cond_time, inference_time, save_time = text_to_speech(
        text=input_text,
        speaker_wav=speaker_file,
        output_path=output_file,
        language="vi"
    )
    
    # Kết thúc đo thời gian tổng thể
    end_total_time = time.time()
    total_time = end_total_time - start_total_time
    
    print(f"Đã tạo file audio tại: {result_file}")
    print("\n--- BÁO CÁO THỜI GIAN ---")
    print(f"1. Thời gian tạo conditioning latents: {cond_time:.2f} giây")
    print(f"2. Thời gian inference: {inference_time:.2f} giây")
    print(f"3. Thời gian lưu file: {save_time:.2f} giây")
    print(f"TỔNG THỜI GIAN XỬ LÝ: {time_load_model + total_time:.2f} giây")