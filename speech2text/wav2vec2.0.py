import torch
import torchaudio
import librosa
import numpy as np
import argparse
import os
import time
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def transcribe_with_wav2vec2(audio_path, model_name="nguyenvulebinh/wav2vec2-base-vietnamese-250h", output_format="txt"):
    """
    Chuyển đổi tiếng nói thành văn bản sử dụng mô hình Wav2Vec 2.0 đã fine-tune cho tiếng Việt
    
    Tham số:
        audio_path (str): Đường dẫn đến file audio cần chuyển đổi
        model_name (str): Tên mô hình hoặc đường dẫn đến mô hình
        output_format (str): Định dạng đầu ra ("txt" hoặc "srt")
    """
    total_start_time = time.time()
    
    # Tải mô hình
    model_load_start = time.time()
    print(f"Đang tải mô hình và processor cho {model_name}...")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    model_load_time = time.time() - model_load_start
    print(f"Thời gian tải mô hình: {model_load_time:.2f} giây")
    
    # Kiểm tra xem có GPU không
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Đang sử dụng thiết bị: {device}")
    if device == "cuda":
        model = model.to(device)
    
    # Tải audio
    audio_load_start = time.time()
    print(f"Đang xử lý file audio: {audio_path}")
    speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
    audio_load_time = time.time() - audio_load_start
    print(f"Thời gian tải và xử lý audio: {audio_load_time:.2f} giây")
    
    # Tokenize
    tokenize_start = time.time()
    input_values = processor(speech_array, sampling_rate=16000, return_tensors="pt").input_values
    if device == "cuda":
        input_values = input_values.to(device)
    tokenize_time = time.time() - tokenize_start
    print(f"Thời gian tokenize: {tokenize_time:.2f} giây")
    
    # Inference
    inference_start = time.time()
    with torch.no_grad():
        logits = model(input_values).logits
    
    # Lấy predicted id và chuyển đổi thành text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    inference_time = time.time() - inference_start
    print(f"Thời gian inference: {inference_time:.2f} giây")
    
    # Lưu kết quả
    output_start = time.time()
    output_base = os.path.splitext(audio_path)[0]
    
    # Xử lý cho output SRT nếu cần
    if output_format == "srt":
        # Tạo segments cho SRT (đơn giản hóa, vì Wav2Vec 2.0 không trực tiếp trả về thời gian)
        # Chúng ta sẽ chia audio thành các đoạn cố định, mỗi đoạn 5 giây
        duration = librosa.get_duration(y=speech_array, sr=sampling_rate)
        segment_duration = 5.0  # 5 giây mỗi segment
        num_segments = int(np.ceil(duration / segment_duration))
        
        segments_start = time.time()
        with open(f"{output_base}.srt", "w", encoding="utf-8") as f:
            # Nếu muốn phụ đề chi tiết hơn, bạn cần sử dụng thêm các kỹ thuật phân đoạn audio
            # Dưới đây là phiên bản đơn giản
            for i in range(num_segments):
                start_time = i * segment_duration
                end_time = min((i + 1) * segment_duration, duration)
                
                # Cắt audio cho đoạn này
                start_sample = int(start_time * sampling_rate)
                end_sample = int(end_time * sampling_rate)
                segment_audio = speech_array[start_sample:end_sample]
                
                # Chỉ xử lý nếu đoạn không quá ngắn
                if len(segment_audio) > 0.5 * sampling_rate:  # Ít nhất 0.5 giây
                    # Xử lý đoạn audio
                    segment_input = processor(segment_audio, sampling_rate=16000, return_tensors="pt").input_values
                    if device == "cuda":
                        segment_input = segment_input.to(device)
                    with torch.no_grad():
                        segment_logits = model(segment_input).logits
                    segment_ids = torch.argmax(segment_logits, dim=-1)
                    segment_text = processor.batch_decode(segment_ids)[0]
                    
                    # Định dạng thời gian cho SRT
                    start_formatted = format_time_srt(start_time)
                    end_formatted = format_time_srt(end_time)
                    
                    # Viết vào file SRT
                    f.write(f"{i+1}\n")
                    f.write(f"{start_formatted} --> {end_formatted}\n")
                    f.write(f"{segment_text.strip()}\n\n")
        
        segments_time = time.time() - segments_start
        print(f"Thời gian tạo phụ đề: {segments_time:.2f} giây")
        print(f"Đã lưu phụ đề vào: {output_base}.srt")
    
    # Lưu văn bản đầy đủ
    with open(f"{output_base}.txt", "w", encoding="utf-8") as f:
        f.write(transcription)
    
    output_time = time.time() - output_start
    print(f"Thời gian xử lý đầu ra: {output_time:.2f} giây")
    print(f"Đã lưu văn bản vào: {output_base}.txt")
    
    # Tổng thời gian
    total_time = time.time() - total_start_time
    
    print("\n===== THỐNG KÊ THỜI GIAN =====")
    print(f"Thời gian tải mô hình: {model_load_time:.2f} giây")
    print(f"Thời gian tải và xử lý audio: {audio_load_time:.2f} giây")
    print(f"Thời gian tokenize: {tokenize_time:.2f} giây")
    print(f"Thời gian inference: {inference_time:.2f} giây")
    print(f"Thời gian xử lý đầu ra: {output_time:.2f} giây")
    if output_format == "srt":
        print(f"Thời gian tạo phụ đề: {segments_time:.2f} giây")
    print(f"Tổng thời gian xử lý: {total_time:.2f} giây")
    print(f"=============================\n")
    
    print("Văn bản được chuyển đổi:")
    print(transcription)
    
    return transcription

def format_time_srt(seconds):
    """Định dạng thời gian theo chuẩn SRT"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuyển đổi audio tiếng Việt thành văn bản với Wav2Vec 2.0")
    parser.add_argument("audio_path", help="Đường dẫn đến file audio")
    parser.add_argument("--model", default="nguyenvulebinh/wav2vec2-base-vietnamese-250h", 
                        help="Tên hoặc đường dẫn đến mô hình Wav2Vec 2.0 cho tiếng Việt")
    parser.add_argument("--format", default="txt", choices=["txt", "srt"], 
                        help="Định dạng đầu ra (txt hoặc srt)")
    
    script_start_time = time.time()
    args = parser.parse_args()
    transcribe_with_wav2vec2(args.audio_path, args.model, args.format)
    script_total_time = time.time() - script_start_time
    print(f"Thời gian chạy toàn bộ script: {script_total_time:.2f} giây")