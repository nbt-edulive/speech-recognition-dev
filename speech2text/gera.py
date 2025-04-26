import torch
import librosa
import time
import os
import argparse
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

def transcribe_with_granite(audio_path, model_name="ibm-granite/granite-speech-3.3-8b", device="cpu", output_format="txt"):
    """
    Chuyển đổi audio thành văn bản sử dụng IBM Granite Speech
    
    Tham số:
        audio_path (str): Đường dẫn đến file audio
        model_name (str): Tên mô hình Granite Speech
        device (str): Thiết bị xử lý ("cpu" hoặc "cuda")
        output_format (str): Định dạng đầu ra ("txt" hoặc "srt")
    """
    total_start_time = time.time()
    
    # Kiểm tra thiết bị
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA không khả dụng, chuyển sang sử dụng CPU")
        device = "cpu"
    
    device_obj = torch.device(device)
    
    # Tải mô hình và processor
    model_load_start = time.time()
    print(f"Đang tải processor và model từ {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    model.to(device_obj)
    model_load_time = time.time() - model_load_start
    print(f"Thời gian tải mô hình: {model_load_time:.2f} giây")
    
    # Tải audio
    audio_load_start = time.time()
    print(f"Đang tải file audio: {audio_path}")
    audio, sample_rate = librosa.load(audio_path, sr=16000)
    audio_load_time = time.time() - audio_load_start
    print(f"Thời gian tải audio: {audio_load_time:.2f} giây")
    
    # Xử lý đầu vào
    process_start = time.time()
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = inputs.to(device_obj)
    process_time = time.time() - process_start
    print(f"Thời gian xử lý đầu vào: {process_time:.2f} giây")
    
    # Inference
    inference_start = time.time()
    with torch.no_grad():
        generated_ids = model.generate(inputs.input_features)
    inference_time = time.time() - inference_start
    print(f"Thời gian inference: {inference_time:.2f} giây")
    
    # Giải mã kết quả
    decode_start = time.time()
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    decode_time = time.time() - decode_start
    print(f"Thời gian giải mã: {decode_time:.2f} giây")
    
    # Lưu kết quả
    output_start = time.time()
    output_base = os.path.splitext(audio_path)[0]
    
    # Lưu văn bản
    with open(f"{output_base}.txt", "w", encoding="utf-8") as f:
        f.write(transcription)
    print(f"Đã lưu văn bản vào: {output_base}.txt")
    
    # Tạo file SRT nếu cần
    if output_format == "srt":
        # Tạo phụ đề đơn giản bằng cách chia văn bản thành các đoạn
        try:
            duration = librosa.get_duration(y=audio, sr=sample_rate)
            words = transcription.split()
            words_per_segment = 10  # Khoảng 10 từ mỗi đoạn
            segments = [words[i:i+words_per_segment] for i in range(0, len(words), words_per_segment)]
            
            # Chia thời gian đều cho mỗi đoạn
            segment_duration = duration / max(len(segments), 1)
            
            with open(f"{output_base}.srt", "w", encoding="utf-8") as f:
                for i, segment_words in enumerate(segments, start=1):
                    start_time = i * segment_duration - segment_duration
                    end_time = min(i * segment_duration, duration)
                    segment_text = " ".join(segment_words)
                    
                    # Định dạng thời gian
                    start_formatted = format_timestamp(start_time)
                    end_formatted = format_timestamp(end_time)
                    
                    # Viết vào file SRT
                    f.write(f"{i}\n")
                    f.write(f"{start_formatted} --> {end_formatted}\n")
                    f.write(f"{segment_text}\n\n")
            
            print(f"Đã lưu phụ đề vào: {output_base}.srt")
        except Exception as e:
            print(f"Không thể tạo file SRT: {e}")
    
    output_time = time.time() - output_start
    print(f"Thời gian xử lý đầu ra: {output_time:.2f} giây")
    
    # Tổng thời gian
    total_time = time.time() - total_start_time
    
    print("\n===== THỐNG KÊ THỜI GIAN =====")
    print(f"Thời gian tải mô hình: {model_load_time:.2f} giây")
    print(f"Thời gian tải audio: {audio_load_time:.2f} giây")
    print(f"Thời gian xử lý đầu vào: {process_time:.2f} giây")
    print(f"Thời gian inference: {inference_time:.2f} giây")
    print(f"Thời gian giải mã: {decode_time:.2f} giây")
    print(f"Thời gian xử lý đầu ra: {output_time:.2f} giây")
    print(f"Tổng thời gian xử lý: {total_time:.2f} giây")
    print(f"=============================\n")
    
    print("Văn bản được chuyển đổi:")
    print(transcription)
    
    return transcription

def format_timestamp(seconds):
    """Định dạng thời gian theo chuẩn SRT (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    millisecs = int((secs - int(secs)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{millisecs:03d}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuyển đổi audio thành văn bản với IBM Granite Speech")
    parser.add_argument("audio_path", help="Đường dẫn đến file audio")
    parser.add_argument("--model", default="ibm-granite/granite-speech-3.3-8b", 
                       help="Tên mô hình Granite Speech")
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"], 
                       help="Thiết bị xử lý (cuda hoặc cpu)")
    parser.add_argument("--format", default="txt", choices=["txt", "srt"], 
                       help="Định dạng đầu ra (txt hoặc srt)")
    
    script_start_time = time.time()
    args = parser.parse_args()
    transcribe_with_granite(args.audio_path, args.model, args.device, args.format)
    script_total_time = time.time() - script_start_time
    print(f"Thời gian chạy toàn bộ script: {script_total_time:.2f} giây")