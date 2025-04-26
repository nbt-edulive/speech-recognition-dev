import time
import argparse
import os
from faster_whisper import WhisperModel

def transcribe_with_faster_whisper(audio_path, model_size="tiny", device="cpu", language="vi", output_format="txt"):
    """
    Chuyển đổi audio thành văn bản sử dụng Faster-Whisper
    
    Tham số:
        audio_path (str): Đường dẫn đến file audio
        model_size (str): Kích thước mô hình ("tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3")
        device (str): Thiết bị xử lý ("cuda" hoặc "cpu")
        language (str): Mã ngôn ngữ
        output_format (str): Định dạng đầu ra ("txt" hoặc "srt")
    """
    total_start_time = time.time()
    
    # Tải mô hình
    model_load_start = time.time()
    print(f"Đang tải mô hình Faster-Whisper {model_size} trên {device}...")
    # Chọn compute_type phù hợp
    compute_type = "float32" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    model_load_time = time.time() - model_load_start
    print(f"Thời gian tải mô hình: {model_load_time:.2f} giây")
    
    # Chuyển đổi
    inference_start = time.time()
    print(f"Đang xử lý file audio: {audio_path}")
    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    # Đảm bảo segments là một list để tránh iterator đã dùng
    segments_list = list(segments)
    inference_time = time.time() - inference_start
    print(f"Thời gian inference: {inference_time:.2f} giây")
    
    # Thông tin về ngôn ngữ
    print(f"Đã phát hiện ngôn ngữ: {info.language} (độ tin cậy: {info.language_probability:.2f})")
    
    # Xử lý kết quả
    output_start = time.time()
    output_base = os.path.splitext(audio_path)[0]
    
    # Xử lý kết quả text
    full_text = ""
    for segment in segments_list:
        full_text += segment.text + " "
    
    # Lưu văn bản
    with open(f"{output_base}.txt", "w", encoding="utf-8") as f:
        f.write(full_text.strip())
    print(f"Đã lưu văn bản vào: {output_base}.txt")
    
    # Tạo file SRT nếu cần
    if output_format == "srt":
        with open(f"{output_base}.srt", "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments_list, start=1):
                # Định dạng thời gian
                start_time = format_timestamp(segment.start)
                end_time = format_timestamp(segment.end)
                
                # Viết vào file SRT
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment.text.strip()}\n\n")
        print(f"Đã lưu phụ đề vào: {output_base}.srt")
    
    output_time = time.time() - output_start
    print(f"Thời gian xử lý đầu ra: {output_time:.2f} giây")
    
    # Tổng thời gian
    total_time = time.time() - total_start_time
    
    print("\n===== THỐNG KÊ THỜI GIAN =====")
    print(f"Thời gian tải mô hình: {model_load_time:.2f} giây")
    print(f"Thời gian inference: {inference_time:.2f} giây")
    print(f"Thời gian xử lý đầu ra: {output_time:.2f} giây")
    print(f"Tổng thời gian xử lý: {total_time:.2f} giây")
    print(f"=============================\n")
    
    # In kết quả
    print("Văn bản được chuyển đổi:")
    print(full_text.strip())
    
    return full_text.strip()

def format_timestamp(seconds):
    """Định dạng thời gian theo chuẩn SRT (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    millisecs = int((secs - int(secs)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{millisecs:03d}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuyển đổi audio thành văn bản với Faster-Whisper")
    parser.add_argument("audio_path", help="Đường dẫn đến file audio")
    parser.add_argument("--model", default="tiny", 
                       choices=["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"], 
                       help="Kích thước mô hình")
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"], 
                       help="Thiết bị xử lý (cuda hoặc cpu)")
    parser.add_argument("--language", default="vi", help="Mã ngôn ngữ")
    parser.add_argument("--format", default="txt", choices=["txt", "srt"], 
                       help="Định dạng đầu ra (txt hoặc srt)")
    
    script_start_time = time.time()
    args = parser.parse_args()
    transcribe_with_faster_whisper(args.audio_path, args.model, args.device, args.language, args.format)
    script_total_time = time.time() - script_start_time
    print(f"Thời gian chạy toàn bộ script: {script_total_time:.2f} giây")