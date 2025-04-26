import whisper
import os
import argparse
import time

def transcribe_audio(audio_path, model_size="base", language=None, output_format="txt"):
    """
    Chuyển đổi file audio thành văn bản sử dụng OpenAI Whisper
    
    Tham số:
        audio_path (str): Đường dẫn đến file audio cần chuyển đổi
        model_size (str): Kích thước mô hình ("tiny", "base", "small", "medium", "large")
        language (str): Mã ngôn ngữ (vd: "vi" cho tiếng Việt, "en" cho tiếng Anh, None để tự động phát hiện)
        output_format (str): Định dạng đầu ra ("txt" hoặc "srt")
    """
    print(f"Đang tải mô hình Whisper {model_size}...")
    load_start_time = time.time()
    model = whisper.load_model(model_size)
    load_time = time.time() - load_start_time
    print(f"Thời gian tải mô hình: {load_time:.2f} giây")
    
    print(f"Đang xử lý file audio: {audio_path}")
    
    # Cấu hình các tùy chọn cho Whisper
    transcribe_options = {}
    if language:
        transcribe_options["language"] = language
    
    # Thực hiện chuyển đổi
    transcribe_start_time = time.time()
    result = model.transcribe(audio_path, **transcribe_options)
    transcribe_time = time.time() - transcribe_start_time
    print(f"Thời gian chuyển đổi: {transcribe_time:.2f} giây")
    
    # Lấy văn bản kết quả
    text = result["text"]
    
    # Lưu kết quả vào file
    output_path = os.path.splitext(audio_path)[0]
    
    if output_format == "txt":
        with open(f"{output_path}.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Đã lưu văn bản vào: {output_path}.txt")
    elif output_format == "srt":
        import datetime
        with open(f"{output_path}.srt", "w", encoding="utf-8") as f:
            for i, segment in enumerate(result["segments"], start=1):
                # Chuyển đổi thời gian từ giây sang định dạng SRT
                start_time = str(datetime.timedelta(seconds=segment["start"])).replace(".", ",")[:11]
                end_time = str(datetime.timedelta(seconds=segment["end"])).replace(".", ",")[:11]
                
                # Viết segment vào file SRT
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment['text'].strip()}\n\n")
        print(f"Đã lưu phụ đề vào: {output_path}.srt")
    
    # Tổng thời gian
    total_time = load_time + transcribe_time
    print(f"Tổng thời gian xử lý: {total_time:.2f} giây")
    
    print("Văn bản được chuyển đổi:")
    print(text)
    return text

if __name__ == "__main__":
    # Cấu hình tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Chuyển đổi audio thành văn bản sử dụng OpenAI Whisper")
    parser.add_argument("audio_path", help="Đường dẫn đến file audio")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"], 
                        help="Kích thước mô hình Whisper")
    parser.add_argument("--language", default=None, help="Mã ngôn ngữ (vd: 'vi' cho tiếng Việt, 'en' cho tiếng Anh)")
    parser.add_argument("--format", default="txt", choices=["txt", "srt"], 
                        help="Định dạng đầu ra (txt hoặc srt cho phụ đề)")
    
    args = parser.parse_args()
    
    # Đo thời gian tổng quát
    start_time = time.time()
    
    # Gọi hàm chuyển đổi
    transcribe_audio(args.audio_path, args.model, args.language, args.format)
    
    total_execution_time = time.time() - start_time
    print(f"Thời gian thực thi toàn bộ script: {total_execution_time:.2f} giây")