import sys
import os
import wave
import json
import argparse
import time
from datetime import datetime, timedelta
from vosk import Model, KaldiRecognizer, SetLogLevel

def get_model_url(model_name):
    """Lấy URL tải xuống cho mô hình dựa trên tên"""
    model_urls = {
        "vosk-model-small-vn-0.3": "https://alphacephei.com/vosk/models/vosk-model-small-vn-0.3.zip",
        "vosk-model-small-vn-0.4": "https://alphacephei.com/vosk/models/vosk-model-small-vn-0.4.zip",
        "vosk-model-vn-0.4": "https://alphacephei.com/vosk/models/vosk-model-vn-0.4.zip"
    }
    
    # Trả về URL mặc định nếu không tìm thấy
    return model_urls.get(model_name, model_urls["vosk-model-small-vn-0.3"])

def download_model_if_not_exists(model_path):
    """Tải xuống mô hình tiếng Việt nếu chưa có"""
    if not os.path.exists(model_path):
        download_start_time = time.time()
        print(f"Mô hình không tìm thấy tại {model_path}, tải xuống...")
        
        # Xác định tên mô hình và URL tải xuống
        model_name = os.path.basename(model_path)
        model_url = get_model_url(model_name)
        zip_file = f"{model_name}.zip"
        
        # Tạo thư mục cha nếu cần
        parent_dir = os.path.dirname(model_path)
        if parent_dir:  # Nếu có thư mục cha
            os.makedirs(parent_dir, exist_ok=True)
        
        import urllib.request
        print(f"Đang tải mô hình tiếng Việt Vosk từ {model_url}...")
        urllib.request.urlretrieve(model_url, zip_file)
        
        import zipfile
        print("Đang giải nén mô hình...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(".")  # Giải nén vào thư mục hiện tại
        
        # Đổi tên thư mục mô hình nếu cần
        if os.path.exists(model_name) and model_path != model_name:
            if os.path.exists(model_path):  # Xóa thư mục đích nếu đã tồn tại
                import shutil
                shutil.rmtree(model_path)
            os.rename(model_name, model_path)
        
        # Xóa file zip sau khi giải nén
        if os.path.exists(zip_file):
            os.remove(zip_file)
        
        download_time = time.time() - download_start_time
        print(f"Tải mô hình hoàn tất! Thời gian tải: {download_time:.2f} giây")

def convert_to_srt_time(seconds):
    """Chuyển đổi thời gian từ giây sang định dạng SRT"""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def transcribe_with_vosk(audio_path, model_path="vosk-model-small-vn-0.3", output_format="txt"):
    """
    Chuyển đổi tiếng nói thành văn bản sử dụng Vosk với mô hình tiếng Việt
    
    Tham số:
        audio_path (str): Đường dẫn đến file audio cần chuyển đổi
        model_path (str): Đường dẫn đến thư mục chứa mô hình Vosk
        output_format (str): Định dạng đầu ra ("txt" hoặc "srt")
    """
    total_start_time = time.time()
    SetLogLevel(-1)  # Tắt log không cần thiết
    
    # Kiểm tra và tải xuống mô hình nếu cần
    model_download_start = time.time()
    download_model_if_not_exists(model_path)
    model_download_time = time.time() - model_download_start
    
    # Kiểm tra file âm thanh
    if not os.path.exists(audio_path):
        print(f"File audio không tồn tại: {audio_path}")
        return
    
    # Nếu không phải file WAV, ta cần chuyển đổi trước
    conversion_start_time = time.time()
    audio_ext = os.path.splitext(audio_path)[1].lower()
    temp_wav_path = None
    
    if audio_ext != ".wav":
        print(f"Chuyển đổi {audio_ext} sang WAV...")
        import subprocess
        temp_wav_path = f"{os.path.splitext(audio_path)[0]}_temp.wav"
        subprocess.call(['ffmpeg', '-i', audio_path, '-ar', '16000', '-ac', '1', temp_wav_path])
        audio_path_for_processing = temp_wav_path
    else:
        audio_path_for_processing = audio_path
    
    conversion_time = time.time() - conversion_start_time
    if audio_ext != ".wav":
        print(f"Thời gian chuyển đổi định dạng: {conversion_time:.2f} giây")
    
    # Mở file audio
    wf = wave.open(audio_path_for_processing, "rb")
    
    # Kiểm tra định dạng audio
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        print("Audio phải là WAV mono 16-bit PCM")
        if temp_wav_path:
            os.remove(temp_wav_path)
        return
    
    # Tải mô hình
    model_load_start = time.time()
    print(f"Đang tải mô hình từ {model_path}...")
    model = Model(model_path)
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)  # Để lấy thời gian cho từng từ
    model_load_time = time.time() - model_load_start
    print(f"Thời gian tải mô hình: {model_load_time:.2f} giây")
    
    # Xử lý audio
    recognition_start = time.time()
    print("Đang chuyển đổi audio thành văn bản...")
    results = []
    
    # Xử lý audio
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            part_result = json.loads(rec.Result())
            results.append(part_result)
    
    part_result = json.loads(rec.FinalResult())
    results.append(part_result)
    
    recognition_time = time.time() - recognition_start
    print(f"Thời gian nhận dạng: {recognition_time:.2f} giây")
    
    # Dọn dẹp file tạm thời
    if temp_wav_path:
        os.remove(temp_wav_path)
    
    # Xử lý kết quả
    processing_start = time.time()
    full_text = ""
    segments = []
    
    for res in results:
        if "result" in res:
            for i, word_info in enumerate(res["result"]):
                if i == 0 or res["result"][i-1]["end"] + 1.0 < word_info["start"]:
                    # Bắt đầu một segment mới
                    if i > 0:
                        segments.append({
                            "start": segments[-1]["start"],
                            "end": res["result"][i-1]["end"],
                            "text": segments[-1]["text"].strip()
                        })
                    
                    segments.append({
                        "start": word_info["start"],
                        "end": word_info["end"],
                        "text": word_info["word"] + " "
                    })
                else:
                    # Thêm từ vào segment hiện tại
                    segments[-1]["end"] = word_info["end"]
                    segments[-1]["text"] += word_info["word"] + " "
        
        if "text" in res:
            full_text += res["text"] + " "
    
    # Tạo tên file đầu ra với thông tin về mô hình đã sử dụng
    model_identifier = os.path.basename(model_path)
    output_base = os.path.splitext(audio_path)[0]
    output_file_base = f"{output_base}_{model_identifier}"
    
    if output_format == "txt":
        with open(f"{output_file_base}.txt", "w", encoding="utf-8") as f:
            f.write(full_text.strip())
        print(f"Đã lưu văn bản vào: {output_file_base}.txt")
    
    elif output_format == "srt":
        with open(f"{output_file_base}.srt", "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments, start=1):
                if not segment["text"].strip():
                    continue
                
                start_time = convert_to_srt_time(segment["start"])
                end_time = convert_to_srt_time(segment["end"])
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment['text'].strip()}\n\n")
        print(f"Đã lưu phụ đề vào: {output_file_base}.srt")
    
    processing_time = time.time() - processing_start
    print(f"Thời gian xử lý kết quả: {processing_time:.2f} giây")
    
    # Tổng thời gian
    total_time = time.time() - total_start_time
    print(f"\n===== THỐNG KÊ THỜI GIAN ({model_identifier}) =====")
    if not os.path.exists(model_path + "_DOWNLOADED"):
        print(f"Thời gian tải mô hình: {model_download_time:.2f} giây")
    print(f"Thời gian tải mô hình vào bộ nhớ: {model_load_time:.2f} giây")
    if audio_ext != ".wav":
        print(f"Thời gian chuyển đổi định dạng: {conversion_time:.2f} giây")
    print(f"Thời gian nhận dạng: {recognition_time:.2f} giây")
    print(f"Thời gian xử lý kết quả: {processing_time:.2f} giây")
    print(f"Tổng thời gian xử lý: {total_time:.2f} giây")
    print(f"=============================\n")
    
    print("Kết quả:")
    print(full_text.strip())
    return full_text.strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuyển đổi audio tiếng Việt thành văn bản sử dụng Vosk")
    parser.add_argument("audio_path", help="Đường dẫn đến file audio")
    parser.add_argument("--model", default="vosk-model-small-vn-0.3", 
                       choices=["vosk-model-small-vn-0.3", "vosk-model-small-vn-0.4", "vosk-model-vn-0.4"],
                       help="Mô hình Vosk tiếng Việt để sử dụng")
    parser.add_argument("--format", default="txt", choices=["txt", "srt"], 
                       help="Định dạng đầu ra (txt hoặc srt)")
    parser.add_argument("--compare", action="store_true", 
                       help="So sánh kết quả từ tất cả các mô hình")
    
    args = parser.parse_args()
    
    script_start_time = time.time()
    
    if args.compare:
        # Chạy tất cả các mô hình và so sánh kết quả
        results = {}
        models = ["vosk-model-small-vn-0.3", "vosk-model-small-vn-0.4", "vosk-model-vn-0.4"]
        
        for model in models:
            print(f"\n{'='*50}")
            print(f"ĐANG XỬ LÝ VỚI MÔ HÌNH: {model}")
            print(f"{'='*50}\n")
            
            model_start_time = time.time()
            results[model] = transcribe_with_vosk(args.audio_path, model, args.format)
            model_total_time = time.time() - model_start_time
            
            print(f"Tổng thời gian xử lý cho {model}: {model_total_time:.2f} giây")
        
        # In bảng so sánh thời gian
        print("\n\n=========== SO SÁNH KẾT QUẢ CÁC MÔ HÌNH ===========")
        for model, result in results.items():
            print(f"\nMô hình: {model}")
            print(f"Kết quả: {result[:150]}..." if len(result) > 150 else f"Kết quả: {result}")
            print(f"{'-'*60}")
    else:
        # Chỉ chạy một mô hình được chỉ định
        transcribe_with_vosk(args.audio_path, args.model, args.format)
    
    script_total_time = time.time() - script_start_time
    print(f"Thời gian chạy toàn bộ script: {script_total_time:.2f} giây")