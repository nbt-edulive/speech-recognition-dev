import speech_recognition as sr
import time
import os

def recognize_from_microphone():
    """Nhận dạng giọng nói từ microphone"""
    # Khởi tạo recognizer
    recognizer = sr.Recognizer()
    
    print("=== NHẬN DẠNG GIỌNG NÓI TỪ MICROPHONE ===")
    print("Hãy chuẩn bị nói...")
    
    # Sử dụng microphone làm nguồn âm thanh
    with sr.Microphone() as source:
        print("Đang hiệu chỉnh cho tiếng ồn xung quanh...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        print("Bắt đầu nói ngay bây giờ!")
        
        try:
            # Nghe âm thanh từ microphone
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            print("Đã ghi âm xong! Đang xử lý...")
            
            # Sử dụng các engine khác nhau để nhận dạng
            try:
                # Google (yêu cầu kết nối internet)
                text = recognizer.recognize_google(audio, language="vi-VN")
                print(f"Google đã nhận dạng: {text}")
            except sr.UnknownValueError:
                print("Google không thể nhận dạng giọng nói")
            except sr.RequestError as e:
                print(f"Không thể kết nối đến Google Speech Recognition service; {e}")
            
        except sr.WaitTimeoutError:
            print("Không phát hiện giọng nói nào trong thời gian chờ")
            
        except Exception as e:
            print(f"Lỗi: {e}")
    
    print("===== KẾT THÚC NHẬN DẠNG =====")

def recognize_from_file(audio_file_path):
    """Nhận dạng giọng nói từ file âm thanh"""
    # Kiểm tra file có tồn tại không
    if not os.path.exists(audio_file_path):
        print(f"File {audio_file_path} không tồn tại!")
        return
    
    # Khởi tạo recognizer
    recognizer = sr.Recognizer()
    
    print(f"=== NHẬN DẠNG GIỌNG NÓI TỪ FILE: {audio_file_path} ===")
    
    # Mở file âm thanh
    with sr.AudioFile(audio_file_path) as source:
        # Đọc toàn bộ file
        audio = recognizer.record(source)
        print("Đang xử lý...")
        
        try:
            # Google (yêu cầu kết nối internet)
            text = recognizer.recognize_google(audio, language="vi-VN")
            print(f"Google đã nhận dạng: {text}")
        except sr.UnknownValueError:
            print("Google không thể nhận dạng giọng nói")
        except sr.RequestError as e:
            print(f"Không thể kết nối đến Google Speech Recognition service; {e}")
    
    print("===== KẾT THÚC NHẬN DẠNG =====")

def record_to_file(output_file, duration=5):
    """Ghi âm và lưu vào file"""
    # Khởi tạo recognizer
    recognizer = sr.Recognizer()
    
    print(f"=== GHI ÂM VÀO FILE: {output_file} ===")
    print(f"Sẽ ghi âm trong {duration} giây...")
    
    # Sử dụng microphone làm nguồn âm thanh
    with sr.Microphone() as source:
        print("Đang hiệu chỉnh cho tiếng ồn xung quanh...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        print("Bắt đầu nói ngay bây giờ!")
        
        try:
            # Nghe âm thanh từ microphone
            audio = recognizer.listen(source, timeout=duration, phrase_time_limit=duration)
            print("Đã ghi âm xong! Đang lưu vào file...")
            
            # Ghi vào file WAV
            with open(output_file, "wb") as file:
                file.write(audio.get_wav_data())
            
            print(f"Đã lưu âm thanh vào file {output_file}")
            
        except sr.WaitTimeoutError:
            print("Không phát hiện giọng nói nào trong thời gian chờ")
            
        except Exception as e:
            print(f"Lỗi: {e}")
    
    print("===== KẾT THÚC GHI ÂM =====")

if __name__ == "__main__":
    while True:
        print("\nCHƯƠNG TRÌNH NHẬN DẠNG GIỌNG NÓI")
        print("1. Nhận dạng từ microphone")
        print("2. Ghi âm và lưu vào file")
        print("3. Nhận dạng từ file WAV")
        print("0. Thoát")
        
        choice = input("\nNhập lựa chọn của bạn: ")
        
        if choice == "1":
            recognize_from_microphone()
        elif choice == "2":
            output_file = input("Nhập tên file để lưu (ví dụ: recording.wav): ")
            duration = int(input("Nhập thời gian ghi âm (giây): "))
            record_to_file(output_file, duration)
        elif choice == "3":
            audio_file = input("Nhập đường dẫn đến file WAV: ")
            recognize_from_file(audio_file)
        elif choice == "0":
            print("Tạm biệt!")
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng thử lại.")