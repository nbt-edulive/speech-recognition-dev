import speech_recognition as sr
import os
from pydub import AudioSegment
import tempfile

class SpeechToText:
    def __init__(self, model_size=None, device=None, compute_type=None, language="vi"):
        """
        Khởi tạo module Speech to Text với Google Speech Recognition
        
        Args:
            model_size (str): Không sử dụng, giữ lại để tương thích với interface cũ
            device (str): Không sử dụng, giữ lại để tương thích với interface cũ
            compute_type (str): Không sử dụng, giữ lại để tương thích với interface cũ
            language (str): Mã ngôn ngữ mặc định ("vi" cho Tiếng Việt)
        """
        self.recognizer = sr.Recognizer()
        self.language = language
    
    def convert_to_wav(self, audio_file):
        """
        Chuyển đổi file âm thanh sang định dạng WAV chuẩn nếu cần
        
        Args:
            audio_file (str): Đường dẫn đến file âm thanh
            
        Returns:
            str: Đường dẫn đến file WAV (file gốc hoặc file tạm nếu cần chuyển đổi)
        """
        # Kiểm tra đuôi file
        file_ext = os.path.splitext(audio_file)[1].lower()
        
        # Nếu không phải WAV hoặc là WAV không chuẩn, chuyển đổi
        if file_ext != '.wav' or self._check_wav_format(audio_file) is False:
            try:
                # Tạo tên file tạm thời
                temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
                
                # Chuyển đổi sang WAV
                sound = AudioSegment.from_file(audio_file)
                sound.export(temp_wav, format="wav", parameters=["-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1"])
                
                return temp_wav
            except Exception as e:
                print(f"Lỗi khi chuyển đổi file âm thanh: {e}")
                # Nếu chuyển đổi thất bại, vẫn trả về file gốc
                return audio_file
        
        # Nếu là WAV chuẩn, trả về đường dẫn gốc
        return audio_file
    
    def _check_wav_format(self, wav_file):
        """
        Kiểm tra xem file WAV có định dạng chuẩn không
        
        Args:
            wav_file (str): Đường dẫn đến file WAV
            
        Returns:
            bool: True nếu file có định dạng chuẩn, False nếu không
        """
        try:
            with sr.AudioFile(wav_file) as source:
                # Thử đọc một chút âm thanh để kiểm tra định dạng
                self.recognizer.record(source, duration=0.1)
            return True
        except Exception:
            return False
    
    def transcribe(self, audio_file):
        """
        Chuyển đổi audio thành văn bản
        
        Args:
            audio_file (str): Đường dẫn đến file audio
            
        Returns:
            str: Văn bản được chuyển đổi
        """
        converted_file = self.convert_to_wav(audio_file)
        temp_file_created = converted_file != audio_file
        
        try:
            with sr.AudioFile(converted_file) as source:
                # Điều chỉnh cho tiếng ồn môi trường
                self.recognizer.adjust_for_ambient_noise(source)
                
                # Ghi âm toàn bộ dữ liệu từ file
                audio_data = self.recognizer.record(source)
                
                # Nhận dạng với Google Speech Recognition
                text = self.recognizer.recognize_google(audio_data, language=self.language)
                
                return text.strip()
        except sr.UnknownValueError:
            return "Không thể nhận dạng giọng nói"
        except sr.RequestError as e:
            return f"Lỗi khi kết nối đến dịch vụ Google Speech Recognition: {e}"
        except Exception as e:
            return f"Lỗi khi xử lý âm thanh: {e}"
        finally:
            # Xóa file tạm nếu đã tạo
            if temp_file_created and os.path.exists(converted_file):
                try:
                    os.remove(converted_file)
                except:
                    pass