from faster_whisper import WhisperModel

class SpeechToText:
    def __init__(self, model_size="tiny", device="cpu", compute_type="int8", language="vi"):
        """
        Khởi tạo module Speech to Text với Whisper
        
        Args:
            model_size (str): Kích thước model ("tiny", "base", "small", "medium", "large")
            device (str): Thiết bị tính toán ("cpu" hoặc "cuda")
            compute_type (str): Kiểu tính toán ("int8", "float16", "float32")
            language (str): Mã ngôn ngữ mặc định ("vi" cho Tiếng Việt)
        """
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.language = language
    
    def transcribe(self, audio_file):
        """
        Chuyển đổi audio thành văn bản
        
        Args:
            audio_file (str): Đường dẫn đến file audio
            
        Returns:
            str: Văn bản được chuyển đổi
        """
        segments, info = self.model.transcribe(audio_file, language=self.language)
        
        full_text = ""
        for segment in segments:
            full_text += segment.text + " "
        
        return full_text.strip()