import os
import string
import torch
import torchaudio
from datetime import datetime
from tqdm import tqdm
from underthesea import sent_tokenize
from unidecode import unidecode

try:
    from vinorm import TTSnorm
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
except:
    print("Không thể import một số thư viện cần thiết")

class TextToSpeech:
    def __init__(self, model_path="model", device=None):
        """
        Khởi tạo module Text to Speech với XTTS
        
        Args:
            model_path (str): Đường dẫn đến thư mục chứa model
            device (str, optional): Thiết bị để chạy model ("cuda" hoặc "cpu")
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.language_code_map = {
            "vietnamese": "vi",
            "english": "en",
            "spanish": "es",
            "french": "fr",
            "german": "de",
            "italian": "it",
            "portuguese": "pt",
            "polish": "pl",
            "turkish": "tr",
            "russian": "ru",
            "dutch": "nl",
            "czech": "cs",
            "arabic": "ar",
            "chinese": "zh-cn",
            "japanese": "ja",
            "hungarian": "hu",
            "korean": "ko",
            "hindi": "hi"
        }
        
        # Danh sách giọng mẫu có sẵn
        self.voices = {
            "vi_female": "model/vi_sample.wav",
            "nbt": "voices/nbt_voice.wav",
            "seren": "voices/Seren2.wav",
            # Thêm các giọng mẫu khác ở đây
        }
        
        # Tự động load model
        self._load_model()
    
    def _clear_gpu_cache(self):
        """Xóa bộ nhớ cache GPU nếu đang sử dụng CUDA"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _load_model(self):
        """Load model XTTS từ checkpoint"""
        self._clear_gpu_cache()
        
        xtts_checkpoint = os.path.join(self.model_path, "model.pth")
        xtts_config = os.path.join(self.model_path, "config.json")
        xtts_vocab = os.path.join(self.model_path, "vocab.json")
        
        if not os.path.exists(xtts_checkpoint) or not os.path.exists(xtts_config) or not os.path.exists(xtts_vocab):
            raise ValueError(f"Không tìm thấy các file model cần thiết trong thư mục {self.model_path}")
        
        try:
            config = XttsConfig()
            config.load_json(xtts_config)
            self.model = Xtts.init_from_config(config)
            print("Đang nạp mô hình XTTS...")
            self.model.load_checkpoint(config,
                                    checkpoint_path=xtts_checkpoint,
                                    vocab_path=xtts_vocab,
                                    use_deepspeed=False)
            
            if self.device == "cuda":
                self.model.cuda()
                
            print("Đã nạp mô hình thành công!")
        except Exception as e:
            print(f"Lỗi khi nạp mô hình XTTS: {str(e)}")
            raise
    
    def get_available_voices(self):
        """
        Lấy danh sách giọng nói có sẵn
        
        Returns:
            dict: Dictionary của tên giọng và đường dẫn file
        """
        return self.voices
    
    def _get_file_name(self, text, max_char=50):
        """
        Tạo tên file từ text đầu vào
        
        Args:
            text (str): Văn bản đầu vào
            max_char (int): Số ký tự tối đa lấy từ text
            
        Returns:
            str: Tên file đã được xử lý
        """
        filename = text[:max_char]
        filename = filename.lower()
        filename = filename.replace(" ", "_")
        filename = filename.translate(str.maketrans("", "", string.punctuation.replace("_", "")))
        filename = unidecode(filename)
        current_datetime = datetime.now().strftime("%m%d%H%M%S")
        filename = f"{current_datetime}_{filename}"
        return filename
    
    def _calculate_keep_len(self, text, lang):
        """
        Tính toán độ dài âm thanh cần giữ lại
        
        Args:
            text (str): Văn bản đầu vào
            lang (str): Mã ngôn ngữ
            
        Returns:
            int: Độ dài âm thanh cần giữ lại
        """
        if lang in ["ja", "zh-cn"]:
            return -1

        word_count = len(text.split())
        num_punct = (
            text.count(".")
            + text.count("!")
            + text.count("?")
            + text.count(",")
        )

        if word_count < 5:
            return 15000 * word_count + 2000 * num_punct
        elif word_count < 10:
            return 13000 * word_count + 2000 * num_punct
        return -1
    
    def _normalize_vietnamese_text(self, text):
        """
        Chuẩn hóa văn bản tiếng Việt
        
        Args:
            text (str): Văn bản cần chuẩn hóa
            
        Returns:
            str: Văn bản đã chuẩn hóa
        """
        try:
            text = (
                TTSnorm(text, unknown=False, lower=False, rule=True)
                .replace("..", ".")
                .replace("!.", "!")
                .replace("?.", "?")
                .replace(" .", ".")
                .replace(" ,", ",")
                .replace('"', "")
                .replace("'", "")
                .replace("AI", "Ây Ai")
                .replace("A.I", "Ây Ai")
                .replace("+", "cộng")
                .replace("-", "trừ")
                .replace("*", "nhân")
                .replace("/", "chia")
                .replace("=", "bằng")
            )
            return text
        except Exception:
            print("Lỗi khi chuẩn hóa văn bản tiếng Việt, sử dụng văn bản gốc")
            return text
    
    def text_to_speech(self, text, language="vietnamese", voice_name=None, voice_path=None, 
                       output_path=None, normalize_text=True, output_chunks=False,
                       temperature=0.3, length_penalty=1.0, repetition_penalty=10.0,
                       top_k=30, top_p=0.85):
        """
        Chuyển đổi văn bản thành giọng nói sử dụng model XTTS
        
        Args:
            text (str): Văn bản cần chuyển đổi
            language (str): Ngôn ngữ của văn bản
            voice_path (str): Đường dẫn đến file giọng mẫu
            output_path (str): Đường dẫn lưu file audio
            normalize_text (bool): Có chuẩn hóa văn bản hay không
            output_chunks (bool): Có lưu từng đoạn âm thanh riêng lẻ không
            temperature (float): Nhiệt độ cho quá trình sinh âm thanh
            length_penalty (float): Hệ số điều chỉnh độ dài
            repetition_penalty (float): Hệ số điều chỉnh sự lặp lại
            top_k (int): Tham số top_k cho việc sinh âm thanh
            top_p (float): Tham số top_p cho việc sinh âm thanh
            
        Returns:
            str: Đường dẫn đến file audio hoặc None nếu lỗi
        """
        if self.model is None:
            raise ValueError("Model chưa được nạp")
        
        # Xử lý ngôn ngữ
        lang = language.lower()
        if lang in self.language_code_map:
            lang_code = self.language_code_map[lang]
        else:
            lang_code = lang  # Giả sử người dùng đã nhập mã ngôn ngữ
        
        # Xử lý voice_name và voice_path
        if voice_path is None and voice_name is None:
            # Không có cả voice_name và voice_path
            if lang_code == "vi":
                voice_path = self.voices.get("vi_female")
            else:
                raise ValueError(f"Cần cung cấp voice_name hoặc voice_path cho ngôn ngữ {language}")
        elif voice_name is not None:
            # Ưu tiên sử dụng voice_name nếu được cung cấp
            # Kiểm tra xem voice_name có trong voices không
            if voice_name in self.voices:
                voice_path = self.voices[voice_name]
            else:
                # Thử dùng voice_name trực tiếp làm đường dẫn
                sample_path = os.path.join(self.model_path, f"{voice_name}.wav")
                if os.path.exists(sample_path):
                    voice_path = sample_path
                else:
                    # Nếu không có sẵn, sử dụng giọng mặc định
                    voice_path = self.voices.get("vi_female")
                    print(f"Không tìm thấy giọng {voice_name}, sử dụng giọng mặc định")
        elif voice_path in self.voices:
            # Nếu voice_path là key trong voices
            voice_path = self.voices[voice_path]
        
        # Kiểm tra file giọng mẫu
        if not os.path.exists(voice_path):
            print(f"Cảnh báo: Không tìm thấy file giọng mẫu: {voice_path}")
            # Tìm file giọng mẫu mặc định trong thư mục model
            default_sample = os.path.join(self.model_path, "vi_sample.wav")
            if os.path.exists(default_sample):
                print(f"Sử dụng file giọng mẫu mặc định: {default_sample}")
                voice_path = default_sample
            else:
                raise ValueError(f"Không tìm thấy file giọng mẫu: {voice_path}")
        
        # Xử lý output path
        if output_path is None:
            output_dir = "./output"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{self._get_file_name(text)}.wav")
        
        try:
            # Lấy conditioning latents từ file giọng mẫu
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                audio_path=voice_path,
                gpt_cond_len=self.model.config.gpt_cond_len,
                max_ref_length=self.model.config.max_ref_len,
                sound_norm_refs=self.model.config.sound_norm_refs,
            )
            
            # Chuẩn hóa văn bản nếu cần
            if normalize_text and lang_code == "vi":
                text = self._normalize_vietnamese_text(text)
            
            # Tách văn bản thành các câu
            if lang_code in ["ja", "zh-cn"]:
                text_segments = text.split("。")
            else:
                text_segments = sent_tokenize(text)
            
            # Sinh âm thanh cho từng đoạn văn bản
            wav_chunks = []
            for segment in tqdm(text_segments):
                if segment.strip() == "":
                    continue
                
                wav_chunk = self.model.inference(
                    text=segment,
                    language=lang_code,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=temperature,
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,
                    top_k=top_k,
                    top_p=top_p,
                )
                
                # Chuyển đổi sang tensor và cắt âm thanh nếu cần
                keep_len = self._calculate_keep_len(segment, lang_code)
                if keep_len > 0 and len(wav_chunk["wav"]) > keep_len:
                    wav = wav_chunk["wav"][:keep_len]
                else:
                    wav = wav_chunk["wav"]
                
                # Đảm bảo là tensor
                if not isinstance(wav, torch.Tensor):
                    wav = torch.tensor(wav)
                
                # Lưu từng đoạn nếu cần
                if output_chunks:
                    chunk_dir = os.path.dirname(output_path)
                    chunk_name = f"{self._get_file_name(segment)}.wav"
                    chunk_path = os.path.join(chunk_dir, chunk_name)
                    torchaudio.save(chunk_path, wav.unsqueeze(0), 24000)
                
                wav_chunks.append(wav)
            
            # Đảm bảo tất cả các đoạn là tensors
            tensor_chunks = []
            for chunk in wav_chunks:
                if not isinstance(chunk, torch.Tensor):
                    chunk = torch.tensor(chunk)
                tensor_chunks.append(chunk)
                
            # Ghép các đoạn âm thanh và lưu kết quả
            if tensor_chunks:
                out_wav = torch.cat(tensor_chunks, dim=0).unsqueeze(0)
                # Đảm bảo thư mục đầu ra tồn tại
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                torchaudio.save(output_path, out_wav, 24000)
            else:
                print("Không có đoạn âm thanh nào được tạo")
            
            return output_path
            
        except Exception as e:
            print(f"Lỗi khi chuyển đổi text to speech: {str(e)}")
            return None