import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()
class GeminiLLM:
    def __init__(self, api_key=None, model_name="gemini-1.5-flash"):
        """
        Khởi tạo module LLM sử dụng Gemini API
        
        Args:
            api_key (str, optional): API key cho Gemini
            model_name (str): Tên model Gemini ("gemini-1.5-pro", "gemini-1.5-flash", etc.)
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Cần cung cấp Gemini API key")
        
        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        
        # System prompt mặc định
        self.system_prompt = """
        Bạn là một trợ lý gia sư thân thiện, chuyên môn trong việc giúp học sinh các cấp học từ tiểu học đến trung học phổ thông.
        Hãy trả lời câu hỏi của học sinh một cách rõ ràng, dễ hiểu và phù hợp với độ tuổi.
        Hãy sử dụng ngôn ngữ đơn giản, thân thiện và khuyến khích học sinh tư duy.
        Khi giải thích các khái niệm khó, hãy sử dụng ví dụ thực tế và liên hệ với cuộc sống hàng ngày.
        Khi trả lời, hãy giữ câu trả lời ngắn gọn, dễ hiểu và súc tích (tối đa 3-4 câu).
        """
    
    def set_system_prompt(self, new_prompt):
        """
        Cập nhật system prompt
        
        Args:
            new_prompt (str): System prompt mới
        """
        self.system_prompt = new_prompt
    
    def get_response(self, user_query, conversation_history=""):
        """
        Lấy phản hồi từ model Gemini
        
        Args:
            user_query (str): Câu hỏi của người dùng
            conversation_history (str): Lịch sử hội thoại trước đó
            
        Returns:
            str: Câu trả lời từ model
        """
        try:
            # Tạo nội dung cho prompt
            conversation_context = f"""
            {self.system_prompt}
            
            Lịch sử trò chuyện:
            {conversation_history}
            
            Câu hỏi của học sinh: {user_query}
            """
            
            response = self.model.generate_content(conversation_context)
            return response.text
        except Exception as e:
            print(f"Lỗi khi truy vấn Gemini API: {str(e)}")
            return f"Xin lỗi, tôi đang gặp vấn đề kỹ thuật: {str(e)}"