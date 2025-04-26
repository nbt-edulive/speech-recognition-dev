import re
import os
import numpy as np
import argparse
from jiwer import wer, cer
import difflib
import json
from tabulate import tabulate
import matplotlib.pyplot as plt

def normalize_text(text):
    """
    Chuẩn hóa văn bản: chuyển thành chữ thường, loại bỏ dấu câu và khoảng trắng thừa
    """
    # Chuyển thành chữ thường
    text = text.lower()
    
    # Loại bỏ dấu câu
    text = re.sub(r'[.,;:!?()\[\]{}"\']', '', text)
    
    # Thay thế nhiều khoảng trắng bằng một khoảng trắng
    text = re.sub(r'\s+', ' ', text)
    
    # Loại bỏ khoảng trắng ở đầu và cuối
    text = text.strip()
    
    return text

def evaluate_asr_result(reference, hypothesis):
    """
    Đánh giá kết quả nhận dạng tiếng nói
    
    Tham số:
        reference (str): Văn bản chuẩn (ground truth)
        hypothesis (str): Văn bản được nhận dạng từ mô hình
        
    Trả về:
        dict: Các chỉ số đánh giá
    """
    # Chuẩn hóa văn bản
    reference_normalized = normalize_text(reference)
    hypothesis_normalized = normalize_text(hypothesis)
    
    # Tính WER (Word Error Rate)
    word_error_rate = wer(reference_normalized, hypothesis_normalized)
    
    # Tính CER (Character Error Rate)
    character_error_rate = cer(reference_normalized, hypothesis_normalized)
    
    # Tính Word Accuracy và Character Accuracy
    word_accuracy = 1 - word_error_rate
    char_accuracy = 1 - character_error_rate
    
    # Phân tích chi tiết sự khác biệt
    reference_words = reference_normalized.split()
    hypothesis_words = hypothesis_normalized.split()
    
    # Sử dụng difflib để tìm sự khác biệt
    d = difflib.Differ()
    diff = list(d.compare(reference_words, hypothesis_words))
    
    # Đếm số lượng các loại lỗi
    substitutions = 0
    deletions = 0
    insertions = 0
    
    for item in diff:
        if item.startswith('- '):
            deletions += 1
        elif item.startswith('+ '):
            insertions += 1
        elif item.startswith('? '):
            substitutions += 1
    
    # Hiển thị các từ sai
    incorrect_words = []
    
    # Sử dụng difflib.SequenceMatcher để tìm các từ sai
    matcher = difflib.SequenceMatcher(None, reference_words, hypothesis_words)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == 'replace':
            for i, j in zip(range(i1, i2), range(j1, j2)):
                incorrect_words.append({
                    'reference': reference_words[i],
                    'hypothesis': hypothesis_words[j]
                })
        elif op == 'delete':
            for i in range(i1, i2):
                incorrect_words.append({
                    'reference': reference_words[i],
                    'hypothesis': '[MISSING]'
                })
        elif op == 'insert':
            for j in range(j1, j2):
                incorrect_words.append({
                    'reference': '[MISSING]',
                    'hypothesis': hypothesis_words[j]
                })
    
    return {
        'wer': word_error_rate,
        'cer': character_error_rate,
        'word_accuracy': word_accuracy,
        'character_accuracy': char_accuracy,
        'substitutions': substitutions,
        'deletions': deletions,
        'insertions': insertions,
        'incorrect_words': incorrect_words,
        'reference_length': len(reference_words),
        'hypothesis_length': len(hypothesis_words)
    }

def print_evaluation_report(model_name, metrics):
    """In báo cáo đánh giá cho một mô hình"""
    print(f"\n===== BÁO CÁO ĐÁNH GIÁ MÔ HÌNH: {model_name} =====")
    print(f"Word Error Rate (WER): {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%)")
    print(f"Character Error Rate (CER): {metrics['cer']:.4f} ({metrics['cer']*100:.2f}%)")
    print(f"Word Accuracy: {metrics['word_accuracy']:.4f} ({metrics['word_accuracy']*100:.2f}%)")
    print(f"Character Accuracy: {metrics['character_accuracy']:.4f} ({metrics['character_accuracy']*100:.2f}%)")
    print("\nPhân tích lỗi:")
    print(f"- Substitutions (thay thế): {metrics['substitutions']}")
    print(f"- Deletions (xóa): {metrics['deletions']}")
    print(f"- Insertions (chèn): {metrics['insertions']}")
    print(f"\nĐộ dài văn bản chuẩn: {metrics['reference_length']} từ")
    print(f"Độ dài văn bản nhận dạng: {metrics['hypothesis_length']} từ")
    
    if metrics['incorrect_words']:
        print("\nCác từ lỗi:")
        for i, word_pair in enumerate(metrics['incorrect_words'], 1):
            print(f"{i}. Chuẩn: '{word_pair['reference']}' → Nhận dạng: '{word_pair['hypothesis']}'")
    
    print("===============================================")

def compare_models(reference, hypotheses):
    """
    So sánh kết quả của nhiều mô hình với văn bản chuẩn
    
    Tham số:
        reference (str): Văn bản chuẩn
        hypotheses (dict): Dictionary chứa tên mô hình và kết quả nhận dạng
    """
    results = {}
    
    # Đánh giá từng mô hình
    for model_name, hypothesis in hypotheses.items():
        results[model_name] = evaluate_asr_result(reference, hypothesis)
    
    # Tạo bảng so sánh
    comparison_table = []
    headers = ['Mô hình', 'WER (%)', 'CER (%)', 'Word Acc. (%)', 'Char Acc. (%)']
    
    for model_name, metrics in results.items():
        row = [
            model_name,
            f"{metrics['wer']*100:.2f}",
            f"{metrics['cer']*100:.2f}",
            f"{metrics['word_accuracy']*100:.2f}",
            f"{metrics['character_accuracy']*100:.2f}"
        ]
        comparison_table.append(row)
    
    # Sắp xếp kết quả theo WER (tăng dần)
    comparison_table.sort(key=lambda x: float(x[1]))
    
    # In bảng so sánh
    print("\n===== SO SÁNH HIỆU SUẤT CÁC MÔ HÌNH =====")
    print(tabulate(comparison_table, headers=headers, tablefmt='grid'))
    
    # Tạo biểu đồ so sánh WER và CER
    plot_model_comparison(results)
    
    return results

def plot_model_comparison(results):
    """Tạo biểu đồ so sánh các mô hình"""
    try:
        model_names = list(results.keys())
        wer_values = [results[model]['wer'] * 100 for model in model_names]
        cer_values = [results[model]['cer'] * 100 for model in model_names]
        
        # Sắp xếp theo WER (tăng dần)
        sorted_indices = np.argsort(wer_values)
        model_names = [model_names[i] for i in sorted_indices]
        wer_values = [wer_values[i] for i in sorted_indices]
        cer_values = [cer_values[i] for i in sorted_indices]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        wer_bars = ax.bar(x - width/2, wer_values, width, label='WER (%)')
        cer_bars = ax.bar(x + width/2, cer_values, width, label='CER (%)')
        
        ax.set_ylabel('Error Rate (%)')
        ax.set_title('Comparison of ASR Models')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        
        # Thêm giá trị trên mỗi cột
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        autolabel(wer_bars)
        autolabel(cer_bars)
        
        fig.tight_layout()
        
        # Lưu biểu đồ
        plt.savefig('asr_model_comparison.png')
        print("Đã lưu biểu đồ so sánh tại: asr_model_comparison.png")
        
        # Hiển thị biểu đồ nếu chạy trong môi trường hỗ trợ
        plt.show()
    except Exception as e:
        print(f"Không thể tạo biểu đồ: {e}")

def read_text_file(file_path):
    """Đọc file văn bản"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Lỗi khi đọc file {file_path}: {e}")
        return None

def save_results_to_json(results, output_file):
    """Lưu kết quả đánh giá vào file JSON"""
    # Chuyển đổi incorrect_words thành danh sách đơn giản
    simplified_results = {}
    for model, metrics in results.items():
        simplified_metrics = metrics.copy()
        incorrect_words_list = []
        for word_pair in metrics['incorrect_words']:
            incorrect_words_list.append({
                'reference': word_pair['reference'],
                'hypothesis': word_pair['hypothesis']
            })
        simplified_metrics['incorrect_words'] = incorrect_words_list
        simplified_results[model] = simplified_metrics
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(simplified_results, f, ensure_ascii=False, indent=2)
        print(f"Đã lưu kết quả đánh giá chi tiết vào: {output_file}")
    except Exception as e:
        print(f"Lỗi khi lưu kết quả vào {output_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá và so sánh kết quả nhận dạng tiếng nói từ nhiều mô hình")
    parser.add_argument("--reference", required=True, help="File văn bản chuẩn (ground truth)")
    parser.add_argument("--models", nargs='+', required=True, help="Các file kết quả từ các mô hình khác nhau")
    parser.add_argument("--names", nargs='+', help="Tên của các mô hình tương ứng (nếu không cung cấp, sẽ sử dụng tên file)")
    parser.add_argument("--output", default="asr_evaluation_results.json", help="File output cho kết quả đánh giá chi tiết (JSON)")
    
    args = parser.parse_args()
    
    # Đọc văn bản chuẩn
    reference_text = read_text_file(args.reference)
    if not reference_text:
        print("Không thể tiếp tục khi không có văn bản chuẩn")
        exit(1)
    
    # Đọc kết quả từ các mô hình
    hypotheses = {}
    model_names = args.names if args.names else [os.path.basename(file_path).split('.')[0] for file_path in args.models]
    
    if len(model_names) != len(args.models):
        print("Số lượng tên mô hình không khớp với số lượng file kết quả")
        exit(1)
    
    for model_name, file_path in zip(model_names, args.models):
        hypothesis_text = read_text_file(file_path)
        if hypothesis_text:
            hypotheses[model_name] = hypothesis_text
        else:
            print(f"Bỏ qua mô hình {model_name} do không đọc được file kết quả")
    
    if not hypotheses:
        print("Không có kết quả mô hình nào để đánh giá")
        exit(1)
    
    # So sánh các mô hình
    results = compare_models(reference_text, hypotheses)
    
    # In báo cáo chi tiết cho từng mô hình
    for model_name, metrics in results.items():
        print_evaluation_report(model_name, metrics)
    
    # Lưu kết quả vào file JSON
    save_results_to_json(results, args.output)