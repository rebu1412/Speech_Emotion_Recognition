import torch
import librosa
from transformers import HubertForSequenceClassification
import os

def process_audio(file_path):
    # Tải âm thanh từ file path sử dụng librosa
    speech, _ = librosa.load(file_path, sr=16000, mono=True)
    # Chuyển đổi dữ liệu âm thanh thành tensor
    speech_tensor = torch.tensor(speech)
    # Tạo attention mask với tất cả giá trị là 1
    attention_mask = torch.ones_like(speech_tensor)
    return {
        "input_values": speech_tensor,
        "attention_mask": attention_mask
    }

def predict_audio(audio_file):
    # Load pre-trained model
    model = HubertForSequenceClassification.from_pretrained("superb/hubert-large-superb-er")
    model.load_state_dict(torch.load("path/to/save/model.pth"))
    model.eval()  # Đặt mô hình vào chế độ đánh giá
    
    # Xử lý âm thanh
    processed_audio = process_audio(audio_file)
    input_values = processed_audio["input_values"].unsqueeze(0)
    attention_mask = processed_audio["attention_mask"].unsqueeze(0) 
    
    # Thực hiện dự đoán
    with torch.no_grad():
        outputs = model(input_values, attention_mask=attention_mask)  # Dự đoán các nhãn
        predictions = predict(outputs, model.config)  # Dự đoán cảm xúc từ outputs và cấu hình mô hình
    
    return predictions

# Hàm predict
def predict(outputs, config):
    logits = outputs.logits[0]  # Lấy logits từ kết quả dự đoán
    predicted_label_id = torch.argmax(logits).item()  # Lấy chỉ mục của nhãn dự đoán
    emotion = get_emotion(predicted_label_id, config)  # Chuyển đổi chỉ mục thành cảm xúc
    return emotion

# Hàm chuyển đổi chỉ mục thành cảm xúc
def get_emotion(label_id, config):
    return config.id2label[label_id]