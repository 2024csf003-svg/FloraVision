import os
import json
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
from PIL import Image
from torchvision import transforms

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), 'model')

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'gif'}
MAX_FILE_SIZE = 5 * 1024 * 1024

with open(os.path.join(DATA_DIR, 'class_names_25.json'), 'r') as f:
    class_names = json.load(f)

with open(os.path.join(DATA_DIR, 'details.json'), 'r') as f:
    flower_details = json.load(f)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=25):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)


model = SimpleCNN(25)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'simple_model.pt'), map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

history = []


@app.route('/')
def welcome():
    return render_template('welcome.html')


@app.route('/home')
def classifier():
    return render_template('classifier.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided', 'code': 'NO_IMAGE'}), 400

    file = request.files['image']
    filename = file.filename.lower()

    if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        return jsonify({'error': 'Invalid format. Use JPG, PNG, WebP, or GIF only.', 'code': 'INVALID_FORMAT'}), 400

    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size > MAX_FILE_SIZE:
        return jsonify({'error': 'File too large. Maximum 5MB allowed.', 'code': 'FILE_TOO_LARGE'}), 400

    try:
        img = Image.open(file.stream)
        img.verify()
        file.seek(0)
        img = Image.open(file.stream).convert('RGB')
    except Exception:
        return jsonify({'error': 'Could not read image. File may be corrupted.', 'code': 'CORRUPT_IMAGE'}), 400

    try:
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, dim=1)
            confidence = confidence.item()
            predicted_class = predicted.item()

        flower_name = class_names[str(predicted_class)]
        is_unknown = confidence < 0.3

        if is_unknown:
            flower_name = "Unknown Flower"

        top_probs, top_indices = torch.topk(probs, 3, dim=1)
        alternatives = []
        for i in range(1, 3):
            alt_idx = top_indices[0][i].item()
            alt_name = class_names[str(alt_idx)]
            alt_conf = top_probs[0][i].item()
            alternatives.append({
                'flower': alt_name,
                'confidence': round(alt_conf * 100, 1)
            })

        result = {
            'flower': flower_name,
            'confidence': round(confidence * 100, 1),
            'is_unknown': is_unknown,
            'description': flower_details.get(flower_name, "A beautiful flower from our collection."),
            'alternatives': alternatives
        }

        history.insert(0, result)
        if len(history) > 10:
            history.pop()

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': 'Processing failed. Please try another image.', 'code': 'PROCESSING_ERROR'}), 500


@app.route('/classes', methods=['GET'])
def get_classes():
    return jsonify({'classes': list(class_names.values())})


@app.route('/history', methods=['GET'])
def get_history():
    total = len(history)
    avg_confidence = round(sum(h['confidence'] for h in history) / total, 1) if total > 0 else 0
    return jsonify({
        'history': history,
        'stats': {
            'total': total,
            'avg_confidence': avg_confidence
        }
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)