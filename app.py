import os
from flask import Flask, request, render_template, redirect, url_for
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import uuid

app = Flask(__name__)

# Set upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Validation transform
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load class mapping
with open("class_mapping.json", "r") as f:
    class_to_idx = json.load(f)

class_names = [None] * len(class_to_idx)
for class_name, idx in class_to_idx.items():
    class_names[idx] = class_name

# Load model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = val_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = torch.argmax(output, 1).item()
        predicted_class = class_names[predicted_idx]
    return predicted_class

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Save file with a unique name
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Predict
            prediction = predict_image(filepath)

            return render_template('index.html', prediction=prediction, image_url=filepath)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)