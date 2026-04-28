# FloraVision - AI Flower Classifier

Build a modern flower identification web app using PyTorch + Flask with fresh design.

## Tech Stack
- Python + PyTorch + Flask
- HTML/CSS/JS (no frameworks)
- Poppins font, Font Awesome icons

## Features
1. **Welcome Page** - Centered card, animated gradient bg, 3 feature cards, "Enter App" button
2. **Classifier Page** - Header, upload zone, preview, results with badges, history sidebar, stats
3. **Glassmorphism UI** - Mint (#00D9A5) + Coral (#FF6B9D) + Purple (#A78BFA) palette
4. **25 flower classes** - Use provided model and class names

## Structure
```
floravision/
├── app.py           # Flask app
├── model/
│   └── simple_model.pt  # Trained CNN
├── data/
│   ├── class_names_25.json
│   └── details.json
├── templates/
│   ├── welcome.html
│   └── classifier.html
└── static/
    └── style.css
```

## Model Architecture (must match saved model)
```python
SimpleCNN(25): Conv2d(3,32,3)→ReLU→MaxPool→Conv2d(32,64,3)→ReLU→MaxPool→Conv2d(64,128,3)→ReLU→MaxPool→Flatten→Linear(128*8*8,256)→ReLU→Linear(256,25)
Input: 64x64, normalize [0.5]*3
```

## Pages
- **welcome.html**: Full-screen centered card, gradient logo, features, CTA
- **classifier.html**: Header, upload/drop zone, preview, result cards (matched/unknown), history panel

## API Endpoints
- `GET /` → welcome.html
- `GET /home` → classifier.html
- `POST /predict` → {flower, confidence, is_unknown, description, alternatives[]}
- `GET /classes` → {classes: [25 names]}

## Data Files (provided)
- `data/class_names_25.json` - {"0":"name",...}
- `data/details.json` - {"flower_name": "description",...}

Build the complete working app with responsive design and smooth animations.