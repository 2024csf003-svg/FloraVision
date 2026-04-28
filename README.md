# FloraVision

AI-powered flower identification web app built with PyTorch + Flask.

## Features

- Instant flower identification from images
- 25 flower species supported
- History tracking
- Statistics dashboard
- Dark modern UI

## Tech Stack

- Python + PyTorch + Flask
- Vanilla HTML/CSS/JS
- Poppins font + Font Awesome

## Running

```bash
cd floravision
pip install torch flask pillow torchvision
python app.py
```

Then open `http://localhost:5000/`

## Endpoints

- `GET /` - Welcome page
- `GET /home` - Classifier
- `POST /predict` - Identify flower
- `GET /history` - Get history
- `GET /classes` - List classes

## License

MIT