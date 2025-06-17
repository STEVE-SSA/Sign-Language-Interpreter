# Sign Language Interpreter

## Data Preparation

This project uses images from `datasets/custom_gestures/` for custom sign language gesture recognition.

### Data Preparation Script
- Loads images from each gesture folder
- Uses MediaPipe to detect and crop hand(s)
- Filters out blurry/moving-hand images
- Augments data for robustness
- Splits into train/val/test sets
- Saves processed data to `data/`

### Usage
```bash
pip install -r requirements.txt
python src/data/prepare_data.py
```

### Next Steps
- Model training and evaluation
- Real-time inference pipeline

---

For best accuracy, ensure your gesture images are clear, well-lit, and include both single and double hand gestures as needed.
