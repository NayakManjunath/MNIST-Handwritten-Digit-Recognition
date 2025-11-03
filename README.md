# MNIST Handwritten Digit Recognition

A deep learning project that recognizes handwritten digits using Convolutional Neural Networks (CNN) trained on the MNIST dataset.

## Project Structure
mnist-digit-recognition/
├── notebooks/ # Jupyter notebooks
├── models/ # Saved models
├── src/ # Source code
├── requirements.txt # Dependencies
└── README.md # This file

text

### Features

- CNN model for digit classification
- 99%+ test accuracy
- Model training and evaluation
- Prediction interface
- Visualization tools

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mnist-digit-recognition.git
cd mnist-digit-recognition
Install dependencies:

bash
pip install -r requirements.txt
Usage
Training the Model
python
from src.train import create_model, load_data

(X_train, y_train), (X_test, y_test) = load_data()
model = create_model()
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
Making Predictions
python
from src.predict import load_model, predict_digit

model = load_model('models/mnist_model.h5')
prediction, confidence = predict_digit(model, your_image)
Using the Notebook
Open notebooks/mnist_digit_recognition.ipynb in Jupyter for interactive exploration.

### Results
Test Accuracy: ~99.2%

Model: CNN with 3 convolutional layers

Training Time: ~5 minutes on CPU

Technologies Used
TensorFlow/Keras

NumPy

Matplotlib

OpenCV

Scikit-learn

Contributing
Feel free to fork this project and submit pull requests!
