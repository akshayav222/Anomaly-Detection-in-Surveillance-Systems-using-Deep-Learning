# Anomaly Detection in Surveillance Systems using Deep Learning

## Project Overview
This project focuses on **Anomaly Detection in Surveillance Systems** using a 3D Convolutional Neural Network (CNN) model built with TensorFlow and Keras. The system processes video input, detects anomalies, and can trigger an alert (e.g., playing a sound) when an anomaly is detected. The model is trained on a labeled dataset of surveillance videos to distinguish between normal and anomalous events.

## Key Features
- **3D CNN Architecture**: Utilizes a 3D Convolutional Neural Network for processing sequential frames in video data.
- **Real-Time Anomaly Detection**: Capable of detecting anomalies in real-time video streams.
- **Audio Alert**: Plays a sound when an anomaly is detected.
- **Frame Sampling**: Processes every nth frame from videos to reduce computational load.
- **Padding/Truncating**: Ensures consistent input size by padding or truncating frames in each video.

## Technologies Used
- **Python**: Main programming language.
- **TensorFlow & Keras**: For building and training deep learning models.
- **OpenCV**: For video frame extraction and preprocessing.
- **NumPy**: For efficient numerical computations.
- **Google Colab**: For training and testing the model using Google Drive for storage.

## Project Structure
```bash
├── data/                   # Directory for dataset (not provided)
├── models/                 # Directory for saved models
├── notebooks/              # Jupyter notebooks for experimentation
├── src/                    # Python scripts for data preprocessing, model, and evaluation
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── evaluate.py
├── README.md               # Project overview and instructions
└── requirements.txt        # List of dependencies
```
# Installation & Setup

### 1. Install the required dependencies by running:
```python
pip install -r requirements.txt
```

### 2. If you're using Google Colab, mount your Google Drive to access datasets and save models:
```python
from google.colab import drive
drive.mount('/content/drive')
```
## Dataset
This project uses the **UCF Crime Trimmed-Dataset** from Kaggle, a subset of the UCF Crime dataset which contains labeled surveillance videos of normal and anomalous events. The dataset should be downloaded from [Kaggle]([https://www.kaggle.com/datasets/lokesh2610/trim-data]) and stored in your Google Drive for use in Google Colab.

## Model Architecture

The model is built using 3D Convolutional layers (`Conv3D`) and `MaxPooling3D` layers to process video data (sequences of frames). Below is the architecture breakdown:

- **Input Layer**: Accepts a batch of video frames with dimensions `(30, 64, 64, 3)`.
- **Convolutional Layers**: Use `Conv3D` layers to extract spatial and temporal features from the video.
- **MaxPooling Layers**: Use `MaxPooling3D` layers to reduce the spatial dimensions.
- **Fully Connected Layers**: Classifies the video as normal or anomalous based on the extracted features.
- **Softmax Output Layer**: Provides the final classification result for each video.

## Results
The model provides predictions in the form of probability scores for each class. Anomalies are detected if the model's confidence in the "anomaly" class exceeds a set threshold.

## Future Improvements
- **Real-Time Processing**: Implement real-time anomaly detection in live video streams.
- **Improved Data Augmentation**: Add more sophisticated augmentation techniques to increase model robustness.
- **Edge Deployment**: Optimize the model for running on edge devices like Raspberry Pi for real-world surveillance applications.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
Special thanks to the TensorFlow and Keras communities for providing open-source libraries that made this project possible.

