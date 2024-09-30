# Road and Lane Detection for Self-driving car
## Introduction
This project aims to develop an efficient and robust road and lane detection system tailored for self-driving vehicles. Utilizing advanced computer vision techniques, this implementation allows vehicles to recognize and navigate road boundaries and lane markings, facilitating safer and more reliable autonomous driving experiences. The system leverages image processing and deep learning algorithms to analyze video feeds from the vehicle's cameras in real-time, ensuring accurate detection in various driving conditions.

<p align="center">
  <img src="https://github.com/Artinmi/Road-and-Lane-Detection/blob/master/Result/result2.gif" width="95%" alt="Leg"/>
</p>

- Real-Time Detection: Processes video frames in real-time to identify road lanes and boundaries.
- Robustness: Capable of functioning in different weather conditions and lighting scenarios.
- Modular Design: Code is organized into modules, making it easy to modify or expand functionality.
- User-Friendly Interface: Simple interface for running the detection system with sample input data.

## Table of Contents
1. [Road and Lane Detection Using Convolutional Neural Networks](#road-and-lane-detection-using-convolutional-neural-networks)
2. [CNN Model Overview](#cnn-model-overview)
3. [Object Detection Using YOLOv8](#object-detection-using-yolov8)
4. [YOLOv8 Model Overview](#yolov8-model-overview)
5. [Getting Started](#getting-started)
6. [Usage](#usage)
7. [Contributions](#contributions)
8. [Contact](#contact)

   
## Road and Lane Detection Using Convolutional Neural Networks
### Description
This project implements a robust road and lane detection system for autonomous driving applications using a Convolutional Neural Network (CNN). The model has been trained on a diverse dataset comprising 10,000 images, which include various road conditions, lane markings, and environmental factors. This extensive training enables the model to generalize effectively and accurately predict road boundaries and lane markings from video captures in real-time.

## CNN Model Overview
The trained model, saved as model.h5, utilizes a CNN architecture specifically designed for image segmentation tasks. This architecture excels at identifying and classifying pixels within an image, making it ideal for detecting road lanes. The model leverages multiple convolutional layers to extract spatial hierarchies of features, followed by fully connected layers that output the predicted lane markings as binary masks overlaying the input image.
### Key Features
- High Accuracy: Achieved through extensive training on a large and diverse dataset, ensuring robust performance across different scenarios.
- Real-Time Processing: Capable of processing video frames in real-time, making it suitable for dynamic driving environments.
- Versatile Use Cases: Applicable for various self-driving applications, including lane keeping assist and autonomous navigation.
### How to Use the Model
1. Load the Model: Utilize the following code to load the trained model:
```
from keras.models import load_model

model = load_model('model.h5')
```
2. Preprocess Input Data: Ensure video frames are preprocessed (resized, normalized, etc.) before feeding them into the model.
3. Predict Lane Markings: Use the model to predict lane markings on video frames:
```
import cv2
import numpy as np

frame = cv2.imread('your_image.jpg')
preprocessed_frame = preprocess(frame)  # Implement your preprocessing function
prediction = model.predict(np.expand_dims(preprocessed_frame, axis=0))

```
### Conclusion
This CNN-based road and lane detection system provides a powerful solution for enhancing the safety and efficiency of autonomous vehicles. By integrating this model into self-driving technology, we can achieve improved lane following and road navigation capabilities.


## Object Detection Using YOLOv8
### Description
In addition to lane detection, this project incorporates a powerful object detection system utilizing the YOLOv8 (You Only Look Once version 8) architecture. This state-of-the-art model is designed for real-time object detection, capable of identifying and localizing various objects in video streams, making it ideal for autonomous driving applications. The YOLOv8 model is known for its high accuracy and speed, allowing it to detect objects in dynamic environments efficiently.

## YOLOv8 Model Overview
The YOLOv8 model is an evolution of the YOLO series, which revolutionized the field of object detection with its single-stage approach. This architecture processes the entire image in one pass, predicting bounding boxes and class probabilities simultaneously. The model is trained on a comprehensive dataset, enabling it to recognize a wide range of objects, including vehicles, pedestrians, traffic signs, and more.

### Key Features
- Real-Time Performance: YOLOv8 is optimized for speed, making it suitable for applications that require immediate responses, such as self-driving cars.
- High Detection Accuracy: Achieved through advanced techniques such as anchor box refinement and multi-scale feature extraction, ensuring reliable detection of objects.
- Wide Range of Object Classes: Capable of detecting numerous object categories, facilitating safer navigation and interaction in complex driving scenarios.

### How to Use the YOLOv8 Model
1. Install YOLOv8: Ensure you have the necessary dependencies installed. You can install the YOLOv8 library using pip:
```
pip install ultralytics

```
2. Load the YOLOv8 Model: Use the following code to load the pre-trained model:
```
from ultralytics import YOLO

model = YOLO('yolov8.pt')  # Load the pre-trained YOLOv8 model

```
3. Process Video Frames: Capture video frames and pass them through the model for object detection:
```
import cv2

video_capture = cv2.VideoCapture('your_video.mp4')

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    results = model.predict(frame)  # Perform object detection
    # Display the results on the frame (optional)
    annotated_frame = results[0].plot()

    cv2.imshow('Object Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

```
### Conclusion
The integration of YOLOv8 for object detection enhances the functionality of the road and lane detection system, providing a comprehensive solution for autonomous driving. By detecting and localizing objects in real-time, the system can significantly improve situational awareness, ensuring safer navigation and interaction with surrounding vehicles, pedestrians, and obstacles.


## Getting Started
To get started with the project, follow these steps:
1. Clone the Repository: Use the following command to clone the repository:
   ```
   git clone https://github.com/Artinmi/Road-and-Lane-Detection.git

   ```
2. Install Dependencies: Install the necessary libraries using:
   ```
   pip install numpy opencv-python moviepy tensorflow scipy ultralytics

   ```


## Usage
After running the application, the system will display the video feed with detected lanes overlaid on the screen. You can adjust parameters in the configuration file to improve detection accuracy based on your specific use case.
follow these steps:
1. place your input video: place your video in the same directory that you have clone the repository. rename it as `input.mp4`
> [!TIP]
> It's highly recommended for you to use Vscode or Pycharm to install dependencies and run your python files.
2. Run the "Road and lane detection.py" : this code will generate a video named as `lane_detected.mp4` which the raod and lines are detected in it with a green path line:
   ```
   python "Road and lane detection.py"

   ```
3. Run the "Object detection.py": this code provides you a real_time object detection in your video:
   ```
   python "Object detection.py"

   ```

## Contributions
Contributions are always welcome! If you'd like to improve the project or add new features:
1. Fork this repository.
2. Create a new branch for your feature or fix.
3. Submit a pull request for review.

### Contact
If you have any questions or suggestions, feel free to reach out:

- Artin Mokhtariha - [artin1382mokhtariha@gmail.com](mailto:artin1382mokhtariha@gmail.com)
- GitHub: [Artinmi](https://github.com/Artinmi)
- Linkedin Post:
