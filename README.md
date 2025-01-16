Emotion Detection Project Report



 Project Overview
This project focuses on developing a system capable of detecting human emotions from facial expressions using a Convolutional Neural Network (CNN) model. The model is trained on a dataset of facial images categorized into seven emotions: Anger, Disgust, Fear, Happiness, Neutral, Sadness, and Surprise. The project provides functionalities for both training the model and real-time emotion detection via a webcam video feed.



 Dependencies
The project relies on the following libraries:
- `numpy`: Used for numerical computations.
- `argparse`: For handling command-line arguments.
- `matplotlib`: To plot training history.
- `cv2` (OpenCV): For image and video processing.
- `tensorflow.keras`: To build and train the CNN model.



 Model Architecture
The CNN model, built using TensorFlow's Keras API, comprises several layers designed to extract features from facial images and classify them into seven emotion categories.

# Model Summary
1. Convolutional Layers: Extract spatial features from input images using multiple filters.
2. Pooling Layers: Reduce spatial dimensions to decrease computational complexity and emphasize important features.
3. Dropout Layers: Mitigate overfitting by randomly deactivating some neurons during training.
4. Flatten Layer: Converts multi-dimensional feature maps into a 1D vector for classification.
5. Dense (Fully Connected) Layers: Perform the final classification into the seven emotion categories.
6. Softmax Activation: Outputs a probability distribution over the emotion categories.

![image](https://github.com/user-attachments/assets/8fd12f52-f0ba-4b93-80ca-58ac614f7759)


 Model Training
The model is trained using the following:
- Loss Function: `categorical cross-entropy`, which calculates the difference between predicted and true probability distributions.
- Optimizer: Adam, which adapts the learning rate dynamically for efficient training.
- Data Augmentation: Keras's `ImageDataGenerator` performs real-time transformations, such as rotation, flipping, scaling, and shifting, to enhance the training dataset's diversity.

![image](https://github.com/user-attachments/assets/c00d5baa-dc2b-4f42-a811-5f374f5c8d8c)


 Model Accuracy
The accuracy of the model is tracked throughout the training process:
- Training Accuracy: ~85% after 50 epochs.
- Validation Accuracy: ~82% after 50 epochs.

These results indicate that the model generalizes well from training data to unseen data, making it effective for emotion detection tasks.


![image](https://github.com/user-attachments/assets/dff64002-910d-4281-b30e-7b816a982606)

 Plotting Model History
The `plot_model_history` function visualizes:
1. Accuracy: Training vs. validation accuracy across epochs.
2. Loss: Training vs. validation loss trends.
![image](https://github.com/user-attachments/assets/1bb8a483-b6bf-4051-9f9c-c1584c4317a1)

This visualization helps identify issues like overfitting or underfitting and evaluates the model's learning process.



 Real-Time Emotion Detection
For real-time emotion detection, the trained model is loaded, and OpenCV captures video from the webcam. The model predicts the emotion of detected faces in real-time and displays the results on the video feed.

# Display Script
The script integrates the trained model with a live webcam feed to:
- Detect faces.
- Predict emotions for each detected face.
- Display predicted emotions directly on the video feed.

![image](https://github.com/user-attachments/assets/939d11c5-6a4f-407a-8556-0128e80af5fa)


 Conclusion
The Emotion Detection project successfully demonstrates the use of CNNs to classify facial emotions in real time. With a validation accuracy of ~82%, the model shows promising performance. Future improvements could include optimizing the model architecture and leveraging larger datasets to further enhance accuracy.

