🌿 Plant Disease Detection Using CNN
Deep Learning Based Image Classification System
## 📌 Project Overview
This project develops a deep learning model to automatically detect plant diseases from leaf images using a Convolutional Neural Network (CNN). Early detection of plant diseases helps farmers take timely action and improve crop productivity.

The model is trained on a dataset of plant leaf images and learns visual patterns associated with healthy and diseased leaves.
## 🧠 Technologies Used
- Python
- TensorFlow / Keras
- Convolutional Neural Networks (CNN)
- NumPy
- Matplotlib
- Jupyter Notebook
  Dataset Description
  _ _ _
## Dataset
The dataset contains plant leaf images arranged in folder format:

data/
│
├── Common_Rust
├── Gray_Leaf_Spot
├── Healthy
├── Northern_Leaf_Blight
Each folder represents one class.

Total:

Training Images: 3352
Validation Images: 836
Classes: 4
---

## 📊 Model Workflow
1. Load plant leaf image dataset
2. Preprocess images (resize, normalization)
3. Build CNN architecture
4. Train the deep learning model
5. Evaluate model performance
6. Predict disease class from images

# 1. Load plant leaf image dataset
* import librsries
<img width="564" height="153" alt="image" src="https://github.com/user-attachments/assets/a804c2fc-8317-43c1-aaed-247113e53b7c" />
* Load dataset
# 2.Preprocess images 
  Images are resized and normalized before training.
img_size = 128
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
Why Preprocessing?
•	Resize images to 128x128
•	Normalize pixel values (0–1)
•	Split into training and validation data

<img width="567" height="400" alt="image" src="https://github.com/user-attachments/assets/b3c1e299-18eb-4ab7-809d-0b2673c18c71" />
------
# 3. Build CNN architecture
<img width="513" height="368" alt="image" src="https://github.com/user-attachments/assets/d18f41ba-9e4e-4850-a709-20156dd3a172" />
The model consists of:

* 3 Convolutional Layers
* 3 MaxPooling Layers
* Batch Normalization
* Fully Connected Dense Layer
* Dropout Layer
* Output Layer (4 classes)

```python
model = Sequential()
```

### Layer Details:

| Layer Type         | Purpose             |
| ------------------ | ------------------- |
| Conv2D             | Feature extraction  |
| MaxPooling         | Reduce dimensions   |
| BatchNormalization | Stabilize training  |
| Flatten            | Convert to 1D       |
| Dense              | Classification      |
| Dropout            | Prevent overfitting |
| Softmax            | Multi-class output  |

Total Parameters: 6.5 Million

---
# Compil and Train the deep learning model

<img width="542" height="356" alt="image" src="https://github.com/user-attachments/assets/eb4e1231-81e9-4823-b2e7-3be44a8d8c5d" />

## Model Compilation

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Explanation:

* Optimizer: Adam
* Loss Function: Categorical Crossentropy
* Metric: Accuracy

---

## Model Training

```python
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)
```

The model is trained for 10 epochs.

---
# 5. Evaluate model performace
**Accuracy**
<img width="347" height="272" alt="image" src="https://github.com/user-attachments/assets/2ee74548-dd98-4804-bcb0-bb7b9399a63a" />
**Loss**
<img width="359" height="279" alt="image" src="https://github.com/user-attachments/assets/1b9a1935-380d-4142-964f-5524b3f02d89" />
## 📊 Model Performance

The training results show that the CNN model learns effectively during training.

- **Accuracy:** Training accuracy increases steadily and reaches around **93%**, while validation accuracy stabilizes around **88–91%**, indicating good learning performance.
- **Loss:** Training loss decreases consistently, showing the model is improving over time. Validation loss slightly fluctuates but remains relatively low.

Overall, the model demonstrates strong learning capability with good classification performance for plant disease detection.
# 6.Predict disease class from images
<img width="525" height="381" alt="image" src="https://github.com/user-attachments/assets/51911967-8eac-4945-ab78-ef5acd61be2c" />
Prediction function:
```python
def predict_image(img_path):
```
Steps:
1. Load image
2. Resize to 128x128
3. Normalize
4. Predict class
5. Return label

Example Output:
```
Prediction: Common_Rust



