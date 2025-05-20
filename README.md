# Image Classification with Neural Networks in Python

This project demonstrates how to build and train a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset with TensorFlow and Keras. It includes data preprocessing, model architecture design, training, evaluation, and model persistence.

## 📁 Dataset

The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset is used in this project. It consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

### Class Labels

```
['airplane', 'automobile', 'bird', 'cat', 'deer', 
 'dog', 'frog', 'horse', 'ship', 'truck']
```

## 🚀 Features

- Image preprocessing and normalization
- Data visualization using `matplotlib`
- CNN architecture with `Conv2D`, `MaxPooling2D`, and `Dense` layers
- Training and evaluation using `sparse_categorical_crossentropy`
- Saving and loading the trained model

## 🧪 Model Architecture

The CNN consists of:

- 3 Convolutional layers with ReLU activation
- 2 Max Pooling layers
- 1 Fully connected (Dense) hidden layer with 64 units
- 1 Output layer with 10 units and softmax activation

## 🛠️ Requirements

- Python 3.6+
- TensorFlow
- NumPy
- OpenCV (`cv2`)
- Matplotlib

Install dependencies:

```bash
pip install tensorflow opencv-python matplotlib numpy
```

## 📊 Training & Evaluation

The model is trained for 10 epochs and validated on the test set.

Example output:

```
accuracy: 0.72
loss: 0.85
```

> *(Your actual results may vary depending on system resources and batch sizes.)*

## 💾 Saving and Loading the Model

The trained model is saved to disk as `image_classification.h5`:

```python
model.save('image_classification.h5')
```

To reload it for inference or further training:

```python
from tensorflow.keras import models
model = models.load_model('image_classification.h5')
```

## 📸 Data Visualization

The script displays the first 16 images from the training dataset along with their labels for a quick sanity check.

## 🔍 Usage

Run the script directly:

```bash
python image_classification.py
```

Make sure the script is saved under the filename `image_classification.py` or update the command accordingly.

## 📂 Project Structure

```
├── image_classification.py
├── image_classification.h5   # Saved model (after training)
├── README.md
```

## 🧠 Future Improvements

- Implement data augmentation
- Add dropout to prevent overfitting
- Experiment with deeper or alternative CNN architectures (e.g. ResNet, MobileNet)
- Add performance plots (loss vs. accuracy)

## 📜 License

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).
