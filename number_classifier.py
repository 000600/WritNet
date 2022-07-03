# Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Resize images into values between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0 

# Define class map
classes = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

# Initialize Adam optimizer
opt = Adam(learning_rate = 0.001)

# Define input and output shapes
input_shape = train_images.shape
output_shape = len(classes)

# Create model
model = Sequential()

# Hidden layers
model.add(Flatten(input_shape = input_shape[1:]))
model.add(Dense(512, activation = 'relu'))

# Output layer
model.add(Dense(output_shape, activation = 'softmax')) # Softmax activation function because the model is a multiclass classifier

# Compile and train model
epochs = 2
model.compile(optimizer = opt, loss = SparseCategoricalCrossentropy(), metrics = ['accuracy'])
history = model.fit(train_images, train_labels, epochs = epochs, validation_data = (test_images, test_labels))

# Visualize  loss and validation loss
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epoch_list = [i for i in range(epochs)]

plt.plot(epoch_list, loss, label = 'Loss')
plt.plot(epoch_list, val_loss, label = 'Validation Loss')
plt.title('Validation and Training Loss Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Visualize accuracy and validation accuracy
accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']

plt.plot(epoch_list, accuracy, label = 'Training Accuracy')
plt.plot(epoch_list, val_accuracy, label =' Validation Accuracy')
plt.title('Validation and Training Accuracy Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# View test accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 0) # Change verbose to 1 or 2 for more information
print(f'\nTest accuracy: {test_acc * 100}%')

# Make predictions
predictions = model.predict(test_images) # Model predicts the class for every image in the test dataset
pred_index = 0 # Change this number to view a different prediction-output set
print(f"Model's predicted value on sample input: {np.argmax(predictions[pred_index])} | Actual label on same input: {test_labels[pred_index]}") # Convert probabilites into a definitive class by taking the highest probability

# Function for viewing image inputs and the model's predictions based on those image inputs
def display_img(index, predictions_array, true_label, img):
  true_label, img = true_label[index], img[index]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap = plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'green' # Green bar if definitive prediction is correct
  else:
    color = 'red' # Red bar if definitive prediction is incorrect

  plt.xlabel("{} {}% ({})".format(classes[predicted_label], 100 * np.max(predictions_array), classes[true_label]), color = color) # Return the prediction value and its probability along with the correct value in parentheses

# Function for displaying multiple predictions and images
def display_array(index, predictions_array, true_label):
  true_label = true_label[index]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  plot = plt.bar(range(10), predictions_array, color = "#0000ff") # Other prediction probabilites shown in blue
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  plot[predicted_label].set_color('red') # Incorrect predictions shown in red
  plot[true_label].set_color('green')# Correct predictions shown in red

# Display incorrect predictions in red, correct predictions in green, and other prediction probabilities in blue
num_rows = 10 # Change this number to view more images and predictions
num_images = num_rows ** 2
plt.figure(figsize = (2 * 2 * num_rows, 2 * num_rows)) # Scale plot to fit all images

# Make a grid with predictions and input images
for ind in range(num_images):
  plt.subplot(num_rows, 2 * num_rows, 2 * ind + 1)
  display_img(ind, predictions[ind], test_labels, test_images)
  plt.subplot(num_rows, 2 * num_rows, 2 * ind + 2)
  display_array(ind, predictions[ind], test_labels)
  filler = plt.xticks(range(10), classes, rotation = 90)

# Display plot (the model's prediction probability distribution based on an image is plotted to the right of that image on the plot
plt.tight_layout()
plt.show()
