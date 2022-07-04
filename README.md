# WritNet

## The Neural Network

This convolutional neural network (CNN) classifies the number someone wrote based on an image of the number and draws its name from its (albeit limited) ability to decipher human handwriting. The model will predict a list of 10 elements (9 indices), where each value in the list represents the probability that the image is a representation of that index number. In other words, given an input image, the model outputs a list [*probability_num_is_zero*, *probability_num_is_one*, *probability_num_is_two*, ... *probability_num_is_eight*, *probability_num_is_nine*]. The element with the highest probability means that the index of that element (an integer 0 - 9) is the model's prediction. Since the model is a multi-label classifier (it classifies which number an image contains), it uses a sparse categorical crossentropy loss function and has 10 output neurons (one for each class). The model uses a standard Adam optimizer with a learning rate of 0.001 and has an architecture consisting of:
- 1 Flatten layer (with an input shape of [28, 28] since each image is 28 x 28 pixels) 
- 1 Hidden layer (with 512 neurons and a ReLU activation function)
- 1 Output layer (with 10 output neurons and a softmax activation function)

Feel free to further tune the hyperparameters or build upon the model!

## The Dataset
The dataset is an MNIST included within keras and contains approximately 70,000 (60,000 images in the train set and 10,000 images in the test set) 28 x 28 pixel images of human handwriting of numbers 0 - 9 and is included within the **number_classifier.py** file.

## Libraries
This neural network was created with the help of the Tensorflow library.
- Tensorflow's Website: https://www.tensorflow.org/
- Tensorflow Installation Instructions: https://www.tensorflow.org/install
