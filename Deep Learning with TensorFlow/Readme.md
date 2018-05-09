## Deep Learning with TensorFlow

### Hand gesture prediction, 2018
Build out a Multi Layer Perceptron model to try to classify hand written digits using TensorFlow using the famous MNIST data set of handwritten digits. The images are black and white images of size 28 x 28 pixels, or 784 pixels total. Features arepixel values for each pixel. Either the pixel is "white" (blank with a 0), or there is some pixel value. The data is stored in a vector format, although the original data was a 2-dimensional matirx with values representing how much pigment was at a certain location.

Predicted what number is written down based solely on the image data in the form of an array. 

Classification modeling to predict hand written digits based on black and white images of size 28 x 28 pixels. Imported vector formatted data stored as 2-dimensional matrix. Defined network parameters and developed multi-layer perceptron model with RELU activation, weights and biases. Minimized the cost (loss) using AdamOptimizer. Achieved 94% accuracy in the predictions. 

### Bank Note Authentication, 2018 
Experimented with effectiveness of neural networks by working on Bank Authentication dataset from UCI library. The data consists of 5 columns:

variance of Wavelet Transformed image (continuous)
skewness of Wavelet Transformed image (continuous)
curtosis of Wavelet Transformed image (continuous)
entropy of image (continuous)
class (integer)

Where class indicates whether or not a Bank Note was authentic.

Analyzed digitized data taken from genuine and forged banknote-like specimens presented in form of variance, skewness, curtosis and entropy of images. Prepared data for neural network and deep learning systems by standardizing. Trained the model using Contriblearn from TensorFlow and classified with deep neural network. Evaluated Model performance with 100% F1-score and precision. Compared with Random forest classifier, DNN was found better with higher accuracy as per classification report and confusion matrix. 
