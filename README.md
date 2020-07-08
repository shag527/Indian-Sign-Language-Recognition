# Indian-Sign-Language-Recognition
Communication is very significant to human beings as it facilitates the spread of knowledge and forms relationships between people. We communicate through speech, facial expressions, hand signs, reading, writing or drawing etc. But speech is the most commonly used mode of communication. However, people with hearing and speaking disability only communicate through signs, which makes them highly dependent on non-verbal forms of communication. India is a vast country, having nearly five million people deaf and hearing impaired. Still very limited work has been done in this research area, beacuse of it's complex nature. Indian Sign Language (ISL) is predominantly used in South Asian countries and sometimes, it is also called as Indo-Pakistani Sign Language (IPSL). The purpose of this project is to recognize all the alphabets (A-Z) and digits (0-9) of Indian sign language using bag of visual words model and convert them to text/speech. Dual mode of recognition is implemeted for better results. Ths system is tested using various  machine learning classifiers like KNN, SVM, logistic regression and a convolutional neural network (CNN) is also implemented for the same. The dataset for this system is created manually in different hand orientations and a train-test ratio of 80:20 is used. The model gives very high accuracy rate.

## Getting Started
### Pre-requisites
Before running this project, make sure you have following dependencies - 
* [Python 3.7](https://www.python.org/downloads/)
* [pip](https://pypi.python.org/pypi/pip)
* [OpenCV](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html)

### Dataset
* [Dataset](https://drive.google.com/drive/folders/1SY67sDO2ROoOoBhTBIIDn17gStS0AvCB?usp=sharing) (Download the images from here)

Now, using ```pip install``` command, include following dependencies 
+ Numpy 
+ Pandas
+ Sklearn
+ Scipy
+ Opencv
+ Tensorflow

### Running
To run the project, perform following steps -

1. Take the dataset folder and all the required python files and put them in the same folder.
2. Required files are - surf_image_processing.py(Image preprocessing folder), preprocessing_surf.py (Bag of features folder), classification.py(classification folder) and visualize_submissions.py(visualization folder). 
3. Run the preprocessing_surf.py file to make the csv file of training data set.
4. classification.py contains the code for svm, knn and many other classifiers.
5. cnn.py contains the code for deep learning as the name suggests. 

## Workflow

<p align="center">
  <br>
  <img align="center" src="https://github.com/imRishabhGupta/Indian-Sign-Language-Recognition/blob/master/Visualization/flowchart.jpg">
</p>

### Image Preprocessing

#### Segmentation:
The main objective of the segmentation phase is to remove the background and noises, leaving only the Region of Interest (ROI), which is the only useful information in the image. This is achieved via Skin Masking defining the threshold on RGB schema and then converting RGB colour space to grey scale image. Finally Canny Edge technique is employed to identify and detect the presence of sharp discontinuities in an image, thereby detecting the edges of the figure in focus.  

<p align="center">
  <br>
<img align="center" src="https://github.com/imRishabhGupta/Indian-Sign-Language-Recognition/blob/master/Processed_images/BGR2HSV.png">       <img align="center" src="https://github.com/imRishabhGupta/Indian-Sign-Language-Recognition/blob/master/Processed_images/masked.png">       <img align="center" src="https://github.com/imRishabhGupta/Indian-Sign-Language-Recognition/blob/master/Processed_images/canny%20edge%20detection.png">
  <br>
  BGR to HSV    &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp   Masked   &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp  Canny Edge
</p>
  <br>
  
#### Feature Extraction:
The Speeded Up Robust Feature (SURF) technique is used to extract descriptors from the segmented hand gesture images. SURF is a novel feature extraction method which is robust against rotation, scaling, occlusion and variation in viewpoint.
<p align="center">
  <br>
  <img align="center" src="https://github.com/imRishabhGupta/Indian-Sign-Language-Recognition/blob/master/Processed_images/SURF_D.png">
</p>

### Classification
The SURF descriptors extracted from each image are different in number with the same dimension (64). However, a multiclass SVM requires uniform dimensions of feature vector as its input. Bag of Features (BoF) is therefore implemented to represent the features in histogram of visual vocabulary rather than the features as proposed. The descriptors extracted are first quantized into 150 clusters using K-means clustering. Given a set of descriptors, where K-means clustering categorizes numbers of descriptors into K numbers of cluster center.

The clustered features then form the visual vocabulary where each feature corresponds to an individual sign language gesture. With the visual vocabulary, each image is represented by the frequency of occurrence of all clustered features. BoF represents each image as a histogram of features, in this case the histogram of 24 classes of sign languages gestures. 

#### Bag of Features model

Following Steps are followed to achieve this:

* The descriptors extracted are first clustered into 150 clusters using K-Means clustering.

* K-means clustering technique categorizes m numbers of descriptors into x number of cluster centre.

* The clustered features form the basis for histogram i-e each image is represented by frequency of occurrence of all clustered features.

* BoF represents each image as a histogram of features, in our case the histogram of 24 classes of sign language is generated.

#### Classifiers

After obtaining the baf of features model, we are set to predict results for new raw images to test our model. Following classifiers are used :
+ Naive Bayes
+ Logistic Regression classifier
+ K-Nearest Neighbours
+ Support Vector Machines
+ Convolution Neaural Network

### Results
Results can be visualized by running file [visualize_submissions.py](https://github.com/imRishabhGupta/Indian-Sign-Language-Recognition/blob/master/Visualization/visualize_submissions.py).

#### Accuracy without SURF

<p align="center">
  <br>
  <img align="center" src="https://github.com/imRishabhGupta/Indian-Sign-Language-Recognition/blob/master/Visualization/accuracy_without_surf.png">
        <br>  
  </p>
  
#### Accuracy with SURF

<p align="center">
  <br>
  <img align="center" src="https://github.com/imRishabhGupta/Indian-Sign-Language-Recognition/blob/master/Visualization/acuracy_with_surf.png">
</p>

### Credits
If there are any doubts or queries with the process, refer these posts, they are amazing -

+ [Post 1](https://ianlondon.github.io/blog/how-to-sift-opencv/)
+ [Post 2](https://ianlondon.github.io/blog/visual-bag-of-words/)

