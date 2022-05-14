# Indian-Sign-Language-Recognition
Communication is very significant to human beings as it facilitates the spread of knowledge and forms relationships between people. We communicate through speech, facial expressions, hand signs, reading, writing or drawing etc. But speech is the most commonly used mode of communication. However, people with hearing and speaking disability only communicate through signs, which makes them highly dependent on non-verbal forms of communication. India is a vast country, having nearly five million people deaf and hearing impaired. Still very limited work has been done in this research area, because of its complex nature. Indian Sign Language (ISL) is predominantly used in South Asian countries and sometimes, it is also called as Indo-Pakistani Sign Language (IPSL). 

The purpose of this project is to recognize all the alphabets (A-Z) and digits (0-9) of Indian sign language using bag of visual words model and convert them to text/speech. Dual mode of recognition is implemented for better results. The system is tested using various machine learning classifiers like KNN, SVM, logistic regression and a convolutional neural network (CNN) is also implemented for the same. The dataset for this system is created manually in different hand orientations and a train-test ratio of 80:20 is used.

#### Research Paper [Indian Sign Language recognition system using SURF with SVM and CNN](https://www.sciencedirect.com/science/article/pii/S2590005622000121?via%3Dihub#!)
If you use this in your research, please cite: 

`Shagun Katoch, Varsha Singh, Uma Shanker Tiwary, Indian Sign Language recognition system using SURF with SVM and CNN, Array, Volume 14, 2022, 100141, ISSN 2590-0056, https://doi.org/10.1016/j.array.2022.100141`.

## Getting Started
### Pre-requisites
Before running this project, make sure you have following dependencies - 
* [pip](https://pypi.python.org/pypi/pip)
* [Python 3.7.1](https://www.python.org/downloads/)
* [OpenCV](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html)
* [Opencv contrib](https://pypi.org/project/opencv-contrib-python/)
* [Anaconda Distribution](https://www.anaconda.com/products/individual)

### Dataset
Download the images from [here](https://drive.google.com/drive/folders/1SY67sDO2ROoOoBhTBIIDn17gStS0AvCB?usp=sharing)

Some images of the dataset are shown below:
<p align="center">
  <br>
<img align="center" src="https://github.com/shag527/Indian-Sign-Language-Recognition/blob/master/Images/dataset.png" width="800" height="750">   
 </p>

### Run Project
To run the project, perform the following steps -
1. ```git clone https://github.com/shag527/Indian-Sign-Language-Recognition.git```
2. ```conda create --name sign python=3.7.1```<br />
3. ```conda activate sign```. 
4. ```pip install -r requirements.txt```. 
5. ```cd to the GitHub Repo till Predict signs folder```. 

Command may look like: ```cd 'C:\Users\.....\Indian-Sign-Language-Recognition\Code\Predict signs\'```

6. ```python main.py```

A tkinter window like this will open.
<p align="center">
<img align="center" src="https://github.com/shag527/Indian-Sign-Language-Recognition/blob/master/Images/login.png" width="400" height="410">
  <br>
 </p>

7. Create your account to access the system.
8. Now, the main tkinter window will open.
<p align="center">
<img align="center" src="https://github.com/shag527/Indian-Sign-Language-Recognition/blob/master/Images/main.png" width="400" height="410">
   <br>
 </p>
9. Click on the desired button to access the respective service.

#### To create your own recognition system
1. To create your own dataset, following the steps given above, go to the create signs panel and create signs.
2. Now, divide the dataset into train and test by running the Dividing_Dataset.ipynb file in the preprocessing folder.
3. To create histograms and saving them to .csv file, run the create_train_hist.py and create_test_hist.py respectively by extrating the SURF features and clustering them using MiniKbatchMeans.
4. Lastly, go to the classification folder and run different python files to check the results. 
5. After saving the model, you can load the model for testing purposes.


## Workflow

### Preprocessing
Here 2 methods for preprocessing are used. First one is the background subtraction using an additive method, in which the first 30 frames are considered as background and any new object in the frame is then filtered out. Second one uses the skin segmentation concept, which is based on the extraction of skin color pixels of the user.

<p align="center">
  <br>
<img align="center" src="https://github.com/shag527/Indian-Sign-Language-Recognition/blob/master/Images/mask.png">       <img align="center" src="https://github.com/shag527/Indian-Sign-Language-Recognition/blob/master/Images/after mask.png">       <img align="center" src="https://github.com/shag527/Indian-Sign-Language-Recognition/blob/master/Images/canny.png">
  <br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Mask &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;After applying mask &nbsp; &nbsp; &nbsp;&nbsp;Canny Edge detection
</p>
  <br>
  
### Feature Detction and Extraction:
The Speeded Up Robust Feature (SURF) technique is used to extract descriptors from the segmented hand gesture images. These descriptors are then clustered to form the similar clusters and then the histograms of visual words are generated, where each image is represented by the frequency of occurrence of all the clustered features. The total classes are 36.
<p align="center">
  <br>
  <img align="center" src="https://github.com/shag527/Indian-Sign-Language-Recognition/blob/master/Images/SURF.png">
 <br>
 &nbsp&nbsp&nbsp&nbsp&nbsp SURF Features
</p>

### Classification
In this phase, various classifiers are used in order to check the best classifier for prediction. The classifiers used are:

+ Naive Bayes
+ Logistic Regression 
+ K-Nearest Neighbours (KNN)
+ Support Vector Machine (SVM)
+ Convolution Neural Network (CNN)

#### Accuracy Plot
The accuracy rate of different classifiers obtained are shown below:
<p align="center">
  <br>
<img align="center" src="https://github.com/shag527/Indian-Sign-Language-Recognition/blob/master/Images/acc.png">
 </p>

### Output
The predicted labels are shown in the form of text as well as speech using the python text to speech conversion library, Pyttsx3.

### Reverse Sign Recognition
Dual mode of communication is implemented. The spoken word is taken as input and the corresponding sign images are shown in sequence. Google speech API is used for this purpose.

### Credits
+ [Bag of Visual Words (BOVW)](https://medium.com/@aybukeyalcinerr/bag-of-visual-words-bovw-db9500331b2f)
+ [Image Classification with Convolutional Neural Networks](https://medium.com/@ksusorokina/image-classification-with-convolutional-neural-networks-496815db12a8)
