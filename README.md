# Human-Gender-and-Age-Detection

# Gender and Age Detection using OpenCV &amp; Deep Learning

##


## **About the Project :**

In this Python Project, We have used Deep Learning to accurately identify the gender and age of a person from a single image of a face.The predicted gender may be one of &#39;Male&#39; and &#39;Female&#39;, and the predicted age may be one of the following ranges- (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100) (8 nodes in the final softmax layer). It is very difficult to accurately guess an exact age from a single image because of factors like makeup, lighting, obstructions, and facial expressions. And so, We made this a classification problem instead of making it one of regression.

<img width="342" alt="1" src="https://user-images.githubusercontent.com/76903537/170078709-ede21a23-bc9d-48da-b489-6dbb244f1633.png">

#### The CNN Architecture

We have used a very simple convolutional neural network architecture, similar to the CaffeNet and AlexNet. The network uses 3 convolutional layers, 2 fully connected layers and a final output layer. The details of the layers are given below.

- Conv1 : The first convolutional layer has 96 nodes of kernel size 7.
- Conv2 : The second conv layer has 256 nodes with kernel size 5.
- Conv3 : The third conv layer has 384 nodes with kernel size 3.
- The two fully connected layers have 512 nodes each.

## **Dataset :**

For this python project, we&#39;ll use the Adience dataset; This dataset serves as a benchmark for face photos and is inclusive of various real-world imaging conditions like noise, lighting, pose, and appearance. The images have been collected from Flickr albums and distributed under the Creative Commons (CC) license. It has a total of 26,580 photos of 2,284 subjects in eight age ranges (as mentioned above) and is about 1GB in size. The models we will use have been trained on this dataset.

## **Additional Python Libraries Required :**

- OpenCV
- argparse

## **The contents of this Project :**

- opencv\_face\_detector.pbtxt
- opencv\_face\_detector\_uint8.pb
- age\_deploy.prototxt
- age\_net.caffemodel
- gender\_deploy.prototxt
- gender\_net.caffemodel
- a few pictures to try the project on
- detect.py

For face detection, we have a .pb file- this is a protobuf file (protocol buffer); it holds the graph definition and the trained weights of the model. We can use this to run the trained model. And while a .pb file holds the protobuf in binary format, one with the .pbtxt extension holds it in text format. These are TensorFlow files. For age and gender, the .prototxt files describe the network configuration and the .caffemodel file defines the internal states of the parameters of the layers.

We use the argparse library to create an argument parser so we can get the image argument from the command prompt. We make it parse the argument holding the path to the image to classify gender and age for.

3. For face, age, and gender, initialize protocol buffer and model.

4. Initialize the mean values for the model and the lists of age ranges and genders to classify from.

5. Now, use the readNet() method to load the networks. The first parameter holds trained weights and the second carries network configuration.

6. Let&#39;s capture video stream in case you&#39;d like to classify on a webcam&#39;s stream. Set padding to 20.

7. Now until any key is pressed, we read the stream and store the content into the names hasFrame and frame. If it isn&#39;t a video, it must wait, and so we call up waitKey() from cv2, then break.

8. Let&#39;s make a call to the highlightFace() function with the faceNet and frame parameters, and what this returns, we will store in the names resultImg and faceBoxes. And if we got 0 faceBoxes, it means there was no face to detect.
 Here, net is faceNet- this model is the DNN Face Detector and holds only about 2.7MB on disk.

- Create a shallow copy of frame and get its height and width.
- Create a blob from the shallow copy.
- Set the input and make a forward pass to the network.
- faceBoxes is an empty list now. for each value in 0 to 127, define the confidence (between 0 and 1). Wherever we find the confidence greater than the confidence threshold, which is 0.7, we get the x1, y1, x2, and y2 coordinates and append a list of those to faceBoxes.
- Then, we put up rectangles on the image for each such list of coordinates and return two things: the shallow copy and the list of faceBoxes.

9. But if there are indeed faceBoxes, for each of those, we define the face, create a 4-dimensional blob from the image. In doing this, we scale it, resize it, and pass in the mean values.

10. We feed the input and give the network a forward pass to get the confidence of the two class. Whichever is higher, that is the gender of the person in the picture.

11. Then, we do the same thing for age.

12. We have added the gender and age texts to the resulting image and display it with imshow().

The code can be divided into four parts:

1. Detect Faces
2. Detect Gender
3. Detect Age
4. Display output

Let us have a look at the code for gender and age prediction using the DNN module in OpenCV.

**Detect Face**

We will use the DNN Face Detector for face detection. The model is only 2.7MB and is pretty fast even on the CPU. More details about the face detector can be found in our blog on [Face Detection](https://learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/) The face detection is done using the function getFaceBox as shown be

###


### **Predict Gender**

We will load the gender network into memory and pass the detected face through the network. The forward pass gives the probabilities or confidence of the two classes. We take the max of the two outputs and use it as the final gender prediction.

**Predict Age**

We load the age network and use the forward pass to get the output. Since the network architecture is similar to the Gender Network, we can take the max out of all the outputs to get the predicted age group.

### **Display Output**

We will display the output of the network on the input images and show them using the imshow function.

##
# **Output**

# **Results**

## **Example 1:**

<img width="473" alt="2" src="https://user-images.githubusercontent.com/76903537/170079193-4ceaa4cd-0e0f-4397-aa9a-f66260c5c0a7.png">

**Output**

<img width="430" alt="1" src="https://user-images.githubusercontent.com/76903537/170079670-95364543-0d1d-4780-944d-d83a151fdc85.png">

**Example 2:**

<img width="474" alt="1" src="https://user-images.githubusercontent.com/76903537/170079846-acee4323-0739-4662-b4b5-c7ec10faa51c.png">

**Output :**

<img width="348" alt="1" src="https://user-images.githubusercontent.com/76903537/170079992-6820b8fe-b4b4-4fa4-8cc8-e131fd13d437.png">

**Example 3:**

<img width="358" alt="1" src="https://user-images.githubusercontent.com/76903537/170080088-c9bd8412-53ee-4e39-9543-82bd3ed5785b.png">

**Output :**

<img width="354" alt="1" src="https://user-images.githubusercontent.com/76903537/170080197-54aec302-e1dd-45c0-b5a7-72d241b1886e.png">

## **Nobelity**  **Results**

We saw above that the network is able to predict both Gender and Age to high level of accuracy. Next, we wanted to do something interesting with this model. Many actors have portrayed the role of the opposite gender in movies.

_We want to check_ _what says about their looks in these roles and whether they are able to fool the AI._

We used some images which shows their actual photographs along with those from the movies in which they changed their gender. Let&#39;s have a look.

**Example 1:**

<img width="288" alt="1" src="https://user-images.githubusercontent.com/76903537/170080378-4f9825a8-c5c5-4e74-a5bd-018485429206.png">

**Output :**

<img width="404" alt="1" src="https://user-images.githubusercontent.com/76903537/170080460-437ea015-0dd6-46fc-a755-e5fc5fbbd3d8.png">

**Example 2:**

<img width="408" alt="1" src="https://user-images.githubusercontent.com/76903537/170080606-5ee8bdfa-dc18-441b-9e03-047baf949723.png">

**Output :**

<img width="399" alt="1" src="https://user-images.githubusercontent.com/76903537/170080718-1eeb0be7-9c60-4383-af2e-477cc0b2abf9.png">

**Example 3:**

<img width="454" alt="1" src="https://user-images.githubusercontent.com/76903537/170080783-2053416f-9cff-4b1e-b7e8-542db3b78692.png">

**Output :**

<img width="415" alt="1" src="https://user-images.githubusercontent.com/76903537/170081171-2fcc1a8f-5a4a-4202-a554-b77913e89268.png">



