# Eigenface-detector

## Brief Description
The purpose of this project is to recognize different faces by using Eigenface with different sizes of training sets (10/20/30). I used Viola and Jones Face Detector with for human face detector and used Eigenface for human face recognition. 
  
## Programme workflow:
Input images --> Prepare training data --> Viola and Jones Face Detector --> Train Eigenface face recognizer --> Use trained recognizer to predict --> Output predicted label

## Dataset description:
I totally used 60 images from three different individuals. The first person is “Song Min Ho” who is a Korean singer and I downloaded 30 images of him from the internet, 10 for training and 20 for testing. The second person is “Lee Dong Wook” who is a Korean actor and I downloaded 40 images of him from the internet, 20 for training and 20 for testing. The third person is “Huang Dong wen”, myself, and I took 50 photos using my laptop, 30 for training and 20 for testing. 

“Song Min Ho” images are labelled as 1 and saved in folder “test-data/s1” and “training-data/s1”.

“Lee Dong Wook” images are labelled as 2 and saved in folder “test-data/s2” and “training-data/s2”.

“Huang Dong Wen” images are labelled as 3 and saved in folder “test-data/s3” and “training-data/s3”.
