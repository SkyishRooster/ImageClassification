# ImageClassification
## This is a project for training a deep learning model for accomplishing an clothing attribute classification job. 
The dataset for training and testing the model is from [FashionAI Attributes Recognition of Apparel Dataset](https://tianchi.aliyun.com/dataset/136948). The dataset can be downloaded directly from the link as long as you’ve registered for a free account.<br>
<br>
There are two training datasets designed for round 1 and round 2 of the competition. Since the categories and labels are the same, I combined the images in these two datasets together as a new dataset. The training job is done baded on the integrated dataset.  

### [AttributeDescription.md](https://github.com/SkyishRooster/ImageClassification/blob/a2dea13dea5c89f4481b5d23cacda3ec77339b48/AttributeDescription.md) displays the information about the attributes contained in the dataset. There are 8 categories in the dataset, and 5 - 10 attributes for each category. Below are the categories and corresponding labels that we look at:  
1. Coat Length: Invisible, High Waist Length,Regular Length,Long Length,Micro Length, Knee Length, Midi Length, Anckle&Floor Length
2. Collar Design: Invisible, Shirt Collar, Peter Pan, Puritan Collar, Rib Collar
3. Lapel Design: Invisible, Notched, Collarless, Shawl Collar, Plus Size Shawl
4. Neck Design: Invisible, Turtle Neck, Ruffle Semi-High Collar, Low Turtle Neck, Draped Collar
5. Neckline Design: Invisible, Strapless Neck, Deep V Neckline, Straight Neck, V Neckline, Square Neckline, Off Shoulder, Round Neckline, Swear Heart Neck, One Shoulder Neckline
6. Pant Length: Invisible, Short Pant, Mid Length, 3/4 Length, Cropped Pant, Full Length
7. Skirt Length: Invisible, Short Length, Knee Length, Midi Length, Ankle Length, Floor Length
8. Sleeve Length: Invisible, Sleeveless, Cup Sleeves, Short Sleeves, Elbow Sleeves, 3/4 Sleeves, Wrist Length, Long Sleeves, Extra Long Sleeves
  
  
### [main.py](https://github.com/SkyishRooster/ImageClassification/blob/433aa4e18fe3083c006d374166fc950c6a934185/main.py) is the main part of the image classification training job. This python script file mainly performs jobs mentioned below:  
1. Load data and Split the data into training and validation set;  
2. Five Data Augmentation layers for choice including RandomFilp, RandomRotation, RandomTranslation, RandomBrightness, and RandomContrast;
3. Use transfer learning to train the model. The default model selection for transfer learning is the ResNet v2 model of 50 layers without fine-tuning, while Inception ResNet v2 model and Fine-tune are optional. The model is trained separately for different categories;
4. Store the weights of the model which has the best performance on the validation set for future use;
5. Visualize the training process by ploting the changes in training, validation loss and accuracy over epoches.  
  
  
### [Preprocess.py](https://github.com/SkyishRooster/ImageClassification/blob/7c04b5222718442c248eaccb09502d697348ba16/Preprocess.py) is used for preprocessing the dataset by grouping images based on labels into subfolders and combining labels based on the distribution and nature of the labels. Main steps to accomplish the task are listed below:  
1. As mentioned above, combine round 1 and round 2 training set together as an integrated training set;  
2. Decode the original labels to eligible ones: Assign corresponding labels based on the position of "y" in the encoded labels. For example, "nnynn" for "Collar Design" category denotes the label "Peter Pan" since its the third label in the category;
3. Create subfolders for each category based on the labels and group images to corresponding folders. This step is designed for using "image_dataset_from_directory" rather than "ImageDataGenerator" for a faster and more efficient training process;
4. Functions for checking the distribution of labels of certian category and preview images of certain label. This step is designed for examining the distribution and nature of labels to help make decisions on what labels to combine;
5. The "regroup" function implement the decisions made for categories by rearrange the subfolders each image belong to. The reason why I combine labels is because the boundaries among labels provided are not very clear in all cases. Even for human being, some labels are difficult to determine using the tagging system with this level of granularity. So this level of granularity is not desired for this project. Also, with fewer output labels, the accuracy of the model would increase inherently.  
