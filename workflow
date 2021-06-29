** Workflow / Ideas:
 - Creating dataset of 26 alphabets - Each 1000 images
  - dataset
   - train - contains folder with images of each alphabet independently
   - test - contains folder with images of each alphabet independently
 - How to create images?
 - Find datasets online? Kaggle?
 - Using data augmentations?
 - Using PyTorch datasetFolder?
 - Transfer Learning - choose pretrained model? EfficientNet or any CNN?
 - Train and testing split
 - Training locally or Colab?
 - Inference - Build proper pipeline and workflow
 - Generic code (see: https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Kaggles/Dog%20vs%20Cat%20Competition)
 - Build demo gradio app and proper README


** Data Creation:
 - Only lower-case letters?
 - Black and white images
 - Dimensions = 56 x 56
 - Shape = [1, 56, 56] - 1 channel (black and white)
 - Color specs:
  - background: white
  - alphabet: black
   (remember during inference too)
 - Let's not worry about transformations right now, we can add more later
 - Using kaggle dataset (https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format)
 - TODO (on my dataset):
  - check out other opencv functions to apply text transformations
  - rotations, fish eye, perspective, zoom, crop, ...
 - It's hard to find different fonts, so my dataset is not so vivid
 - Need to use augmentations before training
 - apply transformations on my dataset 
 - Merge any online available dataset
 - dataset and model created with i/o sizes checking
 - todo: proper pipelining
 - refer for train.py
	- https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Kaggles/DiabeticRetinopathy/utils.py
	- https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Kaggles/DiabeticRetinopathy/train.py

