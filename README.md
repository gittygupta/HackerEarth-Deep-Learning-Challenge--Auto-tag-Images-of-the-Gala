# HackerEarth-Deep-Learning-Challenge--Auto-tag-Images-of-the-Gala
#### Rank - 60
#### f1 score - 84.22499
####

This repository (private until 7th April, 2020, 7:30am IST) contains the codes used to create a simple classifier that classifies input images into Attires, Decorations and Signage, Food and, miscellaneous items. The dataset provided consisted of 5100 odd images in the training set for all the classes and 3100 odd images in the test set, the answers to which were evaluated with f1-score as the metric.

The codes provided use fastai as the framework which has been built as an API over PyTorch.

The basic approach was to use pre-trained deep learning models. The difference here lies in the model used and whether or not it needed to be unfreezed. After 'n' number of trials with AlexNet, ResNet-34, ResNet-50, ResNet152, DenseNet-201, it was found that ResNet-152 provided the best results.

Unfreezing the pretrained models didn't lead to anything good at all, and the loss function didn't seem to converge well, also convergence came with the price of an overfit. On freezed training of ResNet-34 and ResNet-50 for about 12-16 epochs and sometimes early stopping lead to an f1-score of about 83-84 on validation set and 79-81 on the test set.

Going through the dataset thoroughly it was found that classes like DecorationsAndSignage, misc, required much more attention to detail as those classes were very much distributed.

Thereafter, models with higher layer density like ResNet-152 and DenseNet-201 came into use. However, unfreezing didn't improve the performance to any level. With freezed layers and training about 10-12 epochs on ResNet-152 yielded a score of 86-87 on the validation set and 81-83 on the test set, which was the highest that I could possibly reach till now. DensNet-201 was a dissapointment for this dataset.

The above experiments are done on trials.ipynb file

Furthermore, on the lower part of model_train.ipynb file, we see there is a loop. That loop was introduced so that there is proper shuffling of data during training, (such that the model can be trained on not one particular training set but over a random sequence of images each time, and a random percentage being the validation set) and not just before the training, making the model more robust, and free from overfit. Though the loss function took more time to lessen, it was worth the wait, yielding an f1-score of 97 on the validation set, yet a dissapointing 84.22499 on the test set.

Feature extraction was taken care of by ResNet-152 and accordingly very complex and unique features were detected by the final convolutions of the model.

### To use the repo, please follow the steps : 

1. Clone the repo's master branch
2. Run the lines below
````
python organize_dataset.py
````
Make sure you have fastai installed
3. Run the notebook `model_train.ipynb` The model gets saved in a desired location
4. Run the line below
````
python model_test.py
````
This file leads to an output.csv file, which consists of the predictions given out by the trained model

This repository will be updated on a regular basis, depending on an increase in the score, and the approach involved behind it.
