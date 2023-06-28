# Mahalanobis
Implementation of Uncertainty Estimation using the Mahalanobis distance-based method

- _Original Paper:_ Lee et al. 2018, https://arxiv.org/abs/1807.03888
- _Source Repository:_ https://github.com/pokaxpoka/deep_Mahalanobis_detector

In contrast to the other UE methods in this repository, the script structure for training and inference is bit different. The process that needs to be followed here is as follows:
1. Train a regular NN backbone model on custom data with ```usecase_train.py```.
2. Load this pretrained model in ```OOD_Generate_Mahalanobis.py``` to perform clustering in feature space and generate the predictions and Mahalanobis distance scores (for several intermediate feature layers). The scores are stored for a separate, subsequent evaluation step.
3. Run ```OOD_Regression_Mahalanobis.py``` to evaluate the quality of generated scores. This executes OOD detection using a separate Linear Regression module and provides the final output plots.

## Backbone Training
To pretrain and store the backbone network for a Mahalanobis model using the Emblem use case dataset, run:

```
python usecase_train.py --models_dir pre_trained/Emblem_linear \
                        --results_dir train_results/Emblem_linear \
                        --uc_datapath ../../data/EmblemUseCase \
                        --model linear --usecase Emblem \
                        --gpu 1 --epochs 30 --batch_size 16 \
                        --lr 1e-4 --aug --seed 0

```
setting the respective parameters for the dataset path (```--uc_datapath```), the type of use case (```--usecase```), specifying the paths for model and result storage (```--model_dir, --results_dir```), the choice of model (```--model```) among one of: ```[linear, vgg16, resnet34, resnet50, densenet]``` as well as other network training hyperparameters. See the ```usecase_train.py``` script for more details on possible parameters.


## Generation
To generate predictions and distance (i.e., uncertainty) scores for both ID and OOD data using the Mahalanobis method, the pretrained models are loaded in the ```OOD_Generate_Mahalanobis.py``` script. Here, a clustering of the feature space(s) is performed and corresponding Mahalanobis distances are calculated by running: 

```
python OOD_Generate_Mahalanobis.py --models_dir pre_trained/Emblem_linear \
                                    --dataset usecase --usecase Emblem \
                                    --uc_datapath ../../data/EmblemUseCase \
                                    --gpu 1 --model linear --seed 0  
```
referencing the paths to the pretrained model from above (```--models_dir```) and the custom dataset directory (```--uc_datapath```) as well as defining the same parameter settings for ```--data```, ```--usecase```, ```--model```. Generated scores are stored in an "output" folder for subsequent evaluation.

## Evaluation
For evaluation of the previously generated distance scores on ID and OOD data, run the ```OOD_Regression_Mahalanobis.py``` script in the following manner:

```
python OOD_Regression_Mahalanobis.py --dataset usecase --usecase Emblem \                                   --model linear --id 0 --seed 0
```
simply referencing the same parameter settings as above for ```--data```, ```--usecase```, ```--model``` and the model seed ```--seed```. \
Additionally, the ```--final_model``` can be set if a model performance is achieved that you want to compare to other UE method. This saves all relevant data for result plots (ROC, histogram, ...) in a numpy file under the "Integration" directory so that comparative diagrams can be created there. 

In contrast to the other subrepositories, there is no flag dedicated for the output of an uncertainty-sorted image folder. Due to the different script structuring and the separation of Generation and Regression, this image output requires a dedicated script: ```get_images.py```
It can be run by:
```
python get_images.py --models_dir pre_trained/Emblem_linear  \
                    --uc_datapath ../../data/EmblemUseCase \
                    --data usecase --usecase Emblem \ 
                    --batch_size 8 --aug --model linear --seed 0
```
All images will be saved asnamed and sorted according to their assigned uncertainty score in an "OUT_images" folder.