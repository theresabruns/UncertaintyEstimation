# Deep Ensembles
Implementation of Uncertainty Estimation using the Deep Ensemble method

- _Original Paper:_ Lakshminarayanan et al. 2017, https://arxiv.org/abs/1612.01474 
- _Source Repository:_ https://github.com/JavierAntoran/Bayesian-Neural-Networks 

## Training
To train a Deep Ensemble with 4 members using the Emblem use case dataset and storing the best model weights for later inference, run:

```
python train_DeepEnsemble.py --models_dir Ensemble_models/RREmblem_linear \
                            --results_dir Ensemble_results/RREmblem_linear \
                            --uc_datapath ../../data/EmblemUseCase \
                            --data usecase --usecase Emblem --model linear \
                            --num 4 --epochs 30 --batch_size 64 --lr 1e-4 \
                            --weight_decay 1 --aug --seed 0
```
setting the respective parameters for the dataset path (```--uc_datapath```), the type of use case (```--usecase```), specifying the paths for model and result storage (```--model_dir, --results_dir```), the choice of model (```--model```) among one of: ```[linear, vgg16, resnet34, resnet50, densenet]``` as well as other network training hyperparameters. See the ```train_DeepEnsemble.py``` script for more details on possible parameters.


## Evaluation
For evaluation of a trained Deep Ensemble on ID and OOD data, the previously trained and stored models can be loaded and inferenced using the ```pred_DeepEnsemble.py``` script in the following manner:
```
python pred_DeepEnsemble.py --models_dir Ensemble_models/RREmblem_linear \
                            --uc_datapath ../../data/EmblemUseCase \
                            --data usecase --usecase Emblem --model linear \
                            --members 4 --gpu 1 --save_txt \
                            --measure entropy --seed 0
```
referencing the above model path  (```--model_dir```), dataset directory (```--uc_datapath```), as well as the same parameter settings as above for ```--data```, ```--usecase```, ```--model``` and ```--members```. To perform UE using Deep Ensembles, different uncertainty measures are available. Here, for the ```--measure``` parameter, you can choose between ```[entropy, conf, mutualinfo]``` for the entropy of the expected output, the maximum softmax confidence and mutual information, respectively. \
Setting ```--save_txt``` will store the computed results for all ID and OOD metrics in a dedicated txt.-file under a "result_files" folder for later reference and further analysis. Additionally, two flags are available: ```--final_model``` is set if a model performance is achieved that you want to compare to other UE method. This saves all relevant data for result plots (ROC, histogram, ...) in a numpy file under the "Integration" directory so that comparative diagrams can be created there. The ```--save_uncimgs``` flag can be set for further output image analysis: All images will be saved named and sorted according to their assigned uncertainty value in an "OUT_images" folder.