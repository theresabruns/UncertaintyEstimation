# Monte Carlo Dropout
Implementation of Uncertainty Estimation using the MC Dropout method

- _Original Paper:_ Gal and Ghahramani 2016, https://arxiv.org/abs/1506.02142
- _Source Repository:_ https://github.com/JavierAntoran/Bayesian-Neural-Networks 

## Training
To train a MC Dropout model using the Emblem use case dataset and storing the best model weights for later inference, run:

```
python train_MCDropout.py --models_dir MCdrop_models/RREmblem_linear \
                            --results_dir MCdrop_results/RREmblem_linear \
                            --uc_datapath ../../data/EmblemUseCase \
                            --data usecase --usecase Emblem --model linear \
                            --epochs 30 --batch_size 64 --lr 1e-4 \
                            --pdrop 0.3 --weight_decay 1 --aug --seed 0
```
setting the respective parameters for the dataset path (```--uc_datapath```), the type of use case (```--usecase```), specifying the paths for model and result storage (```--model_dir, --results_dir```), the choice of model (```--model```) among one of: ```[linear, vgg16, resnet34, resnet50, densenet]``` as well as other network training hyperparameters including the dropout probability ```--pdrop```. See the ```train_MCDropout.py``` script for more details on possible parameters.


## Evaluation
For evaluation of a trained MCDropout model on ID and OOD data, the previously trained and stored model can be loaded and inferenced using the ```pred_MCDropout.py``` script in the following manner:
```
python pred_MCDropout.py --models_dir MCdrop_models/RREmblem_linear \
                            --uc_datapath ../../data/EmblemUseCase \
                            --data usecase --usecase Emblem --model linear \
                            --pdrop 0.3 --gpu 1 --save_txt \
                            --measure entropy --seed 0
```
referencing the above model path  (```--model_dir```), dataset directory (```--uc_datapath```), as well as the same parameter settings as above for ```--data```, ```--usecase```, ```--model``` and ```--pdrop```. To perform UE using MC Dropout, different uncertainty measures are available. Here, for the ```--measure``` parameter, you can choose between ```[entropy, conf, mutualinfo]``` for the entropy of the expected output, the maximum softmax confidence and mutual information, respectively. \
Setting ```--save_txt``` will store the computed results for all ID and OOD metrics in a dedicated txt.-file under a "result_files" folder for later reference and further analysis. Additionally, two flags are available: ```--final_model``` is set if a model performance is achieved that you want to compare to other UE method. This saves all relevant data for result plots (ROC, histogram, ...) in a numpy file under the "Integration" directory so that comparative diagrams can be created there. The ```--save_uncimgs``` flag can be set for further output image analysis: All images will be saved named and sorted according to their assigned uncertainty value in an "OUT_images" folder.