# Learning Confidence
Implementation of Uncertainty Estimation using the Learning Confidence method

- _Original Paper:_ de Vries and Taylor 2017, https://arxiv.org/abs/1802.04865
- _Source Repository:_ https://github.com/uoguelph-mlrg/confidence_estimation

## Training
To train a Learning Confidence model using the Emblem use case dataset and storing the best model weights for later inference, run:

```
python train.py --uc_datapath ../../data/EmblemUseCase \
                --outdir Emblem_linear --dataset Emblem --model linear \
                --batch_size 16 --epochs 60 --learning_rate 1e-1 \
                --budget 0.1 --gpu 0 --seed 0
```
setting the respective parameters for the dataset path (```--uc_datapath```) and name of output directory (```--outdir```), where trained models will be saved under "checkpoints/outdir" and logs under "logs/outdir". The dataset and model type among one of: ```[linear, vgg16, resnet34, resnet50, densenet]``` are specified using ```--dataset``` and ```--model```, respectively, next to other network training hyperparameters including the budget parameter ```--budget```. See the ```train.py``` script for more details on possible parameters.

## Evaluation
For evaluation of a trained Learning Confidence model on ID and OOD data, the previously trained and stored model can be loaded and inferenced using the ```out_of_distribution_detection.py``` script in the following manner:
```
python out_of_distribution_detection.py --models_dir checkpoints/Emblem_linear \
                                        --uc_datapath ../../data/EmblemUseCase \
                                        --ind_dataset Emblem --ood_dataset Emblem \
                                        --model linear --batch_size 1 --save_txt \
                                        --budget 0.1 --gpu 0 --id 0 --seed 2
```
referencing the above model path  (```--models_dir```), dataset directory (```--uc_datapath```), as well as the same parameter settings as above for the ```--model```, ID dataset ```--ind_dataset``` and corresponding OOD dataset ```--ind_dataset```. \
Setting ```--save_txt``` will store the computed results for all ID and OOD metrics in a dedicated txt.-file under a "result_files" folder for later reference and further analysis. Additionally, two flags are available: ```--final_model``` is set if a model performance is achieved that you want to compare to other UE method. This saves all relevant data for result plots (ROC, histogram, ...) in a numpy file under the "Integration" directory so that comparative diagrams can be created there. The ```--save_uncimgs``` flag can be set for further output image analysis: All images will be saved named and sorted according to their assigned uncertainty value in an "OUT_images" folder.