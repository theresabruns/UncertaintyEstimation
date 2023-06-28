# Deterministic Uncertainty Quantification (DUQ)
Implementation of Uncertainty Estimation using the DUQ method

- _Original Paper:_ van Amersfoort et al. 2020, https://arxiv.org/abs/2003.02037
- _Source Repository:_ https://github.com/y0ast/deterministic-uncertainty-quantification

## Training
To train a DUQ model using the Emblem use case dataset and storing the best model weights for later inference, run:

```
python train_duq.py --output_dir Emblem_linear --architecture linear --data Emblem \
                    --aug --batch_size 16 --learning_rate 5e-2 --weight_decay 5e-4 \
                    --epochs 30 --l_gradient_penalty 0.6 --length_scale 0.2 \
                    --gpu 1 --seed 0
```
setting the respective parameters for the type of use case (```--usecase```), specifying the path for model storage (```--output_dir```), the choice of backbone architecture (```--architecture```) among one of: ```[linear, vgg16, resnet34, resnet50, densenet]``` as well as other network training hyperparameters including the ```--l_gradient_penalty``` and ```--length_scale``` parameters as described in the original paper. See the ```train_duq.py``` script for more details on possible parameters.

When using your own custom dataset, it is necessary to define a dedicated function in ```datasets.py``` that transforms you data, loads it from its directory and creates dataloaders with it. See the respective script for examples.


## Evaluation
For evaluation of a trained DUQ model on ID and OOD data, the previously trained and stored model can be loaded and inferenced using the ```eval_duq.py``` script in the following manner:
```
python eval_duq.py --exp_dir Emblem_linear --data Emblem --architecture linear \
                     --save_txt --length_scale 0.2 --seed 0 --id 0 --gpu 1 
```
referencing the experiment path  (```--exp_dir```) containing the trained model and the same parameter settings as above for ```--data```, ```--architecture``` and ```--length_scale```. \
Setting ```--save_txt``` will store the computed results for all ID and OOD metrics in a dedicated txt.-file under a "result_files" folder for later reference and further analysis. Additionally, two flags are available: ```--final_model``` is set if a model performance is achieved that you want to compare to other UE method. This saves all relevant data for result plots (ROC, histogram, ...) in a numpy file under the "Integration" directory so that comparative diagrams can be created there. The ```--save_uncimgs``` flag can be set for further output image analysis: All images will be saved named and sorted according to their assigned uncertainty value in an "OUT_images" folder.

<hr>

***Note:*** This subrepostiory contains an additional implementation for the Deep Ensemble method that can be executed as a second reference using the ```train_deep_ensemble.py``` and ```eval_deep_ensemble.py```. This implementation is mainly used as a sanity check for debugging purposes and is therefore not further detailed here. See the respective scripts for more details and possible parameter settings. 