# Bayes by Backprop (BBB)
Implementation of Uncertainty Estimation using the BBB method

- _Original Paper:_ Blundell et al. 2015, https://arxiv.org/abs/1505.05424
- _Source Repository:_ https://github.com/JavierAntoran/Bayesian-Neural-Networks 

## Training
To train a BBB model using the Emblem use case dataset and storing the best model weights for later inference, run:

```
python train_BayesByBackprop.py --models_dir BBP_models/Emblem_Gaussian \
                                --results_dir BBP_results/Emblem_Gaussian \
                                --uc_datapath ../../data/EmblemUseCase \
                                --data usecase --usecase Emblem \
                                --model Gaussian_prior --nhid 200 \
                                --weight_decay 5e-4 --prior_sig 0.1 --epochs 30 \ --batch_size 16 --lr 1e-4 --aug --seed 0
```
setting the respective parameters for the dataset path (```--uc_datapath```), the type of use case (```--usecase```), specifying the paths for model and result storage (```--model_dir, --results_dir```), the choice of model (```--model```) among one of: ```[Gaussian_prior, Laplace_prior, GMM_prior, Local_Reparam]``` as well as other network training hyperparameters including the number of hidden units ```--nhid``` or prior standard deviation ```--prior_sig```. Note that here, the backbone model is always linear (FC layers) according to the proposed method. See the ```train_BayesByBackprop.py``` script for more details on possible parameters.

## Evaluation
For evaluation of a trained BBB model on ID and OOD data, the previously trained and stored model can be loaded and inferenced using the ```pred_BayesByBackprop.py``` script in the following manner:
```
python pred_BayesByBackprop.py --models_dir BBP_models/Emblem_Gaussian \
                                --uc_datapath ../../data/EmblemUseCase \
                                --data usecase --ood_data usecase --usecase Emblem \ 
                                --model Gaussian_prior --nhid 200 \
                                --measure entropy --save_txt --gpu 0 \
                                --n_samples 20 --prior_sig 0.1 --seed 0
```
referencing the above model path  (```--model_dir```), dataset directory (```--uc_datapath```), as well as the same parameter settings as above for ```--data```, ```--usecase```, ```--model```, ```--nhid``` and ```--prior_sig```. To perform UE using MC Dropout, different uncertainty measures are available. Here, for the ```--measure``` parameter, you can choose between ```[entropy, conf, mutualinfo]``` for the entropy of the expected output, the maximum softmax confidence and mutual information, respectively. \
Setting ```--save_txt``` will store the computed results for all ID and OOD metrics in a dedicated txt.-file under a "result_files" folder for later reference and further analysis. Additionally, two flags are available: ```--final_model``` is set if a model performance is achieved that you want to compare to other UE method. This saves all relevant data for result plots (ROC, histogram, ...) in a numpy file under the "Integration" directory so that comparative diagrams can be created there. The ```--save_uncimgs``` flag can be set for further output image analysis: All images will be saved named and sorted according to their assigned uncertainty value in an "OUT_images" folder.