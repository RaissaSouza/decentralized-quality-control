# Self-supervised identification and elimination of harmful datasets in distributed machine learning for medical image analysis

<div align="center">

</div>

<p align="center">
<img src="workflow.png?raw=true">
</p>


Implementation for decentralized quality control that is published by the (coming soon): "[Self-supervised identification and elimination of harmful datasets in distributed machine learning for medical image analysis] (DOI).

Our code here is based on the investigation of a Parkinson's disease classification using non-identical distribution across 83 centers, including simulated harmful data samples to investigate when entire centers provide only harmful data and when a single hamrful data sample is added to otherwise good datasets.

If you find our framework, code, or paper useful to your research, please cite us!
```
@article{
}

```
```
Souza, R., 
```

### Abstract 

### Folder organization
1. **dirty_baselines**: load a pre-trained model and continue to train it including all datasets (good and harmful) to evaluate how the model would perform if harmful data samples are included in the training.
2. **generate_harmful_samples**: cointains a python script that load a MRR dataset and invert it to get the inverted MRI sample and add noise and clip the brain tissue to get the pure noise sample.
3. **inference**: has a script that generate the metrics (accuracy, sensitivity, specificity, F1 score, AUROC for the overall dataset and for subgroups based on sex and age) for the models per cycle.
4. **standard_travelling_model**: train a classifier without quality control. Use this one to get the clean baseline and the pre-trained model.
5. **travelling_quality_control**: train a classifier with quality control steps: verification, revisit, elimination. This script alo generate a file that shows false positive rate, false negative rate, and a binary flag that determines if an image was flagged and eliminated.

### Step-by-step implementation
<div align="center">

</div>

<p align="center">
<img src="flowchart.png?raw=true">
</p>

## All scripts have parameters that need to be called with descriptions in the argument parser. An example of how to call all of them:

```
python enc_PD_train_dirt_baseline.py -fn_train ./training_set.csv -en ./path_pretrained_encoder -pd ./path_pretrained_classifier -cycles 30 -epochs 1 -batch_size 5
```

For the generate harmful data samples replace the name of the nifti file in the script.

```
python inference_pd_distributed.py -fn ./test_set.csv -en ./path_encoder -pd ./path_classifier -o filename_to_save
```

For the inference, you can change the loop indices to determine the range of models you want to evaluate. For example, for the clean baseline, you want to check the performance from 0 to 30 (max cycle trained); however, for the dirty baselines, you want to check from 10 to 30 because it uses a pre-trained model.

```
python enc_PD_train_distributed.py -fn_train ./training_set.csv -cycles 30 -epochs 1 -batch_size 5
```
You may change the fixed name to save the models and the folder you want to save them.

```
python enc_PD_quality_control.py -fn_train ./training_set.csv -fn_test ./test_set.csv  -cycles 30 -en ./path_encoder -pd ./path_classifier -revisit 2 -error 0.02 -fn_save filename
```

You may change the number of local epochs (now fixed to 1) and batch_size (now fixed to 5 when dataset >=5).

## Environment 
Our code for the Keras model pipeline used: 
* Python 3.10.6
* pandas 1.5.0
* numpy 1.23.3
* scikit-learn 1.1.2
* simpleitk 2.1.1.1
* tensorflow-gpu 2.10.0
* cudnn 8.4.1.50
* cudatoolkit 11.7.0

GPU: NVIDIA GeForce RTX 3090

Full environment in `requirements.txt`.


## Resources
* Questions? Open an issue or send an [email](mailto:raissa_souzadeandrad@ucalgary.ca?subject=decentralized_quality_control).
