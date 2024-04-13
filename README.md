# ZeroHAR
This is a repository to implement Zero-Shot Wearable Human Activity Recognition.



## Datasets
Because the file sizes are bigger than the maximum size supported by Github and uploading them in Google Drive/DropBox risks revealing author identity, we release pointers to the datasets. We will upload our preprocessed datasets on Google Drive once the review process is complete.

Opportunity: https://archive.ics.uci.edu/dataset/226/opportunity+activity+recognition

PAMAP2: http://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring

harth: https://github.com/ntnu-ai-lab/harth-ml-experiments

wisdm: https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset

uschad: https://sipi.usc.edu/had/

Split the data according to the class splits in Table 7 under Appendix A.2 of the paper. Place X_train.npy, y_train.npy, X_test.npy, y_test.npy under the directory of the corresponding dataset folder under data/.



## Metadata

The metatdata folder contains the following files:

`hyperparameters.json` - hyperparameter config for all datasets

`gpt4_label_description_variations.json` - GPT-4 generated activity descriptions for each activity class of each dataset. There are 10 descriptions to each activity.

`gpt4_label_description_variations_imagebind_embedding.pickle` - ImageBind embeddings of the text in `gpt4_label_description_variations.json`.

`sensor_description.json` - Text data for Stage I training, containing information on measurement axis, sensor name and body placement information.

`sensor_description_imagebind_embedding.pickle` - ImageBind embeddings of the text in `sensor_description.json`.



## Stage I Training

```
python3 stage1_script.py --dataset uschad --zsl --fold fold1
```

This command runs Stage I training on `uschad` dataset for `fold 1`. Folds are defined in Table 7 under Appendix A.2 of the paper. After Stage I training completes, it will save the model and optimizer.



## Stage 2 Training and Evaluation

```
python3 stage2_script.py --dataset uschad --zsl --fold fold1 --pretrained
```

This command runs Stage II training on `uschad` dataset for `fold 1` using the model saved in Stage I. Folds are defined in Table 7 under Appendix A.2 of the paper. It will also run the evaluation on the test set and generate the Accuracy and F1 metrics for the given fold.

