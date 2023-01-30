# IndexGAN

## Train the model 
### Step 1: Install the packages
* python 3.9.6
* pandas 1.3.1
* numpy 1.21.1
* nltk 3.6.2
* ta 0.7.0
* scikit-learn 0.24.2
* keras-Preprocessing 1.1.2
* torch 1.9.0
* tqdm 4.62.0
* matplotlib 3.4.2

### Step 2: Preparation
1. Download pre-trained word vectors from [GloVe](https://nlp.stanford.edu/projects/glove/). Make a directory of ./data/glove/ and save glove.6B.50d.txt.
2. Create a directory ./outputs to save the training log and trained model

### Step 3: Run the model
The models with best performance: ./saved_model/data/final_model.pth

Corresponding parameters: ./saved_model_data/args.txt
#### DJI
```Python
python ./code/train.py --data dji --num_epochs 100 --enc_size 100 --dec_size 200 --w2v_size 6 --freq 5
```
#### SPX
```Python
python ./code/train.py --data spx --num_epochs 80 --enc_size 100 --dec_size 50 --w2v_size 3 --freq 7
```
