import torch
import numpy as np
import pandas as pd
import argparse
from preprocess import *
from train import *
from model import *

torch.manual_seed(0)
parser = argparse.ArgumentParser(description='PyTorch IndexGAN for stock movement prediction')
parser.add_argument('--model', type=str, default = 'IndexGAN')
parser.add_argument('--data', type=str, default= 'SPX')
parser.add_argument('--seq_len', type=int, default=35)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr_D', type=float, default=0.00005)
parser.add_argument('--lr_G', type=float, default=0.0001)
parser.add_argument('--future_step', type=int, default=5)
parser.add_argument('--num_epochs', type=int, default=80)
parser.add_argument('--glove_dim', type=int, default=50)
parser.add_argument('--enc_size', type=int, default=100)
parser.add_argument('--dec_size', type=int, default=50)
parser.add_argument('--w2v_size', type=int, default=3)
parser.add_argument('--clip_value', type=float, default=0.01)
parser.add_argument('--entropy_penalty', type=float, default=1)
parser.add_argument('--mae_penalty', type=float, default=10)
parser.add_argument('--weight_acc_penalty', type=float, default=3)
parser.add_argument('--freq', type=int, default=7)
parser.add_argument('--test_epochs', type=int, default=10)
args = parser.parse_args()

# features
features = ['bbhi', 'bbli', 'rsi_ind', 'macd_diff',
            'open_p', 'close_p', 'high_p', 'low_p',
            'ema5', 'sma13', 'sma21', 'sma50', 'sma200', 'vxd', 'vxd_p']
in_features = len(features)
close_index = features.index('close_p')

# read data
if args.data == 'dji':
    stock = pd.read_csv('./data/DJI.csv')
    vxd = pd.read_csv('./data/VXD.csv')
    stock['vxd'] = vxd.Close
    print("DJIA data is ready")
else: # spx
    stock = pd.read_csv('./data/SPX.csv')
    vxd = pd.read_csv('./data/VIX.csv')
    stock['vxd'] = vxd.Close
    print('SP500 data is ready')

# process data
all_news, embedding_matrix, vocab_size = process_news(args.glove_dim, args.seq_len, args.future_step)
stock = feature_eng(stock)
train, valid, test = load_data(stock, args.seq_len, args.future_step, features, close_index)

# initial D and G
modelD, modelG = build_GAN(in_features, args)
model_cp = torch.load('./outputs/SPX/final_model.pth') # or input other path for saved model
modelD.load_state_dict(model_cp['modelD_state_dict'])
modelG.load_state_dict(model_cp['modelG_state_dict'])

acc, mcc, f1, g_loss, d_loss= [], [], [], [], []
for i in range(args.test_epochs):
    pre, gt, perf = evaluate(test, all_news[2], modelD, modelG, embedding_matrix, args)
    acc.append(perf[0])
    mcc.append(perf[1])
    f1.append(perf[2])
    g_loss.append(perf[-1])
    d_loss.append(perf[-2])
test_perf = pd.DataFrame({'accuracy': acc, 'mcc': mcc, 'f1': f1, 'd_loss': d_loss, 'g_loss': g_loss})

print("[Accuracy: %.4f] [MCC: %.4f] [F1: %.4f] [Loss_D: %.4f] [Loss_G: %.4f]"
      % (np.mean(test_perf.iloc[:,0]), np.mean(test_perf.iloc[:,1]), np.mean(test_perf.iloc[:,2]),
         np.mean(test_perf.iloc[:,3]),np.mean(test_perf.iloc[:,4])))
print("[Accuracy: %.4f] [MCC: %.4f] [F1: %.4f] "
      % (np.std(test_perf.iloc[:,0]), np.std(test_perf.iloc[:,1]), np.std(test_perf.iloc[:,2])))
pd.DataFrame(test_perf).to_csv('test_perf.csv')
