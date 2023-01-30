from tqdm import tqdm
from torch.autograd import Variable
import json
import os
import torch.optim as optim
from datetime import datetime
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
import argparse
from model import *
from preprocess import *
from utils import *

##########################################
##### Training
##########################################

def build_GAN(in_features, args):
    enc = Encoder(args, in_features)
    attn = Attention(args)
    dec_attn = Decoder_attn(args, attn)
    modelG = Generator_wgan_attn(enc, dec_attn)
    modelD = Critic(num_layers=1)
    return modelD, modelG

def generator_loss(fake_output, fake_data, real_data,args):
    entropy = torch.mean(fake_output)
    mae = torch.mean(abs(fake_data-real_data))
    weight_accuracy = (0.8 * torch.sum(torch.sign(fake_data[real_data<0])==torch.sign(real_data[real_data<0]))+
                       0.2 * torch.sum(torch.sign(fake_data[real_data>=0])==torch.sign(real_data[real_data>=0]))) / fake_data.nelement()
    loss = -args.entropy_penalty * entropy + args.mae_penalty * mae - args.weight_acc_penalty * weight_accuracy
    return loss


def trainIter(train, train_news, modelD, modelG, optimizerD, optimizerG, embedding_matrix, args):
    modelD.train()
    modelG.train()

    train_d_loss = 0
    train_g_loss = 0
    n_batch = 0

    for j in range(len(train) // args.batch_size+1):

        idx = j * args.batch_size
        n_batch += 1

        if idx+args.batch_size>len(train):
            idx = len(train)-args.batch_size

        X = torch.stack([train[idx:idx + 1][0][0], train[idx + 1:idx + 2][0][0]], 0)
        for i in train[idx + 2:idx + args.batch_size]:
            X = torch.cat([X, i[0].unsqueeze(0)])  # (batch_size, seq_len, in_features)

        y = torch.stack([train[idx:idx + 1][0][1], train[idx + 1:idx + 2][0][1]], 0)
        for i in train[idx + 2:idx + args.batch_size]:
            y = torch.cat([y, i[1].unsqueeze(0)])  # (batch_size, future_step)

        w2v = torch.stack([train_news[idx:idx + 1][0], train_news[idx + 1:idx + 2][0]], 0)
        for i in train_news[idx + 2:idx + args.batch_size]:
            w2v = torch.cat([w2v, i.unsqueeze(0)])  # (batch_size, seq_len, quan_len)


        ##########################################
        ##### Train Discriminator
        ##########################################

        optimizerD.zero_grad()

        # generate fake data
        # generate noise
        noise = Variable(torch.FloatTensor(torch.rand(args.batch_size, X.size(1), 1)), requires_grad=True)
        X = Variable(torch.FloatTensor(X.float()), requires_grad=True)
        real_data = Variable(torch.FloatTensor(y.float()), requires_grad=True)

        w2v = Variable(torch.LongTensor(w2v.long()))

        # put noise and x into G to generate fake data
        fake_data = modelG(X, w2v, noise, embedding_matrix, real_data)  # (batch_size, future_step)

        # loss on real and fake data
        d_loss = -torch.mean(modelD(real_data)) + torch.mean(modelD(fake_data))

        d_loss.backward()
        optimizerD.step()

        # Clip weights of discriminator
        for p in modelD.parameters():
            p.data.clamp_(-args.clip_value, args.clip_value)

        ##########################################
        ##### Train Generator
        ##########################################

        if j % args.freq == 0:
            optimizerG.zero_grad()

            # feed fake data into optimized Discriminator again
            fake_data = modelG(X, w2v, noise, embedding_matrix, real_data)

            # loss on Generator
            g_loss = generator_loss(modelD(fake_data), fake_data, real_data,args)
            g_loss.backward()
            optimizerG.step()


        train_d_loss += d_loss.item()
        train_g_loss += g_loss.item()


    return train_d_loss / n_batch, train_g_loss / n_batch


##########################################
##### Evaluation
##########################################
def evaluate(eval_data, eval_news, modelD, modelG, embedding_matrix, args):

    # if modelD is not None:
    modelD.eval()
    modelG.eval()

    with torch.no_grad():
        X = torch.stack([eval_data[0:1][0][0], eval_data[1:2][0][0]], 0)
        for i in eval_data[2:len(eval_data)]:
            X = torch.cat([X, i[0].unsqueeze(0)])

        y = torch.stack([eval_data[0:1][0][1], eval_data[1:2][0][1]], 0)
        for i in eval_data[2:len(eval_data)]:
            y = torch.cat([y, i[1].unsqueeze(0)])  # (batch_size, future_step)

        w2v = torch.stack([eval_news[0:1][0], eval_news[1:2][0]], 0)
        for i in eval_news[2:len(eval_data)]:
            w2v = torch.cat([w2v, i.unsqueeze(0)])  # (batch_size, seq_len, quan_len)

        # generate fake data
        # generate noise
        noise = Variable(torch.FloatTensor(torch.rand(len(eval_data), X.size(1), 1)), requires_grad=True)
        X = Variable(torch.FloatTensor(X.float()), requires_grad=True)
        real_data = Variable(torch.FloatTensor(y.float()), requires_grad=True)
        w2v = Variable(torch.LongTensor(w2v.long()))
        # put noise and x into G to generate fake data
        fake_data = modelG(X, w2v, noise, embedding_matrix, real_data)  # (batch_size, future_step)
        d_loss = -torch.mean(modelD(real_data)) + torch.mean(modelD(fake_data))
        g_loss = generator_loss(modelD(fake_data), fake_data, real_data, args)

        # loss on real and fake data
        total_d_loss = d_loss.item()
        total_g_loss = g_loss.item()
        pre_dir, gt_dir, perf = make_prediction(y, fake_data, args.future_step)
        perf.extend([total_d_loss, total_g_loss])
        return pre_dir, gt_dir, perf


def make_prediction(gt, pre, future_step):
    # rolling window
    gt_value = []
    for i in range(len(gt) - future_step + 1):
        x = 0
        for j in range(future_step):
            x += gt[i + j, future_step - 1 - j]
        gt_value.append(x / future_step)

    pre_value = []
    for i in range(len(pre) - future_step + 1):
        x = 0
        for j in range(future_step):
            x += pre[i + j, future_step - 1 - j]
        pre_value.append(x / future_step)

    pre_dir = np.array([1 if i >= 0 else 0 for i in pre_value])
    gt_dir = np.array([1 if i >= 0 else 0 for i in gt_value])
    perf = [np.round(accuracy_score(gt_dir, pre_dir), 4), np.round(matthews_corrcoef(gt_dir, pre_dir), 4),
            np.round(f1_score(gt_dir, pre_dir), 4)]
    return pre_dir, gt_dir, perf

def test_model(in_data, in_news, args, embedding_matrix):
    torch.manual_seed(0)
    modelD, modelG = build_GAN(in_features, args)
    model_cp = torch.load('final_model.pth')
    modelD.load_state_dict(model_cp['modelD_state_dict'])
    modelG.load_state_dict(model_cp['modelG_state_dict'])

    acc, mcc, f1, g_loss, d_loss= [], [], [], [], []
    for i in range(args.test_epochs):
        pre, gt, perf = evaluate(in_data, in_news, modelD, modelG, embedding_matrix, args)
        acc.append(perf[0])
        mcc.append(perf[1])
        f1.append(perf[2])
        g_loss.append(perf[-1])
        d_loss.append(perf[-2])
    test_perf = pd.DataFrame({'accuracy': acc, 'mcc': mcc, 'f1': f1, 'd_loss': d_loss, 'g_loss': g_loss})

    print("[Accuracy: %.4f] [MCC: %.4f] [F1: %.4f] [Loss_D: %.4f] [Loss_G: %.4f]"
          % (np.mean(test_perf.iloc[:,0]), np.mean(test_perf.iloc[:,1]), np.mean(test_perf.iloc[:,2]),
             np.mean(test_perf.iloc[:,3]),np.mean(test_perf.iloc[:,4])))
    print("Standard deviation: [Accuracy: %.4f] [MCC: %.4f] [F1: %.4f] "
          % (np.std(test_perf.iloc[:, 0]), np.std(test_perf.iloc[:, 1]), np.std(test_perf.iloc[:, 2])))

    if np.mean(test_perf.iloc[:,0])>0.58:
        pd.DataFrame(test_perf).to_csv('test_perf.csv')


##########################################
##### Build model
##########################################

if __name__ == '__main__':

    torch.manual_seed(1)

    parser = argparse.ArgumentParser(description='PyTorch IndexGAN for stock movement prediction')
    parser.add_argument('--model', type=str, default = 'IndexGAN')
    parser.add_argument('--data', type=str, default= 'dji')
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
    features = ['bbhi', 'bbli', 'rsi_ind', 'macd_diff', 'open_p', 'close_p', 'high_p', 'low_p',
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
    print("Train, test and valid are completed")

    # initial D and G
    modelD, modelG = build_GAN(in_features, args)

    # initial optimizer
    optimizerD = optim.RMSprop(modelD.parameters(), lr=args.lr_D)
    optimizerG = optim.RMSprop(modelG.parameters(), lr=args.lr_G)

    # saving args
    os.mkdir(os.path.join("./outputs", args.data))
    os.chdir(os.path.join("./outputs", args.data))
    with open('args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # train model
    print("Start to build model: ")
    train_perf = []
    progress_bar = tqdm(range(args.num_epochs))
    for epoch in range(args.num_epochs):
        train_d_loss_avg, train_g_loss_avg = trainIter(train, all_news[0], modelD, modelG, optimizerD, optimizerG,
                                                       embedding_matrix, args)
        valid_pre, valid_gt, valid_perf = evaluate(valid, all_news[1], modelD, modelG, embedding_matrix, args)
        if epoch % 10 == 0:
            print("[Epoch %d/%d] [Train G loss: %.4f] [Train D loss: %.6f] [Valid G loss: %.4f] [Valid D loss: %.6f] "
                        % (epoch, args.num_epochs, train_g_loss_avg, train_d_loss_avg,
                           valid_perf[4], valid_perf[3]))

        valid_perf.extend([train_d_loss_avg, train_g_loss_avg])
        train_perf.append(valid_perf[3:])
        progress_bar.update(1)

    save_final_model(modelD, modelG, optimizerD, optimizerG, train_perf, args)

    # test model
    print("Final model performance on test set: ")
    test_model(test, all_news[2], args, embedding_matrix)



