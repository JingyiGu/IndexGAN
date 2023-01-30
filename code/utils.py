import matplotlib.pyplot as plt
import pandas as pd
import torch
plt.style.use('ggplot')

def save_final_model(modelD, modelG, opt_D, opt_G, train_perf, args):
    print(f"Saving final model...")
    torch.save({'args': args,
                'modelD_state_dict': modelD.state_dict(),
                'modelG_state_dict': modelG.state_dict(),
                'optimizerD_state_dict': opt_D.state_dict(),
                'optimizerG_state_dict': opt_G.state_dict()
                }, 'final_model.pth')

    train_perf = pd.DataFrame(train_perf, columns = ['valid_loss_d', 'valid_loss_g', 'train_loss_d', 'train_loss_g'])
    train_perf.to_csv('train_log.csv')
    save_plots(train_perf)


def save_plots(train_perf):
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_perf.iloc[:,3], color='orange', linestyle='-',label='train loss G')
    plt.plot(train_perf.iloc[:,1], color='red', linestyle='-',label='validation loss G')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_g.png')

