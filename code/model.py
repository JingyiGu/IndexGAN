import torch
import torch.nn as nn
import torch.nn.functional as F

##########################################
##### Generator
##########################################

class Encoder(nn.Module):
    def __init__(self, args, in_features):
        super(Encoder, self).__init__()

        self.enc_size = args.enc_size
        self.seq_len = args.seq_len

        self.model1 = nn.Sequential(
            nn.Linear(args.glove_dim * 372, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256))
        self.batch = nn.BatchNorm1d(256, 0.8)
        self.model2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, args.w2v_size),
            nn.Tanh() )

        self.gru = nn.GRU(1 + in_features + args.w2v_size, self.enc_size, 1, batch_first=True)


    def forward(self, input, w2v, noise, embedding_matrix):
        # blocks for w2v
        self.batch_size = input.size(0)
        w2v_out = nn.Embedding.from_pretrained(embedding_matrix)(w2v).view(self.batch_size, self.seq_len, -1)
            # (batch_size, seq_len, quan_len, dimension) --> # (batch, seq_len, 18600)
        w2v_out = self.model1(w2v_out).permute(0, 2, 1)
        w2v_out = self.model2(self.batch(w2v_out).permute(0, 2, 1))

        # concatenate noise, w2v and price, then gru as encoder
        h = torch.rand(1, self.batch_size, self.enc_size)
        noise_x = torch.cat((noise, input, w2v_out), dim=2)  # noise_x (batch_size, seq_len, 1 + in_features + w2v_size)
        encoder_out, encoder_h = self.gru(noise_x, h)  # output (batch_size, seq_len, enc_size), h (1, batch_size, enc_size)
        return encoder_out, encoder_h


class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.attn = nn.Linear(args.enc_size, args.dec_size, bias=False)
        self.v = nn.Linear(args.dec_size, 1, bias=False)


    def forward(self, enc_out):
        batch_size, seq_len, _ = enc_out.size()

        energy = torch.tanh(self.attn(enc_out))  # [batch_size, seq_len, dec_size]
        attention = self.v(energy).squeeze(2)  # [batch_size, seq_len]

        return F.softmax(attention, dim=1)


class Decoder_attn(nn.Module):
    def __init__(self, args, attn):
        super(Decoder_attn, self).__init__()
        self.attention = attn
        self.fc = nn.Sequential(
            nn.Linear(args.enc_size, args.dec_size),
            nn.Tanh())
        self.rnn = nn.GRU(args.enc_size, args.dec_size, batch_first=True)
        self.fc_out = nn.Linear(args.dec_size, args.future_step)

    def forward(self, dec_h, enc_out):
        a = self.attention(enc_out).unsqueeze(1) # [batch_size, 1, seq_len]
        c = torch.bmm(a, enc_out) # [batch_size, 1, enc_size ]

        dec_h = self.fc(dec_h)  # (1, batch_size, dec_size)
        dec_output, dec_hidden = self.rnn(c, dec_h) # output: (batch_size, 1, dec_size) h: (1, batch_size, dec_size)
        pred = self.fc_out(dec_output.squeeze(1)) # [batch_size, future_step]
        return pred


class Generator_wgan_attn(nn.Module):
    def __init__(self, encoder, decoder_attn):
        super(Generator_wgan_attn, self).__init__()
        self.encoder = encoder
        self.decoder = decoder_attn

    def forward(self, input, w2v, noise, embedding_matrix, target):
        # input: [batch_size, seq_len, in_features]
        # target: [batch_size, future_step]

        # encoder
        enc_out, hidden = self.encoder(input, w2v, noise, embedding_matrix)  # output (batch_size, seq_len, enc_size), h (1, batch_size, dec_size)

        # decoder
        output = self.decoder(hidden, enc_out)

        return output



##########################################
##### Critic
##########################################

class Critic(nn.Module):
    def __init__(self, num_layers=1):
        super(Critic, self).__init__()
        self.num_layers = num_layers
        # self.hidden_s = hidden_s
        self.gru = nn.GRU(1, 8, self.num_layers, batch_first=True)
        self.out = nn.Linear(8, 1)

    def forward(self, input):
        batch_size = input.size(0)
        seq_len1 = input.size(1)
        x = input.view(batch_size, seq_len1, 1)
        h = torch.rand(self.num_layers, batch_size, 8)
        output, hidden = self.gru(x, h)  # output (batch_size, seq_len+1, hidden_dim)
        output1 = self.out(output[:, -1, :].view(batch_size, 8))
        return output1


