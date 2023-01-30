import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import string, re
import ta
from sklearn.preprocessing import MinMaxScaler
import torch
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import itertools



##########################################
##### read and process stock data
##########################################

# news: 73608, ['Date', 'News']
# stock: 1989, ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'], Close is same as Adj Close
# combined: 1989, ['Date', 'Label','Top1',...'Top25']
# reverse columns in stock and news, sort the date from early to lately


def feature_eng(stock):
    # Technical indicators (need to min max scaler later)
    # Volatility：Bollinger Bands (Upper and Lower Bands), Indicator (0,1)
    bb = ta.volatility.BollingerBands(close=stock["Close"], window=20, window_dev=2)
    stock['bbhi'] = bb.bollinger_hband_indicator()
    stock['bbli'] = bb.bollinger_lband_indicator()

    # Trend： Average Directional Movement Index [0,100]
    stock['macd_diff'] = ta.trend.macd_diff(close=stock.Close)

    # Momentum
    stock['rsi'] = ta.momentum.rsi(close=stock.Close)
    stock['rsi_ind'] = np.where(stock['rsi'] < 20, -1, np.where(stock['rsi'] > 80, 1, 0))

    # open, high, low, close: t-1
    stock['close_t-1'] = stock.Close.shift(1)
    stock['open_t-1'] = stock.Open.shift(1)
    stock['high_t-1'] = stock.High.shift(1)
    stock['low_t-1'] = stock.Low.shift(1)
    stock['vxd_t-1'] = stock.vxd.shift(1)

    stock['open_p'] = stock.Open / stock['open_t-1'] - 1
    stock['high_p'] = stock.High / stock['high_t-1'] - 1
    stock['low_p'] = stock.Low / stock['low_t-1'] - 1
    stock['close_p'] = stock.Close / stock['close_t-1'] - 1
    stock['vxd_p'] = stock.vxd/stock['vxd_t-1'] - 1

    # ema5, 13-day, 21-day, 50-day, 200-day SMA/close_t - 1
    stock['ema5'] = ta.trend.EMAIndicator(close=stock.Close, window=5).ema_indicator()
    stock['ema5_t-1'] = stock['ema5'].shift(1)
    stock['ema5_p'] = stock['ema5']/stock['ema5_t-1'] - 1

    # ema5, 13-day, 21-day, 50-day, 200-day SMA/close_t - 1
    stock['ema5'] = ta.trend.EMAIndicator(close=stock.Close, window=5).ema_indicator().shift(1) / stock['Close'] - 1
    sma_list = [13, 21, 50, 200]
    for i in sma_list:
        stock['sma' + str(i)] = ta.trend.SMAIndicator(close=stock.Close, window=i).sma_indicator().shift(1) / stock['Close'] - 1

    # drop the rows with nan, match the index with news
    stock.drop(stock.index[0:286], inplace=True)
    stock.reset_index(drop=True, inplace=True)

    scaler = MinMaxScaler()
    scale_features = ['rsi', 'macd_diff', 'vxd']
    stock[scale_features] = scaler.fit_transform(stock[scale_features])
    return stock


def load_data(data, seq_len, future_step, features, close_index):
    # split into sequence
    seq_data = []
    X = torch.tensor(np.array(data[features]))
    y = X[:, close_index]

    for i in range(len(X) - seq_len - future_step):
        seq_data.append((X[i:i + seq_len],  # x_1 to x_t including close_p
                         y[i + seq_len:i + seq_len + future_step]))  # y_t+1-y_t+5

    print('Total instances:', len(seq_data))

    train_num = int(len(seq_data) * 0.8)
    valid_num = int(len(seq_data) * 0.1)
    test_num = valid_num

    train = seq_data[0:train_num]
    valid = seq_data[train_num:train_num + valid_num]
    test = seq_data[train_num + valid_num:train_num + valid_num + test_num]

    print('Training instances:', len(train))  # 2008-09-25 - 2008-10-01, 2014-12-02 - 2014-12-08
    print('Validation instances:', len(valid))  # 2014-12-03 - 2014-12-09, 2015-09-10 - 2015-09-16
    print('Testing instances:', len(test))  # 2015-09-11 - 2015-09-17, 2016-06-17 - 2016-06-23

    return train, valid, test


##########################################
##### read and process news data
##########################################
def clean_news(combined):
    # combine news in the same day together
    combined_news = combined.drop(['Date', 'Label'], axis=1, inplace=False)
    combined_news.drop(combined_news.index[0:33], inplace=True)
    combined_news.reset_index(drop=True, inplace=True)
    combined_news = combined_news.applymap(lambda x: x[2:] if str(x).startswith('b') else x)
    combined_news['news'] = combined_news.apply(lambda x: ' '.join(x.dropna()), axis=1)
    cleaned_news = [" ".join(remove_stop(combined_news.news[i])) for i in range(len(combined_news.news))]

    return cleaned_news

def remove_stop(doc):
    # turn a doc into clean tokens
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    # tokens = [w.translate(table,string.punctuation) for w in tokens]
    tokens = [regex.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word.lower() for word in tokens if len(word) > 1]
    return tokens


def load_news(data, seq_len, future_step):
    # split into sequence
    seq_data = []

    for i in range(len(data) - seq_len - future_step):
        seq_data.append(torch.tensor(data[i:i + seq_len]))  # x_1 to x_t

    print('Total news instances:', len(seq_data))

    train_num = int(len(seq_data) * 0.8)
    valid_num = int(len(seq_data) * 0.1)
    test_num = valid_num

    train_news = seq_data[0:train_num]  # (n, seq_len, vocab_size)
    valid_news = seq_data[train_num:train_num + valid_num]
    test_news = seq_data[train_num + valid_num:train_num + valid_num + test_num]

    print('Training news instances:', len(train_news))
    print('Validation news instances:', len(valid_news))
    print('Testing news instances:', len(test_news))

    return train_news, valid_news, test_news


def create_embedding_matrix(data, dimension):
    doc_len = [len(doc.split()) for doc in data]
    quantile = 95
    max_length = int(np.percentile(doc_len, quantile))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([doc for doc in itertools.chain(data)])
    vocab_size = len(tokenizer.word_index) + 1
    encoded_docs = pad_sequences(tokenizer.texts_to_sequences(data),
                                 maxlen=max_length, padding='post')

    glove = pd.read_csv('data/glove/glove.6B.50d.txt', sep=" ", quoting=3, header=None, index_col=0)
    glove_embedding = {key: val.values for key, val in glove.T.items()}

    embedding_matrix = np.zeros((vocab_size, dimension))
    for word, i in tokenizer.word_index.items():
        if word in glove_embedding:
            embedding_matrix[i] = glove_embedding[word]

    return embedding_matrix, encoded_docs, max_length  # (vocab_size dimension)


def process_news(glove_dim, seq_len, future_step):
    combined = pd.read_csv('data/Combined_News_DJIA.csv')

    cleaned_news = clean_news(combined)
    embedding_matrix, encoded_docs, quan_len = create_embedding_matrix(cleaned_news, glove_dim)
    train_news, valid_news, test_news = load_news(encoded_docs, seq_len, future_step)

    # vocab_size = embedding_matrix.shape[0]
    embedding_matrix = torch.FloatTensor(embedding_matrix)

    torch.save(train_news, './data/train_news.pt')
    torch.save(valid_news, './data/valid_news.pt')
    torch.save(test_news, './data/test_news.pt')
    torch.save(embedding_matrix, './data/embedding.pt')

    train_news = torch.load("./data/train_news.pt")
    valid_news = torch.load("./data/valid_news.pt")
    test_news = torch.load("./data/test_news.pt")
    all_news = [train_news, valid_news, test_news]
    embedding_matrix = torch.load("./data/embedding.pt")
    vocab_size = embedding_matrix.shape[0]

    return all_news, embedding_matrix, vocab_size


