import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import numpy as np
import tensorflow
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras.utils import plot_model
from numpy.random import seed
import spacy
from spacy import displacy
from dataExchanging import ner_dataset
data = ner_dataset
data = data.drop("Unnamed: 0",axis=1)

from itertools import chain
def get_dict_map(data,token_or_tag):
    tok2idx={}
    idx2tok={}

    if token_or_tag=='token':
        vocab = list(set(data['word'].to_list()))
    else:
        vocab = list(set(data['tag'].to_list()))

    idx2tok = {idx:tok for idx, tok in enumerate(vocab)}
    tok2idx = {tok:idx for idx, tok in enumerate(vocab)}
    return tok2idx,idx2tok


token2idx,idx2token = get_dict_map(data,'token')
tag2idx,idx2tag = get_dict_map(data,'tag')

data['Word_idx'] = data['word'].map(token2idx)
data['Tag_idx'] = data['tag'].map(tag2idx)

# Fill na
data_fillna = data.fillna(method='ffill',axis=0)

# Groupby and collect columns
data_group = data_fillna.groupby(
    ['sentence_idx'],as_index=False
)[['word','pos','tag','Word_idx','Tag_idx']].agg(lambda x: list(x))
print(data_group.head())


def get_pad_train_test_val(data_group, data):
    # get max token and tag length
    n_token = len(list(set(data['word'].to_list())))
    n_tag = len(list(set(data['tag'].to_list())))

    # Pad tokens (X var)
    tokens = data_group['Word_idx'].tolist()
    maxlen = max([len(s) for s in tokens])
    pad_tokens = pad_sequences(tokens, maxlen=maxlen, dtype='int32', padding='post', value=n_token - 1)

    # Pad Tags (y var) and convert it into one hot encoding
    tags = data_group['Tag_idx'].tolist()
    pad_tags = pad_sequences(tags, maxlen=maxlen, dtype='int32', padding='post', value=tag2idx["O"])
    n_tags = len(tag2idx)
    pad_tags = [np_utils.to_categorical(i, num_classes=n_tags) for i in pad_tags]

    # Split train, test and validation set
    tokens_, test_tokens, tags_, test_tags = train_test_split(pad_tokens, pad_tags, test_size=0.1, train_size=0.9,
                                                              random_state=2020)
    train_tokens, val_tokens, train_tags, val_tags = train_test_split(tokens_, tags_, test_size=0.25, train_size=0.75,
                                                                      random_state=2020)

    print(
        'train_tokens length:', len(train_tokens),
        '\ntrain_tokens length:', len(train_tokens),
        '\ntest_tokens length:', len(test_tokens),
        '\ntest_tags:', len(test_tags),
        '\nval_tokens:', len(val_tokens),
        '\nval_tags:', len(val_tags),
    )

    return train_tokens, val_tokens, test_tokens, train_tags, val_tags, test_tags


train_tokens, val_tokens, test_tokens, train_tags, val_tags, test_tags = get_pad_train_test_val(data_group, data)

seed(1)
tensorflow.random.set_seed(2)
input_dim = len(list(set(data['word'].to_list())))+1
output_dim = 64
input_length = max([len(s) for s in data_group['Word_idx'].to_list()])
n_tags = len(tag2idx)

# print('input_dim:',input_dim,'\noutput_dim:',output_dim,'\ninput_length:',input_length,'\nn_tags:',n_tags)


def get_bilstm_lstm_model():
    model = Sequential()

    # Add Embedding layer
    model.add(Embedding(input_dim=input_dim,output_dim=output_dim,input_length=input_length))

    # Add bidirectional LSTM
    model.add(LSTM(units=output_dim,return_sequences=True,dropout=0.5,recurrent_dropout=0.5))

    # Add LSTM
    model.add(LSTM(units=output_dim,return_sequences=True,dropout=0.5,recurrent_dropout=0.5))

    # Add timeDistributed Layer
    model.add(TimeDistributed(Dense(n_tags,activation='relu')))

    #Optimiser
    #adam = k.optimizers.Adam(lr=0.0005,beta_1=0.9,beta_2=0.999)

    # Compile model
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()

    return model


def train_model(X,y,model):
    loss = list()
    for i in range(25):
        # fit model for one epoch on this sequence
        hist = model.fit(X,y,batch_size=1000,verbose=1,epochs=1,validation_split=0.2)
        loss.append(hist.history['loss'][0])

    return loss
results = pd.DataFrame()
model_bilstm_lstm = get_bilstm_lstm_model()
# plot_model(model_bilstm_lstm)
results['with_add_lstm'] = train_model(train_tokens,
np.array(train_tags),model_bilstm_lstm)
print(results)



nlp = spacy.load('en_core_web_sm')

text = nlp(

    'Jim bought 300 shares of Acme Corp. in 2006. '
    'And producing an annotated block of text that highlights '
    'the names of entities: [Jim]Person bought 300 shares of'
    ' [Acme Corp.]Organization in [2006]Time. In this example,'
    ' a person name consisting of one token, '
    'a two-token company name and a temporal expression have been '
    'detected and classified.State-of-the-art NER systems for English'
    ' produce near-human performance. For example, the best system '
    'entering MUC-7 scored 93.39% of F-measure while human '
    'annotators scored 97.60% and 96.95%.[1][2]'
)
displacy.render(text,style = 'ent',jupyter=True)