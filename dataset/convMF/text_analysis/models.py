'''
Created on Dec 8, 2015

@author: donghyun
'''

import numpy as np
np.random.seed(1337)

from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Reshape, Flatten, Dropout
from keras.layers import Input, Embedding, Dense, concatenate
from keras.models import Model, Sequential
from keras.preprocessing import sequence


class CNN_module():
    '''
    classdocs
    '''
    batch_size = 128
    # More than this epoch cause easily over-fitting on our data sets
    nb_epoch = 5

    def __init__(self, output_dimesion, vocab_size, dropout_rate, emb_dim, max_len, nb_filters, init_W=None):

        self.max_len = max_len
        max_features = vocab_size
        vanila_dimension = 200
        projection_dimension = output_dimesion

        filter_lengths = [3, 4, 5]

        doc_input = Input(shape=(max_len,), dtype='int32', name='doc_input')

        '''Embedding Layer'''

        if init_W is None:
            sentence_embeddings = Embedding(output_dim=emb_dim, input_dim=max_features, input_length=max_len, name='sentence_embeddings')(doc_input)
        else:
            sentence_embeddings = Embedding(output_dim=emb_dim, input_dim=max_features, input_length=max_len, weights=[init_W / 20], name='sentence_embeddings')(doc_input)

        '''Reshape Layer'''
        reshape = Reshape(target_shape=(max_len, emb_dim, 1), name='reshape')(sentence_embeddings)  # chanels last

        '''Convolution Layer & Max Pooling Layer'''
        flatten_ = []
        for i in filter_lengths:
            model_internal = Sequential()
            model_internal.add(Conv2D(nb_filters, (i, emb_dim), activation="relu",
                                      name='conv2d_' + str(i), input_shape=(self.max_len, emb_dim, 1)))
            model_internal.add(MaxPooling2D(pool_size=(self.max_len - i + 1, 1), name='maxpool2d_' + str(i)))
            model_internal.add(Flatten())
            flatten = model_internal(reshape)
            flatten_.append(flatten)

        '''Fully Connect Layer & Dropout Layer'''
        fully_connect = Dense(vanila_dimension, activation='tanh',
                              name='fully_connect')(concatenate(flatten_, axis=-1))
        dropout = Dropout(dropout_rate, name='dropout')(fully_connect)
        '''Projection Layer & Output Layer'''
        pj = Dense(projection_dimension, activation='tanh', name='output')  # output layer
        projection = pj(dropout)

        # Output Layergit
        model = Model(inputs=doc_input, outputs=projection)
        model.compile(optimizer='rmsprop', loss='mse')
        self.model = model

    def load_model(self, model_path):
        self.model.load_weights(model_path)

    def save_model(self, model_path, isoverwrite=True):
        self.model.save_weights(model_path, isoverwrite)

    def qualitative_CNN(self, vocab_size, emb_dim, max_len, nb_filters):
        self.max_len = max_len
        max_features = vocab_size

        filter_lengths = [3, 4, 5]
        print("Build model...")
        self.qual_model = Graph()
        self.qual_conv_set = {}
        '''Embedding Layer'''
        self.qual_model.add_input(
            name='input', input_shape=(max_len,), dtype=int)

        self.qual_model.add_node(Embedding(max_features, emb_dim, input_length=max_len, weights=self.model.nodes['sentence_embeddings'].get_weights()),
                                 name='sentence_embeddings', input='input')

        '''Convolution Layer & Max Pooling Layer'''
        for i in filter_lengths:
            model_internal = Sequential()
            model_internal.add(
                Reshape(dims=(1, max_len, emb_dim), input_shape=(max_len, emb_dim)))
            self.qual_conv_set[i] = Convolution2D(nb_filters, i, emb_dim, activation="relu", weights=self.model.nodes[
                                                  'unit_' + str(i)].layers[1].get_weights())
            model_internal.add(self.qual_conv_set[i])
            model_internal.add(MaxPooling2D(pool_size=(max_len - i + 1, 1)))
            model_internal.add(Flatten())

            self.qual_model.add_node(
                model_internal, name='unit_' + str(i), input='sentence_embeddings')
            self.qual_model.add_output(
                name='output_' + str(i), input='unit_' + str(i))

        self.qual_model.compile(
            'rmsprop', {'output_3': 'mse', 'output_4': 'mse', 'output_5': 'mse'})

    def train(self, X_train, V, item_weight, seed):
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
        np.random.seed(seed)
        X_train = np.random.permutation(X_train)
        np.random.seed(seed)
        V = np.random.permutation(V)
        np.random.seed(seed)
        item_weight = np.random.permutation(item_weight)

        print("Train...CNN module")
        history = self.model.fit(x=X_train, y=V,
                                 verbose=0, batch_size=self.batch_size, epochs=self.nb_epoch,
                                 sample_weight=item_weight)

        # cnn_loss_his = history.history['loss']
        # cmp_cnn_loss = sorted(cnn_loss_his)[::-1]
        # if cnn_loss_his != cmp_cnn_loss:
        #     self.nb_epoch = 1
        return history

    def get_projection_layer(self, X_train):
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
        Y = self.model.predict(X_train, batch_size=len(X_train))
        return Y
