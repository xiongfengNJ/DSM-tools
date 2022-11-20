from os import PathLike
from typing import Union

import numpy as np
from importlib_resources import files, as_file
from keras import Model, Input, regularizers, backend
from keras.layers import Dense, GRU, Masking, TimeDistributed, Dropout, ReLU, Layer, Multiply
from keras.metrics import CategoricalAccuracy
from keras.optimizers import RMSprop
from sklearn.preprocessing import LabelEncoder

from .model_abstract import ModelAbstract


class AttentionLayer(Layer):
    def __init__(self, embedding_dim, name):
        self.w_dense = Dense(embedding_dim, name=name, activation='softmax',
                             kernel_regularizer=regularizers.l2(0.001))
        self.supports_masking = True
        super(AttentionLayer, self).__init__()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'w_dense': self.w_dense,
        })
        return config

    def compute_mask(self, i, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, inputs, *args, **kwargs):  # I shape [batch_size, seq_length, dim]
        qk = self.w_dense(inputs)  # activation = "softmax"
        mv = Multiply()([qk, inputs])

        output = backend.sum(mv, axis=1)
        return output, qk

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


class DSMHierarchicalAttentionNetwork(ModelAbstract):
    """
    The class to compile a hierarchical attention network,
    train/load model parameters and use it to encode your neuron sequences, based on keras API.
    Detailed explanations can be found in [the above guide](#deep-sequential-models).

    To use this class, you can start by invoking `DSMHierarchicalAttentionNetwork.load_1282_seu` to get our trained
    model and encode your neuron sequences, provided that they are converted to the shape that this model accepts
    (same as the default parameters of the `DSMDataConverter.convert_for_han`).
    And then use `DSMHierarchicalAttentionNetwork.predict` to get the classification. The result is a one-hot
    matrix. To get their labels, your can find the largest position of each neuron, and invoke
    `DSMHierarchicalAttentionNetwork.label_encoder_1282_seu` to get the label for that position. There are 12 types
    in our trained classification model.

    To use this class from the beginning, after initialization, use `DSMHierarchicalAttentionNetwork.compile` to set up
    a new model, whose dimension should be consistent with your input.
    If you'd like to train your own model, use `DSMAutoencoder.fit`, and `DSMAutoencoder.load` to load a saved one.
    Use `DSMAutoencoder.evaluate` for further evaluation. The label encoder is obtained by
    `DSMDataConverter.convert_for_han`, and it's yours to maintain.

    The training history and the model are available as `DSMHierarchicalAttentionNetwork.history` and
    `DSMHierarchicalAttentionNetwork.model`. If you
    are familiar with keras, you know what to do with them, but we offer some basic functions to plot learning stats and
    save the model parameters. Their paths can be specified in each method.
    """

    def compile(self, seq_len_sentence=300, seq_len_word=20, feature_dim=6, class_num=12,
                last_activation='softmax', index_test=False,
                optimizer=RMSprop(learning_rate=0.001, clipvalue=0.5, decay=0.002)):
        """Compile the HAN model with specified parameters, before you can train, load or predict.

        The dimensions of word and sentence length and feature should be consistent with those in the data conversion,
        used by `DSMDataConverter.convert_for_han`. The class number is dimension of the classification result,
        a one-hot matrix.
        They should also be the same with a trained model, if used, and the default parameters are used by our
        prepared model.

        You can also specify the way how the last activation is calculated, and how word and sentence layers
        are built.

        :param seq_len_sentence: The length of the sentence level, the second dimension of the input array.
        :param seq_len_word: The length of the word level, the third dimension of the input array.
        :param feature_dim: The number of features, the last dimension of the input array.
        :param class_num: The number of classes, the second dimension of the output.
        :param last_activation: The activation function for the last layer.
        :param index_test: Whether to use time distributed rather than GRU to build word and sentence layers.
        :param optimizer: A keras optimizer with learning parameters.
        """

        input_word = Input(shape=(seq_len_word, feature_dim,))  # [batchsize, word_length, dim(8)]
        x_word = Masking(mask_value=0.0)(input_word)
        em1 = 128

        if not index_test:
            x_word = GRU(em1, return_sequences=False, kernel_regularizer=regularizers.l2(0.001), dropout=0.5)(
                x_word)  # LSTM or GRU
        elif index_test:
            x_word = TimeDistributed(Dense(em1, kernel_regularizer=regularizers.l2(0.001)), name='segment_encoder')(
                x_word)

        model_word = Model(input_word, x_word)

        # Sentence part
        input = Input(
            shape=(seq_len_sentence, seq_len_word, feature_dim,))  # [batchsize, sentence_length, word_length, dim(8)]
        x_sentence = TimeDistributed(model_word, name='segment_encoder')(input)
        em2 = 128

        if not index_test:
            x_sentence = GRU(em2, return_sequences=False, kernel_regularizer=regularizers.l2(0.001), dropout=0.5,
                             name='sentence_out')(x_sentence)  # LSTM or GRU
        elif index_test:
            x_sentence = TimeDistributed(Dense(em2, kernel_regularizer=regularizers.l2(0.001)))(x_sentence)

        x_sentence = Dense(64, kernel_regularizer=regularizers.l2(0.01), name='dense_out')(x_sentence)
        x_sentence = Dropout(rate=0.5)(x_sentence)
        x_sentence = ReLU()(x_sentence)

        output = Dense(class_num, activation=last_activation, kernel_regularizer=regularizers.l2(0.01))(x_sentence)
        self._model = Model(inputs=input, outputs=output)
        self._model.summary()
        self._model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                            metrics=[CategoricalAccuracy(name='accuracy')])

    def load(self, path: Union[str, 'PathLike[str]']):
        super(DSMHierarchicalAttentionNetwork, self).load(path)

    def fit(self, x_train, y_train, x_test, y_test, batch_size=64, epochs=400, model_save_path=None, shuffle=False):
        """Model training.

        If a model save path is specified, the model will be saved when improved by loss evaluation. This save path
        can also be formatted as a keras model path. Check keras documentation to write the format string.

        :param x_train: An input array for training, with shape of (#neuron, #seq_len, #feature).
        :param y_train: An output array for training, with shape of (#neuron, #classes).
        :param x_test: An input array for test, with shape of (#neuron, #seq_len, #feature).
        :param y_test: An output array for testing, with shape of (#neuron, #classes).
        :param batch_size: The training batch size.
        :param epochs: The number of training epochs.
        :param model_save_path: The path to save the model by checkpoint.
        :param shuffle: Whether to shuffle the input data.
        :return:
        """

        super(DSMHierarchicalAttentionNetwork, self)._fit(x_train, y_train, x_test, y_test,
                                                          batch_size, epochs, model_save_path, 'val_accuracy', shuffle)

    def plot_learning_curve(self, path: Union[str, 'PathLike[str]'] = None, fig_size=(20, 20), dpi=200):
        """Plot the learning curve using the history data after training.

        It will plot a new figure and return the matplotlib handles.
        If a path is specified, the figure will be saved.

        :param path: The save path of the figure.
        :param fig_size: The size of the figure.
        :param dpi: resolution setting.
        :return: The plotted figure's handles, figure and axes.
        """

        return super(DSMHierarchicalAttentionNetwork, self).\
            _plot_learning_curve(path, fig_size, dpi, ['accuracy', 'val_accuracy', 'loss', 'val_loss'])

    def predict(self, x: np.array):
        return super(DSMHierarchicalAttentionNetwork, self).predict(x)

    def evaluate(self, x, y):
        """Test the model with a given pair of input data and output class one-hot matrix.
        Your input and output array should share their dimensions other than the number of neurons with the model.

        :param x: An array for evaluation, with shape of (#neuron, #sentence_len, #word_len, #feature).
        :param y: A one-hot matrix of class results, with shape of (#neuron, #class).
        :return: The loss and accuracy of evaluation.
        """

        return super(DSMHierarchicalAttentionNetwork, self)._evaluate(x, y)

    @staticmethod
    def load_1282_seu():
        """A static method to build and load an already trained model on 1282 neurons from Southeast University.

        :return: A `DSMHierarchicalAttentionNetwork` instance with the trained model.
        """
        model = DSMHierarchicalAttentionNetwork()
        model.compile()
        with as_file(files('dsmtools.modeling') / 'HAN_1282_SEU.h5') as path:
            model.load(path)
        return model

    @staticmethod
    def label_encoder_1282_seu():
        """A static method to get the label encoder for the trained model on 1282 neurons from Southeast University.
        You can use it to transform the number places inferred from the one-hot matrix back to their labels.

        :return: A fitted `LabelEncoder` instance.
        """
        return LabelEncoder().fit(['CP_GPe', 'CP_SNr', 'ET_MO', 'ET_SS', 'IT_MO', 'IT_SS', 'IT_VIS',
                                   'LGd', 'MG', 'RT', 'VPL', 'VPM'])
