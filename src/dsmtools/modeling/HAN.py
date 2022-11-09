from os import PathLike
from typing import Union

from importlib_resources import files, as_file
from keras import Model, Input, regularizers, Sequential, backend
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
    TODO
    """

    def __init__(self, result_dir: Union[str, 'PathLike[str]'] = None):
        super(DSMHierarchicalAttentionNetwork, self).__init__(result_dir)

    def compile(self, input_dim=6, max_len_sentence=300, max_len_word=20, class_num=12,
                last_activation='softmax', index_test=False,
                optimizer=RMSprop(learning_rate=0.001, clipvalue=0.5, decay=0.002)):
        """
        TODO

        :param max_len_sentence:
        :param max_len_word:
        :param class_num:
        :param last_activation:
        :param index_test:
        :param input_dim:
        :param optimizer:
        :return:
        """

        input_word = Input(shape=(max_len_word, input_dim,))  # [batchsize, word_length, dim(8)]
        x_word = Masking(mask_value=0.0)(input_word)
        em1 = 128

        if not index_test:
            x_word = GRU(em1, return_sequences=False, kernel_regularizer=regularizers.l2(0.001), dropout=0.5)(
                x_word)  # LSTM or GRU
        elif index_test:
            x_word = TimeDistributed(Dense(em1, kernel_regularizer=regularizers.l2(0.001)), name='segment_encoder')(
                x_word)

        # x_word, word_attention_weight = simple_att(em1, 'word_att')(x_word)
        model_word = Model(input_word, x_word)

        # Sentence part
        input = Input(
            shape=(max_len_sentence, max_len_word, input_dim,))  # [batchsize, sentence_length, word_length, dim(8)]
        x_sentence = TimeDistributed(model_word, name='segment_encoder')(input)
        em2 = 128
        # x_sentence = Bidirectional(
        #     GRU(em2, return_sequences=True, kernel_regularizer=regularizers.l2(0.0015), dropout=0.5))(
        #     x_sentence)  # LSTM or GRU

        if not index_test:
            x_sentence = GRU(em2, return_sequences=False, kernel_regularizer=regularizers.l2(0.001), dropout=0.5,
                             name='sentence_out')(x_sentence)  # LSTM or GRU
        elif index_test:
            x_sentence = TimeDistributed(Dense(em2, kernel_regularizer=regularizers.l2(0.001)))(x_sentence)

        # x_sentence, sentence_attention_weight = simple_att(em2, 'sentence_att')(x_sentence)
        # x_sentence = x_sentence[:, -1, :]
        x_sentence = Dense(64, kernel_regularizer=regularizers.l2(0.01), name='dense_out')(x_sentence)
        x_sentence = Dropout(rate=0.5)(x_sentence)
        x_sentence = ReLU()(x_sentence)

        output = Dense(class_num, activation=last_activation, kernel_regularizer=regularizers.l2(0.01))(x_sentence)
        self._model = Model(inputs=input, outputs=output)
        self._model.summary()
        self._model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                            metrics=[CategoricalAccuracy(name='accuracy')])

    def fit(self, x_train, y_train, x_test, y_test, batch_size=64, epochs=400, model_save_path=None, shuffle=False):
        """
        TODO

        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :param batch_size:
        :param epochs:
        :param model_save_path:
        :param shuffle:
        :return:
        """

        super(DSMHierarchicalAttentionNetwork, self)._fit(x_train, y_train, x_test, y_test,
                                                          batch_size, epochs, model_save_path, 'val_accuracy', shuffle)

    def plot_learning_curve(self, path: Union[str, 'PathLike[str]'] = None, fig_size=(20, 20), dpi=200):
        """
        TODO

        :param path:
        :param fig_size:
        :param dpi:
        :return:
        """

        return super(DSMHierarchicalAttentionNetwork, self).\
            _plot_learning_curve(path, fig_size, dpi, ['accuracy', 'val_accuracy', 'loss', 'val_loss'])

    def evaluate(self, x, y):
        """
        TODO

        :param x:
        :param y:
        :return:
        """

        return super(DSMHierarchicalAttentionNetwork, self)._evaluate(x, y)

    @staticmethod
    def load_1282_seu():
        model = DSMHierarchicalAttentionNetwork()
        model.compile()
        with as_file(files('dsmtools.models') / 'HAN_1282_SEU.h5') as path:
            model.load(path)
        return model

    @staticmethod
    def label_encoder_1282_seu():
        return LabelEncoder().fit(['CP_GPe', 'CP_SNr', 'ET_MO', 'ET_SS', 'IT_MO', 'IT_SS', 'IT_VIS',
                                   'LGd', 'MG', 'RT', 'VPL', 'VPM'])