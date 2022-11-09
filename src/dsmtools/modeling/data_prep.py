from typing import Union
from collections import OrderedDict
import os

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler, LabelEncoder
from importlib_resources import files, as_file
from keras.utils import pad_sequences, to_categorical


class DSMDataConverter:
    """
    TODO
    """

    def __init__(self, dataset: Union[dict, OrderedDict], w2v_model: Union[str, 'os.PathLike[str]', Word2Vec] = None):
        """
        TODO

        :param dataset:
        :param w2v_model:
        """

        self._origin = OrderedDict(dataset)
        self._region_vec = OrderedDict()
        self._scaled_loc = OrderedDict()
        self._type_dummy = OrderedDict()

        # load word2vec model
        use_default = False
        if isinstance(w2v_model, Word2Vec):
            model = w2v_model
        elif isinstance(w2v_model, str) or isinstance(w2v_model, os.PathLike):
            model = Word2Vec.load(str(w2v_model))
        else:
            with as_file(files('dsmtools.models') / 'w2v_model_6dim_from_1200_seu.model') as path:
                model = Word2Vec.load(str(path))
                use_default = True
        self._region2vec = dict(zip(model.wv.index_to_key, StandardScaler().fit_transform(model.wv.vectors)))
        if use_default and 'unknow' in self._region2vec:  # sorry, we named it wrong while training
            self._region2vec['unknown'] = self._region2vec.pop('unknow')

        self._preprocess()

    def _preprocess(self):
        """
        TODO

        """
        for key, df in self._origin.items():
            self._region_vec[key] = pd.DataFrame(
                map(lambda w: self._region2vec[w] if w in self._region2vec else self._region2vec['unknown'],
                    df['region'].str.replace('fiber tracts', 'fiber_tracts')), index=df.index)
            self._scaled_loc[key] = pd.DataFrame(StandardScaler().fit_transform(df[['x', 'y', 'z']].values),
                                                 index=df.index)
            self._type_dummy[key] = pd.get_dummies(list(df['node_type']))

    def convert_for_ae(self, pad_seq_max_len=2000, features=('region',)):
        """
        TODO

        :param pad_seq_max_len:
        :param features:
        :return:
        """

        final = {
            'region': pad_sequences([*self._region_vec.values()], maxlen=pad_seq_max_len, dtype='float32'),
            'coord': pad_sequences([*self._scaled_loc.values()], maxlen=pad_seq_max_len, dtype='float32'),
            'topology': pad_sequences([*self._type_dummy.values()], maxlen=pad_seq_max_len, dtype='float32')
        }
        return np.concatenate([final[f] for f in features], axis=-1)

    def convert_for_han(self, max_len_word=20, max_len_sentence=300, features=('region',),
                        labels: dict[str, str] = None):
        """
        TODO

        :param max_len_word:
        :param max_len_sentence:
        :param features:
        :param labels:
        :return:
        """

        region_hierarch = []
        loc_hierarch = []
        type_hierarch = []
        for df, region_vec, scaled_loc, type_dummy in zip(self._origin.values(), self._region_vec.values(),
                                                          self._scaled_loc.values(), self._type_dummy.values()):
            higher_region = []
            higher_loc = []
            higher_type = []
            last_node = df.index[0]
            for terminal in df.index[df['node_type'] == 'T']:  # you can see terminals as punctuations
                higher_region.append(region_vec.loc[last_node:terminal].values.astype(float))
                higher_loc.append(scaled_loc.loc[last_node:terminal].values.astype(float))
                higher_type.append(type_dummy.loc[last_node:terminal].values.astype(float))
                last_node = terminal
            region_hierarch.append(pad_sequences(higher_region, maxlen=max_len_word, dtype='float32'))
            loc_hierarch.append(pad_sequences(higher_loc, maxlen=max_len_word, dtype='float32'))
            type_hierarch.append(pad_sequences(higher_type, maxlen=max_len_word, dtype='float32'))

        final = {
            'region': pad_sequences(region_hierarch, maxlen=max_len_sentence, dtype='float32'),
            'coord': pad_sequences(region_hierarch, maxlen=max_len_sentence, dtype='float32'),
            'topology': pad_sequences(type_hierarch, maxlen=max_len_sentence, dtype='float32')
        }

        x = np.concatenate([final[f] for f in features], axis=-1)

        if labels is None:
            return x

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform([labels[k] for k in self._origin.keys()])
        y = to_categorical(y, num_classes=len(label_encoder.classes_))
        return x, y, label_encoder
