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
    Detailed explanation of this class can be found in the above [guide](#data-conversion).

    It can be used for conversion for both AE and HAN. Their conversion are similar so there is a common
    method. They are different in that HAN needs a hierarchical structure that has one more dimension than that for AE.

    This class uses an already trained word2vec model to convert brain regions into something numeric, but you can
    train one yourself and provide it in initialization. By default, it extracts three features from the original
    dataset: region, node_type and xyz.

    In most cases, 'region' is enough, so it's the default option. The 'node_type' is useful for finding words level
    features in data conversion for HAN. If you need more features, consider making a new class like this one.
    """

    def __init__(self, dataset: Union[dict, OrderedDict], w2v_model: Union[str, 'os.PathLike[str]', Word2Vec] = None):
        """Load the word2vec model and preprocess the dataset(common steps for AE and HAN data preparation).
        Input dataset should be OrderedDict, consistent to the output of the `preprocessing` module.
        If it's just a dict, it will be converted as ordered.

        :param dataset: An OrderedDict for
        :param w2v_model: Another word2vec model rather than the default, can be the model or path.
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
            with as_file(files('dsmtools.modeling') / 'w2v_model_6dim_from_1200_seu.model') as path:
                model = Word2Vec.load(str(path))
                use_default = True
        self._region2vec = dict(zip(model.wv.index_to_key, StandardScaler().fit_transform(model.wv.vectors)))
        if use_default and 'unknow' in self._region2vec:  # sorry, we named it wrong while training
            self._region2vec['unknown'] = self._region2vec.pop('unknow')

        self.preprocess()

    def preprocess(self):
        """Common steps for AE and HAN data preparations: turn the brain regions to vector, scale the coordinates,
        and make the topological type field as dummy. These three features are currently the ones you can choose
        from to make up the deep learning data.
        """

        for key, df in self._origin.items():
            self._region_vec[key] = pd.DataFrame(
                map(lambda w: self._region2vec[w] if w in self._region2vec else self._region2vec['unknown'],
                    df['region'].str.replace('fiber tracts', 'fiber_tracts')), index=df.index)
            self._scaled_loc[key] = pd.DataFrame(StandardScaler().fit_transform(df[['x', 'y', 'z']].values),
                                                 index=df.index)
            self._type_dummy[key] = pd.get_dummies(list(df['node_type']))

    def convert_for_ae(self, seq_len=2000, features=('region',)):
        """Data conversion for autoencoder.

        Specify the sequence length, so that all neuron sequences are padded or truncated to the same length.

        The default parameters are consistent with our trained model. You don't have to change if you are going
        to use.

        :param seq_len: The final sequence length.
        :param features: The features to use, among the three in the preprocessing.
        :return: An array of shape (#neuron, seq_len, #feature)
        """

        final = {
            'region': pad_sequences([*self._region_vec.values()], maxlen=seq_len, dtype='float32'),
            'coord': pad_sequences([*self._scaled_loc.values()], maxlen=seq_len, dtype='float32'),
            'topology': pad_sequences([*self._type_dummy.values()], maxlen=seq_len, dtype='float32')
        }
        return np.concatenate([final[f] for f in features], axis=-1)

    def convert_for_han(self, seq_len_word=20, seq_len_sentence=300, features=('region',),
                        labels: dict[str, str] = None):
        """Data conversion for autoencoder.

        The sequences will be further transformed so that they grouped in a two level structure.
        word and sentence, each of which will be given a sequence length, and they will be padded
        or truncated to the same length.

        The two level structure will use the topological information--terminals to use to break up words, i.e.
        a sequence between two terminals is seen as a word and its length will be unified. A neuron sequence will
        be turned into multiple words to make up a sentence, and the number of words will be seen as the length
        of the sentence and will also be unified.

        If labels are provided as a dict mapping the key from the input dataset to their types, this conversion will
        output the encoded labels as vectors and the label encoder, suitable for training.

        The default parameters are consistent with our trained model. You don't have to change if you are going
        to use.

        :param seq_len_word: The sequence length of word level
        :param seq_len_sentence: The sequence length of sentence level.
        :param features: The features to use, among the three in the preprocessing.
        :param labels: A dict mapping from the key of each sequence to their cell type.
        :return: An array of shape (#neuron, seq_len_sentence, seq_len_word, #feature)
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
            region_hierarch.append(pad_sequences(higher_region, maxlen=seq_len_word, dtype='float32'))
            loc_hierarch.append(pad_sequences(higher_loc, maxlen=seq_len_word, dtype='float32'))
            type_hierarch.append(pad_sequences(higher_type, maxlen=seq_len_word, dtype='float32'))

        final = {
            'region': pad_sequences(region_hierarch, maxlen=seq_len_sentence, dtype='float32'),
            'coord': pad_sequences(region_hierarch, maxlen=seq_len_sentence, dtype='float32'),
            'topology': pad_sequences(type_hierarch, maxlen=seq_len_sentence, dtype='float32')
        }

        x = np.concatenate([final[f] for f in features], axis=-1)

        if labels is None:
            return x

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform([labels[k] for k in self._origin.keys()])
        y = to_categorical(y, num_classes=len(label_encoder.classes_))
        return x, y, label_encoder
