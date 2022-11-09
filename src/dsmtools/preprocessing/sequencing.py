from typing import Optional, Iterable, Union
import traceback
from multiprocessing import Pool
from functools import partial
from warnings import warn
import pickle
from copy import deepcopy
from os import PathLike
from collections import OrderedDict

import pandas as pd

from .swc_qc import SWCQualityControl
from .neuron_tree import NeuronTree
from dsmtools import utils


class NeuronSequenceDataset:
    """
    Generate and prepare dataset from multiple SWC files to be further converted into dataset that can be
    used by our HAN and AE.

    By default, it does a necessary qc before a lesser to greater subtree preorder traversal and return features indeed
    such way.

    If you need specifying some or all of the sequence conversion to your like, you can inherit this class
    and define a new set of function for preprocessing processing, like the way this class's functions do. It's quite
    simple.

    The object of this class, after data generation, is passed to and used by dataset generation processes
    of the models' classes.

    """

    def __init__(self,
                 swc_paths: Iterable[Union[str, 'PathLike[str]']],
                 show_failure=True,
                 debug=False,
                 chunk_size=1,
                 jobs=1,
                 ):
        """
        Set up a dataset class to generate sequence preprocessing of your neuron structure in parallel, with which
        you can start doing classification and autoencoder.

        :param swc_paths: an iterable of file paths to the SWC files to load.
        the classifier's result, for its training. Otherwise, it can be left as None.
        :param show_failure: whether to print failed swc during processing, default as True.
        :param debug: whether to print traceback messages when processes fail, default as False.
        :param chunk_size: the chunk size for parallel processing, default as 10.
        :param jobs: number of jobs for parallel computation, default as 1.
        """

        self._swc_paths: tuple[str] = tuple(str(i) for i in swc_paths)
        self._swc_raw: Optional[OrderedDict[str, pd.DataFrame]] = None
        self._swc_qc: Optional[OrderedDict[str, pd.DataFrame]] = None
        self._sequences: Optional[OrderedDict[str, tuple[pd.DataFrame, str]]] = None
        self.chunk_size = chunk_size
        self.jobs = jobs
        self.show_failure = show_failure
        self.debug = debug

    def pickle_sequences(self, path: Union[str, 'PathLike[str]']):
        """Save the generated sequences as an ordered dict to a pickle file.

        :param path: the path to save the pickle
        """
        with open(path, 'wb') as f:
            pickle.dump(self._sequences, f)

    def pickle_qc(self, path: Union[str, 'PathLike[str]']):
        """Save the SWC dataframes after qc as an ordered dict to a pickle file.

        :param path: the path to save the pickle
        """
        with open(path, 'wb') as f:
            pickle.dump(self._swc_qc, f)

    def result(self, deep=True):
        """Return the sequence dataframe ordered dict. For safety, by default it is a deep copy, just in case you
        modify some items in the original copy unknowingly. If you know what you are doing and intend to save memory,
        your can turn it off.

        :param deep: whether return a deep copy or just reference.
        :return: an ordered dict mapping SWC paths to their sequence dataframes.
        """
        return deepcopy(self._sequences) if deep else self._sequences

    @property
    def swc_after_qc(self):
        return self._swc_qc

    def read_parallel(self):
        """Read all the given SWC files and store them as a list of dataframes within this object.
        The multiprocessing parameters are given by class initialization.
        """

        with Pool(self.jobs) as p:
            res = p.map(self._read_proc, self._swc_paths, chunksize=self.chunk_size)
            res = filter(lambda i: i[1] is not None, zip(self._swc_paths, res))
            self._swc_raw = OrderedDict(res)
        print(f'Successfully loaded {len(self._swc_raw)} SWC files.')

    def qc_parallel(self, qc_len_thr=10, min_node_count=10):
        """Commit quality control on all the readable SWC dataframes and store them as another list.
        The multiprocessing parameters are given by class initialization.

        :param qc_len_thr: the pruning length threshold for quality control.
        :param min_node_count: the minimum allowed count of node of the final SWC.
        """

        if self._swc_raw is None:
            warn("No SWC, running read_parallel..")
            self.read_parallel()
        qc_proc = partial(self._qc_proc, qc_len_thr=qc_len_thr)
        with Pool(self.jobs) as p:
            res = p.map(qc_proc, self._swc_raw.keys(), chunksize=self.chunk_size)
            res = filter(lambda i: i[1] is not None and len(i[1]) >= min_node_count, zip(self._swc_raw.keys(), res))
            self._swc_qc = OrderedDict(res)
        print(f'Successfully quality-controlled {len(self._swc_qc)} SWC files.')

    def make_sequences_parallel(self, ordering='pre', lesser_first=True):
        """Make the sequence dataset that can be further converted to I for other models.

        Before running this function, read & qc are supposed to have been committed.
        It gives you a dataframe containing only features sufficient for our models in the order of
        binary tree preorder traversals of each neuron tree. You can inherit and derive this function as well as its
        multiprocessing unit. Many of the traversal related functions are in BinarySubtree and NeuronSequencer
        classes.

        It makes the sequences for every SWC passing QC, in the order of binary tree preorder traversals, and uses the
        traversal to reindex the SWC and output the features.
        It will also add CCFv3 regions and topological annotations as features corresponding to each node.

        CCF regions features are brain region names given CCFv3 atlas, which there is a word2vec model to give
        a quantitative representation for each. The column name will be `node_type`.

        Topological annotations are made up of four types: GBTR, respectively representing
        general, branch, terminal, and root. The column name will be `region`.

        Some columns of the dataframes are shallow copies of those from SWC after QC, be wary when deriving.

        :param ordering: the traversal type, either 'pre', 'in', or 'post'.
        :param lesser_first: whether traversing from lesser to greater, affect both within and among subtrees.
        """

        if self._swc_qc is None:
            warn("No QC SWC, running qc_parallel with default parameters..")
            self.qc_parallel()
        assert ordering in ['pre', 'in', 'post']
        make_seq = partial(self._make_seq_proc, ordering=ordering, lesser_first=lesser_first)
        with Pool(self.jobs) as p:
            res = p.map(make_seq, self._swc_qc)
            res = filter(lambda i: i[1] is not None, zip(self._swc_qc.keys(), res))
            self._sequences = OrderedDict(res)
        print(f'Successfully made {len(self._sequences)} sequences.')

    def _read_proc(self, path: str):
        try:
            return utils.swc.read(path)
        except:
            if self.show_failure:
                if self.debug:
                    traceback.print_exc()
                print(f'Some error occurred when reading SWC: {path}, better check your file.')
            return None

    def _qc_proc(self, key: str, qc_len_thr: float):
        try:
            swc = self._swc_raw[key].copy(deep=True)
            SWCQualityControl(swc).retain_only_1st_root().adjust_multifurcation().prune_by_len(qc_len_thr)
            return swc
        except:
            if self.show_failure:
                if self.debug:
                    traceback.print_exc()
                print(f'Some error occurred during quality control with SWC: {key}.')
            return None

    def _make_seq_proc(self, key: str, ordering: str, lesser_first: bool):
        try:
            # shallow copy the qc dataframe for new columns
            swc = self._swc_qc[key].copy(deep=False)
            nt = NeuronTree(swc, deep_copy=False)
            axon_index = swc.index[swc['type'] == utils.swc.Type.AXON]
            dendrite_index = swc.index[swc['type'].isin([utils.swc.Type.APICAL, utils.swc.Type.BASAL])]
            nt.find_binary_trees_in(axon_index)
            nt.find_binary_trees_in(dendrite_index)
            traversal = nt.collective_traversal(ordering=ordering, lesser_first=lesser_first)
            # node type, topological annotation
            root = swc.index[swc['parent'] == -1]
            children_counts = swc['parent'].value_counts()
            branch = children_counts.index[children_counts.values > 1]
            terminal = swc.index.difference(swc.loc[filter(lambda x: x != -1, swc['parent'])].index)
            swc['node_type'] = 'G'
            swc.loc[root, "node_type"] = 'R'
            swc.loc[branch, 'node_type'] = 'B'
            swc.loc[terminal, 'node_type'] = 'T'
            # ccf region annotation
            swc['region'] = utils.ccf_atlas.get_region(swc)
            return swc.loc[traversal, ['x', 'y', 'z', 'type', 'node_type', 'region']]
        except:
            if self.show_failure:
                if self.debug:
                    traceback.print_exc()
                print(f'Some error occurred when converting SWC to sequence by default: {key}.')
            return None
