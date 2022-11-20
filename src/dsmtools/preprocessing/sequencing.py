from typing import Optional, Iterable, Union
import traceback
from multiprocessing import Pool
from functools import partial
from warnings import warn
import pickle
from os import PathLike
from collections import OrderedDict

import pandas as pd

from .swc_qc import SWCQualityControl
from .neuron_tree import NeuronTree
from dsmtools import utils


class NeuronSequenceDataset:
    """
    An interface for generating dataset from multiple SWC files to be further converted into a dataset that can be
    used by our deep learning models. See [the above guide](#neuronsequencedataset) for a more detailed description
    of its principles.

    The default usage should be: loading SWC files with path given during initialization, then operating a quality
    control defined by `SWCQualityControl`, and then
    finding binary trees in the neuron tree and committing traversals to get the sequences. They defined as member
    functions are should be invoked in this order.

    All the above procedures are coded as a host function paired with its unit function that can be called in parallel.
    for different SWC dataframes. You'll find 3 pairs of them, including:

    |Host|Unit|
    |----|----|
    |read_parallel|read_proc|
    |qc_parallel|qc_proc|
    |make_sequence_parallel|make_sequence_proc|

    See their own function documentation for more info.

    If you need specifying some or all of the sequence conversion to your like, you can inherit this class
    and derive its functions. Deriving only the unit function will not change how the host invoke them and maintain the
    multiprocessing feature.

    The result of this class is multiple dataframes in an OrderedDict indexed by paths, which can be retrieved
    as a property or saved as a pickle. This result can be further fed
    to the deep learning model class in the `dsmtools.modeling` module. Here, you also have access to
    the SWC dataframes after QC as a member property (also an OrderedDict).

    **Note**: Do not modify the retrieved property data, which can cause bugs. Deep copy them if you need.
    """

    def __init__(self,
                 swc_paths: Iterable[Union[str, 'PathLike[str]']],
                 show_failure=True,
                 debug=False,
                 chunk_size=1,
                 jobs=1):
        """Set up the class, specifying common processing options.

        :param swc_paths: An iterable of file paths to the SWC files to load. Note they will be forced as strings.
        :param show_failure: Whether to print failed swc during processing.
        :param debug: Whether to print traceback messages when processes fail.
        :param chunk_size: The chunk size for parallel processing.
        :param jobs: number of processes for parallel computation.
        """

        self._swc_paths: tuple[str] = tuple(str(i) for i in swc_paths)
        self._swc_raw: Optional[OrderedDict[str, pd.DataFrame]] = None
        self._swc_qc: Optional[OrderedDict[str, pd.DataFrame]] = None
        self._sequences: Optional[OrderedDict[str, tuple[pd.DataFrame, str]]] = None
        self.chunk_size = chunk_size
        self.jobs = jobs
        self.show_failure = show_failure
        self.debug = debug

    def read_parallel(self):
        """
        Read all the given SWC files and store them as an OrderedDict of dataframes within this object.
        The multiprocessing parameters are given by the class initialization.
        """

        with Pool(self.jobs) as p:
            res = p.map(self.read_proc, self._swc_paths, chunksize=self.chunk_size)
            res = filter(lambda i: i[1] is not None, zip(self._swc_paths, res))
            self._swc_raw = OrderedDict(res)
        print(f'Successfully loaded {len(self._swc_raw)} SWC files.')

    def read_proc(self, path: str) -> Optional[pd.DataFrame]:
        """
        Loading of a single SWC file, with a try except mechanism to prevent interruption.
        Failed processes will print a message and return None.

        :param path: Path to the SWC.
        :return:  An SWC dataframe, or None if failed with some error.
        """

        try:
            return utils.swc.read_swc(path)
        except:
            if self.show_failure:
                if self.debug:
                    traceback.print_exc()
                print(f'Some error occurred when reading SWC: {path}, better check your file.')
            return None

    def qc_parallel(self, qc_len_thr=10, min_node_count=10):
        """
        Commit quality control on all the loaded SWC dataframes. This operation is not inplace
        and set up a new OrderedDict, which can be referenced by the property `NeuronSequenceDataset.swc_after_qc`.
        Here, you can specify QC parameters passed to the unit processing function.

        To prevent empty dataframes, this function applies a final filter on the QC results, the parameter of
        which is also customizable.

        Before running this function, it will check if SWC files are loaded. If not, it will invoke
        `NeuronSequenceDataset.read_parallel`.

        The multiprocessing parameters are given by the class initialization.

        :param qc_len_thr: The pruning length threshold for quality control.
        :param min_node_count: The minimum allowed count of nodes for the final SWC.
        """

        if self._swc_raw is None:
            warn("No SWC, running read_parallel..")
            self.read_parallel()
        qc_proc = partial(self.qc_proc, qc_len_thr=qc_len_thr)
        with Pool(self.jobs) as p:
            res = p.map(qc_proc, self._swc_raw.keys(), chunksize=self.chunk_size)
            res = filter(lambda i: i[1] is not None and len(i[1]) >= min_node_count, zip(self._swc_raw.keys(), res))
            self._swc_qc = OrderedDict(res)
        print(f'Successfully quality-controlled {len(self._swc_qc)} SWC files.')

    def qc_proc(self, key: str, qc_len_thr: float) -> Optional[pd.DataFrame]:
        """
        Quality control of a single SWC file, with a try except mechanism to prevent interruption.
        Failed processes will print a message and return None.

        It serially commits retaining tree from the root detected, bifurcation checking, and removing short segments.
        It's advised that this order is maintained.

        :param key: The key of the loaded SWC, should be a string path.
        :param qc_len_thr: The pruning length threshold for quality control.
        :return: An SWC dataframe, or None if failed with some error.
        """

        try:
            swc = self._swc_raw[key].copy(deep=True)
            SWCQualityControl(swc).retain_only_1st_root().degrade_to_bifurcation().prune_by_len(qc_len_thr)
            return swc
        except:
            if self.show_failure:
                if self.debug:
                    traceback.print_exc()
                print(f'Some error occurred during quality control with SWC: {key}.')
            return None

    def make_sequence_parallel(self, ordering='pre', lesser_first=True):
        """Convert all neuron trees passing previous procedures to sequences. This operation is not inplace
        and set up a new OrderedDict, which can be referenced by the property `NeuronSequenceDataset.result`.
        Here, you can specify some traversal parameters passed to the unit processing function.

        Before running this function, it will check if quality control is done. If not, it will invoke
        `NeuronSequenceDataset.qc_parallel`.

        The multiprocessing parameters are given by the class initialization.

        :param ordering: The traversal type, either 'pre', 'in', or 'post'.
        :param lesser_first: Whether traversing from lesser to greater, affect both within and among subtrees.
        """

        if self._swc_qc is None:
            warn("No QC SWC, running qc_parallel with default parameters..")
            self.qc_parallel()
        assert ordering in ['pre', 'in', 'post']
        make_seq = partial(self.make_sequence_proc, ordering=ordering, lesser_first=lesser_first)
        with Pool(self.jobs) as p:
            res = p.map(make_seq, self._swc_qc)
            res = filter(lambda i: i[1] is not None, zip(self._swc_qc.keys(), res))
            self._sequences = OrderedDict(res)
        print(f'Successfully made {len(self._sequences)} sequences.')

    def make_sequence_proc(self, key: str, ordering: str, lesser_first: bool):
        """Convert an SWC dataframe to a feature dataframe indexed by a traversal sequence.

        It uses `NeuronTree` to build binary trees separately for axon and dendrite from an SWC, and traverses the trees
        with the parameters specified, which by default is preorder traversal and lesser subtrees prioritized.

        The final dataframe is similar to an SWC, but different as it's with some additional features and indexed by
        the traversal sequence.

        The added features are CCFv3 region abbreviations retrieved by `dsmtools.utils.ccf_atlas` module (in columns
        'region'), and topological annotations (in column 'node_type') of four types: GBTR,
        representing general, branch, terminal, and root.

        :param key: The key of the loaded SWC, should be a string path.
        :param ordering: The traversal type, either 'pre', 'in', or 'post'.
        :param lesser_first: Whether traversing from lesser to greater, affect both within and among subtrees.
        :return:
        """
        try:
            # shallow copy the qc dataframe for new columns
            swc = self._swc_qc[key].copy(deep=True)
            nt = NeuronTree(swc)
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
            swc['region'] = utils.ccf_atlas.annotate_swc(swc)
            return swc.loc[traversal, ['x', 'y', 'z', 'type', 'node_type', 'region']]
        except:
            if self.show_failure:
                if self.debug:
                    traceback.print_exc()
                print(f'Some error occurred when converting SWC to sequence by default: {key}.')
            return None

    def pickle_qc(self, path: Union[str, 'PathLike[str]']):
        """Save the SWC dataframes after qc as an ordered dict to a pickle file.

        :param path: the path to save the pickle
        """
        with open(path, 'wb') as f:
            pickle.dump(self._swc_qc, f)

    def pickle_result(self, path: Union[str, 'PathLike[str]']):
        """Save the generated sequences as an ordered dict to a pickle file.

        :param path: the path to save the pickle
        """
        with open(path, 'wb') as f:
            pickle.dump(self._sequences, f)

    @property
    def swc_after_qc(self) -> Optional[OrderedDict[str, pd.DataFrame]]:
        """Reference to the SWC files after quality control. If no QC is done, it's None."""
        return self._swc_qc

    @property
    def result(self) -> Optional[OrderedDict[str, pd.DataFrame]]:
        """The sequence dataframe as an OrderedDict, indexed by paths."""
        return self._sequences
