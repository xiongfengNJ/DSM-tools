import pandas as pd
from functools import cache
from typing import Iterable, Optional
import numpy as np

from dsmtools.utils.misc import safe_recursion_bootstrap
from queue import LifoQueue
from dsmtools import utils


class BinarySubtree:
    """
    Node by node tree for the NeuronTreeSequencer class. It's ordered and children are lesser and
    greater instead of left and child.
    The order is based on total path length of subtree of either node, calculated by recursion and the coordinates given
    at initialization time.

    Once given during initialization, the properties shouldn't be modified. As some properties are cached, this is safe.

    When the node isn't bifurcation, the child is always given to the lesser. There is a comparison strategy to ensure
    that.
    """

    def __init__(self, key: int, coord: Iterable[float], children: Iterable):
        """
        Set up an BinarySubtree based on the current node and its children. The child can be an iterable
        not longer than 2. It can also be a generator whose computation can be done at initialization time.

        :param key: the SWC node index.
        :param coord: the 3D coordinate of the node.
        :param children: an iterable that contains at most 2 child BinarySubtree, this can be a generator.
        """
        self._key = key
        self._coord = np.array(coord)
        children = list(children)
        assert len(children) <= 2
        self._lesser: Optional[BinarySubtree] = None
        self._greater: Optional[BinarySubtree] = None
        if len(children) == 2:
            if children[0].tot_path_len(stack=LifoQueue()) <= children[1].tot_path_len(stack=LifoQueue()):
                self._lesser = children[0]
                self._greater = children[1]
            else:
                self._lesser = children[1]
                self._greater = children[0]
        elif len(children) == 1:
            self._lesser = children[0]

    @property
    def key(self):
        return self._key

    @property
    def lesser(self):
        return self._lesser

    @property
    def greater(self):
        return self._greater

    @property
    def coord(self):
        return self._coord

    @cache
    @safe_recursion_bootstrap
    def tot_path_len(self, stack: LifoQueue):
        """Sum of path distance under this node."""
        s = 0.0
        if self._lesser is not None:
            s += (yield self._lesser.tot_path_len(stack=stack)) + np.linalg.norm(self._coord - self._lesser.coord)
        if self._greater is not None:
            s += (yield self._greater.tot_path_len(stack=stack)) + np.linalg.norm(self._coord - self._greater.coord)
        yield s

    @cache
    @safe_recursion_bootstrap
    def preorder(self, lesser_first, stack: LifoQueue) -> tuple[int]:
        a = (yield self._lesser.preorder(lesser_first, stack=stack)) if self._lesser is not None else tuple()
        b = (yield self._greater.preorder(lesser_first, stack=stack)) if self._greater is not None else tuple()
        yield (self._key,) + a + b if lesser_first else (self._key,) + b + a

    @cache
    @safe_recursion_bootstrap
    def inorder(self, lesser_first, stack: LifoQueue) -> tuple[int]:
        a = (yield self._lesser.inorder(lesser_first, stack=stack)) if self._lesser is not None else tuple()
        b = (yield self._greater.inorder(lesser_first, stack=stack)) if self._greater is not None else tuple()
        yield a + (self._key,) + b if lesser_first else b + (self._key,) + a

    @cache
    @safe_recursion_bootstrap
    def postorder(self, lesser_first, stack: LifoQueue) -> tuple[int]:
        a = (yield self._lesser.postorder(lesser_first, stack=stack)) if self._lesser is not None else tuple()
        b = (yield self._greater.postorder(lesser_first, stack=stack)) if self._greater is not None else tuple()
        yield a + b + (self._key,) if lesser_first else b + a + (self._key,)


class NeuronTree:
    """Class generating different ordering traversals for the whole SWC neuron tree.

    - USED FEATURES OF SWC
    This class uses root, bifurcation, terminal to signify topological nodes and traits, and soma, axon, dendrite to
    signify biological features.

    1. Topological features are fixed for a specific tree, and used for the regularization of the tree structure,
    e.g. no nodes can have more than 2 branches other than roots, and subtree backtracking must end before the root
    (as they must be binary tree). Such regularization is in case for inputting SWC with multiple roots
    (common for some automatic reconstruction), which is highly deprecated.

    2. Biological features are node annotation that helps you to cut out subtrees, typically prepared as the type field
    in SWC for most manual reconstructions, but uncommon for some others. Without this info, the program cannot tell
    the difference between axons and dendrites, and will output the very subtrees sprouting from the root.
    You can also define your own type and cut more interesting subtrees anyway you like. It's just annotation.
    """

    def __init__(self, swc: pd.DataFrame, deep_copy=False):
        """
        Initialize a NeuronTreeSequencer object by assigning private properties and annotate the SWC. The dataframe is
        by default referenced and there's no modification to it in this class, but as you are responsible for
        maintaining the dataframe, their modification during this class's usage would cause unpredictable results.

        As recursion will be used, for betting managing the system attributes. Here provides a recursion limit setter.
        Any use of recursion will temporarily set the limit to the value specified. It has to be higher than
        the max depth of your SWC tree. Or you should do down-sampling for your I beforehand.

        :param swc: the SWC dataframe, no modification during usage.
        :param deep_copy: whether to deep copy the original dataframe in case it could be modified during usage.
        """

        self._df = swc[['x', 'y', 'z', 'parent']].copy(deep=deep_copy)
        self._trees = list[BinarySubtree]()
        self._child_dict = utils.swc.get_child_dict(self._df)

    @property
    def binary_trees(self):
        return self._trees

    @safe_recursion_bootstrap
    def _build(self, rt: int, within: set, stack: LifoQueue):
        children = []
        for i in self._child_dict[rt]:
            if i in within:
                children.append((yield self._build(i, within, stack=stack)))
        yield BinarySubtree(rt, self._df.loc[rt, ['x', 'y', 'z']], children)

    def find_binary_trees_in(self, index: Iterable[int], overwrite=False):
        """Merging any founded backtracking within a subset of its nodes, and recursively build subtrees. Trees are
        stored as a list that can be referenced as a property in this class.

        The subtrees are defined as the deepest trees constructed by those nodes. Usually, it's used to differentiate
        between dendrite and axon subtrees. Because the root of some subtrees, for instance, that of axon can be
        diverse, either from the soma or from a dendrite, so finding it top-down isn't effective.

        If the nodes are not contiguous, because this program backtracks from virtual terminals, it will output
        trees as well, with vertical breakups. This feature will be make it useful to preprocessing discrete arbors.

        Note this won't necessarily give you the
        biological stems,  i.e. the main one-level branches of a neuron, unless there's no branch point alongside the
        root. So you should make sure the topology of soma region is precise, or make your own node collection (e.g.
        draw a circle around the soma and exclude the nodes within, like seeing them as roots) and do
        the subtree finding.

        Disabling overwrite gives you the flexibility to build trees multiple times for different node types.

        :param index: an iterable containing nodes including the subtrees to search.
        :param overwrite: whether to overwrite the existing trees or extend, default as False.
        """

        index = pd.Index(index)
        child_count = self._df.loc[index, 'parent'].value_counts()
        ends = index.difference(child_count.index)
        sorted_nodes = {}
        root_index = self._df.index[self._df['parent'] == -1]
        for n in ends:
            backtrack = []
            while n in index and n not in root_index:  # root can't be included in a binary tree, must stop
                backtrack.append(n)
                n = self._df.loc[n, 'parent']
                for root, tree in sorted_nodes.items():
                    if n in tree:
                        sorted_nodes[root] |= set(backtrack)
                        break
                else:  # when current subtrees contain none of backtrack, continue backtrack
                    continue
                break  # complete backtracking, break and start from next terminal
            else:  # execute when cur_id run out of nodes, i.e. meeting a new ending
                sorted_nodes[backtrack[-1]] = set(backtrack)

        trees = [self._build(root, nodes, stack=LifoQueue()) for root, nodes in sorted_nodes.items()]
        if overwrite:
            self._trees = trees
        else:
            self._trees.extend(trees)

    def collective_traversal(self, ordering='pre', lesser_first=True) -> tuple[int]:
        """Obtain traversals of three orders for a list of trees and combine them into one by some priority.

        For some applications, like classification, one ordering is quite enough. Yet reconstructing the original tree
        requires the inorder with either the pre- or the postorder traversal.

        The priority is whether the lesser or the greater subtree will be traversed as the left or right child.

        This function also chains the traversal of each binary tree into one, by the same order with the traversal in
        each subtree.

        :param ordering: the traversal type, either 'pre', 'in', or 'post'.
        :param lesser_first: whether traversing from lesser to greater, affect both within and among subtrees.
        :return: a list of node index.
        """

        trees = sorted(self._trees, key=lambda x: x.tot_path_len(stack=LifoQueue()), reverse=not lesser_first)
        if ordering == 'pre':
            traversal = (t.preorder(lesser_first, stack=LifoQueue()) for t in trees)
        elif ordering == 'in':
            traversal = (t.inorder(lesser_first, stack=LifoQueue()) for t in trees)
        elif ordering == 'post':
            traversal = (t.postorder(lesser_first, stack=LifoQueue()) for t in trees)
        else:
            raise "Must specify an ordering among 'pre', 'in' and 'post'."
        return sum(traversal, ())
