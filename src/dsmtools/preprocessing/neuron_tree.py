import pandas as pd
from functools import cache
from typing import Optional, Sequence
import numpy as np

from dsmtools.utils.misc import safe_recursion_bootstrap
from queue import LifoQueue
from dsmtools.utils.swc import get_child_dict


class BinarySubtree:
    """Binary tree implemented as linked nodes.
    See [the above guide](#binarysubtree) for a more detailed description of its principles.

    The reason why the tree is reconstructed instead of using the form of SWC comes from the benefit of node-level
    caching and easy recursion design.

    Its definition is still different from a general binary tree in some ways. It keeps an order of children by weight
    rather, named as 'lesser' and 'greater' than left and right. When it comes to a linear connected node the child will
    be lesser.

    The weight is calculated as the total path length of subtrees, so
    the determination of the subtree structure is necessary to build the parent. The construction of the tree has to be
    recursive, and once it's completed, there should be no more modification.

    This way, all properties obtained by recursion is definite, so we can apply caching on them, such as the total
    path length itself, and traversal.
    """

    def __init__(self, key: int, coord: Sequence[float], children: Sequence):
        """Set up a tree node with already constructed children nodes.

        :param key: The SWC node ID.
        :param coord: The 3D coordinate of the node.
        :param children: A sequence containing at most 2 already constructed child BinarySubtree.
        """
        self._key = key
        self._coord = np.array(coord)
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
    def key(self) -> int:
        """Node ID."""
        return self._key

    @property
    def coord(self) -> Sequence[float]:
        """Node coordinate."""
        return self._coord

    @property
    def lesser(self) -> 'BinarySubtree':
        """Child tree node of the subtree with lesser total path length."""
        return self._lesser

    @property
    def greater(self) -> 'BinarySubtree':
        """Child tree node of the subtree with greater total path length."""
        return self._greater

    @cache
    @safe_recursion_bootstrap
    def tot_path_len(self, stack: LifoQueue) -> float:
        """
        Recursive sum of path length of the subtree, implemented as recursion safe, and with a cache feature.
        It can be used as a normal recursion, but must be provided with a new stack of `LifoQueue`.

        Usually this value is already calculated and cached after the tree construction completes.

        :param stack: A newly initialized `LifoQueue` passed by keyword, will be passed down in the recursion.
        :return: Total path length of this subtree.
        """
        s = 0.0
        if self._lesser is not None:
            s += (yield self._lesser.tot_path_len(stack=stack)) + np.linalg.norm(self._coord - self._lesser.coord)
        if self._greater is not None:
            s += (yield self._greater.tot_path_len(stack=stack)) + np.linalg.norm(self._coord - self._greater.coord)
        yield s

    @cache
    @safe_recursion_bootstrap
    def preorder(self, lesser_first, stack: LifoQueue) -> tuple[int]:
        """
        Preorder traversal for this subtree, implemented as recursion safe, and with a cache feature.
        It can be used as a normal recursion, but must be provided with a new stack of `LifoQueue`.

        :param lesser_first: Whether to traverse the lesser subtree before the greater.
        :param stack: A newly initialized `LifoQueue` passed by keyword, will be passed down in the recursion.
        :return: A tuple of node IDs.
        """
        a = (yield self._lesser.preorder(lesser_first, stack=stack)) if self._lesser is not None else tuple()
        b = (yield self._greater.preorder(lesser_first, stack=stack)) if self._greater is not None else tuple()
        yield (self._key,) + a + b if lesser_first else (self._key,) + b + a

    @cache
    @safe_recursion_bootstrap
    def inorder(self, lesser_first, stack: LifoQueue) -> tuple[int]:
        """
        Inorder traversal for this subtree, implemented as recursion safe, and with a cache feature.
        It can be used as a normal recursion, but must be provided with a new stack of `LifoQueue`.

        :param lesser_first: Whether to traverse the lesser subtree before the greater.
        :param stack: A newly initialized `LifoQueue` passed by keyword, will be passed down in the recursion.
        :return: A tuple of node IDs.
        """
        a = (yield self._lesser.inorder(lesser_first, stack=stack)) if self._lesser is not None else tuple()
        b = (yield self._greater.inorder(lesser_first, stack=stack)) if self._greater is not None else tuple()
        yield a + (self._key,) + b if lesser_first else b + (self._key,) + a

    @cache
    @safe_recursion_bootstrap
    def postorder(self, lesser_first, stack: LifoQueue) -> tuple[int]:
        """
        Postorder traversal for this subtree, implemented as recursion safe, and with a cache feature.
        It can be used as a normal recursion, but must be provided with a new stack of `LifoQueue`.

        :param lesser_first: Whether to traverse the lesser subtree before the greater.
        :param stack: A newly initialized `LifoQueue` passed by keyword, will be passed down in the recursion.
        :return: A tuple of node IDs.
        """
        a = (yield self._lesser.postorder(lesser_first, stack=stack)) if self._lesser is not None else tuple()
        b = (yield self._greater.postorder(lesser_first, stack=stack)) if self._greater is not None else tuple()
        yield a + b + (self._key,) if lesser_first else b + a + (self._key,)


class NeuronTree:
    """Interface for generating binary trees and commit traversals for the whole SWC neuron tree.
    See [the above guide](#neurontree) for a more detailed description of its principles.

    With this interface, you can provide a set of nodes to its binary tree find function
    `NeuronTree.find_binary_trees_in`. The method will take your input as the foreground, and do a backtracking and
    merge to generate multiple trees (the set of nodes given can be disconnected).

    For tree generation, it implements a tree building method `NeuronTree.build_a_tree` using `BinarySubtree` as units.
    At last, the root `BinarySubtree` nodes of the built trees will be saved.

    You can repeat this process multiple times and the interface will aggregate your results by default,
    which is accessible as `NeuronTree.binary_trees`. You are free to use this property to traverse them with the
    `BinaryTree` methods yourself or use `NeuronTree.collective_traversal` to get it at once.
    """

    def __init__(self, swc: pd.DataFrame, tree_node_class=BinarySubtree):
        """
        Initialize by assigning an SWC dataframe as a reference.
        It will only use the xyz and parent fields, and will check if they exist at this stage.
        Any change of the original copy of the dataframe is deprecated, as the initialization info would be outdated.

        :param swc: An SWC dataframe, no modification takes place.
        :param tree_node_class: The binary tree node class to use for building, as you can provide your own.
        """

        self._df = swc[['x', 'y', 'z', 'parent']]
        self._tree_class = tree_node_class
        self._trees = list[tree_node_class]()
        self._child_dict = get_child_dict(self._df)

    @property
    def binary_trees(self):
        return self._trees

    @safe_recursion_bootstrap
    def build_a_tree(self, rt: int, within: set, stack: LifoQueue):
        """Build a binary tree from a root using a specified tree node class.

        :param rt: The root to start with, and to proceed in recursion.
        :param within: The range of node IDs to build trees from, children outside will be ignored.
        :param stack: A new queue.LifoQueue to handle recursion in decorators.
        :return: The root of the binary tree as the specified tree node class.
        """
        children = []
        for i in self._child_dict[rt]:
            if i in within:
                children.append((yield self.build_a_tree(i, within, stack=stack)))
        yield self._tree_class(rt, self._df.loc[rt, ['x', 'y', 'z']], children)

    def find_binary_trees_in(self, index: Sequence[int], overwrite=False):
        """Find the largest binary trees within a subset of nodes of an SWC.

        With this method, you can build binary trees only upon the 'foreground' nodes in the SWC, so you can get
        traversals for specific structures.
        Naturally, if the nodes are not contiguous, there will be multiple binary trees. Here, the root of the
        whole neuron will not be counted even if provided.

        By merging any founded backtracking, it determines which tree each node belongs to, and ensures the trees
        are as deep as they can be, working like a union-find. The backtracking starts from 'terminals',
        the nodes without children in the set, rather than the real terminals of the SWC.

        After backtracking, it invokes `NeuronTree.build_a_tree` to build the trees as linked nodes recursively. By
        default, it uses `BinarySubtree` as the tree node class. You can change it during initialization.

        This method will save the results in the property `NeuronTree.binary_trees`. It's a list and every time you
        invoke the function, the new results will be appended, unless you specify to overwrite the whole list.

        :param index: A sequence containing nodes including the subtrees to search.
        :param overwrite: Whether to overwrite the existing trees or extend.
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

        trees = [self.build_a_tree(root, nodes, stack=LifoQueue()) for root, nodes in sorted_nodes.items()]
        if overwrite:
            self._trees = trees
        else:
            self._trees.extend(trees)

    def collective_traversal(self, ordering='pre', lesser_first=True) -> tuple[int]:
        """Obtain binary tree traversals of three orders for a list of trees and combine them into one by some priority.

        You can specify which of pre, in and postorder traversal is performed, as well as whether the lesser subtrees
        are traversed first than the greater.
        This method will also chain the traversals of each binary tree into one, by the same order with the traversal in
        each subtree.

        :param ordering: The traversal type, either 'pre', 'in', or 'post'.
        :param lesser_first: Whether traversing from lesser to greater, affect both within and among subtrees.
        :return: A list of node IDs.
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
