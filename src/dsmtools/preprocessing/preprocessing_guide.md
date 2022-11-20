One of the major features of this package, also as the prerequisite for you to run a neural network on neuron morphology
data, is to convert SWC into sequences of quantifiable features. This module provides classes to:

* commit quality control on a batch of SWC dataframes to make sure the structures can be converted to sequence.
* convert a batch of SWC into sequences of tree node traversals annotated with useful features.

The classes under this module also use multiprocessing to boost the speed as the tasks are highly homogeneous.

# Traversal of Binary Tree

A binary tree can be traversed by **preorder**, **inorder** and **postorder**, which is the simplest idea of
converting a tree structure to sequence. Here, we don't include the breadth- and depth-first methods that produce 
more diverse results for general tree structures.

How do you get each of them? As a simple explanation, when you come into a tree node, w/o children typically recognized
as its subtrees, do the following steps **recursively**:

| Order | Step 1                    | Step 2                     | Step 3                     |
|-------|---------------------------|----------------------------|----------------------------|
| pre   | output the current node   | traverse the left subtree  | traverse the right subtree |
| in    | traverse the left subtree | output the current node    | traverse the right subtree |
| post  | traverse the left subtree | traverse the right subtree | output the current node    |

This workflow is executed from the root to the leaves for any node in the tree. As the structure of the tree and its
subtrees are comparable, this is usually implemented with **recursion**.

Any binary tree can and only be uniquely represented by either **pre+inorder** or **post+inorder** traversal pairs.
A single traversal of any kind is attainable from a bunch of binary trees,
the number of which is given by the **Catalan number**.

The matter of left or right is decided by the builder of the tree. They are essentially interchangeable,
so you would call where you start the left.

[A full explanation of tree traversal](https://en.wikibooks.org/wiki/A-level_Computing/AQA/Paper_1/Fundamentals_of_algorithms/Tree_traversal),
if you are interested.

As you may note, these 3 orders of traversal make sense only for **binary trees**.
A node with #children more than 2 would give more combinations of where to put the current node in the sequence.
So how do we deal with neuron tree structures, which can have nodes with many children, especially at the soma?
It's answered in the [following section](#neuron-tree-to-sequence).

# Neuron Tree to Sequence

To make a neuron tree traversable, some preprocessing steps has to be done. In quality control, the program detects
any node with #children more than 2 except for the root node (soma), and turns them into a series of bifurcations
by a specific priority (further explained in the [QC section](#reasons-for-qc)).

Now, the whole neuron tree is not yet binary, but all of its subtrees, so we can reconstruct multiple binary trees 
with a subset of nodes excluding the root. After getting traversal sequence for each tree, it
chains the sequences into one by a specific priority. Or can interpret it as an
<span id="extended-traversal">**extended definition of traversal**</span>:

| Order         | Step 1  | Step 2  | Step 3  | ... | Step n    |
|---------------|---------|---------|---------|-----|-----------|
| extended pre  | parent  | child 1 | child 2 | ... | child n-1 |
| extended post | child 1 | child 2 | child 3 | ... | parent    |

You may still wonder:

1. What does the **specific priority** mean?  
By the total path length of the subtree, which you can take as weight. So every time running into a node, we compare
the weight of the 2 subtrees and decide either the lesser(default) or greater will be traversed first. Same is true
for the final chaining of sequences, and we use this priority for converting the branch nodes to binary ones.

2. What kind of **traversal order** are you using?  
Though we provide the API for all the 3 orders, we only do preorder for our dataset generation. A preorder can be mapped
to multiple different tree structures. That being said, we found that one order is quite enough for information 
embedding in our application, and there is no significant difference between them.

3. Now that we need only 1 traversal, **why must binary trees**?  
You may have found, a long as the tree is traversed by a determined priority and only 1 order is enough, 
whether the tree is binary matters no more. That's true, but we still expect this package to give you a chance to
try the complete representation of a tree by sequence for your own application. Another reason is bifurcations take up
the majority of neuron tree branch nodes, and we are more interested in bifurcations for analysis.

4. **How many binary trees** are there?  
It depends on the number of stems your neuron has. A stem is the branch emerging from the root, so one binary tree
corresponds to one stem. However, our program adopts a more interesting strategy to build binary trees, and you
can actually get more than that. In our package, by default, we build trees for axon and dendrite separately. For 
neurons where axon can emerge from the dendrite not as a stem, this makes a difference, in that the axon would otherwise
be counted as part of dendrite in a binary tree. The [Customization section](#sequence-customization) has a more instructive guide.

# Reasons for Quality Control

Before converting your SWC files to our sequences, some prerequisites should be met. They are:

1. 1 tree for 1 SWC, i.e. no more than 1 root node in an SWC file.
2. Short segments are removed.
3. All non-root branching nodes are binary.

## remove multiple trees

As a usual practice, a single SWC would only be given a single neuron tree, but it may not be true 
for some automatic tracing algorithms and tracings with error.

Although our final transformation of neuron tree to sequence doesn't require there to be only one connected component,
as you may find out, there are still some processing steps that rely on this, so we would suggest, if you need all 
the unconnected components, separate them beforehand.

Otherwise, our program will mark the first root in the table and retain the component it's connected to.

## guarantee bifurcation

As described above, nodes with many children would give multiple traversing choices. Although we don't really need a
binary tree for getting a pre/postorder sequence of [the extended definition](#extended-traversal), the nature of neuron 
branching and the unique representation of binary tree encourage us to assume that.

As already suggested, this QC step will consider the subtree total path length as branching priority. To be exact,
a subtree branch of a lesser scale would be adjusted closer to the root. This way, if you traverse the tree 
from the lesser to the greater, the lesser adjusted part will always be visited first, equivalent to the way it would be
without the adjustment. If you start from the greater it's the same. Yet we still implement the traversing unit as 
binary, so the adjustment is necessary.

Together with this adjustment, the tree nodes will be sorted so that node numbers increase from the root to the leaves,
and the maximum node ID equals the tree size. This step is favorable as we need to assign new nodes for the adjustment,
and the final result will be sorted again. As this sort_swc function doesn't connect tree components (unlike the 
[Vaa3D](https://alleninstitute.org/what-we-do/brain-science/research/products-tools/vaa3d/) plugin 
[*sort_neuron_swc*](https://github.com/Vaa3D/vaa3d_tools/tree/master/released_plugins/v3d_plugins/sort_neuron_swc)),
it requires the input SWC to have only 1 root, so the first [QC step](#remove-multiple-trees) is necessary.

## prune short segments

We found the main skeletons of a neuron is sufficient for encoding the tree structure, and removing the short segments
can reduce the computation load. This program will prune the tree iteratively until there is no segment shorter than
a given threshold. The pruning wouldn't cause a tree to 'shrink', i.e. as far as a path is longer than the threshold,
it wouldn't be affected at all. You can see this pruning as removing branches that look like one. Its logic is
identical to that of the [Vaa3D](https://alleninstitute.org/what-we-do/brain-science/research/products-tools/vaa3d/) 
plugin [*sort_neuron_swc*](https://github.com/Vaa3D/vaa3d_tools/tree/master/released_plugins/v3d_plugins/pruning_swc).

# Sequence Customization

The data preparation module designed for our deep learning models can be easily understood and modified.
You are encouraged to learn from the processing code and use the infrastructures to customize subtree sequences, 
in terms of not only different traversal orders or their combination, but also the node types.

There is a trend that people are interested in disintegrating the neuron into multiple subunits
by some algorithm and quantify their features, going further than the conventional axon-dendrite dichotomy.
We provide the function of this very dichotomy, but you can follow it to try your own partition of neuron nodes.
Given any set of nodes, even if they are unconnected, or without terminals, the program can find binary trees for you.

Three classes are designed for data preparation except QC: `NeuronSequenceDataset`, `NeuronTree` and `BinarySubtree`.

## NeuronSequenceDataset

`NeuronSequenceDataset` is a high-level pipeline execution interface utilizing parallel computation.
It commits data loading, quality control and traversal serially but in parallel for each step.

As a good template to start with, it contains instances of parallel computation by pairing the host processing
function (ending with `_parallel`) with its own unit processing function (ending with `_proc`), for all the three steps.
In most cases where you are satisfied with the multiprocessing design, 
you only need to inherit and derive the unit functions, and let the host invoke them for you.

In the traversal step that you might be mostly interested in for customization, 
the high-level class calls a child process to execute binary tree reconstruction, traversal,
and retrieval of spatial features indexed by the traversal within one SWC, where `NeuronTree` is put in use.

## NeuronTree

`NeuronTree` is an SWC-level interface to find binary subtrees by sets of nodes and commit traversal upon them.
After initialization, you can provide the interface with a set of nodes to find subtrees.

The number of trees you can get for a given node set depends on the connectivity.
If you specify all the nodes within a neuron tree, as the root will always be ignored, it will return subtrees
for each stem. When you have multiple sets of nodes, like axon & dendrite, 
you can invoke this discovery process multiple times, each time
for one set. e.g., you can find axon trees first, and the class will store the found trees, and then you
invoke that for dendrite, and this time the dendrite trees will be added to the storage. 
The ultimate subtree number can be larger than the that of the node set.

After the discovery process, you can use this interface to do traversal by pre, in or postorder in a single shot.
Here, you can also specify whether to do it from lesser to greater subtrees or the other way around.
If you'd like to make a more customizable traversal, you can retrieve the binary trees and do traversal for
each one of them yourself and assemble as you like.

This class only allows SWC of bifurcations except for the root, as it uses `BinarySubtree` for tree building.

## BinarySubtree

`BinarySubtree` is the basic class that undertakes recursive searching of node, total subtree path length and traversal.
Its instances are basically tree nodes pointing to their children, so the built subtrees exposed by `NeuronTree` are 
actually their roots. 

All the recursion functions are decorated to cache data prevent stack overflow caused by Python's stack limit. 
Note, the caching behavior applies to any node, and can cause bugs if you modify the tree structure yourself, 
so you should keep it as what `NeuronTree` builds.

If you'd like to have other stats of the binary subtree, you can write your own recursion functions. Please follow
the instructions of `dsmtools.utils.misc.safe_recursion_bootstrap` if you want to bypass the stack problem. Caching
can only be used by a tree node member function, so ideally you could inherit `BinarySubtree`, 
but building would require refactoring `NeuronTree`.

---

Below are the class API documentations of the above classes for you to learn more about their usage and implementation.
Have fun!
