



# Effecient implementation of Recursive Neural Nets using level-wise batching in TF

Recursive Neural Nets follows a tree structure recursively to build a consise representation of a sentence or a paragraph. That means that every sentence has its own tree structure, the result of its parsing. So, if we want to implement such model in TF, we have to build a graph serperatly and iteratively for each training sentence, which is very time consuming. Luckly TF provides tools for building flexible models. These tools are `tf.while_loop` , `tf.cond` and `tf.gather`. In [this github repo](https://github.com/bogatyy/cs224d/tree/master/assignment3), the author modified the naive implementation of RvNN to use `tf.while_loop` inside of TF graph structure to respresent the idea of a tree recursively rather than building a new model recursively  for every tree through a regural while loop. This made the code run x16 times faster. 

In this project we'll improve it more, making it from 6X to 500X for various batching sizes from 1 to the full dataset, that makes it 16 *(6 or 500) faster than the naive implementation. Since each layer in the recursive tree is dependent on the previous layer but not dependent on other nodes in the same level, it makes sense to group tree operations by level, so that we can make compute several nodes in parallel. This can boost the performance by a factor up to $n / log_{2}(n)$, as a binary tree of $n$ nodes have at least $log_{2}(n)$ levels and at most $n$ levels, reflecting the two cases of a balanced binary tree, and RecurrentNN-like tree (each level link all the previous words with the next word in one node). The performance boost will show more in long sentences, since those, on average, will have higher factor of number of nodes to number of levels. For example, a sentence of length $m$ words, will have $2m - 1$ nodes and from $m$ to $log_{2}(m) + 1$ levels. So we hope that at least we get $(2m -1)/ m$ increase in performance  and at most  $(2m - 1)/( log_{2}(m) + 1)$ increase.  

In order to implement the above idea we can break down the necessary steps into the following`:
#### Step 1:
First we need to know the level each node instant. In this part we'll need to modify the basic Node class that is part of `tree.py`, this part of the code that is provided with the assignment of Stanford Deep Learning course mentioned in the aforementioned github to parse and load data from SST data-set.
We will just add this line to `Node` class constructor:
```python
class Node:
    def __init__(self, ...):
        ...
        self.level = None
        ...
````
Then we'll populate this levels recursively for each node
```python
def generate_levels(node):
    if node.isLeaf:
        node.level = 0
        return node.level 
    node.level = max(generate_levels(node.left),
                     generate_levels(node.right)) + 1
    return node.level
```
And call this function on the root of a tree during its instansiation in class `Tree`
```python
class Tree:
    def __init__(self, ...):
        ....
        self.max_level = generate_levels(self.root)
        ....
```

#### Step 2
Add a placeholder to represent the level of each node, and populate it in the feed dict by looping through all the nodes in a tree:
```python
self.node_level_placeholder = tf.placeholder(tf.int32, (None), name='node_level_placeholder')
feed_dict = {...,
             self.node_level_placeholder: [node.level for node in nodes_list],
             ....}
```
Where `nodes_list` is a list of all the nodes in a tree.
The key difference here is that the while_loop won't run from 0 to the number of nodes, but from 0 to number of levels. To get the max level within a single tree, one can simply use `tf.reduce_max`, we can return the level of the root, but since we want to use this later in batching, it's better to use the maximum height in general. Then looping over these levels, we check which nodes are of that level, gather them by `tf.gather_nd` apply the appropriate operation, embedding for level 0 and recursion for the rest, then scatter the value in `tensor_array` to be used in the next level.

```python
max_level = tf.reduce_max(self.node_level_placeholder)
loop_cond = lambda tensor_array, i:  tf.less(i, tf.add(1, max_level))
def loop_body(tensor_array, i):
    level = tf.equal(i, self.node_level_placeholder) # of size 1 x number of nodes 
    indeces = tf.where(level)
    node_word_index = tf.gather_nd(self.node_word_indices_placeholder, indeces) 
    left_children = tf.gather_nd(self.left_children_placeholder, indeces)
    right_children = tf.gather_nd(self.right_children_placeholder, indeces)
    indeces = tf.reshape(indeces, [-1])
    
    node_tensor = tf.cond(
        tf.equal(i, 0),
        lambda: embed_words(tf.reshape(node_word_index, [-1, 1])),
        lambda: combine_children(tensor_array.gather(left_children), tensor_array.gather(right_children)))
    tensor_array = tensor_array.scatter(indeces, node_tensor)
    i = tf.add(i, 1)
    return tensor_array, i
```
Only `embed_words` will change to using `tf.gather_nd` instead of `tf.gather`, so we can read multiple embedding at once from the embedding matrix. `combine_children` won't need a change, it will just return a $d\times E$ tensor, where d is the number of nodes in a given level, instead of $1\times E$ representing a single word. 
```python
def embed_words(word_indeces):
    return tf.gather_nd(embeddings, word_indeces)

def combine_children(left_tensor, right_tensor):
    # left_tensor is of size  d x E
    # right_tensor is of size  d x E
    concatenated_children = tf.concat([left_tensor, right_tensor], 1) # d x 2E
    return tf.nn.relu(tf.matmul(concatenated_children, W1) + b1)# d x E
```
### Results:
If everything else fixed and we train the whole binary data-set with 6920 data points using both methods, the one mentioned here and the one mentioned in the github repo mentioned above, the following running time differences is obtained.
|   |Static Graph (the repo solution) | Level-wise batches   |  
|---------------------------------------------|:------:|:---------:|
| Time for one Epoch (6920 sentences) in secs | 1940 secs ~ 32 mins |**330** secs ~ 5 mins |
| Time for the shortest 700 sentences in secs | 7.1 secs |**3.3** secs|
|   |   |   |
--
In the next table we show a comparison between the obtained speedup, the best possible speedup and the least possible speedup, for both the whole dataset and the shortest 700 sentences. As we can see the longer the sentence the higher the maximum and actual speedup we can reach, making the difference in performance more pronounced in the case of the full epoch (for 32 mins to 5 mins).

|                                   |Full Epoch                           | Shortest 700 Sentence               |  
|-----------------------------------|:-----------------------------------:|:-----------------------------------:|
| Average length (n)                | $19$                                |$7$                                  |
| Average number of levels          | $10$                                | $5$                                 |
| Actual improving factor           | $1940/330 = 5.87$                   |  $7.1/3.3 = 2.15$                   |
| Worst possible improving factor   | $(2n-1)/n = 1.94$                     |  $(2n-1)/n = 1.85$                    |
| Best possible improving factor    | $(2n - 1)/( log_{2}(n) + 1) = 6.1$ | $(2n - 1)/( log_{2}(n) + 1) = 3.25$ |

------
#### Step 3:
Now that a tree is parallelised in terms of its levels, we can batch the whole data-set. In the blog mentioned above, the author says the following

> It should not be too hard to add batching to the static graph implementation, speeding it up even further. We would have to pad the placeholders up to the length of the longest tree in the batch, and in the loop body replace [`tf.cond(...)`](https://www.tensorflow.org/versions/r0.10/api_docs/python/control_flow_ops.html#cond) on a single value with [`tf.select(...)`](https://www.tensorflow.org/versions/r0.10/api_docs/python/control_flow_ops.html#select) on the whole batch.
>  
I don't believe this is possible, since node x in sentence 1 and node x in sentence 2 don't have to be of the same types (both leaf or both internal nodes), and thus cannot be parallelised or computed at the same time. But I might be missing something, or not get what he means fully.

We'll still batch the data-set, but we will follow the same ideas we used to batch per-level. For every example in the batch, the same level will be parallelised across the whole batch. So if the batch has 2 trees, one with 2 levels and one with 10 levels, the first two levels will be computed together and the rest will only be computed for the higher tree. This means that no padding will be involved to make the shorter one as long as the high on in terms of levels or nodes. The lack of padding also means saving time doing computation on extra padding, especially when there's a huge difference in size. 
We will  implement that as follows. Instead of having   `tensor_array` carries only one tree nodes, it will carry the whole batch, in order. So the whole size of it will be the summation of the number of nodes in a tree in the whole batch.  To keep track of which element in the tensor_array is the root(s), as we need them to measure root_only accuracy, will have a placeholder with the right indices. We will also add also `number_of_examples_placeholder` to keep track of how many trees we are processing,  so we can calculate the avg. loss (avg in terms of trees not nodes) across the whole batch, this is basically the batch size except for the last batch, which might be smaller (unless the batch generator rotates through the data set), so better feed the length of each batch than using a const.
```python
self.root_indeces_placeholder = tf.placeholder(tf.int32, (None), name='root_indeces_placeholder')
self.number_of_examples_placeholder = tf.placeholder(tf.float32, (), name='number_of_examples_placeholder')

feed_dict = {...
            self.root_indeces_placeholder: [node_to_index[node] 
                                            for node in batch_node_lists
                                            if node.isRoot],
            self.number_of_examples_placeholder: len(trees),
            ....
            }
```
`isRoot` is `False` by default, and only set to `True` when `parent` of the node is `None`.
The while_loop code and its associated parts are exactly the same. The change will be only when we need to predict the sentiment of batch of trees, rather than only one. 
```python
root_indeces = tf.reshape(self.root_indeces_placeholder, [-1, 1])
self.root_logits =  tf.gather_nd(self.logits, root_indeces)
``` 
The following table is the running time for various batch size, including processing all the dataset at one go.

Batch size | time 
|:-----------:|:-------:|
1 | 330 secs
8| 77 secs
16| 45 secs
32| 30 secs
64 |20 secs
128| 12 secs
526| 6 secs
All (6920) | 4.1 secs


On CPU the whole data-set (6920 examples) is trained in **4.1 secs**, when giving max batch size (the whole data-set) . That's 500x faster than the 35 mins required for one epoch, and on a personal laptop CPU.  Of course we won't necessary use the whole data-set in one batch,  but in case of hyper parameter tuning, where the training for multiple epochs will be repeated for every variation of hyper parameters, having one epoch in half an hour won't make us able to experiment much.  While with full batch size we can have more than 850 epoch in an hour, so even if we decreased the batch size, there's still a huge difference between the running times.
