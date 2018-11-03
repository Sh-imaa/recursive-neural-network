import os, sys, shutil, time, itertools
import math, random
from collections import OrderedDict, defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope, init_ops

import utils
import tree

SAVE_DIR = './weights/'
LOG_DIR = './logs/'



class Config(object):
    def __init__(self, model_name='RvNN', lr=0.01, l2=0, embed_size=35,
                 label_size=2, batch_size=32, max_epochs=20):
        self.model_name = model_name
        self.lr = l2
        self.l2 = l2
        self.embed_size = embed_size
        self.label_size = label_size
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.model_string = '{}_lr={}_l2={}_embed_size={}_label_size={}_batch_size={}'.format(
                        model_name, lr, l2, embed_size, label_size, batch_size)

class RecursiveNN(object):
    def __init__(self, config, train_data, dev_data, test_data=None):
        self.config = config
        self.train_data, self.dev_data, self.test_data = train_data, dev_data, test_data

        self.vocab = utils.Vocab()
        train_sents = [t.get_words() for t in self.train_data]
        self.vocab.construct(itertools.chain.from_iterable(train_sents))

        self.build_graph()

    def build_graph(self):    
        # placeholders
        with tf.variable_scope('Input'):
            self.node_level_placeholder = tf.placeholder(
                tf.int32, (None), name='node_level_placeholder')
            self.left_children_placeholder = tf.placeholder(
                tf.int32, (None), name='left_children_placeholder')
            self.right_children_placeholder = tf.placeholder(
                tf.int32, (None), name='right_children_placeholder')
            self.node_word_indices_placeholder = tf.placeholder(
                tf.int32, (None), name='node_word_indices_placeholder')
            self.root_indeces_placeholder = tf.placeholder(
                tf.int32, (None), name='root_indeces_placeholder')
            self.labels_placeholder = tf.placeholder(
                tf.int32, (None), name='labels_placeholder')
            self.number_of_examples_placeholder = tf.placeholder(
                tf.float32, (), name='number_of_examples_placeholder')

        with tf.variable_scope('Embeddings'):
            embeddings = tf.Variable(
                tf.random_normal(
                    [len(self.vocab), self.config.embed_size],
                    stddev=0.1),
                name="embeddings")

        with tf.variable_scope('Composition'):
            W1 = tf.Variable(
                tf.random_normal(
                    [2 * self.config.embed_size, self.config.embed_size],
                    stddev=0.001),
                name="W1")
            b1 = tf.Variable(tf.zeros([1, self.config.embed_size]), name='b1')

        with tf.variable_scope('Projection'):
            U = tf.Variable(
                tf.random_normal(
                    [self.config.embed_size, self.config.label_size],
                    stddev=0.001),
                name="U")
            bs = tf.Variable(tf.zeros([1, self.config.label_size]), name='bs')

        tensor_array = tf.TensorArray(
            tf.float32,
            size=0,
            dynamic_size=True,
            clear_after_read=False,
            infer_shape=False)
        
        def embed_words(word_indeces):
            return tf.gather_nd(embeddings, tf.reshape(word_indeces, [-1, 1]))

        def combine_children(left_tensor, right_tensor):
            # left: d x E
            # right: d X E
            concatenated_children = tf.concat([left_tensor, right_tensor], 1) # d x 2E
            activation = tf.nn.relu(tf.matmul(concatenated_children, W1) + b1)
            return activation # d x E

        def loop_body(tensor_array, i):
            ind = tf.where(tf.equal(self.node_level_placeholder, i))
            word_indeces = tf.gather_nd(self.node_word_indices_placeholder, ind)
            left_children = tf.gather_nd(self.left_children_placeholder, ind)
            right_children = tf.gather_nd(self.right_children_placeholder, ind)
            tensor_node = tf.cond(
                        tf.equal(i, 0),
                        lambda: tf.gather_nd(embeddings, tf.reshape(word_indeces, [-1, 1])),
                        lambda: combine_children(tensor_array.gather(left_children),
                                                 tensor_array.gather(right_children)))
            tensor_array = tensor_array.scatter(tf.cast(
                tf.reshape(ind, [-1]), tf.int32), tensor_node)

            i = tf.add(i, 1)

            return tensor_array, i

        max_level = tf.reduce_max(self.node_level_placeholder)
        loop_cond = lambda tensor_array, i: tf.less(i, tf.add(max_level, 1))
        self.tensor_array, _ = tf.while_loop(loop_cond, loop_body,
            [tensor_array, tf.constant(0, dtype=tf.int32)],
            parallel_iterations=1)

        # add prediction layer
        with tf.variable_scope('Prediction'):
            # ta -> n x E, concat -> n * E X 1
            # U -> E x label 
            self.logits = tf.matmul(self.tensor_array.stack(), U) + bs
            root_indeces = tf.reshape(self.root_indeces_placeholder, [-1, 1])
            self.root_logits =  tf.gather_nd(self.logits, root_indeces)
            self.root_prediction = tf.cast(tf.argmax(self.root_logits, 1), tf.int32)
            self.root_labels = tf.gather_nd(self.labels_placeholder, root_indeces)
            self.root_acc = tf.reduce_mean(tf.cast(tf.equal(
                self.root_prediction, self.root_labels), tf.float32))

        # add loss layer
        with tf.variable_scope('Loss'):
            regularization_loss = self.config.l2 * (
                tf.nn.l2_loss(W1) + tf.nn.l2_loss(U))
            included_indices = tf.where(tf.not_equal(self.labels_placeholder, 2))
            self.full_loss = regularization_loss + tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=tf.gather_nd(self.logits, included_indices),
                labels=tf.gather_nd(self.labels_placeholder, included_indices)))
            self.full_loss = tf.divide(self.full_loss, self.number_of_examples_placeholder)
            self.root_loss = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.root_logits, labels=self.root_labels))
            self.root_loss = tf.divide(self.root_loss,self.number_of_examples_placeholder)

        # add training op
        with tf.variable_scope('Train'):
            self.train_op = tf.train.GradientDescentOptimizer(self.config.lr).minimize(
                self.full_loss)

        # with tf.variable_scope('Summaries'):
        #     total_loss_summary =  tf.summary.scalar('total_loss', self.full_loss)
        #     root_loss_summary = tf.summary.scalar('root_loss', self.root_loss)
        #     root_acc_summary = tf.summary.scalar('root_acc', self.root_acc)

        #     tf.summary.histogram('W1', W1)
        #     tf.summary.histogram('b1', b1)
        #     tf.summary.histogram('U', U)
        #     tf.summary.histogram('bs', bs)
        #     tf.summary.histogram('Embeddings', embeddings)

        #     tf.summary.image('W1', tf.expand_dims(tf.expand_dims(W1, 0), 3))
        #     tf.summary.image('U', tf.expand_dims(tf.expand_dims(U, 0), 3))

        #     self.merge_train_summaries = tf.summary.merge_all()
        #     self.merge_val_summaries = tf.summary.merge(
        #         [total_loss_summary, root_loss_summary, root_acc_summary])

    def generate_batch(self, train=True):
        # the generator is just to train batch by batch not to avoid storing
        # the data-set in ram
        i1 = 0
        if train:
            data = self.train_data
            data_size = len(self.train_data)
        else:
            data = self.dev_data
            data_size = len(self.dev_data)

        while True:
            i2 = min(i1 + self.config.batch_size, data_size)
            new_batch = data[i1:i2]
            i1 = i2 % data_size
            yield new_batch

    def build_feed_dict(self, trees):
        batch_node_lists = []
        for tree_instance in trees:
            tree_root = tree_instance.root
            nodes_list = []
            tree.leftTraverse(tree_root,
                lambda tree_root, args: args.append(tree_root), nodes_list)
            batch_node_lists.extend(nodes_list)
    
        node_to_index = OrderedDict()
        num_nodes = len(batch_node_lists)
        for i in xrange(num_nodes):
            node_to_index[batch_node_lists[i]] = i
    
        feed_dict = {
            self.node_level_placeholder: [node.level for node in batch_node_lists],
            self.root_indeces_placeholder: [node_to_index[node] 
                                            for node in batch_node_lists
                                            if node.isRoot],
            self.node_word_indices_placeholder: [self.vocab.encode(node.word)
                                                 if node.word else -1
                                                 for node in batch_node_lists],
            self.left_children_placeholder: [node_to_index[node.left]
                                             if node.left else -1
                                             for node in batch_node_lists],
            self.right_children_placeholder: [node_to_index[node.right]
                                              if node.right else -1
                                              for node in batch_node_lists],
            self.labels_placeholder: [node.label for node in batch_node_lists],
            self.number_of_examples_placeholder: len(trees)}
        return feed_dict

    def run_epoch(self, epoch, new_model=False, verbose=True):
        random.shuffle(self.train_data)
        feed_dict_dev = self.build_feed_dict(self.dev_data)
        with tf.Session() as sess:
            # train_writer = tf.summary.FileWriter(os.path.join(
            #     LOG_DIR, self.config.model_string, 'train'), sess.graph)
            # val_writer = tf.summary.FileWriter(os.path.join(
            #     LOG_DIR, self.config.model_string, 'val'), sess.graph)
            if new_model:
                sess.run(tf.global_variables_initializer())
            else:
                saver = tf.train.Saver()
                saver.restore(sess, SAVE_DIR + '%s.temp' % self.config.model_string)

            m = len(self.train_data)
            num_batches = int(m / self.config.batch_size) + 1
            batch_generator = self.generate_batch()
            now = time.time()
            for batch in xrange(num_batches):
                feed_dict = self.build_feed_dict(batch_generator.next())
                # loss_value, train_summary, acc, _ = sess.run(
                #     [self.full_loss, self.merge_train_summaries, self.root_acc, self.train_op],
                #     feed_dict=feed_dict)
                # loss_value, acc, _ = sess.run(
                #     [self.full_loss, self.root_acc, self.train_op],
                #     feed_dict=feed_dict)
                loss_value, _ = sess.run(
                    [self.full_loss, self.train_op],
                    feed_dict=feed_dict)


                # train_writer.add_summary(train_summary, num_batches * epoch + batch )
                # write val summary every 10 batches
                # if batch % 10 == 0:
                #     val_summary = sess.run(self.merge_val_summaries, feed_dict=feed_dict_dev)
                #     val_writer.add_summary(val_summary, (num_batches * epoch) + batch )
                if verbose:
                    sys.stdout.write('\r{} / {} :    loss = {}'.format(
                        batch, num_batches, loss_value))
                    sys.stdout.flush()
            overall = time.time() - now
            print '--------------------------'
            print overall
            print '--------------------------'
            saver = tf.train.Saver()
            if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)
            saver.save(sess, SAVE_DIR + '%s.temp' % self.config.model_string)
            train_preds, train_losses = self.predict(
                self.train_data,SAVE_DIR + '%s.temp' % self.config.model_string, get_loss=True)
            val_preds, val_losses = self.predict(
                self.dev_data, SAVE_DIR + '%s.temp' % self.config.model_string, get_loss=True)

        train_labels = [t.root.label for t in self.train_data]
        val_labels = [t.root.label for t in self.dev_data]
        train_acc = np.equal(train_preds, train_labels).mean()
        val_acc = np.equal(val_preds, val_labels).mean()

        print
        print 'Training acc (only root node): {}'.format(train_acc)
        print 'Valiation acc (only root node): {}'.format(val_acc)
        print self.make_conf(train_labels, train_preds)
        print self.make_conf(val_labels, val_preds)
        return train_acc, val_acc, train_losses, val_losses

    def train(self):
        for epoch in xrange(self.config.max_epochs):
            print 'epoch {}'.format(epoch)
            if epoch == 0:
                train_acc, val_acc, train_loss, val_loss = self.run_epoch(epoch, new_model=True)
            else:
                train_acc, val_acc, train_loss, val_loss = self.run_epoch(epoch)



    def predict(self, trees, weights_path, get_loss=False):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, weights_path)
            feed_dict = self.build_feed_dict(trees)
            if get_loss:
                root_prediction, root_loss = sess.run(
                    [self.root_prediction, self.root_loss], feed_dict=feed_dict)
                return root_prediction, root_loss
            else:
                root_prediction = sess.run(self.root_prediction, feed_dict=feed_dict)
                return root_prediction

    def make_conf(self, labels, predictions):
        confmat = np.zeros([2, 2])
        for l, p in itertools.izip(labels, predictions):
            confmat[l, p] += 1
        return confmat

train_data, dev_data, _ = tree.load_data_binary()
random.shuffle(train_data)

config = Config(batch_size=len(train_data) + 1)
model = RecursiveNN(config, train_data, dev_data)
model.train()
