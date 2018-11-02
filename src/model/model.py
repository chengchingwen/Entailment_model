import os
from time import time

import tensorflow as tf
import numpy as np
from tqdm import tqdm

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parentdir)

from .nn import embedded, mask, highway_network, multihead_attention, normalize, char_conv, ffn
from util.util import tprint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class EntailModel:
    def __init__(self,
                 name,
                 hidden_dim,
                 num_heads,
                 filter_size,
                 dataset,
                 learning_rate=0.000005,
                 model_path="../../model/",
                 retrain=True,
                 char=True,
                 pos=True,
                 use_placeholder=False,
                 dropout=False,
                 keep_rate= 0.9,
                 l2=True,
                 clip=True,
                 clip_value=1.0,
    ):
        self.name = name
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.filter_size = filter_size
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.model_path = model_path
        self.retrain = retrain
        self.char = char
        self.pos = pos
        self.use_placeholder = use_placeholder
        self.dropout = dropout
        self.keep_rate_value = keep_rate
        self.l2 = l2
        self.clip = clip
        self.clip_value = clip_value

    def dataprovider(self):
        if not self.use_placeholder:
            #sentence
            self.sentence1 = self.dataset.sentence1
            self.sentence2 = self.dataset.sentence2

            #labels
            self.labels = self.dataset.label

            #preprocess info
            self.antonym1  = tf.expand_dims(self.dataset.antonym1, -1)
            self.antonym2  = tf.expand_dims(self.dataset.antonym2, -1)
            self.exact1to2 = tf.expand_dims(self.dataset.exact1to2, -1)
            self.exact2to1 = tf.expand_dims(self.dataset.exact2to1, -1)
            self.synonym1  = tf.expand_dims(self.dataset.synonym1, -1)
            self.synonym2  = tf.expand_dims(self.dataset.synonym2, -1)

            #pos
            if self.pos:
                self.pos1 = self.dataset.pos1
                self.pos2 = self.dataset.pos2

            #char
            if self.char:
                self.sent1char = self.dataset.sent1char
                self.sent2char = self.dataset.sent2char

        else:
            #sentence
            self.sentence1 = tf.placeholder(tf.int64, shape=[None, None], name="premise")
            self.sentence2 = tf.placeholder(tf.int64, shape=[None, None], name="hypothesis")

            ###labels
            self.labels = tf.placeholder(tf.int64, shape=[None], name="label")

            #preprocess info
            self.antonym1  = tf.expand_dims(tf.placeholder(tf.float32, shape=[None, None], name="antonym1"), -1)
            self.antonym2  = tf.expand_dims(tf.placeholder(tf.float32, shape=[None, None], name="antonym2"), -1)
            self.exact1to2 = tf.expand_dims(tf.placeholder(tf.float32, shape=[None, None], name="exact1to2"), -1)
            self.exact2to1 = tf.expand_dims(tf.placeholder(tf.float32, shape=[None, None], name="exact2to1"), -1)
            self.synonym1  = tf.expand_dims(tf.placeholder(tf.float32, shape=[None, None], name="synonym1"), -1)
            self.synonym2  = tf.expand_dims(tf.placeholder(tf.float32, shape=[None, None], name="synonym2"), -1)

            #pos
            if self.pos:
                self.pos1 = tf.placeholder(tf.int64, shape=[None, None], name="pos1")
                self.pos2 = tf.placeholder(tf.int64, shape=[None, None], name="pos2")

            #char
            if self.char:
                self.sent1char = tf.placeholder(tf.int64, shape=[None, None, self.dataset.char_pad], name="sent1char")
                self.sent2char = tf.placeholder(tf.int64, shape=[None, None, self.dataset.char_pad], name="sent2char")

        #masks
        self.sent1_mask = tf.cast(tf.sign(self.sentence1), dtype=tf.float32)
        self.sent2_mask = tf.cast(tf.sign(self.sentence2), dtype=tf.float32)
        self.sent1_len = tf.reduce_sum(self.sent1_mask, -1)
        self.sent2_len = tf.reduce_sum(self.sent2_mask, -1)

        #dropout
        self.keep_rate = tf.placeholder_with_default(tf.constant(self.keep_rate_value),
                                                     shape=(),
                                                     name="keep_rate"
        )

    def word_embedding(self):
        with tf.variable_scope("word_embedding"):
            self.glove_embedding = embedded(self.dataset.embedding)
            self.embedding_pre = self.glove_embedding(self.sentence1)
            self.embedding_hyp = self.glove_embedding(self.sentence2)

            if self.dropout:
                self.embedding_pre = tf.nn.dropout(self.embedding_pre, self.keep_rate)
                self.embedding_hyp = tf.nn.dropout(self.embedding_hyp, self.keep_rate)

    def char_embedding(self):
        with tf.variable_scope("char_embedding"):
            self.char_embedding = embedded(self.dataset.char_embedding, name="char")
            self.char_embedding_pre = self.char_embedding(self.sent1char)
            self.char_embedding_hyp = self.char_embedding(self.sent2char)

            with tf.variable_scope("conv") as scope:
                self.conv_pre = char_conv(self.char_embedding_pre, filter_size=self.filter_size)
                scope.reuse_variables()
                self.conv_hyp = char_conv(self.char_embedding_hyp, filter_size=self.filter_size)

                if self.dropout:
                    self.conv_pre = tf.nn.dropout(self.conv_pre, self.keep_rate)
                    self.conv_hyp = tf.nn.dropout(self.conv_hyp, self.keep_rate)

    def pos_embedding(self):
        with tf.variable_scope("pos_embedding"):
            self.pos_embedding = embedded(self.dataset.pos_embedding, name="pos")
            self.pos_embedding_pre = self.pos_embedding(self.pos1)
            self.pos_embedding_hyp = self.pos_embedding(self.pos2)

            if self.dropout:
                self.pos_embedding_pre = tf.nn.dropout(self.pos_embedding_pre, self.keep_rate)
                self.pos_embedding_hyp = tf.nn.dropout(self.pos_embedding_hyp, self.keep_rate)

    def embedding(self):
        tprint("building embedding")
        self.word_embedding()

        s1_e = [self.embedding_pre, self.antonym1, self.exact1to2, self.synonym1]
        s2_e = [self.embedding_hyp, self.antonym2, self.exact2to1, self.synonym2]

        if self.char:
            self.char_embedding()
            s1_e.append(self.conv_pre)
            s2_e.append(self.conv_hyp)

        if self.pos:
            self.pos_embedding()
            s1_e.append(self.pos_embedding_pre)
            s2_e.append(self.pos_embedding_hyp)

        self.embed_pre = tf.concat(s1_e, -1)
        self.embed_hyp = tf.concat(s2_e, -1)


    def encoding(self):
        tprint("building highway encoder")
        self.hout_pre = highway_network(self.embed_pre, 2, [tf.nn.sigmoid] * 2, "premise")
        self.hout_hyp = highway_network(self.embed_hyp, 2, [tf.nn.sigmoid] * 2, "hypothesis")

        #dim reduction
        self.hout_pre = normalize(tf.layers.dense(self.hout_pre, self.hidden_dim, activation=tf.nn.sigmoid))
        self.hout_hyp = normalize(tf.layers.dense(self.hout_hyp, self.hidden_dim, activation=tf.nn.sigmoid))

        if self.dropout:
            self.hout_pre = tf.nn.dropout(self.hout_pre, self.keep_rate)
            self.hout_hyp = tf.nn.dropout(self.hout_hyp, self.keep_rate)

        self.hout_pre = mask(self.hout_pre, self.sent1_mask)
        self.hout_hyp = mask(self.hout_hyp, self.sent2_mask)


    def attention(self):
        tprint("build attention")
        self.pre_atten = multihead_attention(self.hout_pre,
                                             self.hout_pre,
                                             self.hout_pre,
                                             h = self.num_heads,
                                             scope="pre_atten"
        )

        self.hyp_atten = multihead_attention(self.hout_hyp,
                                             self.hout_hyp,
                                             self.hout_hyp,
                                             h = self.num_heads,
                                             scope="hyp_atten"
        )

        self.p2h_atten = multihead_attention(self.pre_atten,
                                             self.hyp_atten,
                                             self.hyp_atten,
                                             h = self.num_heads,
                                             scope="p2h_atten"
        )

        self.h2p_atten = multihead_attention(self.hyp_atten,
                                             self.pre_atten,
                                             self.pre_atten,
                                             h = self.num_heads,
                                             scope="h2p_atten"
        )

    def _align_aggregate(self, s, a):
        concat = tf.concat((s,a), -1)
        mul = s * a
        sub = s - a
        return tf.concat((concat, mul, sub), -1)

    def attention_integration(self):
        tprint("build attention integration")
        self.P_ = ffn(self._align_aggregate(self.hout_pre, self.pre_atten), self.hidden_dim, act=tf.nn.tanh)
        self.H_ = ffn(self._align_aggregate(self.hout_hyp, self.hyp_atten), self.hidden_dim, act=tf.nn.tanh)

        self.P_ = mask(self.P_, self.sent1_mask)
        self.H_ = mask(self.H_, self.sent2_mask)

        self.PH_ = ffn(self._align_aggregate(self.hout_pre, self.p2h_atten), self.hidden_dim, act=tf.nn.tanh)
        self.HP_ = ffn(self._align_aggregate(self.hout_hyp, self.h2p_atten), self.hidden_dim, act=tf.nn.tanh)

        self.PH_ = mask(self.PH_, self.sent1_mask)
        self.HP_ = mask(self.HP_, self.sent2_mask)

        self.P = tf.concat([self.P_, self.PH_], -1)
        self.H = tf.concat([self.H_, self.HP_], -1)

        if self.dropout:
            self.P = tf.nn.dropout(self.P, self.keep_rate)
            self.H = tf.nn.dropout(self.H, self.keep_rate)

    def interaction(self):
        tprint("build rnn")
        self.rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=self.hidden_dim)
        self.p_outputs, self.p_state = tf.nn.dynamic_rnn(self.rnn_cell, self.P,
                                               sequence_length=self.sent1_len,
                                               dtype=tf.float32)

        self.h_outputs, self.h_state = tf.nn.dynamic_rnn(self.rnn_cell, self.H,
                                               sequence_length=self.sent2_len,
                                               initial_state=self.p_state,
                                               dtype=tf.float32)

        self.ph = tf.concat((self.p_state, self.h_state), 1)

    def classify(self):
        tprint("build classifier")
        self.clf = highway_network(self.ph, 3, [tf.nn.sigmoid] * 3, "clf")
        self.y = tf.layers.dense(self.clf, 3)

    def loss(self):
        tprint("build loss")
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                                                  logits=self.y))
        if self.l2:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            self.loss += l2_loss * 9e-5


    def optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        if self.clip:
            self.tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.tvars), self.clip_value)
            self.train_op = self.optimizer.apply_gradients(zip(grads, self.tvars))
        else:
            self.train_op = self.optimizer.minimize(loss)

    def metrics(self):
        self.predictlabel = tf.argmax(self.y, axis=1)
        self.correctlabel = tf.cast(tf.equal(self.predictlabel, self.labels), dtype=tf.float32)
        self.correctnumber = tf.reduce_sum(self.correctlabel)
        self.correctPred = tf.reduce_mean(self.correctlabel)

    def initialize(self):
        tprint("initializing")
        self.saver = tf.train.Saver()

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        if self.retrain:
            init = tf.global_variables_initializer()
            self.sess.run(init)
        else:
            self.saver.restore(self.sess, os.path.join(self.model_path, self.name))

        tprint("done")

    def build(self):
        tprint("building graph")
        BST = time()
        self.dataprovider()
        self.embedding()
        self.encoding()
        self.attention()
        self.attention_integration()
        self.interaction()
        self.classify()
        self.loss()
        self.optimizer()
        self.metrics()
        tprint(f"finish build graph. take {time()-BST} seconds.")
        self.para_num = sum([np.prod(v.get_shape()) for v in self.tvars]).value
        tprint(f"parameters num: {self.para_num}")
        self.initialize()

    def _set_feed_dict(self, feed_dict, samples):
        feed_dict["premise:0"] = [j["sentence1"] for j in samples]
        feed_dict["hypothesis:0"] = [j["sentence2"] for j in samples]
        feed_dict["labels:0"] = [j["label"] for j in samples]
        feed_dict["antonym1:0"] = [j["antonym11"] for j in samples]
        feed_dict["antonym2:0"] = [j["antonym2"] for j in samples]
        feed_dict["exact1to2:0"] = [j["exact1to2"] for j in samples]
        feed_dict["exact2to1:0"] = [j["exact2to1"] for j in samples]
        feed_dict["synonym1:0"] = [j["synonym1"] for j in samples]
        feed_dict["synonym2:0"] = [j["synonym2"] for j in samples]

        #pos
        if self.pos:
            feed_dict["pos1:0"] = [j["pos1"] for j in samples]
            feed_dict["pos2:0"] = [j["pos2"] for j in samples]

        #char
        if self.char:
            feed_dict["sent1char:0"] = [j["sent1char"] for j in samples]
            feed_dict["sent2char:0"] = [j["sent2char"] for j in samples]

        return feed_dict


    def run(self, init=None, e=1, train=False, name="", printnum=500):
        for epoch in range(e):
            total_loss = 0.
            batch_number = 0
            total_pred = 0.  # total_pred for one epoch
            local_pred = 0.
            local_loss = 0.

            # init_trainset
            if not self.use_placeholder:
                init(self.sess)
            while True:
                try:
                    feed_dict = {}
                    if train:
                        run_op = (self.train_op, self.loss, self.correctPred)
                    else:
                        run_op = (self.loss, self.correctPred)
                        feed_dict[self.keep_rate] = 1.0

                    if self.use_placeholder:
                        samples = self.dataset.get_minibatch()
                        feed_dict = self._set_feed_dict(feed_dict, samples)

                    *_, loss_value, pred = self.sess.run(run_op, feed_dict)

                    total_loss += loss_value
                    local_loss += loss_value
                    total_pred += pred
                    local_pred += pred
                    batch_number += 1

                    if batch_number % printnum == 0:
                        tprint(f"{name}> average_loss:{local_loss/printnum}, local_accuracy:{local_pred/printnum}")
                        local_pred = 0.
                        local_loss = 0.
                except tf.errors.OutOfRangeError:
                    break

            tprint(f"{name}> total_loss:{total_loss/batch_number}, total_accuracy:{total_pred/batch_number}"
)

