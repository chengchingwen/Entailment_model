#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from time import time

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from dataset import MultiNli
from util.util import tprint
from model.model import EntailModel

######parameters

keep_prob = 1
learning_rate = 0.00005 
batch_num = 64
max_len = None
filter_size = 3
num_heads = 8 #for transformer
hidden_dim = 300 #a dim reduction after highway network
char_emb_dim=8

##############################

tprint("start loading dataset")
mnli = MultiNli("../data/embedding/glove.txt.gz", "../data/multinli_0.9",
                max_len = max_len,
                batch=batch_num,
                train_epoch=1,
                dev_epoch=1,
                char_emb_dim=char_emb_dim,
                pad2=False
                #all_printable_char=True,
                #trainfile="multinli_0.9_train_5000.jsonl",
)

model = EntailModel("base", hidden_dim, num_heads, filter_size, mnli, learning_rate = learning_rate)
model.build()

for i in tqdm(range(1000)):
    tprint(f"train epoch: {i}")
    model.run(init=model.dataset.train, train=True, name="train")
    tprint(f"evaluate on dev_matched")
    model.run(init=model.dataset.dev_matched, name="matched")
    tprint(f"evaluate on dev_mismatched")
    model.run(init=model.dataset.dev_mismatched, name="mismatched")

tprint("done!")



