# Author   : Orange
# Coding   : Utf-8
# @Time    : 2021/10/18 6:59 下午
# @File    : main.py
import argparse
from functools import partial

import numpy as np
# import torch
from fastNLP import Vocabulary
from fastNLP.io.loader import CSVLoader
from torch.utils.data import Dataset, DataLoader

from model import CNN_Text
from utils import MyDataset, collate_fn, args4textcnn
from train import *

args = args4textcnn()

train_path = 'train_orange.txt'
test_path = 'test_orange.txt'
dataset_loader = CSVLoader(headers=None, sep='\t')
testset_loader = CSVLoader(headers=None, sep='\t')

"""
print(tr_data)

+-------------------------------------------+-------+
| text                                      | label |
+-------------------------------------------+-------+
| 就是你那那是租房子是吗对你是中介还是个...        | 0     |
| 你好你好你好这样的我们这边有客户想要租...        | 0     |
| 你好你好你好请问是夏先生吗对对对我看到...        | 0     |
| 你好你好你那个和看一看房子要租掉吗对是...        | 0     |
| 你好你好我你那边有房子出租是吗对是几楼...        | 0     |
| 对你好你好我在网上看到你那个信息你现在...        | 0     |
| 你好你好我看我你在网上有个二手房的信息...        | 0     |
| 行我问你是不是刘光你是不是有房子出租对...        | 0     |
| 你好你好我看你那个网上有房子是吧我想看...        | 0     |
| 你好你好哪里您好您您这边是刚才你打过电...        | 0     |
| 你好请问那边那个两那房子现在还有吗有的...        | 0     |
| 对你好你好你好我想看一下咱那个八角南里...        | 0     |
| ...                                       | ...   |
+-------------------------------------------+-------+
"""
tr_data = dataset_loader._load(train_path)
te_data = testset_loader._load(test_path)


# 这里我们是用的是 char-level 你也可以选用你喜欢的分词器 比如 jieba
def tokenizer(sent):
	return [token for token in sent]


tr_data.apply_field(tokenizer, field_name='text', new_field_name='token')
te_data.apply_field(tokenizer, field_name='text', new_field_name='token')
tr_data.apply_field(int, field_name='label', new_field_name='target')
te_data.apply_field(int, field_name='label', new_field_name='target')

vocab = Vocabulary()
target = Vocabulary(padding=None, unknown=None)

vocab.from_dataset(tr_data, field_name='token', no_create_entry_dataset=[te_data])
target.from_dataset(tr_data, field_name='target', no_create_entry_dataset=[te_data])

vocab.index_dataset(tr_data, field_name='token', new_field_name='ids')
vocab.index_dataset(te_data, field_name='token', new_field_name='ids')
np.save('my_file.npy', vocab.word2idx)

args.embed_num = len(vocab.word2idx)
args.num_class = len(target.word2idx)
args.padding_idx = vocab.padding_idx


assert len(tr_data.field_arrays['ids'].content) == len(
	tr_data.field_arrays['target'].content), f"trainset : target don't match ids"
assert len(te_data.field_arrays['ids'].content) == len(
	te_data.field_arrays['target'].content), f"testset  : target don't match ids"

print(type(tr_data.field_arrays['ids'].content,))

trainSet = MyDataset(tr_data.field_arrays['ids'].content, tr_data.field_arrays['target'].content)
testSet = MyDataset(te_data.field_arrays['ids'].content, te_data.field_arrays['target'].content)

func = partial(collate_fn, padding_idx=vocab.padding_idx, min_length=max(args.kernel_sizes))
train_iter = DataLoader(trainSet, batch_size=16, shuffle=False, collate_fn=func)
test_iter = DataLoader(testSet, batch_size=16, shuffle=True, collate_fn=func)

# 用以下方式可以访问pad 和 unknown 对应的索引
# print(vocab.padding_idx)
# print(vocab.unknown_idx)

model = CNN_Text(args)
trainer(model, args, train_iter, test_iter)



