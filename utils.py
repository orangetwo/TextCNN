# Author   : Orange
# Coding   : Utf-8
# @Time    : 2021/10/18 10:00 下午
# @File    : utils.py
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import argparse
import pickle


def args4textcnn():
	parser = argparse.ArgumentParser()
	args = parser.parse_args()
	args.dropout = 0.1
	args.static = False
	args.kernel_sizes = [3, 4, 5]
	args.embed_num = -1
	args.embed_dim = 128
	args.padding_idx = -1
	args.num_class = -1
	args.kernel_num = 32
	args.epochs = 6
	args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	return args


def save_obj(obj, name):
	with open(name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
	with open(name + '.pkl', 'rb') as f:
		return pickle.load(f)


class MyDataset(Dataset):
	def __init__(self, texts, labels):
		self.data = [(text, label) for text, label in zip(texts, labels)]

	def __len__(self):
		return len(self.data)

	def __getitem__(self, item):
		return self.data[item]


def collate_fn(examples, padding_idx=0, max_length=None, min_length=None):
	"""
    process the batch samples
    """

	labels = torch.LongTensor([sample[1] for sample in examples])
	if min_length:
		features = [sample[0] + (min_length - len(sample[0]))*[padding_idx] for sample in examples]
	else:
		features = [sample[0] for sample in examples]

	# length = max([len(sample[0]) for sample in examples])
	if max_length is None:
		inputs = [torch.LongTensor(sample) for sample in features]
	else:
		assert isinstance(max_length, int), f"attention the type of max length!!!"
		inputs = [torch.LongTensor(sample[:max_length]) for sample in features]

	out = pad_sequence(inputs, padding_value=padding_idx, batch_first=True)

	return out, labels


if __name__ == '__main__':

	examples = [([2,3,4], 1), ([5,4,5,6], 0)]
	x, y = collate_fn(examples, min_length=5)
	print(x, y)


