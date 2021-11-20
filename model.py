# Author   : Orange
# Coding   : Utf-8
# @Time    : 2021/10/18 10:23 下午
# @File    : model.py

import torch
import time
import torch.nn as nn
import torch.nn.functional as F

"""
args.droput
args.static
args.kernel_sizes
args.embed_num
args.embed_dim
args.padding_idx
args.num_class
args.kernel_num
"""


class CNN_Text(nn.Module):

	def __init__(self, agrs):  # vocab_size, embed_dim, num_class, kernel_num):
		super(CNN_Text, self).__init__()

		self.Ci = 1
		self.dropout = agrs.dropout
		self.static = agrs.static
		self.Kernel_sizes = agrs.kernel_sizes

		# idx = 1 对应 <pad>
		self.embed = nn.Embedding(agrs.embed_num, agrs.embed_dim,
		                          padding_idx=agrs.padding_idx)  # vocab_size, embed_dim, padding_idx=1)

		# self.embed.weight.requires_grad = False
		# self.embed = self.embed.from_pretrained(vectors, freeze=True)
		# self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])

		self.convs = nn.ModuleList(
			[nn.Conv2d(self.Ci, agrs.kernel_num, (K, agrs.embed_dim)) for K in self.Kernel_sizes])
		# 上面是个for循环，不好理解写成下面也是没问题的。
		# self.conv13 = nn.Conv2d(Ci, Co, (2, D))
		# self.conv14 = nn.Conv2d(Ci, Co, (3, D))
		# self.conv15 = nn.Conv2d(Ci, Co, (4, D))
		# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.dropout = nn.Dropout(self.dropout)
		self.fc1 = nn.Linear(len(self.Kernel_sizes) * agrs.kernel_num, agrs.num_class)

		if self.static:
			self.embed.weight.requires_grad = False

	def forward(self, x):
		x = self.embed(x)  # (N, W, D)

		x = x.unsqueeze(1)  # (N, Ci, W, D) batch_size, channels, sequence_length, embedding_dimension
		#
		x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

		'''
                最大池化也可以拆分理解
                x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
                x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
                x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
                x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
		x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

		x = torch.cat(x, 1)

		x = self.dropout(x)  # (N, len(Ks)*Co)
		logit = self.fc1(x)  # (N, C)
		return logit


if __name__ == '__main__':
	from utils import args4textcnn

	args = args4textcnn()

	args.embed_num = 50
	args.num_class = 2
	args.padding_idx = 1

	model = CNN_Text(args)
	print(model)