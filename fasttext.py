# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class fasttext(nn.Module):
    def __init__(self, class_3, num_embeddings):
        super(fasttext, self).__init__()
        self.hidden_size = 768
        self.embedding = nn.Embedding(num_embeddings, self.hidden_size, padding_idx= 0)
        self.embedding_ngram2 = nn.Embedding(250499, self.hidden_size)
        self.embedding_ngram3 = nn.Embedding(250499, self.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(config.embed * 3, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):

        out_word = self.embedding(x[0])
        out_bigram = self.embedding_ngram2(x[2])
        out_trigram = self.embedding_ngram3(x[3])
        out = torch.cat((out_word, out_bigram, out_trigram), -1)

        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out