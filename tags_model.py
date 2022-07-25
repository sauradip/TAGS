# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.transformer import SnippetEmbedding
import yaml

with open("./config/anet.yaml", 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = yaml.load(tmp, Loader=yaml.FullLoader)



class TAGS(nn.Module):
    def __init__(self):
        super(TAGS, self).__init__()
        self.len_feat = config['model']['feat_dim']
        self.temporal_scale = config['model']['temporal_scale']
        self.num_classes = config['dataset']['num_classes']+1
        self.n_heads = config['model']['embedding_head']
        self.embedding = SnippetEmbedding(self.n_heads, self.len_feat, self.len_feat, self.len_feat, dropout=0.3)

        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feat, out_channels=self.num_classes, kernel_size=1,
            padding=0)
        )

        self.global_mask = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=1024, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3,padding=1),
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=256, out_channels=self.temporal_scale, kernel_size=1,stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, snip):

        ### Snippet Embedding Module ###
        snip = snip.permute(0,2,1)
        out = self.embedding(snip,snip,snip)
        out = out.permute(0,2,1)
        features = out
        ### Classifier Branch ###
        top_br = self.classifier(features)
        ### Global Segmentation Mask Branch ###
        bottom_br = self.global_mask(features)

        return top_br, bottom_br, features





