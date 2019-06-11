# -*- coding: utf-8 -*-


"""
models
__author__ = 'krikit (krikit@naver.com)'
__copyright__ = 'No copyright, just copyleft!'
"""


###########
# imports #
###########
import logging

import torch
from torch import nn
from torch.nn import functional as F
from pytorch_pretrained_bert import BertModel


#########
# types #
#########
class ToxicityModel(nn.Module):
    """
    toxicity classification model
    """
    def __init__(self):
        super().__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-cased')
        bert_dim = self.bert_model.config.hidden_size
        self.hidden1 = nn.Linear(bert_dim, bert_dim)
        self.hidden2 = nn.Linear(bert_dim, bert_dim // 2)
        self.hidden3 = nn.Linear(bert_dim // 2, 1)

    def forward(self, batch):    # pylint: disable=arguments-differ
        """
        Args:
            batch:  input of batch
        """
        logging.debug('inside(input): %s', batch.size())
        # BERT 모델은 두번째 차원(index: 1)이 배치 차원이다. (batch sencond)
        bert_out, _ = self.bert_model(batch, output_all_encoded_layers=False)
        classifier = F.selu(bert_out[0])
        hdn1_out = F.selu(self.hidden1(F.dropout(classifier)))
        hdn2_out = F.selu(self.hidden2(F.dropout(classifier + hdn1_out)))
        hdn3_out = torch.sigmoid(self.hidden3(F.dropout(hdn2_out)))    # pylint: disable=no-member
        logging.debug('inside(output): %s', hdn3_out.size())
        # data parallel을 위해 배치 차원을 다시 복원해야 한다.
        return hdn3_out.transpose(0, 1)
