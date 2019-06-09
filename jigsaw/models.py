# -*- coding: utf-8 -*-


"""
models
__author__ = 'krikit (krikit@naver.com)'
__copyright__ = 'No copyright, just copyleft!'
"""


###########
# imports #
###########
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
        bert_out, _ = self.bert_model(batch.comment_text, output_all_encoded_layers=False)
        classifier = F.selu(bert_out[0])
        hdn1_out = F.selu(self.hidden1(F.dropout(classifier)))
        hdn2_out = F.selu(self.hidden2(F.dropout(classifier + hdn1_out)))
        hdn3_out = torch.sigmoid(self.hidden3(F.dropout(hdn2_out)))    # pylint: disable=no-member
        return hdn3_out
