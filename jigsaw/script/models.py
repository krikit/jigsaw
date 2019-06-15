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


#############
# constants #
#############
# BERT_MODEL_PATH = 'bert-large-cased'
BERT_MODEL_PATH = '../input/bertpt/bertpt/bert-large-cased/bert-large-cased.tar.gz'




#########
# types #
#########
class ToxicityModel(nn.Module):
    """
    toxicity classification model
    """
    def __init__(self, model_path: str = BERT_MODEL_PATH):
        super().__init__()
        self.bert_model = BertModel.from_pretrained(model_path)
        bert_dim = self.bert_model.config.hidden_size
        self.hdn1_drop = nn.Dropout()
        self.hidden1 = nn.Linear(bert_dim, bert_dim)
        self.hdn2_drop = nn.Dropout()
        self.hidden2 = nn.Linear(bert_dim, bert_dim // 2)
        self.hdn3_drop = nn.Dropout()
        self.hidden3 = nn.Linear(bert_dim // 2, 1)

    def forward(self, batch):    # pylint: disable=arguments-differ
        """
        Args:
            batch:  input of batch
        """
        with torch.no_grad():    # use BERT as feature
            _, pooled_out = self.bert_model(batch, output_all_encoded_layers=False)
        hdn1_out = F.relu(self.hidden1(self.hdn1_drop(pooled_out)))
        hdn2_out = F.relu(self.hidden2(self.hdn2_drop(pooled_out + hdn1_out)))
        logit = self.hidden3(self.hdn3_drop(hdn2_out))
        return logit

    def train(self, mode: bool = True):
        """
        override train method
        """
        super().train(mode)
        self.bert_model.eval()    # use BERT as feature

    def load(self, path: str):
        """
        load model from file
        Args:
            path:  model path
        """
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        stripped = {}
        for key, val in state_dict.items():
            if key.startswith('module.'):    # I don't know why all keys have "module." prefix
                key = key[7:]
            stripped[key] = val
        self.load_state_dict(stripped)
        if torch.cuda.is_available():
            self.cuda()
