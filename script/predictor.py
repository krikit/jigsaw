"""
predictor
__author__ = 'krikit (krikit@naver.com)'
__copyright__ = 'No copyright, just copyleft!'
"""


###########
# imports #
###########
from argparse import Namespace
import logging
from typing import TextIO

from pytorch_pretrained_bert.modeling import BertForSequenceClassification
import torch
from torchtext.data import BucketIterator
from tqdm import tqdm

import dataset
from dataset import BertField


#########
# types #
#########
class Predictor:
    """
    predictor class
    """
    def __init__(self, cfg: Namespace):
        """
        Args:
            model_path:  model path
        """
        self.cfg = cfg
        bert_path = cfg.bert_path if cfg.bert_path else 'bert-base-cased'
        self.model = BertForSequenceClassification.from_pretrained(bert_path, num_labels=2)
        state_dict = torch.load(cfg.model_path, map_location=lambda storage, loc: storage)
        stripped = {}
        for key, val in state_dict.items():
            if key.startswith('module.'):    # I don't know why all keys have "module." prefix
                key = key[7:]
            stripped[key] = val
        self.model.load_state_dict(stripped)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
        self.bert_field = BertField()

    def predict_test(self, path: str, batch_size: int, fout: TextIO):
        """
        predict from test dataset
        Args:
            path:  test dataset path
            fout:  output file
        """
        data = dataset.load_test(path)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    # pylint: disable=no-member
        tst_itr = BucketIterator(data, device=device, batch_size=batch_size, shuffle=False,
                                 train=False, sort_within_batch=True,
                                 sort_key=lambda exam: -len(exam.comment_text))
        print('id,prediction', file=fout)
        for step, batch in enumerate(tqdm(tst_itr, mininterval=1, ncols=100), start=1):
            if step % 1000 == 0:
                logging.info('%dk-th step..')
            with torch.no_grad():
                outputs = self.model(batch.comment_text)
                for id_, output in zip(batch.id, torch.softmax(outputs, dim=1)):    # pylint: disable=no-member
                    print(f'{id_},{output[1].item()}', file=fout)

    def predict(self, text: str) -> float:
        """
        predict a single text
        Args:
            text:  input text
        Returns:
            toxicity score
        """
        tokens = self.bert_field.tokenize(text, do_preproc=True)
        tensor = self.bert_field.to_tensor(tokens)
        with torch.no_grad():
            output = torch.softmax(self.model(tensor.unsqueeze(0)), dim=1)    # pylint: disable=no-member
        return output.squeeze(0)[1].item()
