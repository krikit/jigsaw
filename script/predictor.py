"""
predictor
__author__ = 'krikit (krikit@naver.com)'
__copyright__ = 'No copyright, just copyleft!'
"""


###########
# imports #
###########
import logging
from typing import TextIO

import torch
from torchtext.data import BucketIterator
from tqdm import tqdm

import dataset
from dataset import BertField
from models import ToxicityModel


#########
# types #
#########
class Predictor:
    """
    predictor class
    """
    def __init__(self, model_path: str):
        """
        Args:
            model_path:  model path
        """
        self.model = ToxicityModel()
        self.model.load(model_path)
        self.model.eval()
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
                for id_, output in zip(batch.id, torch.sigmoid(outputs)):    # pylint: disable=no-member
                    print(f'{id_},{output.item()}', file=fout)

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
            output = torch.sigmoid(self.model(tensor.unsqueeze(0)))    # pylint: disable=no-member
        return output.squeeze(0).item()
