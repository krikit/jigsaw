# -*- coding: utf-8 -*-


"""
trainer
__author__ = 'krikit (krikit@naver.com)'
__copyright__ = 'No copyright, just copyleft!'
"""


###########
# imports #
###########
from typing import TextIO

import torch

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

    def predict_test(self, path: str, fout: TextIO):
        """
        테스트 데이터셋으로부터 예측을 하여 출력한다.
        Args:
            path:  test dataset path
            fout:  output file
        """

    def predict(self, text: str) -> float:
        """
        하나의 텍스트에 대한 예측값을 리턴한다.
        Args:
            text:  input text
        Returns:
            toxicity score
        """
        tokens = self.bert_field.txt2tok(text, do_preproc=True)
        tensor = self.bert_field.to_tensor(tokens)
        with torch.no_grad():
            output = self.model(tensor.unsqueeze(0))
        return output.squeeze(0).item()
