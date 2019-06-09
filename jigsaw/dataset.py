# -*- coding: utf-8 -*-


"""
dataset
__author__ = 'krikit (krikit@naver.com)'
__copyright__ = 'No copyright, just copyleft!'
"""


###########
# imports #
###########
from collections import Counter
from datetime import datetime
import logging
import os
import pickle
from typing import List

from pytorch_pretrained_bert import BertTokenizer
import torch
from torchtext.data import Dataset, Field, TabularDataset
from torchtext.vocab import Vocab


#############
# constants #
#############
_BERT_TOK = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)    # BERT 토크나이저


#########
# types #
#########
class BertVocab(Vocab):
    """
    vocabulary for BERT fields
    """
    def __init__(self):
        super().__init__(Counter(), specials=[])
        self.stoi = _BERT_TOK.vocab
        self.stoi['<pad>'] = _BERT_TOK.vocab['[PAD]']
        self.stoi['<unk>'] = _BERT_TOK.vocab['[UNK]']


class BertField(Field):
    """
    text field processed by BERT
    """
    def __init__(self):
        super().__init__(use_vocab=True, tokenize=_BERT_TOK.tokenize,
                         preprocessing=BertField.preproc)
        self.vocab = BertVocab()

    @classmethod
    def preproc(cls, tokens: List[str]) -> List[str]:
        """
        맨 앞에 [CLS] 토큰을, 맨 뒤에 [SEP] 토큰을 붙여준다.
        Args:
            tokens:  토큰 리스트
        Returns:
            토큰 리스트
        """
        logging.debug('tokens: %s', tokens)
        return ['[CLS]', ] + tokens[:510] + ['[SEP]', ]


class IntField(Field):
    """
    integer field
    """
    def __init__(self, is_target: bool = False):
        super().__init__(dtype=torch.long, use_vocab=False, sequential=False, is_target=is_target,    # pylint: disable=no-member
                         preprocessing=IntField.preproc)

    @classmethod
    def preproc(cls, val: str) -> int:
        """
        preprocessor before numericalizing
        Args:
            val:  string value
        Returns:
            integer value
        """
        return int(float(val)) if val else 0


class FloatField(Field):
    """
    float field
    """
    def __init__(self, is_target: bool = False):
        super().__init__(dtype=torch.float32, use_vocab=False, sequential=False,    # pylint: disable=no-member
                         is_target=is_target, preprocessing=FloatField.preproc)

    @classmethod
    def preproc(cls, val: str) -> float:
        """
        preprocessor before numericalizing
        Args:
            val:  string value
        Returns:
            float value
        """
        return float(val) if val else 0.0


class DateTimeField(Field):
    """
    datetime field
    """
    def __init__(self):
        super().__init__(dtype=torch.float64, use_vocab=False, sequential=False,    # pylint: disable=no-member
                         preprocessing=DateTimeField.preproc)

    @classmethod
    def preproc(cls, val: str) -> float:
        """
        preprocessor before numericalizing
        Args:
            val:  datetime value
        Returns:
            float value
        """
        return datetime.strptime(val + '00', '%Y-%m-%d %H:%M:%S.%f%z').timestamp() if val else 0.0


#############
# constants #
#############
RATING_FIELD = Field(sequential=False)

TRAIN_FIELDS = [
    ('id', IntField()),
    ('target', FloatField(is_target=True)),
    ('comment_text', BertField()),
    ('severe_toxicity', FloatField()),
    ('obscene', FloatField()),
    ('identity_attack', FloatField()),
    ('insult', FloatField()),
    ('threat', FloatField()),
    ('asian', FloatField()),
    ('atheist', FloatField()),
    ('bisexual', FloatField()),
    ('black', FloatField()),
    ('buddhist', FloatField()),
    ('christian', FloatField()),
    ('female', FloatField()),
    ('heterosexual', FloatField()),
    ('hindu', FloatField()),
    ('homosexual_gay_or_lesbian', FloatField()),
    ('intellectual_or_learning_disability', FloatField()),
    ('jewish', FloatField()),
    ('latino', FloatField()),
    ('male', FloatField()),
    ('muslim', FloatField()),
    ('other_disability', FloatField()),
    ('other_gender', FloatField()),
    ('other_race_or_ethnicity', FloatField()),
    ('other_religion', FloatField()),
    ('other_sexual_orientation', FloatField()),
    ('physical_disability', FloatField()),
    ('psychiatric_or_mental_illness', FloatField()),
    ('transgender', FloatField()),
    ('white', FloatField()),
    ('created_date', DateTimeField()),
    ('publication_id', IntField()),
    ('parent_id', IntField()),
    ('article_id', IntField()),
    ('rating', RATING_FIELD),
    ('funny', FloatField()),
    ('wow', FloatField()),
    ('sad', FloatField()),
    ('likes', FloatField()),
    ('disagree', FloatField()),
    ('sexual_explicit', FloatField()),
    ('identity_annotator_count', IntField()),
    ('toxicity_annotator_count', IntField()),
]

TEST_FIELDS = [
    ('id', IntField()),
    ('comment_text', BertField()),
]


#############
# functions #
#############
def load(path: str) -> Dataset:
    """
    load dataset
    Args:
        path:  file path
    Returns:
        dataset object
    """
    pkl_path = f'{path}.pkl'
    if os.path.exists(pkl_path) and os.path.getmtime(path) < os.path.getmtime(pkl_path):
        return Dataset(pickle.load(open(pkl_path, 'rb')), TRAIN_FIELDS)
    tbl_ds = TabularDataset(path, format='csv', skip_header=True, fields=TRAIN_FIELDS)
    pickle.dump(tbl_ds.examples, open(pkl_path, 'wb'))
    return tbl_ds