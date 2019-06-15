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
from typing import List, Tuple

from pytorch_pretrained_bert import BertTokenizer
import torch
from torch import Tensor    # pylint: disable=no-name-in-module
from torchtext.data import Dataset, Field, TabularDataset
from torchtext.vocab import Vocab


#############
# constants #
#############
# BERT_TOK_PATH = 'bert-large-cased'
BERT_TOK_PATH = '../input/bertpt/bertpt/bert-large-cased/bert-large-cased-vocab.txt'


#########
# types #
#########
class BertVocab(Vocab):
    """
    vocabulary for BERT fields
    """
    def __init__(self, vocab_path: str = BERT_TOK_PATH):
        super().__init__(Counter(), specials=[])
        # BERT tokenizer
        self.bert_tok = BertTokenizer.from_pretrained(vocab_path, do_lower_case=False)
        self.stoi = self.bert_tok.vocab
        self.stoi['<pad>'] = self.bert_tok.vocab['[PAD]']
        self.stoi['<unk>'] = self.bert_tok.vocab['[UNK]']

    def tokenize(self, text: str, do_preproc: bool = False) -> List[str]:
        """
        tokenizer
        Args:
            text:  text to tokenize
            do_preproc:  whether do preproc or not
        Returns:
            list of tokens
        """
        tokens = self.bert_tok.tokenize(text)
        return BertField.preproc(tokens) if do_preproc else tokens


class BertField(Field):
    """
    text field processed by BERT
    """
    def __init__(self, bert_vocab_path: str = BERT_TOK_PATH):
        self.vocab = BertVocab(bert_vocab_path)
        super().__init__(use_vocab=True, tokenize=self.vocab.tokenize, batch_first=True,
                         preprocessing=BertField.preproc)

    @classmethod
    def preproc(cls, tokens: List[str]) -> List[str]:
        """
        add '[CLS]' token at first, add '[SEP]' token at last
        Args:
            tokens:  token list
        Returns:
            preprocessed token list
        """
        logging.debug('tokens: %s', tokens)
        return ['[CLS]', ] + tokens[:510] + ['[SEP]', ]

    def to_tensor(self, tokens: List[str]) -> Tensor:
        """
        make tensor from tokens
        Args:
            tokens:  token list
        Returns:
            tensor
        """
        nums = [self.vocab.stoi[token] for token in tokens]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    # pylint: disable=no-member
        return torch.tensor(nums, device=device)    # pylint: disable=not-callable


class IntField(Field):
    """
    integer field
    """
    def __init__(self, is_target: bool = False):
        super().__init__(dtype=torch.long, use_vocab=False, sequential=False, is_target=is_target,    # pylint: disable=no-member
                         batch_first=True, preprocessing=IntField.preproc)

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


class BoolField(Field):
    """
    boolean field
    """
    def __init__(self, is_target: bool = False):
        super().__init__(dtype=torch.float32, use_vocab=False, sequential=False,    # pylint: disable=no-member
                         is_target=is_target, batch_first=True, preprocessing=BoolField.preproc)

    @classmethod
    def preproc(cls, val: str) -> bool:
        """
        preprocessor before numericalizing
        Args:
            val:  string value
        Returns:
            boolean value
        """
        prob = float(val)
        assert 0.0 <= prob <= 1.0
        return 1.0 if prob >= 0.5 else 0.0


class FloatField(Field):
    """
    float field
    """
    def __init__(self, is_target: bool = False):
        super().__init__(dtype=torch.float32, use_vocab=False, sequential=False,    # pylint: disable=no-member
                         batch_first=True, is_target=is_target, preprocessing=FloatField.preproc)

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
                         batch_first=True, preprocessing=DateTimeField.preproc)

    @classmethod
    def preproc(cls, val: str) -> float:
        """
        preprocessor before numericalizing
        Args:
            val:  datetime value
        Returns:
            float value
        """
        return datetime.strptime(val[:19], '%Y-%m-%d %H:%M:%S').timestamp() if val else 0.0


#############
# constants #
#############
RATING_FIELD = Field(sequential=False, batch_first=True)

TRAIN_FIELDS = [
    ('id', IntField()),
    ('target', BoolField(is_target=True)),
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
def _load(path: str, fields: List[Tuple[str, Field]]) -> Dataset:
    """
    load dataset
    Args:
        path:  file path
    Returns:
        dataset object
    """
    pkl_path = f'{path}.pkl'
    if os.path.exists(pkl_path) and os.path.getmtime(path) < os.path.getmtime(pkl_path):
        return Dataset(pickle.load(open(pkl_path, 'rb')), fields)
    tbl_ds = TabularDataset(path, format='csv', skip_header=True, fields=fields)
    pickle.dump(tbl_ds.examples, open(pkl_path, 'wb'))
    return tbl_ds


def load_train(path: str) -> Dataset:
    """
    load train dataset
    Args:
        path:  file path
    Returns:
        dataset object
    """
    return _load(path, TRAIN_FIELDS)


def load_test(path: str) -> Dataset:
    """
    load test dataset
    Args:
        path:  file path
    Returns:
        dataset object
    """
    return _load(path, TEST_FIELDS)
