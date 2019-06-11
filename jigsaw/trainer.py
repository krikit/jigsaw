# -*- coding: utf-8 -*-


"""
trainer
__author__ = 'krikit (krikit@naver.com)'
__copyright__ = 'No copyright, just copyleft!'
"""


###########
# imports #
###########
from argparse import Namespace
import logging

import torch
from torch import nn, optim
from torchtext.data import Dataset, BucketIterator
from tqdm import tqdm

from parallel import DataParallelModel, DataParallelCriterion

from dataset import RATING_FIELD
from models import ToxicityModel


#########
# types #
#########
class Trainer:
    """
    trainer class
    """
    def __init__(self, cfg: Namespace, data: Dataset):
        self.cfg = cfg
        self.train, self.valid = data.split(0.8)
        RATING_FIELD.build_vocab(self.train)

        device = 'cpu'
        batch_size = self.cfg.batch_size
        if torch.cuda.is_available():
            device = 'cuda'
            batch_size *= torch.cuda.device_count()
        self.trn_itr = BucketIterator(self.train, device=device, batch_size=batch_size,
                                      shuffle=True, sort_within_batch=True,
                                      sort_key=lambda exam: -len(exam.comment_text))
        self.vld_itr = BucketIterator(self.valid, device=device, batch_size=(batch_size // 2),
                                      shuffle=False, sort_within_batch=True,
                                      sort_key=lambda exam: -len(exam.comment_text))
        self.log_step = 1000
        if len(self.trn_itr) < 100:
            self.log_step = 10
        elif len(self.trn_itr) < 1000:
            self.log_step = 100

        self.model = ToxicityModel()
        self.criterion = nn.MSELoss()
        if torch.cuda.is_available():
            # 두번째 차원(index: 1)이 배치이므로 이를 지정해준다. (batch second)
            self.model = DataParallelModel(self.model.cuda(), dim=1)
            self.criterion = DataParallelCriterion(self.criterion)
        self.optimizer = optim.Adam(self.model.parameters(), 0.001)

    def run(self):
        """
        do train
        """
        min_loss = 9e10
        min_epoch = -1
        for epoch in range(self.cfg.epoch):
            train_loss = self._train_epoch(epoch)
            valid_loss = self._evaluate(epoch)
            min_loss_str = f' > {min_loss:.6f}'
            if valid_loss < min_loss:
                min_loss_str = ' is min'
                min_loss = valid_loss
                min_epoch = epoch
                torch.save(self.model.state_dict(), self.cfg.model_out)
            logging.info('EPOCH[%d]: train loss: %.6f, valid loss: %.6f%s', epoch, train_loss,
                         valid_loss, min_loss_str)
            if (epoch - min_epoch) >= self.cfg.patience:
                logging.info('early stopping...')
                break
        logging.info('epoch: %d, valid loss: %.6f', min_epoch, min_loss)

    def _train_epoch(self, epoch: int) -> float:
        """
        train single epoch
        Args:
            epoch:  epoch number
        Returns:
            average loss
        """
        self.model.train()
        losses = []
        for step, batch in enumerate(tqdm(self.trn_itr, f'EPOCH[{epoch}]', mininterval=1,
                                          ncols=100), start=1):
            logging.debug('outside(input): %s', batch.comment_text.size())
            outputs = self.model(batch.comment_text)
            # DataParallelModel로 감싼 모델의 출력은 각 GPU로부터 나온 출력들의 리스트이다.
            logging.debug('outside(output): %s', outputs[0].size())
            logging.debug('target: %s', batch.target.size())
            # DataParallelCriterion의 입력은 튜플의 리스트이므로 리스트를 풀어서 각 엘리먼트를 튜플로 감싸준다.
            # ToxicityModel의 출력은 0~1의 1차원 값이므로 [1 x batch]인 출력의 첫번째 차원을 squeeze로 없앤다.
            loss = self.criterion([(output.squeeze(0), ) for output in outputs], batch.target)
            losses.append(loss.item())
            if step % self.log_step == 0:
                last_loss = sum(losses[-self.log_step:]) / self.log_step
                tqdm.write(f'loss[{epoch}|{step / 1000}]: {last_loss}')
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return sum(losses) / len(losses)

    def _evaluate(self, epoch: int) -> float:
        """
        evaluate on validation data
        Args:
            epoch:  epoch number
        Returns:
            validation loss
        """
        self.model.eval()
        losses = []
        for batch in tqdm(self.vld_itr, f' EVAL[{epoch}]', mininterval=1, ncols=100):
            outputs = self.model(batch.comment_text)
            loss = self.criterion([(output.squeeze(0), ) for output in outputs], batch.target)
            losses.append(loss.item())
        return sum(losses) / len(losses)
