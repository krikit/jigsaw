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

from torch import nn, optim
from torchtext.data import Dataset, BucketIterator
from tqdm import tqdm

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

        device = 'cpu' if self.cfg.gpu_num < 0 else f'cuda:{self.cfg.gpu_num}'
        self.trn_itr = BucketIterator(self.train, batch_size=self.cfg.batch_size, device=device,
                                      shuffle=True, sort_within_batch=True,
                                      sort_key=lambda exam: -len(exam.comment_text))
        self.vld_itr = BucketIterator(self.valid, batch_size=self.cfg.batch_size, device=device,
                                      sort_within_batch=True,
                                      sort_key=lambda exam: -len(exam.comment_text))
        self.log_step = 1000
        if len(self.trn_itr) < 100:
            self.log_step = 10
        elif len(self.trn_itr) < 1000:
            self.log_step = 100

        self.model = ToxicityModel()
        if cfg.gpu_num >= 0:
            self.model.cuda(cfg.gpu_num)
        self.criterion = nn.MSELoss()
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
            logging.info('EPOCH[%d]: train loss: %.6f, valid loss: %.6f%s', epoch, train_loss,
                         valid_loss, min_loss_str)
            if (epoch - min_epoch) > self.cfg.patience:
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
            output = self.model(batch)
            output.requires_grad_()
            loss = self.criterion(output.squeeze(1), batch.target)
            losses.append(loss.item())
            if step % self.log_step == 0:
                last_loss = sum(losses[-self.log_step:]) / self.log_step
                tqdm.write(f'loss[{epoch}|{step / 1000}]: {last_loss}')
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
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
            output = self.model(batch)
            loss = self.criterion(output.squeeze(1), batch.target)
            losses.append(loss.item())
        return sum(losses) / len(losses)
