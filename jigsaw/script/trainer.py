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
        """
        Args:
            cfg:  configuration
            data:  train dataset
        """
        self.cfg = cfg
        self.train, self.valid = data.split(0.8)
        RATING_FIELD.build_vocab(self.train)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    # pylint: disable=no-member
        self.batch_size = cfg.batch_size
        if torch.cuda.is_available():
            self.batch_size *= torch.cuda.device_count()

        self.trn_itr = BucketIterator(self.train, device=self.device, batch_size=self.batch_size,
                                      shuffle=True, train=True, sort_within_batch=True,
                                      sort_key=lambda exam: -len(exam.comment_text))
        self.vld_itr = BucketIterator(self.valid, device=self.device, batch_size=self.batch_size,
                                      shuffle=False, train=False, sort_within_batch=True,
                                      sort_key=lambda exam: -len(exam.comment_text))
        self.log_step = 1000
        if len(self.trn_itr) < 100:
            self.log_step = 10
        elif len(self.trn_itr) < 1000:
            self.log_step = 100

        self.model = ToxicityModel()
        pos_weight = (len([exam for exam in self.train.examples if exam.target < 0.5])
                      / len([exam for exam in self.train.examples if exam.target >= 0.5]))
        pos_wgt_tensor = torch.tensor([pos_weight, ], device=self.device)    # pylint: disable=not-callable
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_wgt_tensor)
        if torch.cuda.is_available():
            self.model = DataParallelModel(self.model.cuda())
            self.criterion = DataParallelCriterion(self.criterion)
        self.optimizer = optim.Adam(self.model.parameters(), cfg.learning_rate)

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
        log_step = self.log_step
        progress = tqdm(self.trn_itr, f'EPOCH[{epoch}]', mininterval=1, ncols=100)
        losses = []
        for step, batch in enumerate(progress, start=1):
            outputs = self.model(batch.comment_text)
            # output of model wrapped with DataParallelModel is a list of outputs from each GPU
            # make input of DataParallelCriterion as a list of tuples
            # since output of ToxicityModel is a scalar value fro 0 to 1,
            #   squeeze first dim of output which size is [batch, 1]
            if isinstance(self.model, DataParallelModel):
                loss = self.criterion([(output.squeeze(1), ) for output in outputs], batch.target)
            else:
                loss = self.criterion(outputs.squeeze(1), batch.target)
            losses.append(loss.item())
            if step % log_step == 0:
                last_loss = sum(losses[-log_step:]) / log_step
                progress.set_description(f'EPOCH[{epoch}] ({last_loss:.6f})')
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
        log_step = self.log_step // 2
        progress = tqdm(self.vld_itr, f' EVAL[{epoch}]', mininterval=1, ncols=100)
        losses = []
        for step, batch in enumerate(progress, start=1):
            with torch.no_grad():
                outputs = self.model(batch.comment_text)
                if isinstance(self.model, DataParallelModel):
                    loss = self.criterion([(output.squeeze(1), ) for output in outputs],
                                          batch.target)
                else:
                    loss = self.criterion(outputs.squeeze(1), batch.target)
                losses.append(loss.item())
                if step % log_step == 0:
                    last_loss = sum(losses[-log_step:]) / log_step
                    progress.set_description(f' EVAL[{epoch}] ({last_loss:.6f})')
        return sum(losses) / len(losses)
