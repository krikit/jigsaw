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
import random

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

        device = 'cpu'
        batch_size = self.cfg.batch_size
        if torch.cuda.is_available():
            device = 'cuda'
            batch_size *= torch.cuda.device_count()
        self.trn_itr = BucketIterator(self.train, device=device, batch_size=batch_size,
                                      shuffle=True, sort_within_batch=True,
                                      sort_key=lambda exam: -len(exam.comment_text))
        self.vld_itr = BucketIterator(self.valid, device=device, batch_size=batch_size,
                                      shuffle=False, sort_within_batch=True,
                                      sort_key=lambda exam: -len(exam.comment_text))
        self.log_step = 1000
        if len(self.trn_itr) < 100:
            self.log_step = 10
        elif len(self.trn_itr) < 1000:
            self.log_step = 100

        self.model = ToxicityModel()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.9, ]))    # pylint: disable=not-callable
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
                # self.model.save(self.cfg.model_out)
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
        trn_itr = self._train_iter()
        self.model.train()
        log_step = self.log_step
        progress = tqdm(trn_itr, f'EPOCH[{epoch}]', mininterval=1, ncols=100)
        losses = []
        for step, batch in enumerate(progress, start=1):
            outputs = self.model(batch.comment_text)
            # DataParallelModel로 감싼 모델의 출력은 각 GPU로부터 나온 출력들의 리스트이다.
            # DataParallelCriterion의 입력은 튜플의 리스트이므로 리스트를 풀어서 각 엘리먼트를 튜플로 감싸준다.
            # ToxicityModel의 출력은 0~1의 1차원 값이므로 [batch x 1]인 출력의 첫번째 차원을 squeeze로 없앤다.
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

    def _train_iter(self):
        """
        한번의 epoch에 적용할 training example의 iterator
        원래는 전체 train 데이터를 사용해야 하나, output class imbalance 문제로 (pos:neg = 1:9),
        pos:neg 비율을 1:1로 랜덤하게 negative example을 추출한 셋을 사용한다.
        """
        pos_exams = [exam for exam in self.train.examples if exam.target > 0.0]
        neg_exams = [exam for exam in self.train.examples if exam.target <= 0.0]
        if len(neg_exams) <= len(pos_exams):
            return self.trn_itr
        neg_exams = random.sample(neg_exams, len(pos_exams))
        all_exams = pos_exams + neg_exams
        random.shuffle(all_exams)
        dataset = Dataset(all_exams, self.train.fields)
        batch_size = self.cfg.batch_size
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
            batch_size *= torch.cuda.device_count()
        return BucketIterator(dataset, device=device, batch_size=batch_size, shuffle=False,
                              sort_within_batch=True, sort_key=lambda exam: -len(exam.comment_text))

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
