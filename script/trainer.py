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
from typing import Dict, List

from pytorch_pretrained_bert.modeling import BertForSequenceClassification
import torch
from torch import nn, optim
from torchtext.data import Dataset, BucketIterator
from tqdm import tqdm

from parallel import DataParallelModel, DataParallelCriterion

from dataset import RATING_FIELD


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
        if len(self.vld_itr) < 100:
            self.log_step = 10
        elif len(self.vld_itr) < 1000:
            self.log_step = 100

        bert_path = cfg.bert_path if cfg.bert_path else 'bert-base-cased'
        self.model = BertForSequenceClassification.from_pretrained(bert_path, num_labels=2)
        pos_weight = (len([exam for exam in self.train.examples if exam.target < 0.5])
                      / len([exam for exam in self.train.examples if exam.target >= 0.5]))
        pos_wgt_tensor = torch.tensor([1.0, pos_weight], device=self.device)    # pylint: disable=not-callable
        self.criterion = nn.CrossEntropyLoss(weight=pos_wgt_tensor)
        if torch.cuda.is_available():
            self.model = DataParallelModel(self.model.cuda())
            self.criterion = DataParallelCriterion(self.criterion)
        self.optimizer = optim.Adam(self.model.parameters(), cfg.learning_rate)

    def run(self):
        """
        do train
        """
        max_f_score = -9e10
        max_epoch = -1
        for epoch in range(self.cfg.epoch):
            train_loss = self._train_epoch(epoch)
            metrics = self._evaluate(epoch)
            max_f_score_str = f' < {max_f_score:.2f}'
            if metrics['f_score'] > max_f_score:
                max_f_score_str = ' is max'
                max_f_score = metrics['f_score']
                max_epoch = epoch
                torch.save(self.model.state_dict(), self.cfg.model_path)
            logging.info('EPOCH[%d]: train loss: %.6f, valid loss: %.6f, acc: %.2f,' \
                         ' F: %.2f%s', epoch, train_loss, metrics['loss'],
                         metrics['accuracy'], metrics['f_score'], max_f_score_str)
            if (epoch - max_epoch) >= self.cfg.patience:
                logging.info('early stopping...')
                break
        logging.info('epoch: %d, f-score: %.2f', max_epoch, max_f_score)

    def _train_epoch(self, epoch: int) -> float:
        """
        train single epoch
        Args:
            epoch:  epoch number
        Returns:
            average loss
        """
        self.model.train()
        progress = tqdm(self.trn_itr, f'EPOCH[{epoch}]', mininterval=1, ncols=100)
        losses = []
        for step, batch in enumerate(progress, start=1):
            outputs = self.model(batch.comment_text)
            # output of model wrapped with DataParallelModel is a list of outputs from each GPU
            # make input of DataParallelCriterion as a list of tuples
            if isinstance(self.model, DataParallelModel):
                loss = self.criterion([(output, ) for output in outputs], batch.target)
            else:
                loss = self.criterion(outputs, batch.target)
            losses.append(loss.item())
            if step % self.log_step == 0:
                avg_loss = sum(losses) / len(losses)
                progress.set_description(f'EPOCH[{epoch}] ({avg_loss:.6f})')
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return sum(losses) / len(losses)

    def _evaluate(self, epoch: int) -> Dict[str, float]:
        """
        evaluate on validation data
        Args:
            epoch:  epoch number
        Returns:
            metrics
        """
        self.model.eval()
        progress = tqdm(self.vld_itr, f' EVAL[{epoch}]', mininterval=1, ncols=100)
        losses = []
        preds = []
        golds = []
        for step, batch in enumerate(progress, start=1):
            with torch.no_grad():
                outputs = self.model(batch.comment_text)
                if isinstance(self.model, DataParallelModel):
                    loss = self.criterion([(output, ) for output in outputs],
                                          batch.target)
                    for output in outputs:
                        preds.extend([(0 if o[0] < o[1] else 1) for o in output])
                else:
                    loss = self.criterion(outputs, batch.target)
                    preds.extend([(0 if output[0] < output[1] else 1) for output in outputs])
                losses.append(loss.item())
                golds.extend([gold.item() for gold in batch.target])
                if step % self.log_step == 0:
                    avg_loss = sum(losses) / len(losses)
                    progress.set_description(f' EVAL[{epoch}] ({avg_loss:.6f})')
        metrics = self._get_metrics(preds, golds)
        metrics['loss'] = sum(losses) / len(losses)
        return metrics

    @classmethod
    def _get_metrics(cls, preds: List[float], golds: List[float]) -> Dict[str, float]:
        """
        get metric values
        Args:
            preds:  predictions
            golds:  gold standards
        Returns:
            metric
        """
        assert len(preds) == len(golds)
        true_pos = 0
        false_pos = 0
        false_neg = 0
        true_neg = 0
        for pred, gold in zip(preds, golds):
            if pred >= 0.5:
                if gold >= 0.5:
                    true_pos += 1
                else:
                    false_pos += 1
            else:
                if gold >= 0.5:
                    false_neg += 1
                else:
                    true_neg += 1
        accuracy = (true_pos + true_neg) / (true_pos + false_pos + false_neg + true_neg)
        precision = 0.0
        if (true_pos + false_pos) > 0:
            precision = true_pos / (true_pos + false_pos)
        recall = 0.0
        if (true_pos + false_neg) > 0:
            recall = true_pos / (true_pos + false_neg)
        f_score = 0.0
        if (precision + recall) > 0.0:
            f_score = 2.0 * precision * recall / (precision + recall)
        return {
            'accuracy': 100.0 * accuracy,
            'precision': 100.0 * precision,
            'recall': 100.0 * recall,
            'f_score': 100.0 * f_score,
        }
