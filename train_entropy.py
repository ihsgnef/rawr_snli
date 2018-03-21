import os
import sys
import glob
import pickle
import argparse
import numpy as np
import logging
import scipy
import scipy.stats
import itertools
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
from torchtext import datasets
from torch.autograd import Variable

from model import SNLIClassifier
from distribution import entropy
from rawr import Batch


class EntIterator:
    
    def __init__(self, data, batch_size, device=-1, evaluation=False):
        self.batch_size = batch_size
        self.device = device
        self.evaluation = evaluation
        self.batches = self.create_batches(data, evaluation)
        self.batch_idx = 0

    def __len__(self):
        return len(self.batches)

    def init_epoch(self):
        self.batch_idx = 0
        if not self.evaluation:
            random.shuffle(self.batches)
    
    def create_batches(self, data, evaluation):
        data = list(itertools.chain(*data))
        if not evaluation:
            data = sorted(data, key=lambda x: x['premise'].shape[0])
        idx = 0
        batches = []
        while True:
            if idx >= len(data):
                break
            batch = data[idx: idx + self.batch_size]
            hypos, prems = [], []
            prem_len = max(len(x['premise']) for x in batch)
            hypo_len = max(len(x['reduced_hypothesis']) for x in batch)
            for i, x in enumerate(batch):
                prem = x['premise']
                if prem.shape[0] < prem_len:
                    zeros = torch.LongTensor(np.zeros(prem_len - prem.shape[0]))
                    prem = torch.cat([prem, zeros], 0)
                prems.append(prem)
                
                hypo = x['reduced_hypothesis']
                if hypo.shape[0] < hypo_len:
                    zeros = torch.LongTensor(np.zeros(hypo_len - hypo.shape[0]))
                    hypo = torch.cat([hypo, zeros], 0)
                hypos.append(hypo)
            if len(hypos) > 0 and len(prems) > 0:
                hypos = torch.stack(hypos, 1)
                prems = torch.stack(prems, 1)
                batches.append((prems, hypos))
            idx += self.batch_size
        return batches
    
    def next(self):
        if self.batch_idx % len(self.batches) == 0:
            if not self.evaluation:
                random.shuffle(self.batches)
            self.batch_idx = 0
        prems, hypos = self.batches[self.batch_idx]
        batch = Batch(Variable(prems), Variable(hypos))
        if self.device != -1:
            batch.premise = batch.premise.cuda()
            batch.hypothesis = batch.premise.cuda()
        self.batch_idx += 1
        return batch

def main():
    from args import args
    os.makedirs(args.run_dir, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args.load_model_dir = parser.parse_args().model
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(args.run_dir, 'output.log'))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    log.addHandler(fh)
    log.addHandler(ch)
    log.info('===== {} ====='.format(args.timestamp))

    with open(os.path.join(args.run_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    input_field = data.Field(lower=args.lower)
    output_field = data.Field(sequential=False)
    train_reg, dev_reg, test = datasets.SNLI.splits(
            input_field, output_field, root=args.data_root)
    input_field.build_vocab(train_reg, dev_reg, test)
    output_field.build_vocab(train_reg)
    input_field.vocab.vectors = torch.load(args.vector_cache)
    
    train_reg_iter, dev_reg_iter = data.BucketIterator.splits(
            (train_reg, dev_reg), batch_size=300, device=args.gpu)

    with open('pkls/rawr.train.pkl', 'rb') as f:
        train_ent = pickle.load(f)['data']
    with open('pkls/rawr.dev.pkl', 'rb') as f:
        dev_ent = pickle.load(f)['data']

    train_ent_iter = EntIterator(
            train_ent, batch_size=300, device=args.gpu)
    dev_ent_iter = EntIterator(
            dev_ent, batch_size=300, device=args.gpu,
            evaluation=True)

    model = torch.load(args.load_model_dir, 
            map_location=lambda storage, location: storage.cuda(args.gpu))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ent_optimizer = torch.optim.Adam(model.parameters(), lr=args.ent_lr)


    ''' initial evaluation '''
    model.eval()
    dev_reg_iter.init_epoch()
    n_dev_correct, dev_loss = 0, 0
    total = 0
    for dev_batch_idx, dev_batch in enumerate(dev_reg_iter):
        total += dev_batch.hypothesis.shape[1]
        answer = model(dev_batch)
        n_dev_correct += (torch.max(answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
        dev_acc = 100. * n_dev_correct / total
    log.info('dev acc {:.4f}'.format(dev_acc))

    dev_ent_iter.init_epoch()
    avg_entropy = 0
    total = 0
    for dev_batch_idx in range(len(dev_ent_iter)):
        dev_batch = dev_ent_iter.next()
        total += dev_batch.hypothesis.shape[1]
        output = model(dev_batch)
        ent = entropy(F.softmax(output, 1)).sum()
        avg_entropy += ent.data.cpu().numpy()[0]
    log.info('dev entropy {:.4f}'.format(avg_entropy / total))

    best_dev_acc = -1
    train_ent_iter.init_epoch()
    for epoch in range(args.epochs):
        epoch_loss = []
        epoch_entropy = []
        n_reg = 0
        n_ent = 0
        train_reg_iter.init_epoch()
        for i_reg, batch in enumerate(train_reg_iter):
            model.train()
            output = model(batch)
            loss = criterion(output, batch.label)
            epoch_loss.append(loss.data.cpu().numpy()[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_reg += 1

            if n_reg % args.n_reg_per_ent == 0:
                model.train()
                for j in range(args.n_ent_per_reg):
                    batch_ent = train_ent_iter.next()
                    output = model(batch_ent)
                    ent = entropy(F.softmax(output, 1)).sum()
                    epoch_entropy.append(ent.data.cpu().numpy()[0])
                    loss = - args.gamma * ent
                    ent_optimizer.zero_grad()
                    loss.backward()
                    ent_optimizer.step()

            if n_reg % args.n_report == 0:
                if len(epoch_loss) != 0 and len(epoch_entropy) != 0:
                    log.info('epoch [{}] batch [{}, {}] loss [{:.4f}] entropy [{:.4f}]'.format(
                        epoch, i_reg, n_ent, sum(epoch_loss) / len(epoch_loss),
                        sum(epoch_entropy) / len(epoch_entropy)))

            if n_reg % args.n_eval == 0:
                model.eval()
                dev_reg_iter.init_epoch()
                n_dev_correct, dev_loss = 0, 0
                total = 0
                for dev_batch_idx, dev_batch in enumerate(dev_reg_iter):
                    total += dev_batch.hypothesis.shape[1]
                    answer = model(dev_batch)
                    n_dev_correct += (torch.max(answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
                    dev_acc = 100. * n_dev_correct / total
                log.info('dev acc {:.4f}'.format(dev_acc))

                dev_ent_iter.init_epoch()
                avg_entropy = 0
                total = 0
                for dev_batch_idx in range(len(dev_ent_iter)):
                    dev_batch = dev_ent_iter.next()
                    total += dev_batch.hypothesis.shape[1]
                    output = model(dev_batch)
                    ent = entropy(F.softmax(output, 1)).sum()
                    avg_entropy += ent.data.cpu().numpy()[0]
                log.info('dev entropy {:.4f}'.format(avg_entropy / total))

                if dev_acc > best_dev_acc:
                    snapshot_path = os.path.join(args.run_dir, 'best_model.pt')
                    torch.save(model, snapshot_path)
                    best_dev_acc = dev_acc
                    log.info('save best model {}'.format(best_dev_acc))

        snapshot_path = os.path.join(args.run_dir, 'checkpoint_epoch_{}.pt'.format(epoch))
        torch.save(model, snapshot_path)
        log.info('save model {}'.format(snapshot_path))

if __name__ == '__main__':
    main()
