import os
import pickle
import argparse
from tqdm import tqdm
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
from torchtext import datasets
from torch.autograd import Variable

from util import prepare_output_dir


class Batch:
    
    def __init__(self, premise=None, hypothesis=None, label=None):
        if premise is not None:
            self.premise = premise
            self.batch_size = premise.shape[1]
        self.hypothesis = hypothesis
        self.label = label


def get_onehot_grad(model, batch):
    criterion = nn.CrossEntropyLoss()
    extracted_grads = {}
    
    def extract_grad_hook(name):
        def hook(grad):
            extracted_grads[name] = grad
        return hook
    
    batch.hypothesis.volatile = False
    model.eval()
    output = model(batch, embed_grad_hook=extract_grad_hook('embed'))
    label = torch.max(output, 1)[1]
    loss = criterion(output, label)
    loss.backward()
    embed_grad = extracted_grads['embed']
    embed = model.embed(batch.hypothesis)
    length = batch.hypothesis.shape[0]
    onehot_grad = embed.view(-1) * embed_grad.view(-1)
    onehot_grad = onehot_grad.view(length, batch.batch_size, -1).sum(-1)
    return onehot_grad


def real_length(x):
    # length of vector without padding
    if isinstance(x, Variable):
        return sum(x.data.cpu().numpy() != 1)
    else:
        return sum(x.cpu().numpy() != 1)
    
def remove_one(model, batch, n_beams, indices, removed_indices, max_beam_size):
    n_examples = len(n_beams)
    onehot_grad = get_onehot_grad(model, batch).data.cpu().numpy()
        
    start = 0
    new_hypo = []
    new_prem = []
    new_n_beams = []
    new_indices = []
    new_removed_indices = []
    
    # batch, length
    prem = batch.premise.transpose(0, 1)
    hypo = batch.hypothesis.transpose(0, 1)
    onehot_grad = onehot_grad.T
    hypo_lengths = [real_length(x) for x in hypo]
    
    for example_idx in range(n_examples):
        if n_beams[example_idx] == 0:
            new_n_beams.append(0)
            continue
        
        coordinates = []
        for i in range(start, start + n_beams[example_idx]):
            if hypo_lengths[i] <= 1:
                continue
            order = np.argsort(- onehot_grad[i][:hypo_lengths[i]])
            coordinates += [(i, j) for j in order[:max_beam_size]]
            
        if len(coordinates) == 0:
            new_n_beams.append(0)
            start += n_beams[example_idx]
            continue
        
        coordinates = np.asarray(coordinates)
        scores = onehot_grad[coordinates[:, 0], coordinates[:, 1]]
        scores = sorted(zip(coordinates, scores), key=lambda x: -x[1])
        coordinates = [x for x, _ in scores[:max_beam_size]]
        
        assert all(j < hypo_lengths[i] for i, j in coordinates)

        cnt = 0
        for i, j in coordinates:
            h = []
            if j > 0:
                h.append(hypo[i][:j])
            if j + 1 < hypo[i].shape[0]:
                h.append(hypo[i][j + 1:])
            if len(h) > 0:
                new_hypo.append(torch.cat(h, 0))
                new_prem.append(prem[i])
                new_removed_indices.append(removed_indices[i] + [indices[i][j]])
                new_indices.append(indices[i][:j] + indices[i][j+1:])
                cnt += 1
        new_n_beams.append(cnt)
        start += n_beams[example_idx]
    
    batch = Batch(torch.stack(new_prem, 1), torch.stack(new_hypo, 1))
    return batch, new_n_beams, new_indices, new_removed_indices

def get_rawr(model, batch, target=None, max_beam_size=5):
    batch = Batch(batch.premise, batch.hypothesis, batch.label)
    n_examples = batch.batch_size
    n_beams = [1 for _ in range(n_examples)] # current number of beams for each example
    indices = [list(range(real_length(x))) for x in batch.hypothesis.transpose(0, 1)]
    removed_indices = [[] for _ in range(n_examples)]
    
    final_hypothesis = [[x.data] for x in batch.hypothesis.transpose(0, 1)]
    final_removed = [[[]] for _ in range(n_examples)]
    final_length = [real_length(x) for x in batch.hypothesis.transpose(0, 1)]
    
    while True:
        max_beam_size = min(batch.hypothesis.shape[0], 5)
        batch, n_beams, indices, removed_indices = remove_one(
                model, batch, n_beams,  indices, removed_indices, max_beam_size)
        prediction = torch.max(model(batch), 1)[1].data.cpu().numpy()
        
        start = 0
        new_hypo, new_prem, new_indices, new_removed = [], [], [], []
        # batch, length
        prem = batch.premise.transpose(0, 1)
        hypo = batch.hypothesis.transpose(0, 1)
        for example_idx in range(n_examples):
            beam_size = 0
            for i in range(start, start + n_beams[example_idx]):
                if prediction[i] == target[example_idx]:
                    new_length = real_length(hypo[i])
                    if new_length == final_length[example_idx]:
                        final_hypothesis[example_idx].append(hypo[i].data)
                        final_removed[example_idx].append(removed_indices[i])
                    elif new_length < final_length[example_idx]:
                        final_hypothesis[example_idx] = [hypo[i].data]
                        final_removed[example_idx] = [removed_indices[i]]
                        final_length[example_idx] = new_length
                    if new_length == 1:
                        beam_size = 0
                        break
                    else:
                        beam_size += 1
                        new_hypo.append(hypo[i])
                        new_prem.append(prem[i])
                        new_indices.append(indices[i])
                        new_removed.append(removed_indices[i])
            start += n_beams[example_idx]
            n_beams[example_idx] = beam_size
        
        if len(new_hypo) == 0:
            break
        
        batch = Batch(torch.stack(new_prem, 1), torch.stack(new_hypo, 1))
        indices = new_indices
        removed_indices = new_removed
    return final_hypothesis, final_removed

def remove_pad(x):
    return x[:real_length(x)]

def process(model, batches):
    checkpoint = []

    # batch = next(iter(dev_iter))
    for batch_i, batch in enumerate(tqdm(batches)):
        if batch_i >= len(batches):
            break
        # if batch_i > 5:
        #     break
        n_examples = batch.batch_size
        output = F.softmax(model(batch), 1)
        target_scores, target = torch.max(output, 1)
        target = target.data.cpu().numpy()
        target_scores = target_scores.data.cpu().numpy()
        
        reduced_hypothesis, removed_indices = get_rawr(model, batch,
                target=target, max_beam_size=5)
        model.eval()
        for i in range(n_examples):
            checkpoint.append([])
            for j, hypo in enumerate(reduced_hypothesis[i]):
                test_input = Batch(
                        batch.premise[:, i].unsqueeze(1),
                        Variable(hypo.unsqueeze(1)).cuda())
                output = F.softmax(model(test_input), 1)
                pred_scores, pred = torch.max(output, 1)
                pred = pred.data.cpu().numpy()[0]
                pred_scores = pred_scores.data.cpu().numpy()[0]
                checkpoint[-1].append(
                        {'premise': remove_pad(batch.premise[:, i].data.cpu()),
                         'original_hypothesis': remove_pad(batch.hypothesis[:, i].data.cpu()),
                         'reduced_hypothesis': remove_pad(hypo.cpu()),
                         'label': batch.label[i].data.cpu().numpy()[0],
                         'original_prediction': target[i],
                         'original_score': target_scores[i],
                         'reduced_prediction': pred,
                         'reduced_score': pred_scores,
                         'removed_indices': removed_indices[i][j]})

    return checkpoint


def main():
    from args import args
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', required=True)
    parser.add_argument('--model', required=True)
    _args = parser.parse_args()
    args.fold = _args.fold
    args.load_model_dir = _args.model

    out_dir = prepare_output_dir(args, 'results')
    print('Generating [{}] rawr data from [{}].'.format(args.fold, args.load_model_dir))
    print(out_dir)

    file_dir = os.path.join(out_dir, '{}.pkl'.format(args.fold))
    print('Saving to {}'.format(file_dir))

    input_field = data.Field(lower=args.lower)
    output_field = data.Field(sequential=False)
    train, dev, test = datasets.SNLI.splits(
            input_field, output_field, root=args.data_root)
    input_field.build_vocab(train, dev, test)
    output_field.build_vocab(train)
    input_field.vocab.vectors = torch.load(args.vector_cache)
    
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
            (train, dev, test), batch_size=300, device=args.gpu)

    config = args
    config.n_embd = len(input_field.vocab)
    config.d_out = len(output_field.vocab)
    config.n_cells = config.n_layers
    if config.birnn:
        config.n_cells *= 2
        
    model = torch.load(args.load_model_dir, map_location=lambda storage, location: storage.cuda(args.gpu))
    iters = {'train': train_iter, 'dev': dev_iter}
    reduced = process(model, iters[args.fold])
    checkpoint = {'data': reduced,
                  'input_vocab': input_field.vocab.itos,
                  'output_vocab': output_field.vocab.itos}

    print(sum(len(x[0]['reduced_hypothesis']) for x in reduced) / len(reduced))

    with open(file_dir, 'wb') as f:
        pickle.dump(checkpoint, f)

if __name__ == '__main__':
    main()
