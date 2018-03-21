import os
import argparse
from datetime import datetime

args  =  argparse.Namespace()

args.batch_size = 128
args.birnn = True
args.d_embed = 100
args.d_hidden = 300
args.d_proj = 300
args.dev_every = 1000
args.dp_ratio = 0.2
args.epochs = 50
args.fix_emb = True
args.gpu = 0
args.log_every = 50
args.lower = True
args.lr = 5e-04
args.n_layers = 1
args.projection = True,
args.save_every = 1000
args.save_path = 'results',
args.word_vectors = 'glove.6B.100d'

args.start_ent = 400 # number of regular batches before entropy
args.gamma = 2e-05
args.ent_lr = 5e-04 # default: 1e-03
args.n_ent_per_reg = 5
args.n_reg_per_ent = 5
args.n_report = 100
args.n_eval = 1000

args.root_dir = open('root').readline().strip() 
args.data_root = os.path.join(args.root_dir, '.data/')
args.vector_cache = os.path.join(args.root_dir, '.vector_cache/input_vectors.pt')

now = datetime.now()
args.timestamp = '{}-{}-{}-{}-{}'.format(now.month, now.day, now.hour, now.minute, now.second)
args.run_dir = os.path.join(args.root_dir, args.timestamp)
args.log_dir = os.path.join(args.run_dir, 'output.log')
