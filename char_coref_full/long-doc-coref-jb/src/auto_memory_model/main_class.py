import argparse
import os
from os import path
import hashlib
import logging
from collections import OrderedDict
import sys
import pdb

sys.path.insert(1, '/content/drive/MyDrive/NLP_CW/LH/long-doc-coref_edit/src')
from auto_memory_model.experiment import Experiment
from mention_model.utils import get_mention_model_name

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class main:
  def __init__(self, new_args):
     self.args = {'base_data_dir': '../data/', 'base_model_dir': '../models', 'dataset': 'litbank_person_only', 
                 'conll_scorer': '../reference-coreference-scorer/scorer.pl', 'model_size': 'large', 'doc_enc': 'overlap', 'pretrained_bert_dir': '../resources', 
                 'max_segment_len': 512, 'max_span_width': 20, 'ment_emb': 'attn', 'top_span_ratio': 0.3, 'mem_type': 'learned', 'num_cells': 20, 'mlp_size': 3000, 
                 'mlp_depth': 1, 'entity_rep': 'wt_avg', 'emb_size': 20, 'cross_val_split': 3, 'new_ent_wt': 2.0, 'num_train_docs': None, 'max_training_segments': 5,
                 'sample_invalid': 0.25, 'dropout_rate': 0.3, 'label_smoothing_wt': 0.0, 'max_epochs': 1, 'seed': 0, 'init_lr': 0.0002, 'no_singletons': False, 
                 'eval': False, 'slurm_id': None, 'eval_model': 'paper_model', 
                 'mention_model': 'ment_litbank_person_only_width_20_mlp_3000_model_large_emb_attn_type_spanbert_enc_overlap_segment_512', 
                 'delete': True, 'trainer': 'James', 'crossval': True}
  
  for new_arg in new_args.keys:
    self.args[new_arg] = new_args[new_arg]
  


  if self.args.dataset == 'litbank':
      self.args.max_span_width = 20
  elif self.args.dataset == 'ontonotes':
      self.args.max_span_width = 30
  else:
      self.args.max_span_width = 20

  # Get model directory name
  opt_dict = OrderedDict()
  # Only include important options in hash computation
  imp_opts = ['model_size', 'max_segment_len',  # Encoder params
              'ment_emb', "doc_enc", 'max_span_width', 'top_span_ratio', # Mention model
              'mem_type', 'num_cells', 'entity_rep', 'mlp_size', 'mlp_depth', 'emb_size',  # Memory params
              'dropout_rate', 'seed', 'init_lr',
              "new_ent_wt", 'sample_invalid',  'max_training_segments', 'label_smoothing_wt',  # weights & sampling
              'dataset', 'num_train_docs', 'cross_val_split',   # Dataset params
              ]
  for key, val in vars(args).items():
      if key in imp_opts:
          opt_dict[key] = val

  str_repr = str(opt_dict.items())
  hash_idx = hashlib.md5(str_repr.encode("utf-8")).hexdigest()
  if not self.args.eval:
    self.model_name = "coref_" + str(hash_idx)
  else:
    self.model_name = self.args.eval_model
  if self.args.trainer:
    self.model_dir = path.join(self.args.base_model_dir,'../',self.args.trainer,'models', self.model_name)
  else:
    self.model_dir = path.join(self.args.base_model_dir, self.model_name)
  self.args.model_dir = self.model_dir
  print(self.model_dir)
  self.best_model_dir = path.join(self.model_dir, 'best_models')
  self.args.best_model_dir = self.best_model_dir
  if not path.exists(self.model_dir):
      os.makedirs(self.model_dir)
  if not path.exists(self.best_model_dir):
      os.makedirs(self.best_model_dir)

  if (self.args.dataset == 'litbank') | (self.args.dataset == 'litbank_person_only'):
      self.args.data_dir = path.join(self.args.base_data_dir, f'{self.args.dataset}/{self.args.doc_enc}/{self.args.cross_val_split}')
      self.args.conll_data_dir = path.join(self.args.base_data_dir, f'{self.args.dataset}/conll/{self.args.cross_val_split}')
  else:
      self.args.data_dir = path.join(self.args.base_data_dir, f'{self.args.dataset}/{self.args.doc_enc}')
      self.args.conll_data_dir = path.join(self.args.base_data_dir, f'{self.args.dataset}/conll')

  print(self.args.data_dir)



  # Get mention model name

  '''
  if not args.eval:
    args.pretrained_mention_model = path.join(
        path.join(args.base_model_dir, get_mention_model_name(args)), "best_models/model.pth")
    print(args.pretrained_mention_model)
  else:
    args.pretrained_mention_model = '/content/drive/MyDrive/NLP_CW/LH/long-doc-coref_edit/models/paper_mention/best_models/model.pth'
    print(args.pretrained_mention_model)
  '''
  self.args.pretrained_mention_model = path.join(self.args.base_model_dir, self.args.mention_model,"best_models/model.pth" )

  print(self.args.pretrained_mention_model)

  # Log directory for Tensorflow Summary
  self.log_dir = path.join(self.model_dir, "logs")
  if not path.exists(self.log_dir):
      os.makedirs(self.log_dir)

  self.config_file = path.join(self.model_dir, 'config')
  with open(self.config_file, 'w') as f:
      for key, val in opt_dict.items():
          logging.info('%s: %s' % (key, val))
          f.write('%s: %s\n' % (key, val))
  if self.args.crossval == True:
    return Experiment(self.args, **vars(self.args))
  else:
    Experiment(self.args, **vars(self.args))

