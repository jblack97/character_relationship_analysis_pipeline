# -*- coding: utf-8 -*-
"""crossval2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oJS_lAAv_xDx-4lMnMc5QEkDuiIgcE40
"""

from google.colab import drive
import itertools
drive.mount('/content/drive')

!pip install transformers
!pip install -r requirements.txt
!git clone https://github.com/conll/reference-coreference-scorers
!git clone https://github.com/dbamman/lrec2020-coref

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/NLP_CW/LH/long-doc-coref_edit/src

from auto_memory_model.main_class import main_class

params = {'sample_invalid' : [.275, .3, .325] , 'new_ent_wt':[1.5, 2, 2.5], 'droupout':[.3, .35],  'ovr_loss_wts': [1, 1.5, 2]}
param_grid = list(itertools.product([2.75, 3, 3.25], [1.5, 2, 2.5],[.3, .35], [1, 1.5, 2]))
hyper_params = param_grid[len(param_grid)//4: 2*len(param_grid)//4]
default_params = {'dataset': 'litbank_person_only', 'mem_type' : 'learned', 'num_cells' : 20, 'top_span_ratio':0.3, 'max_span_width':20, 'max_epochs':25, 'dropout_rate' :0.3,
                  'sample_invalid':0.25, 'new_ent_wt':2.0, 'cross_val_split':3, 'max_training_segments':5, 'seed' :0, 'mention_model':'ment_litbank_person_only_width_20_mlp_3000_model_large_emb_attn_type_spanbert_enc_overlap_segment_512',
                  'trainer':'luke','eval':False, 'delete':True, 'crossval':True}
with open('/content/drive/MyDrive/NLP_CW/LH/long-doc-coref_edit/models/best_precision/perf.json') as json_file:
    data = json.load(json_file)
best_prec = data['dev']['precision']

def cross_val(param_names, hyper_params):
  for idx in range(len(hyper_params)):
    arguments = default_params
    for  i, name in enumerate(param_names):
      arguments[name] = hyper_params[idx][i]
    arguments['seed'] = i
    try:
      model = main_class(arguments)
      res = model.run()
    except:
      continue
    prec = res['dev']['precision']
    if (prec > best_prec):
      #json file overwritten by training script
      best_prec = prec
      with open('/content/drive/MyDrive/NLP_CW/LH/long-doc-coref_edit/models/best_precision/perf.json') as json_file:
        data = json.load(json_file)
      best_prec = data['dev']['precision']
      print('\n \n New Best Params:\n', arguments, '\n \n ')
      best_arguments = arguments
      best_res = res
  print('\n \n Best Params:\n', arguments, '\n \n ')
  print('\n \n Best Precision :\n', best_res['test'], '\n \n ')
  return best_arguments, best_res['dev'], best_res['test']

cross_val(params.keys(), hyper_params)