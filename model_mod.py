import json
import argparse
import os
import torch
from dataloader import Dataloader
from jointBERT import JointBERT

parser = argparse.ArgumentParser(description="Train a model.")
parser.add_argument('--config_path',
                    help='path to config file',
                    default='config/crosswoz_all_base.json')

args = parser.parse_args()
config = json.load(open(args.config_path))
model_dir = config['model_dir']
data_dir = config['data_dir']
DEVICE = torch.device('cuda:0')
intent_vocab = json.load(open(os.path.join(data_dir, 'intent_vocab.json')))
dataloader = Dataloader(intent_vocab=intent_vocab,
                        pretrained_weights=config['model']['pretrained_weights'])
model = JointBERT(config['model'], DEVICE, dataloader.intent_dim)
pretrained_dict = torch.load(os.path.join(model_dir, 'pytorch_model.bin'), DEVICE)
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                   k in model_dict and 'intent_classifier' not in k and 'intent_hidden' not in k and 'intent_loss_fct' not in k}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
print(model.parameters())
