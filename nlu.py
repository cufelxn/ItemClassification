import os
import zipfile
import json

import requests
import torch

# from convlab2.util.file_util import cached_path
from tqdm import tqdm

from IntentNLU import NLU
from dataloader import Dataloader
from jointBERT import JointBERT
from postprocess import recover_intent
from preprocess import preprocess


class BERTNLU(NLU):
    def __init__(self, config_file='crosswoz_all_base.json',model_file='bert_crosswoz.zip'):
        # assert mode == 'usr' or mode == 'sys' or mode == 'all'
        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config/{}'.format(config_file))
        config = json.load(open(config_file))
        DEVICE = config['DEVICE']
        root_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(root_dir, config['data_dir'])
        output_dir = os.path.join(root_dir, config['output_dir'])

        # if not os.path.exists(os.path.join(data_dir, 'intent_vocab.json')):
        #     preprocess()

        intent_vocab = json.load(open(os.path.join(data_dir, 'intent_vocab.json')))
        dataloader = Dataloader(intent_vocab=intent_vocab,
                                pretrained_weights=config['model']['pretrained_weights'])

        print('intent num:', len(intent_vocab))

        best_model_path = os.path.join(output_dir, 'pytorch_model.bin')
        # if not os.path.exists(best_model_path):
        #     if not os.path.exists(output_dir):
        #         os.makedirs(output_dir)
        #     print('Load from model_file param')
        #     # archive_file = cached_path(model_file)
        #     archive_file = os.path.join(output_dir, model_file)
        #     archive = zipfile.ZipFile(archive_file, 'r')
        #     archive.extractall(root_dir)
        #     archive.close()
        print('Load from', best_model_path)
        model = JointBERT(config['model'], DEVICE, dataloader.intent_dim)
        model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin'), DEVICE))
        model.to(DEVICE)
        model.eval()

        self.model = model
        self.dataloader = dataloader
        print("BERTNLU loaded")

    def predict(self, utterance, context=list()):
        ori_word_seq = self.dataloader.tokenizer.tokenize(utterance)
        # ori_tag_seq = ['O'] * len(ori_word_seq)
        context_size = 1
        context_seq = self.dataloader.tokenizer.encode('[CLS] ' + ' [SEP] '.join(context[-context_size:]))
        intents = []
        da = {}

        word_seq, new2ori = ori_word_seq, None
        batch_data = [[ori_word_seq, intents, da, context_seq,
                       new2ori, word_seq, self.dataloader.seq_intent2id(intents)]]

        pad_batch = self.dataloader.pad_batch(batch_data)
        pad_batch = tuple(t.to(self.model.device) for t in pad_batch)
        word_seq_tensor, intent_tensor, word_mask_tensor, context_seq_tensor, context_mask_tensor = pad_batch
        context_seq_tensor, context_mask_tensor = None, None
        intent_logits = self.model.forward(word_seq_tensor, word_mask_tensor,
                                           context_seq_tensor=context_seq_tensor,
                                           context_mask_tensor=context_mask_tensor)[0]
        intent_logits = intent_logits.detach().cpu().numpy()
        intent = recover_intent(self.dataloader, intent_logits[0],
                                batch_data[0][0], batch_data[0][-4])
        return intent


if __name__ == '__main__':
    nlu = BERTNLU(config_file='crosswoz_all_base.json')
    print('人才绿卡A卡的办理条件',nlu.predict('人才绿卡A卡的办理条件'))

    with open('data/intent_test.json', 'r') as f:
        intent_dict = json.load(f)

    N_total = len(intent_dict)
    N_error = 0
    # intent_path = "http://localhost:8000/intent?text={}"

    for i in intent_dict.values():
        # if i['module'] == 'FAQ':
        #     continue
        first_utterance = i['content']
        intent_res = nlu.predict(first_utterance)
        intent_class = intent_res[0][0]
        if intent_class.lower() != i['module'].lower():
            N_error += 1
            print(first_utterance + " label:" + i['module'] + " predict:" + intent_class)

    print( 1 - N_error / N_total)
