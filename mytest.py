import torch
from transformers import BertTokenizer
# sentence = "我想要办理种畜禽生产经营许可-种畜禽生产经营许可"
# sentence = "矿业勘查探矿采矿许可-开采矿产资源审批-采矿权变更登记"
# sentence = "我要查看装修厂房备案的指南"
# sentence = "保存渔业船舶船员证书核发怎样办理"
# sentence = "在线评价设置医疗机构许可和执业登记、变更-设置戒毒医疗机构或者医疗机构从事戒毒治疗业务的许可-医疗机构从事戒毒治疗业务许可需要什么材料"
# sentence = "我想取消收藏计量标准考评员查询的指南"
# sentence = "我要查询医疗、药品、医疗器械、保健食品、特殊医学用途配方食品和农药、兽药广告许可-农药广告许可的办事指南"
# sentence = "评价高校毕业生就业服务如何预约"
# sentence = "关于个人出入境，港澳台居民旅居证明签发的步骤和具体要求有什么？"


# tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
tokenizer = BertTokenizer.from_pretrained("voidful/albert_chinese_base")


# print(word_mask_tensor)
from jointBERT import JointBERT
from dataloader import Dataloader
import json
import os
import numpy as np

config = json.load(open("config/crosswoz_all.json"))
DEVICE = config['DEVICE']
data_dir = config['data_dir']
output_dir = config['output_dir']
intent_vocab = json.load(open(os.path.join(data_dir, 'intent_vocab.json')))
dataloader = Dataloader(intent_vocab=intent_vocab,
                            pretrained_weights=config['model']['pretrained_weights'])
model = JointBERT(config['model'], DEVICE, dataloader.intent_dim)
model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin'), DEVICE))
model.to(DEVICE)
model.eval()

import time
while True:
    sentence = input("请输入问题：")
    start_time = time.time()

    if sentence == '':
        break
    tokens = tokenizer.tokenize(sentence)
    words = ['[CLS]'] + tokens + ['[SEP]']
    # print(tokens)
    word_seq_tensor = torch.zeros((1, 256), dtype=torch.long)
    word_mask_tensor = torch.zeros((1, 256), dtype=torch.long)
    indexed_tokens = tokenizer.convert_tokens_to_ids(words)
    word_seq_tensor[0 , : len(words)] = torch.LongTensor(indexed_tokens)
    word_mask_tensor[0 , : len(words)] = torch.LongTensor([1] * len(words))
    intent_logits = model.forward(word_seq_tensor,word_mask_tensor,None,None,None)
    intent_logits = intent_logits[0].detach().cpu().numpy()
    index = np.argmax(intent_logits)
    intent = dataloader.id2intent[index]
    end_time = time.time()
    print(intent)
    print(end_time - start_time)