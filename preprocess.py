import json
import os
from collections import Counter
from transformers import BertTokenizer,AlbertTokenizer,AutoTokenizer


def read_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def preprocess():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(cur_dir, 'mydata')
    # processed_data_dir = os.path.join(cur_dir, 'processed_mydata_tiny')
    processed_data_dir = os.path.join(cur_dir, 'processed_mydata_electra')
    print(processed_data_dir)
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
        print('yes')
    data_key = ['train', 'dev', 'test']
    data = {}
    for key in data_key:
        data[key] = read_json(os.path.join(data_dir, key + '.json'))
        print('load {}, size {}'.format(key, len(data[key])))

    processed_data = {}
    all_intent = []

    context_size = 1

    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-electra-180g-small-discriminator")
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # tokenizer = AutoTokenizer.from_pretrained("voidful/albert_chinese_base")
    # tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    # tokenizer = AutoTokenizer.from_pretrained("voidful/albert_chinese_tiny")

    for key in data_key:
        processed_data[key] = []
        for no, sess in data[key].items():
            context = []
            # for i, turn in enumerate(sess['messages']):
            utterance = sess['content']
            tokens = tokenizer.tokenize(utterance)
            golden = []
            intents = [sess['intent']]
            golden.append(intents)
            processed_data[key].append([tokens, intents, golden, context[-context_size:]])
            all_intent += intents
            context.append(sess['content'])
        # 去重
        all_intent = [x[0] for x in dict(Counter(all_intent)).items()]
        print('loaded {}, size {}'.format(key, len(processed_data[key])))
        json.dump(processed_data[key],
                  open(os.path.join(processed_data_dir, '{}_data.json'.format(key)), 'w', encoding='utf-8'),
                  indent=2, ensure_ascii=False)

    print('sentence label num:', len(all_intent))
    print(all_intent)
    json.dump(all_intent, open(os.path.join(processed_data_dir, 'intent_vocab.json'), 'w', encoding='utf-8'), indent=2,
              ensure_ascii=False)


if __name__ == '__main__':
    preprocess()
