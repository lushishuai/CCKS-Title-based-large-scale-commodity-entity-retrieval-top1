import pandas as pd
import numpy as np
import json
from transformers import BertTokenizer, BertModel, BertConfig
BERT_BASE_MODEL = "/home/msqin/bert-base-chinese"
BERT_TINY_MODEL = "/home/msqin/bert-base-chinese"

def seq_padding(X, maxlen=None, padding_value=0, debug=False):
    L = [len(x) for x in X]
    if maxlen is None:
        maxlen = max(L)
    pad_X = np.array([
        np.concatenate([x, [padding_value] * (maxlen - len(x))]) if len(x) < maxlen else x[: maxlen] for x in X
    ])
    if debug:
        print("[!] before pading {}\n".format(X))
        print("[!] after pading {}\n".format(pad_X))
    return pad_X
def get_len(text_lens, max_len=510, min_len=30):
    """
    戒断过长文本你的长度，小于30不在戒断，大于30按比例戒断
    :param text_lens: 列表形式 data 字段中每个 predicate+object 的长度
    :param max_len: 最长长度
    :param min_len: 最段长度
    :return: 列表形式 戒断后每个 predicate+object 保留的长度
            如 input：[638, 10, 46, 9, 16, 22, 10, 9, 63, 6, 9, 11, 34, 10, 8, 6, 6]
             output：[267, 10, 36, 9, 16, 22, 10, 9, 42, 6, 9, 11, 31, 10, 8, 6, 6]
    """
    new_len = [min_len] * len(text_lens)
    sum_len = sum(text_lens)
    del_len = sum_len - max_len
    del_index = []
    for i, l in enumerate(text_lens):
        if l > min_len:
            del_index.append(i)
        else:
            new_len[i] = l
    del_sum = sum([text_lens[i] - min_len for i in del_index])
    for i in del_index:
        new_len[i] = text_lens[i] - int(((text_lens[i] - min_len) / del_sum) * del_len) - 1
    return new_len


def get_text(en_data, max_len=510, min_len=30):
    '''
    根据data字段数据生成描述文本，将 predicate项与object项相连，在将过长的依据规则戒断
    :param en_data: kb里面的每个实体的data数据
    :param max_len: 每个 predicate+object 的最大长度
    :param min_len: 每个 predicate+object 的最小长度
    :return: 每个实体的描述文本
    '''
    texts = []
    text = ''
    for data in en_data:
        # if data['object']:
        texts.append(str(data['predicate']) + '为' + str(data['object']) + '，')
            # texts.append(str(data['object']) + '，')
    text_lens = []
    for t in texts:
        text_lens.append(len(t))
    if sum(text_lens) < max_len:
        for t in texts:
            text = text + t
    else:
        new_text_lens = get_len(text_lens, max_len=max_len, min_len=min_len)
        for t, l in zip(texts, new_text_lens):
            text = text + t[:l]
    return text[:max_len]

def get_tokens_id(texts):
    tokenizer = BertTokenizer.from_pretrained(BERT_TINY_MODEL)
    tokens = tokenizer.batch_encode_plus(texts, add_special_tokens=True,max_length=256)
    return tokens['input_ids']
def train_to_token():

    texts = []
    text_ids = []
    with open('../data/train.txt') as f:
        for index, line in enumerate(f):
            line = json.loads(line)
            text = line['text']
            text_id = line['text_id']
            texts.append(text)
            text_ids.append(text_id)

    tokens_id = get_tokens_id(texts)
    text_id_token = dict(zip(text_ids, tokens_id))
    pd.to_pickle(text_id_token, '../data/train_ids.pkl')


def test_to_token():

    texts = []
    text_ids = []
    with open('/home/msqin/CCKS6/data/dev.txt') as f:
        for index, line in enumerate(f):
            text = str(line)
            texts.append(text)
            text_ids.append(index)
    tokens_id = get_tokens_id(texts)
    text_id_token = dict(zip(text_ids, tokens_id))
    pd.to_pickle(text_id_token, '../data/test_ids.pkl')
def kb_to_token():
    subject_ids = []
    texts = []
    del_subject = get_del_subject()
    with open('../data/entity_kb.txt', ) as f:
        for line in f:
            line = json.loads(line, encoding='utf-8')
            subject_id = line['subject_id']
            subject = line['subject']
            data = line['data']
            type_ = line['type']
            if type_ == 'Medical':
                if subject_id in del_subject:
                    continue
                text = get_text(data)
                text = str(subject)+','+text
                texts.append(text)
                subject_ids.append(subject_id)
    tokens_id = get_tokens_id(texts)
    text_id_token = dict(zip(subject_ids, tokens_id))
    print(len(text_id_token))
    pd.to_pickle(text_id_token, '../data/kb_ids.pkl')

def get_type():
    subject_type = {}

    with open('../data/entity_kb.txt', ) as f:
        for line in f:
            line = json.loads(line, encoding='utf-8')
            subject_id = line['subject_id']

            data = line['data']
            type_ = line['type']
            if type_=='Medical':
                subject_type[subject_id] = type_
    return subject_type

def get_train_kb():
    texts = []
    text_ids = []
    with open('/data-rbd/pan/python/CCKS/CCKS6/data/train.txt') as f:
        for index, line in enumerate(f):
            line = json.loads(line)
            text = line['text']
            text_id = line['text_id']
            texts.append(text)
            text_ids.append(text_id)

    train_id_text = dict(zip(text_ids, texts))

    subject_ids = []
    texts = []

    with open('/data-rbd/pan/python/CCKS/CCKS6/data/entity_kb.txt', ) as f:
        for line in f:
            line = json.loads(line, encoding='utf-8')
            subject_id = line['subject_id']
            subject = line['subject']
            data = line['data']
            type_ = line['type']
            if type_ == 'Medical':
                text = get_text(data)
                text = str(subject) + ',' + text
                texts.append(text)
                subject_ids.append(subject_id)

    kb_id_token = dict(zip(subject_ids, texts))
    return train_id_text,kb_id_token
def kb_to_info():
    subject_ids = []
    texts = []
    predicate_set = set()
    with open('../data/entity_kb.txt', ) as f:
        for line in f:
            line = json.loads(line, encoding='utf-8')
            subject_id = line['subject_id']
            subject = line['subject']
            data = line['data']
            type_ = line['type']
            if type_ == 'Medical':
                for d in data:
                    predicate_set.add(d['predicate'])
                    print(d)
                # input()
    print(predicate_set)







                # input()
    #             text = get_text(data)
    #             text = str(subject)+','+text
    #             texts.append(text)
    #             subject_ids.append(subject_id)
    # tokens_id = get_tokens_id(texts)
    # text_id_token = dict(zip(subject_ids, tokens_id))
    # pd.to_pickle(text_id_token, '../data/kb_ids.pkl')


def get_del_subject():
    dataset = []
    kb_subject = {}
    with open('../data/entity_kb.txt', ) as f:
        for line in f:
            line = json.loads(line, encoding='utf-8')
            subject_id = line['subject_id']
            subject = line['subject']
            data = line['data']
            type_ = line['type']
            if type_ == 'Medical':
                if subject in kb_subject:
                    kb_subject[subject].append(subject_id)
                else:
                    kb_subject[subject] = []
                    kb_subject[subject].append(subject_id)
    # for k in kb_subject:
    #     print(k, len(kb_subject[k]))

    subject_id_dict = dict()
    del_subject = set()
    with open('../data/train.txt') as f:
        for index, line in enumerate(f):
            line = json.loads(line)
            text= line['text']
            implicit_entity = line['implicit_entity']
            for entity in implicit_entity:
                subject = entity['subject']
                subject_id = entity['subject_id']
                if subject in kb_subject and len(kb_subject[subject])>1:
                    for sub_id in kb_subject[subject]:

                        if sub_id !=subject_id:
                            del_subject.add(sub_id)
                    # print(text,subject_id,subject)
                # if subject_id in subject_id_dict:
                #     subject_id_dict[subject_id] += 1
                # else:
                #     subject_id_dict[subject_id] = 1
    # for k in del_subject:
    #     print(k)
    print(len(del_subject))
    return del_subject
if __name__ == '__main__':
    # pass
    get_del_subject()


    kb_to_info()
    kb_to_token()
    print(len(pd.read_pickle( '../data/kb_ids.pkl')))
    get_del_subject()
    train_to_token()
    test_to_token()



    kb_ids = pd.read_pickle('../data/kb_ids.pkl')
    print(len(kb_ids))
    for l in kb_ids:
        print(l)
        if len(kb_ids[l])>2560:

            print(len(kb_ids[l]))