from models.bert_model import BERTModel
import json
import numpy as np
import math
import pandas as pd
import torch.nn as nn
import random
import torch
from processors.data_utils import *
from tqdm import tqdm
from models.callbacks import *
import torch.nn.functional as F
import os

from models.progressbar import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def get_dataset():
    dataset = []
    kb_ids = pd.read_pickle('data/kb_ids.pkl')
    with open('data/train.txt') as f:
        for index, line in enumerate(f):
            line = json.loads(line)
            text_id = line['text_id']
            implicit_entity = line['implicit_entity']
            for entity in implicit_entity:
                subject_id = entity['subject_id']
                if subject_id in kb_ids:
                    dataset.append((text_id, subject_id))
    return dataset
def get_test():

    dataset = []
    with open('data/dev.txt') as f:
        for index, line in enumerate(f):
            line = str(line)
            dataset.append((index, 269144))
    return dataset

def get_negative(top10: dict, pos_id: str):
    while True:
        neg_id = random.choice(top10)
        if neg_id != pos_id:
            break
    return neg_id


class DataLoaderTrain(object):
    def __init__(self, dataset, batch_size=5, shuffle=True):
        self.dataset = dataset

        self.kb_id_token = pd.read_pickle('data/kb_ids.pkl')
        self.query_id_token = pd.read_pickle('data/train_ids.pkl')
        self.recall_top10 = pd.read_pickle('data/recall_top10.pkl')
        self.batch_size = batch_size
        self.data_length = len(self.dataset)
        self.steps_per_epoch = math.ceil(self.data_length / batch_size)
        self.indexes = np.arange(self.data_length)
        self.shuffle = shuffle
        self.indexes_no = list(np.arange(self.data_length))

        self.length = self.steps_per_epoch

    def __len__(self):

        return self.length

    def get_indexes(self, i):
        return self.indexes[i * self.batch_size:(i + 1) * self.batch_size]

    def __iter__(self):
        # bert input
        anchor_ids = []
        anchor_mask = []
        positive_ids = []
        positive_mask = []
        negative_ids = []
        negative_mask = []
        batch_index = [i for i in range(len(self.dataset))]
        if self.shuffle:
            np.random.shuffle(batch_index)
        for j, i in enumerate(batch_index):
            data = self.dataset[i]
            text_id = data[0]
            positive_id = data[1]
            if positive_id not in self.kb_id_token:
                continue
            negative_id = get_negative(self.recall_top10[text_id], positive_id)

            an_token = self.query_id_token[text_id]
            an_mask = [1] * len(an_token)

            pos_token = self.kb_id_token[positive_id]
            pos_mask = [1] * len(pos_token)

            neg_token = self.kb_id_token[negative_id]
            neg_mask = [1] * len(neg_token)

            anchor_ids.append(an_token)
            anchor_mask.append(an_mask)
            positive_ids.append(pos_token)
            positive_mask.append(pos_mask)
            negative_ids.append(neg_token)
            negative_mask.append(neg_mask)

            if len(anchor_ids) == self.batch_size or j == len(batch_index) - 1:
                # torch.tensor(batch[i], dtype=torch.long)

                anchor_ids = torch.tensor(seq_padding(anchor_ids), dtype=torch.long)
                anchor_mask = torch.tensor(seq_padding(anchor_mask), dtype=torch.long)
                positive_ids = torch.tensor(seq_padding(positive_ids), dtype=torch.long)
                positive_mask = torch.tensor(seq_padding(positive_mask), dtype=torch.long)
                negative_ids = torch.tensor(seq_padding(negative_ids), dtype=torch.long)
                negative_mask = torch.tensor(seq_padding(negative_mask), dtype=torch.long)
                yield anchor_ids, anchor_mask, positive_ids, positive_mask, negative_ids, negative_mask

                anchor_ids = []
                anchor_mask = []
                positive_ids = []
                positive_mask = []
                negative_ids = []
                negative_mask = []


class DataLoaderTest(object):
    def __init__(self, dataset, batch_size=5, shuffle=True):
        self.dataset = dataset

        self.kb_id_token = pd.read_pickle('data/kb_ids.pkl')
        self.query_id_token = pd.read_pickle('data/test_ids.pkl')
        self.recall_top10 = pd.read_pickle('data/test_recall_top10.pkl')
        self.batch_size = batch_size
        self.data_length = len(self.dataset)
        self.steps_per_epoch = math.ceil(self.data_length / batch_size)
        self.indexes = np.arange(self.data_length)
        self.shuffle = shuffle
        self.indexes_no = list(np.arange(self.data_length))

        self.length = self.steps_per_epoch

    def __len__(self):

        return self.length

    def get_indexes(self, i):
        return self.indexes[i * self.batch_size:(i + 1) * self.batch_size]

    def __iter__(self):
        # bert input
        anchor_ids = []
        anchor_mask = []
        positive_ids = []
        positive_mask = []
        negative_ids = []
        negative_mask = []
        batch_index = [i for i in range(len(self.dataset))]
        if self.shuffle:
            np.random.shuffle(batch_index)
        for j, i in enumerate(batch_index):
            data = self.dataset[i]
            text_id = data[0]
            positive_id = data[1]
            if positive_id not in self.kb_id_token:
                continue
            negative_id = get_negative(self.recall_top10[text_id], positive_id)

            an_token = self.query_id_token[text_id]
            an_mask = [1] * len(an_token)

            pos_token = self.kb_id_token[positive_id]
            pos_mask = [1] * len(pos_token)

            neg_token = self.kb_id_token[negative_id]
            neg_mask = [1] * len(neg_token)

            anchor_ids.append(an_token)
            anchor_mask.append(an_mask)
            positive_ids.append(pos_token)
            positive_mask.append(pos_mask)
            negative_ids.append(neg_token)
            negative_mask.append(neg_mask)

            if len(anchor_ids) == self.batch_size or j == len(batch_index) - 1:
                # torch.tensor(batch[i], dtype=torch.long)

                anchor_ids = torch.tensor(seq_padding(anchor_ids), dtype=torch.long)
                anchor_mask = torch.tensor(seq_padding(anchor_mask), dtype=torch.long)
                positive_ids = torch.tensor(seq_padding(positive_ids), dtype=torch.long)
                positive_mask = torch.tensor(seq_padding(positive_mask), dtype=torch.long)
                negative_ids = torch.tensor(seq_padding(negative_ids), dtype=torch.long)
                negative_mask = torch.tensor(seq_padding(negative_mask), dtype=torch.long)
                yield anchor_ids, anchor_mask, positive_ids, positive_mask, negative_ids, negative_mask

                anchor_ids = []
                anchor_mask = []
                positive_ids = []
                positive_mask = []
                negative_ids = []
                negative_mask = []
class DataLoaderKb(object):
    def __init__(self, dataset, batch_size=5, ):
        self.dataset = dataset
        self.kb_id_token = pd.read_pickle('data/kb_ids.pkl')
        self.batch_size = batch_size
        self.data_length = len(self.dataset)
        self.steps_per_epoch = math.ceil(self.data_length / batch_size)
        self.indexes = np.arange(self.data_length)
        self.length = self.steps_per_epoch

    def __len__(self):

        return self.length

    def get_indexes(self, i):
        return self.indexes[i * self.batch_size:(i + 1) * self.batch_size]

    def __iter__(self):
        # bert input
        kb_token_ids = []
        kb_token_mask = []
        batch_index = [i for i in range(len(self.dataset))]
        for j, i in enumerate(batch_index):
            data = self.dataset[i]
            kb_id = data
            token_id = self.kb_id_token[kb_id]
            token_mask = [1] * len(token_id)
            kb_token_ids.append(token_id)
            kb_token_mask.append(token_mask)
            if len(kb_token_ids) == self.batch_size or j == len(batch_index) - 1:
                kb_token_ids = torch.tensor(seq_padding(kb_token_ids), dtype=torch.long)
                kb_token_mask = torch.tensor(seq_padding(kb_token_mask), dtype=torch.long)
                yield kb_token_ids, kb_token_mask
                kb_token_ids = []
                kb_token_mask = []


def train(index):
    model = BERTModel(bert_pre_model=BERT_BASE_MODEL)

    dataset = get_dataset()
    # train_dataset = dataset[:index * 10000] + dataset[(index + 1) * 10000:]
    # val_dataset = dataset[index * 10000:(index + 1) * 10000]
    train_dataset = dataset[:500]
    val_dataset = dataset[500:1000]

    print(len(dataset))
    print(len(train_dataset))
    print(len(val_dataset))

    train_loader = DataLoaderTrain(train_dataset, batch_size=4)
    val_loader = DataLoaderTrain(val_dataset, batch_size=4)

    device = 'cuda'
    model.to(device)
    bert_model_params = list(map(id, model.bert_model.parameters()))
    base_params = filter(lambda p: id(p) not in bert_model_params,
                         model.parameters())
    params_list = [
        {"params": base_params, 'lr': 5e-04},
        {'params': model.bert_model.parameters(), 'lr': 3e-05}
    ]
    device_ids = range(torch.cuda.device_count())
    print(device_ids)
    model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(params_list, lr=3e-05)  # torch.optim.Adam(params=model.parameters(), lr=2e-05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=1,
                                                           min_lr=5e-6, )
    earlystopping = EarlyStopping(model)
    mc = ModelCheckpoint(model, 'data/model' + str(index) + '_to1.pt')
    loss_function = torch.nn.TripletMarginLoss(margin=3)
    epochs = 1
    for i in range(epochs):
        model.train()
        # tqdm_batch_iterator = tqdm(train_loader)
        ave_loss = 0
        pbar = ProgressBar(n_total=len(train_loader), desc=f'training{i}/{epochs}')
        for batch_index, batch in enumerate(train_loader):
            batch = [i.to('cuda') for i in batch]
            anchor_ids = batch[0]
            anchor_mask = batch[1]
            positive_ids = batch[2]
            positive_mask = batch[3]
            negative_ids = batch[4]
            negative_mask = batch[5]
            optimizer.zero_grad()
            anchor_out = model([anchor_ids, anchor_mask])
            positive_out = model([positive_ids, positive_mask])
            negative_out = model([negative_ids, negative_mask])

            loss = loss_function(anchor_out, positive_out, negative_out)
            ave_loss += loss.item() / len(train_loader)
            loss.backward()
            optimizer.step()
            pbar(step=batch_index, info={'loss': loss})
        model.eval()
        val_loss = 0.0
        preds, labels = [], []
        pos_distance = []
        neg_distance = []
        with torch.no_grad():
            tqdm_batch_iterator = val_loader
            for batch_index, batch in enumerate(tqdm_batch_iterator):
                batch = [i.to('cuda') for i in batch]
                anchor_ids = batch[0]
                anchor_mask = batch[1]
                positive_ids = batch[2]
                positive_mask = batch[3]
                negative_ids = batch[4]
                negative_mask = batch[5]
                anchor_out = model([anchor_ids, anchor_mask])
                positive_out = model([positive_ids, positive_mask])
                negative_out = model([negative_ids, negative_mask])

                loss = loss_function(anchor_out, positive_out, negative_out)
                val_loss += loss.item() / len(val_loader)

                dist_pos = F.pairwise_distance(anchor_out, positive_out, p=2).detach().cpu().numpy()
                pos_distance.extend(dist_pos)
                dist_neg = F.pairwise_distance(anchor_out, negative_out, p=2).detach().cpu().numpy()
                neg_distance.extend(dist_neg)

        scheduler.step(val_loss)
        earlystopping.step(val_loss)
        mc.epoch_step(val_loss)
        print('val loss', val_loss)


def batched_index_select(t, dim, inds):
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy)  # b x e x f
    return out
def eval():
    start = time.time()
    model = BERTModel(bert_pre_model=BERT_BASE_MODEL)
    model = nn.DataParallel(model)
    kb_id_token = pd.read_pickle('data/kb_ids.pkl')
    query_id_token = pd.read_pickle('data/train_ids.pkl')
    subject_ids = list(kb_id_token.keys())
    dataset = get_dataset()

    val_dataset = dataset[70000:]
    merge_scores = []
    recall_top10 = pd.read_pickle('data/recall_top10.pkl')
    for index in range(1):

        model.load_state_dict(torch.load('data/model' + str(index) + '_to1.pt'))
        model.cuda()
        model.eval()
        kb_preds = []
        kb_loader = tqdm(DataLoaderKb(subject_ids, batch_size=8))
        for batch_index, batch in enumerate(kb_loader):
            batch = [i.to('cuda') for i in batch]
            anchor_out = model(batch).detach()
            kb_preds.append(anchor_out)
        kb_preds = torch.cat(kb_preds, dim=0)

        kb_preds_dict = dict(zip(subject_ids, kb_preds))


        val_loader = DataLoaderTrain(val_dataset, batch_size=4, shuffle=False)

        tqdm_batch_iterator = tqdm(val_loader)
        val_preds = []
        for batch_index, batch in enumerate(tqdm_batch_iterator):
            batch = [i.to('cuda') for i in batch]
            anchor_ids = batch[0]
            anchor_mask = batch[1]
            anchor_out = model([anchor_ids, anchor_mask]).detach()
            for i in range(len(anchor_out)):
                val_preds.append(anchor_out[i])

        scores = []
        for pred, data in zip(val_preds, val_dataset):
            query_id = data[0]
            subject_id = data[1]
            recall_10 = np.array(recall_top10[query_id][:3])
            recall_10_pred = [kb_preds_dict[recall_id].view(1, pred.size(0)) for recall_id in recall_10]
            recall_10_pred = torch.cat(recall_10_pred, dim=0)
            score = F.pairwise_distance(pred, recall_10_pred, p=2).cpu().numpy()
            scores.append(score)
        scores = np.array(scores)
        merge_scores.append(scores)
    recall_num = 0
    recall_top1 = {}
    merge_scores = np.mean(merge_scores, axis=0)
    for data, scores in zip(val_dataset, merge_scores):
        text_id = data[0]
        subject_id = data[1]
        recall_10 = np.array(recall_top10[text_id])
        indices = scores.argsort()[:2]
        recall_subject_ids = [recall_10[index] for index in indices]
        if subject_id in recall_subject_ids:
            recall_num += 1
        recall_top1[text_id] = recall_subject_ids
    print(recall_num / len(val_dataset))
    print(recall_num)
    pd.to_pickle(recall_top1, 'data/recall_top1.pkl')
    print(time.time() - start)

def predict():
    start = time.time()
    model = BERTModel(bert_pre_model=BERT_TINY_MODEL)
    model = nn.DataParallel(model)
    kb_id_token = pd.read_pickle('data/kb_ids.pkl')
    subject_ids = list(kb_id_token.keys())
    dataset = get_test()

    merge_scores = []
    recall_top10 = pd.read_pickle('data/test_recall_top10.pkl')
    for index in range(1):

        model.load_state_dict(torch.load('data/model' + str(index) + '_to10.pt'))
        model.cuda()
        model.eval()
        kb_preds = []
        kb_loader = tqdm(DataLoaderKb(subject_ids, batch_size=8))
        for batch_index, batch in enumerate(kb_loader):
            batch = [i.to('cuda') for i in batch]
            anchor_out = model(batch).detach()
            kb_preds.append(anchor_out)
        kb_preds = torch.cat(kb_preds, dim=0)

        kb_preds_dict = dict(zip(subject_ids, kb_preds))

        val_loader = DataLoaderTest(dataset, batch_size=4, shuffle=False)

        tqdm_batch_iterator = tqdm(val_loader)
        val_preds = []
        for batch_index, batch in enumerate(tqdm_batch_iterator):
            batch = [i.to('cuda') for i in batch]
            anchor_ids = batch[0]
            anchor_mask = batch[1]
            anchor_out = model([anchor_ids, anchor_mask]).detach()
            for i in range(len(anchor_out)):
                val_preds.append(anchor_out[i])

        scores = []
        for pred, data in zip(val_preds, dataset):
            query_id = data[0]
            subject_id = data[1]
            recall_10 = np.array(recall_top10[query_id])
            recall_10_pred = [kb_preds_dict[recall_id].view(1, pred.size(0)) for recall_id in recall_10]
            recall_10_pred = torch.cat(recall_10_pred, dim=0)
            score = F.pairwise_distance(pred, recall_10_pred, p=2).cpu().numpy()
            scores.append(score)
        scores = np.array(scores)
        merge_scores.append(scores)
    recall_num = 0
    recall_top1 = {}
    merge_scores = np.mean(merge_scores, axis=0)
    for data, scores in zip(dataset, merge_scores):
        text_id = data[0]
        recall_10 = np.array(recall_top10[text_id])
        indices = scores.argsort()[:1]
        recall_subject_ids = [recall_10[index] for index in indices]
        recall_top1[text_id] = recall_subject_ids
    print(recall_num / len(dataset))
    print(recall_num)
    pd.to_pickle(recall_top1, 'data/test_recall_top1.pkl')
    print(time.time() - start)



if __name__ == '__main__':
    # for i in range(8):
    #     train(i)
    train(0)
    eval()
    predict()
