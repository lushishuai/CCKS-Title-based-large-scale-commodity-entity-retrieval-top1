# coding=utf-8
# coding=utf-8
import torch.nn as nn
from transformers import BertModel, RobertaModel
import torch.autograd as autograd
from torch.autograd import Variable
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
BERT_BASE_MODEL = "/home/msqin/bert-base-chinese"
BERT_TINY_MODEL = "/home/msqin/bert-base-chinese"


def batched_index_select(t, dim, inds):
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy)  # b x e x f
    return out

class BERTModel(nn.Module):

    def __init__(self, bert_pre_model=BERT_TINY_MODEL):
        super(BERTModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(bert_pre_model)
        for param in self.bert_model.parameters():
            param.requires_grad = True

    def forward(self, x):
        input_ids, masks = x
        bert_output = self.bert_model(input_ids, attention_mask=masks)
        h_embedding, pooled = bert_output
        out_mean = torch.mean(bert_output['last_hidden_state'], dim=1)
        # out = torch.cat([out_mean,pooled],dim=1)
        return out_mean


class SBERTModel(nn.Module):

    def __init__(self, bert_pre_model=BERT_TINY_MODEL):
        super(SBERTModel, self).__init__()
        self.bert_model = BERTModel(bert_pre_model)
        self.feature_size = list(self.bert_model.state_dict().values())[-1].size(0)
        for param in self.bert_model.parameters():
            param.requires_grad = True
        self.dropout = torch.nn.Dropout(0.1)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.fc1 = nn.Linear(self.feature_size * 3, 1)

        # self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        anchor_ids, anchor_mask, positive_ids, positive_mask, negative_ids, negative_mask = x

        anchor_out = self.bert_model([anchor_ids, anchor_mask])
        positive_out = self.bert_model([positive_ids, positive_mask])
        negative_out = self.bert_model([negative_ids, negative_mask])

        pos_out = torch.cat([anchor_out, positive_out, anchor_out - positive_out], dim=1)

        pos_out = self.dropout(pos_out)
        pos_out = self.fc1(pos_out)
        # pos_out = nn.ReLU()(pos_out)
        # pos_out = self.dropout(pos_out)
        # pos_out = self.fc2(pos_out)
        # pos_out = nn.Sigmoid()(pos_out)

        neg_out = torch.cat([anchor_out, negative_out, anchor_out - negative_out], dim=1)
        neg_out = self.dropout(neg_out)
        neg_out = self.fc1(neg_out)
        # neg_out = nn.ReLU()(neg_out)
        # neg_out = self.dropout(neg_out)
        # neg_out = self.fc2(neg_out)
        # neg_out = nn.Sigmoid()(neg_out)
        return pos_out, neg_out

    def predict(self, query_out, doc_out):
        out = torch.cat([query_out, doc_out, query_out - doc_out], dim=1)

        out = self.fc1(out)
        # out = nn.ReLU()(out)
        #
        # out = self.fc2(out)
        # out = nn.Sigmoid()(out)
        return out
class BERTBinery(nn.Module):
    def __init__(self, bert_pre_model=BERT_TINY_MODEL):
        super(BERTBinery, self).__init__()
        self.bert_model = BertModel.from_pretrained(bert_pre_model)
        self.feature_size = list(self.bert_model.state_dict().values())[-1].size(0)
        for param in self.bert_model.parameters():
            param.requires_grad = True
        self.dropout = torch.nn.Dropout(0.1)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.fc1 = nn.Linear(self.feature_size, 128)

        self.fc2 = nn.Linear(128, 1)
    def forward(self, x):
        input_ids, masks,type_ = x
        bert_output = self.bert_model(input_ids, attention_mask=masks,token_type_ids=type_)
        h_embedding, pooled = bert_output
        out = self.fc1(pooled)
        out = nn.ReLU()(out)
        out = self.fc2(out)
        out = nn.Sigmoid()(out)
        return out