import os
import json
import time
import math
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import random
from copy import copy, deepcopy
import sys
from itertools import chain
from datetime import datetime
from sklearn import metrics
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from nltk.corpus import wordnet
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import corenlp
from torchnlp.encoders import LabelEncoder

sys.path.append("../")
from transformers import BertTokenizer, BertForMaskedLM, BertConfig, RobertaTokenizer
from model import AttBERTForPolarity, ROBERTAForPolarity
from data_utils import MyDataset
from loss_func import CrossEntropy
import torch.nn.functional as F

MAX_LENGTH = 512
# trans_dict = {"[positive]": "[negative]", "[negative]": "[positive]",
#               "[entailment]": "[contradiction]", "[contradiction]": "[entailment]",
#               "[0]": "[1]", "[1]": "[0]"}

trans_label = {
    "SST2": {
        "1": "positive",
        "0": "negative"
    },
    "CR": {
        "1": "positive",
        "0": "negative"
    },
    "TREC": {
        "0": "description abstract concepts",
        "1": "entity",
        "2": "abbreviation",
        "3": "human",
        "4": "location",
        "5": "numeric"
    },
    "SUBJ": {
        "0": "subjective",
        "1": "objective"
    },
    "procon": {
        "positive": "positive",
        "negative": "negative"
    }
}


def collate_fn(batch, label_length, label_start, label_end, LABEL_CLASS):
    tot_label_length = sum(label_length)
    out = [[] for _ in range(len(batch[0]) + 1)]
    for row in batch:
        inputs_id = row[0]  # (bs, sl)
        keywords_labels = row[3]
        label_mask = row[5]

        SL = len(inputs_id)
        cur_index = [i for i in range(1, 1 + LABEL_CLASS)]
        random.shuffle(cur_index)
        cur_index_expand = []
        for idx in cur_index:
            cur_index_expand.extend(list(range(label_start[idx-1] + 1, label_end[idx-1] + 1)))
        assert(len(cur_index_expand) == tot_label_length)
        cur_all_index = [0] + cur_index_expand + [i for i in range(1 + tot_label_length, SL)]
        inputs_id = np.take(inputs_id, cur_all_index)  # torch.gather
        keywords_labels = np.take(keywords_labels, cur_all_index)
        label_mask = np.take(label_mask, cur_all_index)
        row[0] = inputs_id
        row[3] = keywords_labels
        row[5] = label_mask

        for idx in range(len(batch[0])):
            out[idx].append(row[idx])
        cur_all_order = np.argsort(np.asarray(cur_all_index))
        out[-1].append(cur_all_order)
    return [torch.as_tensor(e) for e in out]


class Instructor():
    ''' Model training and evaluation '''
    def __init__(self, opt):  # prepare for training the model
        train_data, test_data = self.load_data(opt.dataset, directory=opt.directory, percentage=opt.percentage)
        self._initialization(opt, train_data, test_data)

        self._print_args()

    @staticmethod
    def load_data(dataset,
                  directory,
                  train=True,
                  test=True,
                  train_file='Train.json',
                  test_file='Test.json',
                  percentage=1.0,
                  ):
        datasets = [
            'SST2',
            'CR',
            'procon',
            'SUBJ',
            'TREC',
        ]
        if dataset not in datasets:
            raise ValueError('dataset: {} not in support list!'.format(dataset))

        ret = []
        splits = [
            '_'.join([dataset, fn_]) for (requested, fn_) in [(train, train_file), (test, test_file)]
            if requested
        ]
        for split_file in splits:
            full_filename = os.path.join(directory, split_file)
            examples = []
            if full_filename.endswith("Train.json") and percentage > 1:
                examples = [[] for _ in range(len(trans_label[dataset].keys()))]
            with open(full_filename, 'r', encoding="utf-8") as f:
                tmp = f.readlines()
                for idx, j in enumerate(tmp):
                    N = len(tmp)
                    a_data = json.loads(j)
                    sent = a_data["sentence"].lower()
                    a_data["sentence"] = " ".join(sent.split(" ")[:MAX_LENGTH - 4])
                    a_data["polarity"] = str(a_data["polarity"])
                    # a_data["gate"] = a_data["gate"][:MAX_LENGTH - 4]
                    # a_data["pos"] = a_data["pos"][:MAX_LENGTH - 4]
                    if full_filename.endswith("Train.json"):
                        if percentage <= 1:
                            examples.append(a_data)
                            if idx >= int(percentage * N):
                                break
                        elif percentage > 1:
                            label2idx = {e: idx for idx, e in enumerate(list(trans_label[dataset].keys()))}
                            if len(examples[label2idx[a_data["polarity"]]]) < percentage:
                                examples[label2idx[a_data["polarity"]]].append(a_data)
                            if min([len(e) for e in examples]) >= percentage:
                                break
                    else:
                        examples.append(a_data)
            if percentage > 1 and full_filename.endswith("Train.json"):
                examples = list(chain(*examples))
            ret.append(examples)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)

    def _initialization(self, opt, train_data, test_data):
        # sentiment label encoder
        senti_label_corpus = [trans_label[opt.dataset][row["polarity"]] for row in train_data]
        senti_label_encoder = LabelEncoder(senti_label_corpus, reserved_labels=[], unknown_index=None)
        opt.label_class = len(senti_label_encoder.vocab)

        # our model
        opt.gumbel_softmax = False 
        if opt.model_type.lower() == "bert":
            model = AttBERTForPolarity(opt).to(opt.device)
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif opt.model_type.lower() == "roberta":
            model = ROBERTAForPolarity(opt).to(opt.device)
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        else:
            raise ValueError("model type should be either bert or roberta")

        max_length = 0
        label_token2idx = senti_label_encoder.token_to_index
        list_labels = list(label_token2idx.keys())
        if opt.class_use_bert_embedding == 1:
            pass
        else:
            label_token2idx = {f"[{idx}]": v for idx, v in enumerate(label_token2idx.values())}
            list_labels = list(label_token2idx.keys())
            tokenizer.add_tokens(list_labels, special_tokens=True)
            model.resize_vocab(tokenizer)
        label_length = self.get_label_length(tokenizer, list_labels)
        label_start = [0]
        label_end = [label_length[0]]
        for idx in range(len(label_length)-1):
            label_start.append(label_start[-1] + label_length[idx])
            label_end.append(label_end[-1] + label_length[idx+1])
        tot_label_length = sum(label_length)

        N_train, N_test = len(train_data), len(test_data)
        for idx, a_data in enumerate(chain(train_data, test_data)):
            label = a_data["polarity"]
            sent = a_data["sentence"]
            sent_list = sent.split(" ")
            if opt.model_type == "bert":
                sent_list = ["[CLS]"] + ["[PAD]"]*opt.label_class + ["[SEP]"] + sent_list + ["[SEP]"]
            elif opt.model_type == "roberta":
                sent_list = ["<s>"] + ["<pad>"] * opt.label_class + ["</s>", "</s>"] + sent_list + ["</s>"]
            else:
                raise ValueError("wrong model type!")
            for k, v in label_token2idx.items():
                sent_list[v + 1] = k
            tokens_list = tokenizer.tokenize(" ".join(sent_list))
            tokens_length = len(tokens_list)
            max_length = max(max_length, tokens_length)
            a_data["tokens_length"] = tokens_length
            a_data["tokens_list"] = tokens_list

        for idx, a_data in enumerate(chain(train_data, test_data)):
            tokens_list = a_data["tokens_list"]
            tokens_length = a_data["tokens_length"]
            label = a_data["polarity"]

            inputs_id = tokenizer.convert_tokens_to_ids(tokens_list)
            inputs_id = inputs_id + [0] * (max_length - tokens_length)
            attention_mask = [1] * tokens_length + [0] * (max_length - tokens_length)

            if idx < N_train:
                if opt.model_type == "bert":
                    token_type_ids = [0] * (2 + tot_label_length) + [1] * (tokens_length - tot_label_length - 2) + \
                                     [0] * (max_length - tokens_length)
                elif opt.model_type == "roberta":
                    token_type_ids = [0] * max_length
                keywords_labels = copy(inputs_id)

                if opt.class_use_bert_embedding == 1:
                    idx_label = list_labels.index(trans_label[opt.dataset][label])
                    label = senti_label_encoder.encode(trans_label[opt.dataset][label])
                else:
                    label = senti_label_encoder.encode(trans_label[opt.dataset][label])
                    idx_label = list_labels.index("[" + str(label.item()) + "]")
                idx_label_start, idx_label_end = label_start[idx_label], label_end[idx_label]
                label_mask = np.asarray(([0] * max_length))
                label_mask[idx_label_start + 1: idx_label_end + 1] = 1

                a_data = [inputs_id, attention_mask, token_type_ids, keywords_labels, label, label_mask]
                train_data[idx] = a_data
            else:
                label = senti_label_encoder.encode(trans_label[opt.dataset][label])
                inputs_id = np.asarray(inputs_id)
                attention_mask = np.asarray(attention_mask)
                label = np.asarray(label)
                a_data = [inputs_id, attention_mask, label]
                test_data[idx - N_train] = a_data
        trainset = MyDataset(train_data)
        testset = MyDataset(test_data)
        drop_last = True if len(trainset) > opt.batch_size else False
        train_dataloader = DataLoaderX(dataset=trainset, batch_size=opt.batch_size,
                                      shuffle=True, num_workers=4, pin_memory=True, drop_last=drop_last,
                                      collate_fn=lambda x: collate_fn(x, label_length, label_start,
                                                                      label_end, opt.label_class))  # training dataloader
        test_dataloader = DataLoaderX(dataset=testset, batch_size=64, shuffle=False,
                                      num_workers=4, pin_memory=True)  # test dataloader
        self.opt = opt
        self.model = model
        self.tokenizer = tokenizer
        self.senti_label_encoder = senti_label_encoder
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.label_token2idx = label_token2idx
        self.label_length = label_length
        self.label_start = label_start
        self.label_end = label_end

    def get_label_length(self, tokenizer, labels):
        if opt.model_type == "bert":
            label_length = [len(tokenizer.tokenize(label)) for label in labels]
        elif opt.model_type == "roberta":
            label_tokenized = tokenizer.tokenize(" " + " ".join(labels))
            label_length = [0] * len(labels)
            idx_label = 0
            cur_label = ""
            for idx_sub, sub_word in enumerate(label_tokenized):
                if sub_word.startswith("\u0120"):
                    sub_word = sub_word.replace("\u0120", "")
                    if not sub_word:
                        continue
                    cur_label += (" " + sub_word)
                else:
                    cur_label += sub_word
                if cur_label.strip() == labels[idx_label]:
                    label_length[idx_label] = idx_sub + 1
                    idx_label += 1
                    cur_label = ""
            label_length = [label_length[0]] + [l1 - l0 for l1, l0 in zip(label_length[1:], label_length[:-1])]
        return label_length

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        if self.opt.device == 'cuda':
            print(f"cuda memory allocated: {torch.cuda.memory_allocated(self.opt.device.index)}")
        print(f"n_trainable_params: {int(n_trainable_params)}, n_nontrainable_params: {int(n_nontrainable_params)}")
        print('training arguments:')
        for arg in vars(self.opt):
            print(f">>> {arg}: {getattr(self.opt, arg)}")

    def _reset_params(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.opt.model_type not in name:
                if 'embedding' in name:  # treat embedding matrices as special cases
                        weight = torch.nn.init.xavier_uniform_(torch.zeros_like(param))  # use xavier_uniform to initialize embedding matrices
                        weight[0] = torch.tensor(0, dtype=param.dtype, device=param.device)  # the vector corresponding to padding index should be zero
                        setattr(param, 'data', weight)  # update embedding matrix
                else:
                    if len(param.shape) > 1:
                        torch.nn.init.xavier_uniform_(param)  # use xavier_uniform to initialize weight matrices
                    else:
                        stdv = 1. / math.sqrt(param.size(0))
                        torch.nn.init.uniform_(param, a=-stdv, b=stdv)  # use uniform to initialize bias vectors

    def _train(self, optimizer, criterion, scaler, warm_up):
        if self.opt.model_type == "bert":
            MASK_ID = self.tokenizer.convert_tokens_to_ids(["[MASK]"])[0]
        elif self.opt.model_type == "roberta":
            MASK_ID = self.tokenizer.convert_tokens_to_ids(["<mask>"])[0]
        LABEL_CLASS = self.opt.label_class
        SENTENCE_BEGIN = sum(self.label_length) + 2
        VOCAB_SIZE = self.tokenizer.vocab_size

        n_train, n_train_loss, n_ce_loss, n_mlm_loss, n_correct = 0, 0, 0, 0, 0
        list_n_con_loss = [0] * len(self.opt.contrast_mode)
        t = time.time()

        self.model.train()  # switch model to training mode
        for sample_batched in self.train_dataloader:  # mini-batch optimization
            if self.opt.device == "cuda":
                inputs = list(map(lambda x: x.cuda(non_blocking=True), sample_batched))
            else:
                inputs = list(sample_batched)
            inputs_id, attention_mask, token_type_ids, keywords_labels, labels, label_mask, inputs_id_order = inputs
            if self.opt.model_type == "roberta":
                token_type_ids = None
            inputs_id_ori = deepcopy(inputs_id)

            train_loss = 0
            with torch.no_grad():
                if warm_up:
                    mask_pos = label_mask.type(torch.bool)
                else:
                    words_mask = torch.zeros_like(attention_mask).to(self.opt.device)
                    words_mask[:, 4:] = attention_mask[:, 4:]
                    label_words_mask = (label_mask + words_mask).type(torch.bool)

                    general_mask = torch.rand(inputs_id.shape).to(self.opt.device)
                    general_mask = general_mask < 0.2

                    rand_for_mask = torch.rand(inputs_id.shape).to(self.opt.device)
                    mask_pos = rand_for_mask < 0.8
                    mask_pos = mask_pos * label_words_mask * general_mask
                    mask_pos = mask_pos.to(self.opt.device).type(torch.bool)

                    rand_words_pos = rand_for_mask > 0.9
                    rand_words_pos = rand_words_pos * label_words_mask * general_mask
                    rand_words_pos = rand_words_pos.to(self.opt.device).type(torch.bool)

                    rand_for_words = torch.rand(inputs_id.shape).to(self.opt.device)
                    rand_words = (rand_for_words * VOCAB_SIZE).type(torch.long)
            inputs_id[mask_pos] = MASK_ID
            if warm_up:
                keywords_labels = torch.where(mask_pos==False,
                                              -100*torch.ones_like(mask_pos), keywords_labels) 
                del mask_pos
                optimizer.zero_grad()  # clear gradient accumulators
                with torch.cuda.amp.autocast():
                    # masked language model
                    if self.opt.alpha1 > 1e-32:
                        outputs = self.model.forward_language_model(input_ids=inputs_id, token_type_ids=token_type_ids,
                                                                    attention_mask=attention_mask, labels=keywords_labels)
                        mlm_loss = self.opt.alpha1 * outputs.loss
                        train_loss += mlm_loss
                    # label feature and cls feature
                    outputs = self.model([inputs_id_ori, attention_mask])  # compute outputs
                    word_feature, cls_feature = outputs.last_hidden_state, outputs.pooler_output
                    BS, SL, HS = word_feature.shape
                    # word feature
                    word_feature = torch.gather(word_feature, dim=1, index=inputs_id_order.unsqueeze(-1).expand(-1, -1, HS))
                    # label feature
                    label_feature = word_feature[:, 1: SENTENCE_BEGIN-1, :]
                    label_feature = self._join_label_feature(label_feature, self.label_length, LABEL_CLASS)  # (bs, label_class, 768)
                    label_feature = self.model.label_dropout(
                        self.model.label_activation(self.model.label_trans(label_feature)))
                    # cls feature
                    cls_feature = self.model.cls_dropout(
                        self.model.cls_activation(self.model.cls_trans(cls_feature)))
                    if self.opt.sentence_mode == "cls":
                        pass
                    elif self.opt.sentence_mode == "mean":
                        # mean pooling over sentence embeddings
                        word_feature = (word_feature * attention_mask.unsqueeze(-1))[:, SENTENCE_BEGIN:, :]
                        text_len_wo_head = torch.sum(attention_mask, dim=1, keepdim=True) - SENTENCE_BEGIN  # (bs, )
                        if self.opt.saliency_mode == "baseline":
                            cls_feature = torch.div(torch.sum((word_feature), dim=1), text_len_wo_head)  # (bs, 768)
                        elif self.opt.saliency_mode == "attention":
                            # query = label_feature
                            # key = value = word_feature
                            attention_scores = torch.bmm(label_feature, word_feature.permute(0, 2, 1))
                            attention_scores = attention_scores / math.sqrt(HS)  # (bs, class_label, sl)
                            attention_mask_wo_head = attention_mask[:, SENTENCE_BEGIN:].unsqueeze(1).expand(-1,
                                                                                                            LABEL_CLASS,
                                                                                                            -1)
                            attention_mask_wo_head = torch.where(attention_mask_wo_head == 1,
                                                                 torch.zeros_like(attention_mask_wo_head),
                                                                 -10000 * torch.ones_like(attention_mask_wo_head))
                            attention_scores = attention_scores + attention_mask_wo_head
                            attention_probs = nn.Softmax(dim=-1)(attention_scores)
                            attention_probs = self.model.fc_dropout(attention_probs)
                            label_feature = torch.bmm(attention_probs, word_feature)
                            cls_feature = torch.div(torch.sum((word_feature), dim=1), text_len_wo_head)
                    else:
                        raise ValueError("wrong sentence mode!")
                    # contrast loss
                    if self.opt.alpha2 > 1e-32 and len(torch.unique(labels)) > 1:
                        list_con_loss = self._contrast_loss(cls_feature, label_feature, labels)
                        list_con_loss = [self.opt.alpha2 * loss for loss in list_con_loss]
                        train_loss += sum(list_con_loss)
                # train_loss.backward()  # compute gradients through back-propagation
                scaler.scale(train_loss).backward()
                # optimizer.step()  # update model parameters
                scaler.step(optimizer)
                scaler.update()

                if self.opt.alpha1 > 1e-32:
                    n_mlm_loss += mlm_loss.item() * len(labels)
                if self.opt.alpha2 > 1e-32 and len(torch.unique(labels)) > 1:
                    list_n_con_loss += [con_loss.item() * len(labels) for con_loss in list_con_loss]
                n_train_loss += train_loss.item() * len(labels)
                n_train += len(labels)  # update train sample number
            else:
                inputs_id = torch.where(rand_words_pos==False, inputs_id, rand_words)
                keywords_labels = torch.where((mask_pos | rand_words_pos)==False,
                                              -100*torch.ones_like(mask_pos), keywords_labels)  # 只计算某些位置的loss
                del rand_words_pos, rand_words, mask_pos
                optimizer.zero_grad()  # clear gradient accumulators

                with torch.cuda.amp.autocast():
                    # masked language model
                    if self.opt.alpha1 > 1e-32:
                        outputs = self.model.forward_language_model(input_ids=inputs_id, token_type_ids=token_type_ids,
                                                                    attention_mask=attention_mask, labels=keywords_labels)
                        mlm_loss = self.opt.alpha1 * outputs.loss
                        train_loss += mlm_loss
                    # label feature and cls feature
                    outputs = self.model([inputs_id_ori, attention_mask])  # compute outputs
                    word_feature, cls_feature = outputs.last_hidden_state, outputs.pooler_output
                    BS, SL, HS = word_feature.shape
                    # word feature
                    word_feature = torch.gather(word_feature, dim=1, index=inputs_id_order.unsqueeze(-1).expand(-1, -1, HS))
                    # label feature
                    label_feature = word_feature[:, 1: SENTENCE_BEGIN-1, :]
                    label_feature = self._join_label_feature(label_feature, self.label_length, LABEL_CLASS)  # (bs, label_class, 768)
                    label_feature = self.model.label_dropout(
                        self.model.label_activation(self.model.label_trans(label_feature)))
                    # cls feature
                    cls_feature = self.model.cls_dropout(
                        self.model.cls_activation(self.model.cls_trans(cls_feature)))
                    if self.opt.sentence_mode == "cls":
                        pass
                    elif self.opt.sentence_mode == "mean":
                        # mean pooling over sentence embeddings
                        word_feature = (word_feature * attention_mask.unsqueeze(-1))[:, SENTENCE_BEGIN:, :]
                        text_len_wo_head = torch.sum(attention_mask, dim=1, keepdim=True) - SENTENCE_BEGIN  # (bs, )
                        if self.opt.saliency_mode == "baseline":
                            cls_feature = torch.div(torch.sum((word_feature), dim=1), text_len_wo_head)  # (bs, 768)
                        elif self.opt.saliency_mode == "attention":
                            # query = label_feature
                            # key = value = word_feature
                            attention_scores = torch.bmm(label_feature, word_feature.permute(0, 2, 1))
                            attention_scores = attention_scores / math.sqrt(HS)  # (bs, class_label, sl)
                            attention_mask_wo_head = attention_mask[:, SENTENCE_BEGIN:].unsqueeze(1).expand(-1,
                                                                                                            LABEL_CLASS,
                                                                                                            -1)
                            attention_mask_wo_head = torch.where(attention_mask_wo_head==1,
                                                                 torch.zeros_like(attention_mask_wo_head),
                                                                 -10000 * torch.ones_like(attention_mask_wo_head))
                            attention_scores = attention_scores + attention_mask_wo_head
                            attention_probs = nn.Softmax(dim=-1)(attention_scores)
                            attention_probs = self.model.fc_dropout(attention_probs)
                            label_feature = torch.bmm(attention_probs, word_feature)
                            cls_feature = torch.div(torch.sum((word_feature), dim=1), text_len_wo_head)
                    else:
                        raise ValueError("wrong sentence mode!")
                    # ce loss
                    predicts = torch.bmm(label_feature, self.model.fc_dropout(cls_feature.unsqueeze(-1))).squeeze(-1)
                    ce_loss = criterion([predicts, None, None], labels)  # compute batch loss
                    train_loss += ce_loss
                    # contrast loss
                    if self.opt.alpha2 > 1e-32 and len(torch.unique(labels)) > 1:
                        list_con_loss = self._contrast_loss(cls_feature, label_feature, labels)
                        list_con_loss = [self.opt.alpha2 * loss for loss in list_con_loss]
                        train_loss += sum(list_con_loss)
                    # autoencoder loss
                    # not implemented...
                # train_loss.backward()  # compute gradients through back-propagation
                scaler.scale(train_loss).backward()
                # optimizer.step()  # update model parameters
                scaler.step(optimizer)
                scaler.update()

                n_ce_loss += ce_loss.item() * len(labels)
                if self.opt.alpha1 > 1e-32:
                    n_mlm_loss += mlm_loss.item() * len(labels)
                if self.opt.alpha2 > 1e-32 and len(torch.unique(labels)) > 1:
                    list_n_con_loss = [con_loss.item() * len(labels) + prev_con_loss for con_loss, prev_con_loss in zip(list_con_loss, list_n_con_loss)]
                n_train_loss += train_loss.item() * len(labels)
                n_correct += (torch.argmax(predicts, -1) == labels).sum().item()  # update correct sample number
                n_train += len(labels)  # update train sample number
        return n_train_loss / n_train, n_ce_loss / n_train, n_mlm_loss / n_train, [n_con_loss / n_train for n_con_loss in list_n_con_loss], \
               n_correct / n_train, time.time() - t

    def _join_label_feature(self, label_feature, label_length, LABEL_CLASS):
        BS, _, HS = label_feature.shape
        out = torch.zeros((BS, LABEL_CLASS, HS)).to(self.opt.device)
        start, end, idx_label = 0, label_length[0], 0
        for idx in range(LABEL_CLASS):
            out[:, idx, :] = torch.mean(label_feature[:, start: end, :], dim=1)
            if idx != LABEL_CLASS - 1:
                start += label_length[idx]
                end += label_length[idx+1]
        return out

    def _contrast_loss(self, cls_feature, label_feature, labels):
        normed_cls_feature = F.normalize(cls_feature, dim=-1)
        normed_label_feature = F.normalize(label_feature, dim=-1)
        list_con_loss = []
        BS, LABEL_CLASS, HS = normed_label_feature.shape
        normed_positive_label_feature = torch.gather(normed_label_feature, dim=1,
                                                     index=labels.reshape(-1, 1, 1).expand(-1, 1, HS)).squeeze(1)  # (bs, 768)
        if "1" in self.opt.contrast_mode:
            loss1 = self._calculate_contrast_loss(normed_positive_label_feature, normed_cls_feature, labels)
            list_con_loss.append(loss1)
        if "2" in self.opt.contrast_mode:
            loss2 = self._calculate_contrast_loss(normed_cls_feature, normed_positive_label_feature, labels)
            list_con_loss.append(loss2)
        if "3" in self.opt.contrast_mode:
            loss3 = self._calculate_contrast_loss(normed_positive_label_feature, normed_positive_label_feature, labels)
            list_con_loss.append(loss3)
        if "4" in self.opt.contrast_mode:
            loss4 = self._calculate_contrast_loss(normed_cls_feature, normed_cls_feature, labels)
            list_con_loss.append(loss4)
        return list_con_loss

    def _calculate_contrast_loss(self, anchor, target, labels, mu=1.0):
        BS = len(labels)
        with torch.no_grad():
            labels = labels.reshape(-1, 1)
            mask = torch.eq(labels, labels.T)  # (bs, bs)
            # compute temperature using mask
            temperature_matrix = torch.where(mask == True, mu * torch.ones_like(mask),
                                             1 / self.opt.temperature * torch.ones_like(mask)).to(self.opt.device)
#             # mask-out self-contrast cases
#             logits_mask = torch.scatter(
#                 torch.ones_like(mask),
#                 1,
#                 torch.arange(BS).view(-1, 1).to(self.opt.device),
#                 0
#             )
#             mask = mask * logits_mask
        # compute logits
        anchor_dot_target = torch.multiply(torch.matmul(anchor, target.T), temperature_matrix)  # (bs, bs)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_target, dim=1, keepdim=True)
        logits = anchor_dot_target - logits_max.detach()  # (bs, bs)
        # compute log_prob
        exp_logits = torch.exp(logits)  # (bs, bs)
        exp_logits = exp_logits - torch.diag_embed(torch.diag(exp_logits))
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)  # (bs, bs)
        # in case that mask.sum(1) has no zero
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_sum.detach()
        loss = - mean_log_prob_pos.mean()
        return loss

    def _test(self, criterion, scaler):
        LABEL_CLASS = self.opt.label_class
        SENTENCE_BEGIN = sum(self.label_length) + 2
        test_loss, n_correct, n_test = 0, 0, 0  # reset counters
        labels_all, predicts_all = None, None  # initialize variables

        self.model.eval()  # switch model to training mode
        with torch.no_grad():
            test_loss = 0
            for sample_batched in self.test_dataloader:  # mini-batch optimization
                if self.opt.device == "cuda":
                    inputs = list(map(lambda x: x.cuda(non_blocking=True), sample_batched))
                else:
                    inputs = list(sample_batched)
                inputs_id, attention_mask, labels = inputs

                # cross entropy
                outputs = self.model([inputs_id, attention_mask])  # compute outputs
                word_feature, cls_feature = outputs.last_hidden_state, outputs.pooler_output
                BS, SL, HS = word_feature.shape
                # label feature
                label_feature = word_feature[:, 1: SENTENCE_BEGIN-1, :]
                label_feature = self._join_label_feature(label_feature, self.label_length, LABEL_CLASS)  # (bs, label_class, 768)
                label_feature = self.model.label_dropout(
                    self.model.label_activation(self.model.label_trans(label_feature)))
                # cls feature
                cls_feature = self.model.cls_dropout(
                    self.model.cls_activation(self.model.cls_trans(cls_feature)))
                if self.opt.sentence_mode == "cls":
                    pass
                elif self.opt.sentence_mode == "mean":
                    # mean pooling over sentence embeddings
                    word_feature = (word_feature * attention_mask.unsqueeze(-1))[:, SENTENCE_BEGIN:, :]
                    text_len_wo_head = torch.sum(attention_mask, dim=1, keepdim=True) - SENTENCE_BEGIN  # (bs, )
                    if self.opt.saliency_mode == "baseline":
                        cls_feature = torch.div(torch.sum((word_feature), dim=1), text_len_wo_head)  # (bs, 768)
                    elif self.opt.saliency_mode == "attention":
                        # query = label_feature
                        # key = value = word_feature
                        attention_scores = torch.bmm(label_feature, word_feature.permute(0, 2, 1))
                        attention_scores = attention_scores / math.sqrt(HS)  # (bs, class_label, sl)
                        attention_mask_wo_head = attention_mask[:, SENTENCE_BEGIN:].unsqueeze(1).expand(-1,
                                                                                                        LABEL_CLASS,
                                                                                                        -1)
                        attention_mask_wo_head = torch.where(attention_mask_wo_head == 1,
                                                             torch.zeros_like(attention_mask_wo_head),
                                                             -10000 * torch.ones_like(attention_mask_wo_head))
                        attention_scores = attention_scores + attention_mask_wo_head
                        attention_probs = nn.Softmax(dim=-1)(attention_scores)
                        attention_probs = self.model.fc_dropout(attention_probs)
                        label_feature = torch.bmm(attention_probs, word_feature)
                        cls_feature = torch.div(torch.sum((word_feature), dim=1), text_len_wo_head)
                else:
                    raise ValueError("wrong sentence mode!")
                predicts = torch.bmm(label_feature, self.model.fc_dropout(cls_feature.unsqueeze(-1))).squeeze(-1)
                ce_loss = criterion([predicts, None, None], labels)  # compute batch loss

                test_loss += ce_loss.item() * len(labels)
                n_correct += (torch.argmax(predicts, -1) == labels).sum().item()
                n_test += len(labels)
                labels_all = torch.cat((labels_all, labels), dim=0) if labels_all is not None else labels
                predicts_all = torch.cat((predicts_all, predicts), dim=0) if predicts_all is not None else predicts
        macro_f1 = metrics.f1_score(labels_all.detach().cpu(), torch.argmax(predicts_all, -1).detach().cpu(), average='macro')  # compute f1 score
        precision = metrics.precision_score(labels_all.detach().cpu(), torch.argmax(predicts_all, -1).detach().cpu(), average='macro')
        recall = metrics.recall_score(labels_all.detach().cpu(), torch.argmax(predicts_all, -1).detach().cpu(), average='macro')
        return test_loss / n_test, n_correct / n_test, macro_f1, precision, recall

    def _run(self):
        all_best_acc = 0
        _params = [p for name, p in self.model.named_parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)
        criterion = CrossEntropy(self.opt)  # loss function implemented as described in paper

        self._reset_params()  # reset model parameters
        best_test_acc, best_test_f1 = 0, 0  # record the best acc and f1 score on testing set
        best_test_precision, best_test_recall = 0, 0
        patience = 0

        print(f"warm up lm model for the first {self.opt.warm_up_epoch} epochs")
        warm_up = True
        for p in optimizer.param_groups:
            p['lr'] = 2 * self.opt.lr
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.opt.num_epoch):
            if epoch >= self.opt.warm_up_epoch:  # (opt.num_epoch + opt.warm_up_epoch)//2 ~ num_epoch
                warm_up = False
                cur_lr = 2 * self.opt.lr - (epoch - self.opt.warm_up_epoch) / \
                         (self.opt.num_epoch - self.opt.warm_up_epoch) * self.opt.lr  # 希望学习率从2lr降到lr
                for p in optimizer.param_groups:
                    p['lr'] = cur_lr
            cur_lr = optimizer.state_dict()['param_groups'][0]['lr']

            train_loss, ce_loss, mlm_loss, list_con_loss, train_acc, used_time = self._train(optimizer, criterion, scaler, warm_up)
            test_loss, test_acc, test_f1, precision, recall = self._test(criterion, scaler)
            list_con_loss = [round(loss, 4) for loss in list_con_loss]
            print("Epoch: {} | train_loss: {:.4f} | train_time: {:.4f} | lr: {:.8f}"
                  "\ttrain_acc: {:.4f} | test_loss: {:.4f} | test_acc: {:.4f} | test_f1: {:.4f}".format(
                epoch, train_loss, used_time, cur_lr, train_acc, test_loss, test_acc, test_f1
            ))
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                best_test_f1 = test_f1
                best_test_precision = precision
                best_test_recall = recall
                patience = 0
                if best_test_acc > all_best_acc:
                    all_best_acc = best_test_acc
                    print("model weights saved!")
            else:
                patience += 1
                if patience > 50:
                    print(f"Early stopping at epoch {epoch+1}!")
                    break
        print('#' * 50)
        print(f"best test acc: {best_test_acc:.4f}, best test f1: {best_test_f1:.4f}, "
              f"best test precision: {best_test_precision:.4f}, best test recall: {best_test_recall:.4f}")
        return


def _main(opt):
    ins = Instructor(opt)
    ins._run()
    return


if __name__ == "__main__":
    ''' hyperparameters '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='SST2', type=str, help='Restaurants, Laptops, SST2, CR'
                                                                    'TREC, IMDB, snli_1.0, yahoo, agnews')
    parser.add_argument('--directory', default='./datasets_manual', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--percentage', default=50, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--warm_up_epoch', default=0, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--word_dim', default=768, type=int)
    parser.add_argument('--fc_dropout', default=0.1, type=float)
    parser.add_argument('--eps', default=1e-2, type=float)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--run_times', default=1, type=int)
    parser.add_argument('--cuda_device', default=0, type=int, help='0, 1, 2, 3')

    parser.add_argument('--sentence_mode', default="cls", type=str, help='mean, cls')
    parser.add_argument('--saliency_mode', default="baseline", type=str, help='baseline, drop_input, mean, cls')
    parser.add_argument('--alpha1', default=0.01, type=float)  # mlm loss
    parser.add_argument('--alpha2', default=0.01, type=float)  # contrast loss
    parser.add_argument('--contrast_mode', default="12", type=str, help='1234')
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--model_type', default="roberta", type=str, help='bert, roberta')
    parser.add_argument('--class_use_bert_embedding', default=1, type=int, help='fake bool')

    opt = parser.parse_args()
    assert (opt.word_dim == 768)
    if opt.saliency_mode == "attention":
        assert (opt.sentence_mode == "mean")

    # # anomaly detection
    # torch.autograd.set_detect_anomaly(True)

    class DataLoaderX(DataLoader):
        def __iter__(self):
            return BackgroundGenerator(super().__iter__())

    opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if opt.device == "cuda":
        ''' if you are using cudnn '''
        torch.backends.cudnn.deterministic = True  # Deterministic mode can have a performance impact
        torch.backends.cudnn.benchmark = False 

    if not os.path.exists("results_polarity"):
        os.mkdir("results_polarity")
    opt.out_dir = "results_polarity"
    _main(opt)
