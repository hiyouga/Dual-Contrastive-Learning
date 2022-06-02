import os
import json
import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):

    def __init__(self, raw_data, label_dict, tokenizer, model_name, method):
        dataset = list()
        for data in raw_data:
            tokens = data['text'].lower().split(' ')
            if model_name == 'bert':
                cls_token, sep_token = ['[CLS]'], ['[SEP]']
            elif model_name == 'roberta':
                cls_token, sep_token = ['<s>'], ['</s>']
            if method in ['ce', 'scl']:
                tokens = cls_token + tokens + sep_token
            else:
                tokens = cls_token + list(label_dict.keys()) + sep_token + tokens + sep_token
            label_id = label_dict[data['label']]
            dataset.append((tokens, label_id))
        self._dataset = dataset
        self._num_classes = len(label_dict)

    def __getitem__(self, index):
        tokens, label_id = self._dataset[index]
        return tokens, label_id

    def __len__(self):
        return len(self._dataset)


def my_collate(batch, tokenizer):
    tokens, label_ids = map(list, zip(*batch))
    text_ids = tokenizer(tokens,
                         padding=True,
                         truncation=True,
                         max_length=256,
                         is_split_into_words=True,
                         add_special_tokens=False,
                         return_tensors='pt')
    return text_ids, torch.tensor(label_ids)


def load_data(dataset, data_dir, tokenizer, train_batch_size, test_batch_size, model_name, method, workers):
    if dataset == 'sst2':
        train_data = json.load(open(os.path.join(data_dir, 'SST2_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'SST2_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset == 'trec':
        train_data = json.load(open(os.path.join(data_dir, 'TREC_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'TREC_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'description': 0, 'entity': 1, 'abbreviation': 2, 'human': 3, 'location': 4, 'numeric': 5}
    elif dataset == 'cr':
        train_data = json.load(open(os.path.join(data_dir, 'CR_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'CR_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset == 'subj':
        train_data = json.load(open(os.path.join(data_dir, 'SUBJ_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'SUBJ_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'subjective': 0, 'objective': 1}
    elif dataset == 'pc':
        train_data = json.load(open(os.path.join(data_dir, 'procon_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'procon_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1}
    else:
        raise ValueError('unknown dataset')
    trainset = MyDataset(train_data, label_dict, tokenizer, model_name, method)
    testset = MyDataset(test_data, label_dict, tokenizer, model_name, method)
    train_dataloader = DataLoader(trainset, train_batch_size, shuffle=True, num_workers=workers,
                                  collate_fn=partial(my_collate, tokenizer=tokenizer), pin_memory=True)
    test_dataloader = DataLoader(testset, test_batch_size, shuffle=False, num_workers=workers,
                                 collate_fn=partial(my_collate, tokenizer=tokenizer), pin_memory=True)
    return train_dataloader, test_dataloader
