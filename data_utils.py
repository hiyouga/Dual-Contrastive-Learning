import os
import re
import json
import pickle
import string
import numpy as np
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, dataset):
        self._dataset = dataset
    def __getitem__(self, index):
        return self._dataset[index]
    def __len__(self):
        return len(self._dataset)


def load_data(dataset='SST2',
              directory='datasets_processed',
              train=True,
              test=True,
              train_file='Train.json',
              test_file='Test.json',
              load_pos=False,
              augmented_mode=None,
              percentage=1.0,
              ):
    """
    Load the Restaurants14 dataset.

    Args:
        dataset : xxx.
        directory (str, optional): Directory to cache the dataset.
        train (bool, optional): If to load the training split of the dataset.
        test (bool, optional): If to load the test split of the dataset.
        train_file (str, optional): xxx.
        test_file (str, optional): xxx.

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset`
        Returns between one and all dataset splits (train, dev and test) depending on if their
        respective boolean argument is ``True``.

    """
    datasets = [
        'SST2',
        'CR',
        'procon',
        'SUBJ',
        'TREC',
        'Restaurants',
        'Restaurants16',
        'Laptops',
        'Tweets',
        "IMDB"
    ]
    if dataset not in datasets:
        raise ValueError('dataset: {} not in support list!'.format(dataset))

    if directory == 'datasets_processed':
        assert (augmented_mode == None)
    if directory in ('datasets_processed', 'datasets_augmented',
                     'datasets_augmented_pure_eda', 'datasets_augmented_manual'):
        directory = os.path.join(directory, dataset)

    if augmented_mode and train_file:
        train_file = augmented_mode + '_' + str(percentage) + '_' + train_file
    else:
        train_file = str(percentage) + '_' + train_file

    ret = []
    splits = [
        '_'.join([dataset, fn_]) for (requested, fn_) in [(train, train_file), (test, test_file)]
        if requested
    ]
    for split_file in splits:
        full_filename = os.path.join(directory, split_file)
        examples = []
        with open(full_filename, 'r', encoding="utf-8") as f:
            for j in f.readlines():
                a_data = json.loads(j)
                sent = a_data["sentence"].lower()
                label = a_data["polarity"]
                if load_pos:
                    pos = a_data["pos"]
                    examples.append([[sent, pos], label])
                else:
                    examples.append([sent, label])
        ret.append(examples)
    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)


def build_embedding_matrix(vocab, dataset, word_dim=300, directory='datasets_processed',
                           augmented_mode=None, percentage=1.0):

    if directory == 'datasets_processed':
        assert (augmented_mode == None)

    if augmented_mode:
        dataset = dataset + augmented_mode + '_' + str(percentage)
    else:
        dataset = dataset + '_' + str(percentage)

    if not os.path.exists(os.path.join('dats', directory)):
        os.mkdir(os.path.join('dats', directory))
    if not os.path.exists(os.path.join(os.path.join('dats', directory), dataset)):
        os.mkdir(os.path.join(os.path.join('dats', directory), dataset))
    parent_dir = os.path.join(os.path.join('dats', directory), dataset)

    data_file = os.path.join(parent_dir, f"{dataset}_embedding_matrix.dat")  # embedding matrix cache
    glove_file = os.path.join('glove', 'glove.840B.300d.txt') # pre-trained glove embedding file
    if os.path.exists(data_file):
        print(f"loading embedding matrix: {data_file}")
        embedding_matrix = pickle.load(open(data_file, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.random.uniform(-0.25, 0.25, (len(vocab), word_dim)).astype('float32') # sample from U(-0.25,0.25)
        word_vec = _load_wordvec(glove_file, word_dim, vocab)
        for i in range(len(vocab)):
            vec = word_vec.get(vocab[i])
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(data_file, 'wb'))
    return embedding_matrix


def _load_wordvec(data_path, word_dim, vocab=None):
    with open(data_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        word_vec = dict()
        word_vec['<pad>'] = np.zeros(word_dim).astype('float32')  # embedding vector for <pad> is always zero
        for line in f:
            tokens = line.rstrip().split()
            if (len(tokens) - 1) != word_dim:
                continue
            if tokens[0] == '<pad>' or tokens[0] == '<unk>': # avoid them
                continue
            if vocab is None or tokens[0] in vocab:
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
        return word_vec


class PosClassEncoder():
    def __init__(self, pos2class, class2pos):
        '''
        pos2class: {'NN':1, 'JJ':1...}
        class2pos: {1:('NN', 'JJ')...}
        '''
        self.pos2class = pos2class
        self.class2pos = class2pos
        self.pos_class = len(pos2class.keys())

    @classmethod
    def from_path(cls, pos_path):
        pos2class, class2pos = dict(), defaultdict(set)
        with open(pos_path, "r") as fo:
            for ind, line in enumerate(fo.readlines()):
                if ind == 0:
                    continue
                line = line.split()
                pos2class[line[1]] = int(line[-1])
        pos2class['unknown'] = 1
        for pos, cla in pos2class.items():
            class2pos[cla].add(pos)
        return cls(pos2class, class2pos)

    def encode(self, list_pos):
        list_class = []
        for pos in list_pos:
            if pos not in self.pos2class.keys():
                list_class.append(self.pos2class['unknown'])
            else:
                list_class.append(self.pos2class[pos])
        return list_class


def _tfidf(dataset='SST2',
              directory='datasets_processed',
              train=True,
              test=False,
              train_file='Train.json',
              test_file='None',
              ):

    datasets = [
        'SST2',
        'CR',
        'procon',
        'SUBJ',
        'TREC'
    ]
    if directory == 'datasets_processed':
        directory = os.path.join(directory, dataset)
    if dataset not in datasets:
        raise ValueError('dataset: {} not in support list!'.format(dataset))
    splits = [
        '_'.join([dataset, fn_]) for (requested, fn_) in [(train, train_file), (test, test_file)]
        if requested
    ]
    for split_file in splits:
        full_filename = os.path.join(directory, split_file)
        all_sents = defaultdict(str)
        labels_list = []
        sents_list = []
        with open(full_filename, 'r', encoding="utf-8") as f:
            for j in f.readlines():
                a_data = json.loads(j)
                sent = a_data["sentence"].lower()
                label = a_data["polarity"]
                all_sents[label] += (sent + " ")
        for label, sents in all_sents.items():
            sents = " ".join(sents.split())
            labels_list.append(label)
            sents_list.append(sents)

    tfidf_vec = TfidfVectorizer()
    tfidf_matrix = tfidf_vec.fit_transform(sents_list)

    # tfidf_vec.get_feature_names()
    # tfidf_vec.vocabulary_
    # tfidf_matrix.toarray()

    with open(os.path.join(directory, f'{dataset}_vocabulary.json'), 'w') as fw1:
        fw1.write(json.dumps(tfidf_vec.vocabulary_))
    np.save(os.path.join(directory, f'{dataset}_matrix.npy'), tfidf_matrix.toarray())
    print(f">>> {dataset} TF-IDF vocabulary and matrix saved! ")


if __name__ == "__main__":
    dataset = 'CR'
    _tfidf(dataset)