import re
import os
import json
import shutil
import numpy as np
import random
import string
import difflib

import corenlp
os.environ["CORENLP_HOME"] = r"/home/LAB/chenqb/stanford-corenlp-4.0.0"
special_tokens_procon = ["\.{3,}", "\.{1}", "\!{1,}", "\?{1,}", "\,{1,}", "\'s", "\(", "\)", "&amp;", "#.*;", ";"]
to_tokens_procon = [",", ".", "!", "?", ",", "'s", "(", ")", "", "", ""]
special_tokens_IMDB = ["<br /><br />", "\.{1}", "\!{1,}", "\?{1,}", '\\"', "\'"]
to_tokens_IMDB = ["", ".", "!", "?", '"', ""]
PUNC_LIST = string.punctuation


def preprocess_pos_ratio(dataset, load_directory='datasets',
                         save_directory='datasets_processed',
                         percentage=1.0, train=False, test=False, if_pos=False):
    '''
    preprocess data by adding part-of-speech and choosing a certain percentage of data
    '''
    datasets = [
        'SST2',
        'CR',
        'procon',
        'SUBJ',
        'TREC',
    ]
    assert (0. <= percentage <= 1.)
    if load_directory == 'datasets_processed':
        load_directory = os.path.join(load_directory, dataset)
    if save_directory == 'datasets_processed':
        save_directory = os.path.join(save_directory, dataset)
    if dataset not in datasets:
        raise ValueError('dataset: {} not in support list!'.format(dataset))
    modes = []
    if train:
        modes.append("Train")
    if test:
        modes.append("Test")

    with corenlp.CoreNLPClient(annotators=['pos'], be_quiet=True, timeout=30000,
                               memory='4G', endpoint='http://localhost:1337') as client:
        for mode in modes:
            if mode == 'Train':
                full_filename_in = dataset + "_" + mode + ".json"
                full_filename_out = dataset + f"_{float(percentage)}_" + mode + ".json"
            else:
                full_filename_in = dataset + "_" + mode + ".json"
                full_filename_out = dataset + "_" + mode + ".json"
            # if os.path.exists(os.path.join(save_directory, full_filename_out)):
            #     continue
            with open(os.path.join(load_directory, full_filename_in), "r", encoding="utf-8") as fo:
                with open(os.path.join(save_directory, full_filename_out), "w", encoding="utf-8") as fw:
                    lines = fo.readlines()
                    random.shuffle(lines)
                    for ind_l, j in enumerate(lines):
                        a_data = json.loads(j)
                        sent = a_data["sentence"].lower()
                        if dataset == "procon":
                            for ind, special_token in enumerate(special_tokens_procon):
                                sent = re.sub(special_token, " " + to_tokens_procon[ind] + " ", sent)
                            sent = " ".join(sent.split())

                            ann = client.annotate(sent)
                            pos = []
                            sent_new = []
                            for ann_sent in ann.sentence: 
                                for token in ann_sent.token:
                                    pos.append(token.pos)
                                    sent_new.append(token.word)
                        elif dataset == "IMDB":
                            for ind, special_token in enumerate(special_tokens_IMDB):
                                sent = re.sub(special_token, " " + to_tokens_IMDB[ind] + " ", sent)
                            sent = " ".join(sent.split())

                            ann = client.annotate(sent)
                            pos = []
                            sent_new = []
                            for ann_sent in ann.sentence:
                                for token in ann_sent.token:
                                    pos.append(token.pos)
                                    sent_new.append(token.word)
                        else:
                            ann = client.annotate(sent)
                            ann_sent = ann.sentence[0]
                            pos = [token.pos for token in ann_sent.token]
                            sent_new = [token.word for token in ann_sent.token]
                        label = a_data["polarity"]
                        sent_new = " ".join(sent_new)

                        if if_pos and "gate" in a_data.keys() and len(pos) != len(a_data["gate"]):
                            raise ValueError(f"found invalid data: {a_data['sentence']}, "
                                             f"length {len(a_data['sentence'].split())}, "
                                             f"pos {len(pos)}, gate {len(a_data['gate'])}")

                        a_data["sentence"] = sent_new
                        if pos:
                            a_data["pos"] = pos
                        # json.dump({"sentence": sent, "pos": pos, "polarity": label}, fw)
                        fw.write(f'{json.dumps(a_data)}\n')

                        if mode == 'Train' and ind_l > percentage * len(lines):
                            break
    print(f"{percentage*100}% {dataset} data preprocessed by adding PoS !")


def preprocess_alsc(dataset, load_directory='datasets_raw', save_directory='datasets'):
    '''
    preprocess ALSC data by removing P signals in all datasets and those datasets with contradictory labels
    '''
    datasets = [
        'Restaurants',
        'Laptops',
        'Tweets',
        'Restaurants16',
    ]
    if dataset not in datasets:
        raise ValueError('dataset: {} not in support list!'.format(dataset))
    if dataset in ('Restaurants', 'Laptops'):
        has_keywords = True
    else:
        has_keywords = False
    modes = ["Train", "Test"]
    for mode in modes:
        full_filename_in = dataset + '_' + mode + '.json'
        full_filename_out = full_filename_in
        if not os.path.exists(os.path.join(save_directory, dataset)):
            os.mkdir(os.path.join(save_directory, dataset))
        with open(os.path.join(load_directory, full_filename_in), "r", encoding="utf-8") as fo:
            with open(os.path.join(save_directory, full_filename_out), "w", encoding="utf-8") as fw:
                cur_sentence = None
                cur_polarities = set()
                for j in fo.readlines():
                    a_data = json.loads(j)
                    sentence = a_data['sentence']
                    sentence = re.sub('<p>', '', sentence)
                    sentence = re.sub('</p>', '', sentence)
                    sentence = ' '.join(sentence.split())
                    polarity = a_data['polarity']
                    if has_keywords and mode == 'Train':
                        keywords = a_data['keywords']
                        
                    if not cur_sentence:
                        cur_sentence = sentence
                        cur_polarities.add(polarity)
                        if has_keywords and mode == 'Train':
                            cur_keywords = keywords

                    if sentence == cur_sentence:
                        cur_polarities.add(polarity)
                    else:
                        if len(cur_polarities) == 1:
                            if has_keywords and mode == 'Train':
                                fw.write(f'{json.dumps({"sentence": cur_sentence, "keywords": cur_keywords, "polarity": cur_polarities.pop()})}\n')
                            else:
                                fw.write(f'{json.dumps({"sentence": cur_sentence, "polarity": cur_polarities.pop()})}\n')
                        cur_polarities = set()
                        cur_polarities.add(polarity)
                        cur_sentence = sentence
                        if has_keywords and mode == 'Train':
                            cur_keywords = keywords
    print(f"{dataset} data preprocessed by removing p signals and label-contradictory datasets !")


def cut_percentage(dataset, load_directory='datasets_raw', save_directory='datasets',
                   percentage=1.0):
    full_filename_in = dataset + "_Train.json"
    full_filename_out = dataset + "_" + str(percentage) + "_Train.json"
    with open(os.path.join(load_directory, full_filename_in), "r", encoding="utf-8") as fo:
        with open(os.path.join(save_directory, full_filename_out), "w", encoding="utf-8") as fw:
            lines = fo.readlines()
            N = len(lines)
            for idx, line in enumerate(lines):
                a_data = json.loads(line)
                fw.write(f"{json.dumps(a_data)}\n")
                if idx >= N * percentage:
                    break
    print("done")
    return


def repath_imdb(path, save_dir="datasets"):
    for sub_path in ("train", "test"):
        full_path = os.path.join(path, sub_path)

        pos_files = os.listdir(full_path + '/pos')
        neg_files = os.listdir(full_path + '/neg')

        pos_all = []
        neg_all = []
        for pf, nf in zip(pos_files, neg_files):
            with open(full_path + '/pos' + '/' + pf, encoding='utf-8') as f:
                s = f.read()
                pos_all.append(s)
            with open(full_path + '/neg' + '/' + nf, encoding='utf-8') as f:
                s = f.read()
                neg_all.append(s)

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        X_orig = np.array(pos_all + neg_all)
        Y_orig = np.array([1 for _ in range(len(pos_all))] + [0 for _ in range(len(neg_all))])

        if sub_path == "train":
            sub_path = "Train"
        else:
            sub_path = "Test"
        print(sub_path)

        with open(os.path.join(save_dir, f"IMDB_{sub_path}.json"), "w", encoding="utf-8") as fw:
            for x, y in zip(X_orig, Y_orig):
                fw.write(f'{json.dumps({"sentence": x, "polarity": str(y)})}\n')


manual_datasets = [
    'SST2'
]
other_datasets = [
    'TREC',
    'CR',
    'SUBJ',
    'procon'
]
wanted_datasets = manual_datasets + other_datasets


def preprocess_data_new(dataset='SST2',
              load_directory='./datasets_manual',
              save_directory="./datasets_processed",
              port=9000,
              train=True,
              test=True
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
    from transformers import BertTokenizer
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if dataset not in wanted_datasets:
        raise ValueError('dataset: {} not in support list!'.format(dataset))
    full_load_filenames, full_save_filenames = [], []
    if train:
        if dataset.lower() == "snli_1.0":
            full_load_filenames.append(os.path.join(load_directory, dataset + "_train.jsonl"))
        else:
            full_load_filenames.append(os.path.join(load_directory, dataset + "_Train.json"))
        full_save_filenames.append(os.path.join(save_directory, dataset + "_Train.json"))
    if test:
        if dataset.lower() == "snli_1.0":
            full_load_filenames.append(os.path.join(load_directory, dataset + "_test.jsonl"))
        else:
            full_load_filenames.append(os.path.join(load_directory, dataset + "_Test.json"))
        full_save_filenames.append(os.path.join(save_directory, dataset + "_Test.json"))
    if not full_load_filenames:
        print("wrong args for 'train' and 'test'! ")
        return

    with corenlp.CoreNLPClient(properties={'annotators': ['tokenize', 'pos'],
                                           'tokenize.whitespace': 'True'},
                               endpoint=f'http://localhost:{port}',
                               be_quiet=True,
                               memory='32G',
                               timeout=10000000) as client:
        for load_filename, save_filename in zip(full_load_filenames, full_save_filenames):
            with open(load_filename, 'r', encoding="utf-8") as fr:
                with open(save_filename, 'w', encoding="utf-8") as fw:
                    tmp = fr.readlines()
                    N = len(tmp)
                    for idx, j in enumerate(tmp):
                        a_data = json.loads(j)
                        sent = a_data["sentence"]
                        if dataset not in ("snli_1.0", "agnews", "yahoo"):
                            sent = sent.lower()
                        sub_sent_list = bert_tokenizer.tokenize(sent)
                        sent_list = reformulate_sub_sent_list(sub_sent_list)
                        sent = " ".join(sent_list)

                        if "train" in load_filename.lower():
                            pos = []
                            new_sent_str = ""
                            ann = client.annotate(sent)
                            for ann_sent in ann.sentence:
                                for token in ann_sent.token:
                                    pos.append(str(token.pos))  # (sl, )
                                    new_sent_str += (" " + sent[token.beginChar:token.endChar])
                            sent = new_sent_str.strip()
                            a_data["sentence"] = sent
                            a_data["pos"] = pos

                            if len(sent.split(" ")) != len(pos):
                                print(f"wrong data {sent} {sent_list} {pos}")
                                print(f"{len(sent_list)} {len(pos)}")
                        else:
                            new_sent_str = ""
                            ann = client.annotate(sent)
                            for ann_sent in ann.sentence:
                                for token in ann_sent.token:
                                    new_sent_str += (" " + sent[token.beginChar:token.endChar])
                            sent = new_sent_str.strip()
                            a_data["sentence"] = sent

                        if "keywords" in a_data.keys():
                            sent_list = sent.split(" ")
                            keywords = [k.lower() for k in a_data["keywords"]]
                            gate = []
                            for w in sent_list:
                                g = max([string_similar(w, k) for k in keywords])
                                g = 1 if g > 0.9 else 0
                                gate.append(g)
                            a_data["gate"] = gate
                            a_data["keywords"] = keywords

                        fw.write(f'{json.dumps(a_data)}\n')
    print("done")
    return


def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()


def reformulate_sub_sent_list(sub_sent_list):
    sent_list = []
    for sub_word in sub_sent_list:
        if sub_word.startswith("##"):
            sub_word = sub_word.replace('##', '')
            sent_list[-1] += sub_word
        else:
            sent_list.append(sub_word)
    return sent_list


def preprocess_snli_data(dataset,
                         load_directory="./datasets_raw/snli_1.0",
                         save_directory="./datasets_processed/"):
    full_load_filenames = [os.path.join(load_directory, dataset + "_train.jsonl"),
                           os.path.join(load_directory, dataset + "_test.jsonl")]
    full_save_filenames = [os.path.join(save_directory, dataset + "_Train.json"),
                           os.path.join(save_directory, dataset + "_Test.json")]
    with corenlp.CoreNLPClient(annotators=['ssplit', 'pos']) as client:
        for load_filename, save_filename in zip(full_load_filenames, full_save_filenames):
            with open(load_filename, 'r') as fo:
                with open(save_filename, 'w') as fw:
                    tmp = fo.readlines()
                    for idx, j in enumerate(tmp):
                        j = json.loads(j)
                        sentence1, sentence2, label = j["sentence1"], j["sentence2"], j["gold_label"]
                        sentence1, sentence2 = sentence1.lower().strip(PUNC_LIST), sentence2.lower().strip(PUNC_LIST)
                        # sent = ["[CLS]"] + sentence1.split(" ") + ["[SEP]"] + sentence2.split(" ") + ["[SEP]"]
                        sent = sentence1.split(" ") + ["[SEP]"] + sentence2.split(" ")
                        sent = " ".join(sent)
                        sent = re.sub(r"[\']+", "", sent)
                        sent = re.sub(r"[-]+", " ", sent)
                        sent = re.sub(r"!+", "!", sent)
                        sent = re.sub(r"\.+", ".", sent)
                        sent_list = sent.split(" ")
                        sent_list = [token for token in sent_list if token not in PUNC_LIST]
                        sent = " ".join(sent_list)
                        fw.write(f'{json.dumps({"sentence": sent, "polarity": label})}\n')
    return

if __name__ == "__main__":
    # Restaurants, Laptops, SST2

    # dataset = "Restaurants"
    # for percentage in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]:
    #     cut_percentage(dataset, load_directory='datasets_pred_bert_adaptive_dropout',
    #                    save_directory='datasets_pred_bert_adaptive_dropout',
    #                    percentage=percentage)

    # dataset = "SST2"
    # # preprocess_alsc(dataset, load_directory='datasets_raw', save_directory='datasets')
    # for percentage in (0.05, 0.2, 0.5, 1.0):
    #     preprocess_pos_ratio(dataset, load_directory='datasets',
    #                          save_directory='datasets_processed', percentage=percentage)

    # path = r'./aclImdb'
    # save_dir = r'./datasets'
    # repath_imdb(path, save_dir)

    # preprocess_snli_data(dataset="snli_1.0",
    #                      load_directory="./datasets_raw/snli_1.0",
    #                      save_directory="./datasets_manual/")

#     dataset = "procon"
#     load_directory = "./datasets_manual"
#     save_directory = "./datasets_processed"
#     port = 5555
#     train = True
#     test = True
#     preprocess_data_new(dataset, load_directory, save_directory, port, train, test)
