# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import pickle
import json
import numpy as np
import pandas as pd
from transformers import BertTokenizer
import random

from sql import SqlKeyWords
from sql import SqlQueryLoader
from constant import data_dir
from constant import spider_dir
from graph import SqlGraphConverter

from database import Database
from database import NodeType


def format_sample_t1(questions, sql_jsons, db_ids, slots, database):
    db_name_list_map = {}
    for db_id in database:
        db = database[db_id]
        name_list = [node.get_full_name() for node in db.tbl_nodes + db.col_nodes]
        db_name_list_map[db_id] = name_list

    sample_size = len(questions)
    bert_inputs = [[], [], [], [], []]
    slot_mapping = []
    print(sample_size)
    for sample_idx in range(sample_size):
        if sample_idx % 100 == 0:
            print(sample_idx)
        db_id = db_ids[sample_idx]
        db = database[db_id]
        que = questions[sample_idx]
        sql_json = sql_jsons[sample_idx]
        slot = slots[sample_idx]

        # Bert
        name_list = db_name_list_map[db_id]
        input_ids, input_mask, segment_ids, indicator_ids, que_tok_map = \
            IDConverter.get_bert_input(name_list, que)

        if not input_ids:
            continue

        bert_inputs[0].append(input_ids)
        bert_inputs[1].append(input_mask)
        bert_inputs[2].append(segment_ids)
        bert_inputs[3].append(indicator_ids)
        bert_inputs[4].append(que_tok_map)

        # Slot
        slot = IDConverter.convert_slot(slot, db)
        slot_mapping.append(slot)
        assert max(slot) <= len(db_name_list_map[db_id]), \
            "max slot: %d; name size: %d" % (max(slot), len(db_name_list_map[db_id]))


    bert_inputs = [np.array(c) for c in bert_inputs]
    slot_mapping = np.array(slot_mapping)
    return {"bert_input": [*bert_inputs],
            "slot": [slot_mapping]}


def format_bert_input(questions, db_ids, database):
    db_name_list_map = {}
    for db_id in database:
        db = database[db_id]
        db_name_list_map[db_id] = [node.get_full_name() for node in db.tbl_nodes + db.col_nodes] #???

    data_size = len(questions)
    bert_inputs = [[], [], [], [], []]

    for sample_idx in range(data_size):
        # if sample_idx == 2000:
        #     break

        db_id = db_ids[sample_idx]
        que = questions[sample_idx]

        name_list = db_name_list_map[db_id]
        input_ids, input_mask, segment_ids, indicator_ids, que_tok_map = \
            IDConverter.get_bert_input(name_list, que)
        #t = [len(input_ids), len(input_mask),len(segment_ids), len(indicator_ids), len(que_tok_map)]
        #print(t)
        # if not input_ids:
        #     continue

        bert_inputs[0].append(input_ids)
        bert_inputs[1].append(input_mask)
        bert_inputs[2].append(segment_ids)
        bert_inputs[3].append(indicator_ids)
        bert_inputs[4].append(que_tok_map)

    bert_inputs = [np.array(c) for c in bert_inputs]
    return bert_inputs


def format_sample_t2(sql_json, db_id, database, bert_input):
    assert len(sql_json) == len(db_id)
    assert len(sql_json) == len(bert_input[0])
    data_size = len(sql_json)
    cl_his, cl_node, cl_label = [], [], []
    cl_inputs = [[] for _ in range(len(bert_input))]
    for i in range(data_size):
        if not bert_input[0][i]:
            continue

        sql = SqlQueryLoader.load(sql_json[i], database[db_id[i]])
        batch_his, batch_node, batch_label = format_cl_ins(sql)
        cl_his.extend(batch_his)
        cl_node.extend(batch_node)
        cl_label.extend(batch_label)

        batch_size = len(batch_his)
        for _ in range(batch_size):
            for comp_i in range(len(bert_input)):
                cl_inputs[comp_i].append(bert_input[comp_i][i])

    cl_length = [len(i) for i in cl_his]
    return cl_inputs, cl_his, cl_length, cl_node, cl_label


def format_sample_t3(sql_json, db_id, database, bert_input):
    assert len(sql_json) == len(db_id)
    data_size = len(sql_json)

    kw_ins, kw_label = [], []
    kw_inputs = [[] for _ in range(len(bert_input))]
    for i in range(data_size):
        sql = SqlQueryLoader.load(sql_json[i], database[db_id[i]])
        kw_ins_batch, kw_label_batch = format_kw_ins(sql)
        kw_ins.extend(kw_ins_batch)
        kw_label.extend(kw_label_batch)

        kw_batch_size = len(kw_ins)
        for _ in range(kw_batch_size):
            for comp_i in range(len(bert_input)):
                kw_inputs[comp_i].append(bert_input[comp_i][i]) #???

    return kw_inputs, kw_ins, kw_label


def format_cl_ins(sql):
    db = sql.db
    t = SqlGraphConverter.sql2graph(sql)

    pos_node, neg_node = [], []
    pos_his, neg_his = [], []
    pos_label, neg_label = [], []
    t_n_id_list = [i for i in t.expand_tree() if i != "root"]
    for t_n_id in t_n_id_list:
        his_t = list(reversed(list(t.rsearch(t_n_id))))
        cur_his = []
        root_bias = 1  # add a node at the beginning of node list
        for i in his_t:
            g_id = 0 if i == "root" else db.get_global_node_id(t[i].data.node) + root_bias
            cur_his.append(g_id)

        pos_sample, neg_sample = sample_cl(t_n_id, t, db)
        for db_g_id in pos_sample:
            pos_his.append(cur_his)
            pos_node.append(db_g_id+root_bias)
            pos_label.append(1)

        for db_g_id in neg_sample:
            neg_his.append(cur_his)
            neg_node.append(db_g_id+root_bias)
            neg_label.append(0)

    his = pos_his+neg_his
    node = pos_node+neg_node
    label = pos_label+neg_label

    return his, node, label


def sample_cl(t_id, tree, db):
    db_node = tree[t_id].data.node
    n_id, n_type = db_node.get_node_id(), db_node.get_node_type()

    pos_t_id = list(tree.children(t_id))
    pos_db_node = [i.data.node for i in pos_t_id]
    pos_sample = [db.get_global_node_id(n) for n in pos_db_node]
    pos_num = len(pos_sample)
    if not pos_num:
        return [], []

    if t_id == "root":  # table child
        candidates = [db.get_global_node_id(tbl) for tbl in db.tbl_nodes]
        candidates = [i for i in candidates if i not in pos_sample]

    elif n_type == NodeType.TBL:  # column child
        tbl_id = db_node.get_node_id()
        candidates = [db.get_global_node_id(col) for col in db.col_nodes if db.col_tbl[col.get_node_id()] == tbl_id]
        candidates = [c_id for c_id in candidates if c_id not in pos_sample]

    elif n_type == NodeType.COL or n_type == NodeType.VAL:  # free child
        candidates = [db.get_global_node_id(n) for n in db.tbl_nodes+db.col_nodes]
        candidates = [i for i in candidates if i not in candidates]

    else:
        assert False, ("Bad node type, id:", t_id)

    neg_num = min(pos_num, len(candidates))
    return pos_sample, random.sample(candidates, neg_num)


def format_kw_ins(sql):
    db = sql.db
    t = SqlGraphConverter.sql2graph(sql)
    node_id = [n for n in t.expand_tree() if n[:4] != "root"]
    node_kw = [t[n].data.get_kws() for n in t.expand_tree() if n[:4] != "root"]
    node_his = [list(reversed(list(t.rsearch(i)))) for i in node_id]
    node_his = [[db.get_global_node_id(t[j].data.node) for j in i if t[j].data.node != "root"] for i in node_his]
    node_kw = [[SqlKeyWords.get_index(SqlKeyWords.unify(kw)) for kw in kws] for kws in node_kw]
    return node_his, node_kw


def format_attr_ins(sql): #???
    db = sql.db
    t = SqlGraphConverter.sql2graph(sql)
    node_id = [n for n in t.expand_tree() if n[:4] != "root"]
    node_kw = [t[n].data.get_kws() for n in t.expand_tree() if n[:4] != "root"]
    node_attr = [t[n].data.get_attrs() for n in t.expand_tree() if n[:4] != "root"]
    node_his = [list(reversed(list(t.rsearch(i)))) for i in node_id]
    node_his = [[db.get_global_node_id(t[j].data.node) for j in i if t[j].data.node != "root"] for i in node_his]

    attr_node_his, attr_kw, attr_label = [], [], []
    for h, k, a in zip(node_his, node_kw, node_attr):
        for _k, _a in zip(k, a):
            attr_node_his.append(h)
            attr_kw.append(_k)
            attr_label.append(_a)

    return attr_node_his, attr_kw, attr_label


def load_glove_tokenizer(vocab_size=6, emb_size=100):
    dict_dir = '{}pretrain/{}B.{}d_idx.pkl'.format(data_dir, vocab_size, emb_size)
    with open(dict_dir, "rb") as file:
        tokenizer = pickle.load(file)

    return tokenizer


def padding(seq, max_length, pad_tok=None):
    if type(seq) != list:
        seq = seq.tolist()
    return (seq + [pad_tok] * max_length)[:max_length]


def vis_counter(counter, fig_name="image"):
    color, line, mark = "r", "", "-"
    counter = sorted(counter.items(), key=lambda x: x[1])
    counter.reverse()
    labels, values = zip(*counter)
    indexes = np.arange(len(labels))
    plt.plot(indexes, values, color + line + mark, label='type1')
    plt.savefig(fig_name)
    plt.clf()


def get_levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


class IDConverter:
    MAX_QUE_LEN = 50
    MAX_DBS_NODE_NUM = 380
    MAX_NODE_TOK_NUM = 5
    WORD_PAD_TOK = "<UNK>"
    UNCASED = './bert-base-uncased'
    VOCAB = 'vocab.txt'
    import os
    bert_tokenizer = BertTokenizer.from_pretrained(os.path.join(UNCASED,VOCAB))
    glove_tokenizer = load_glove_tokenizer()


    @classmethod
    def get_bert_input(cls, node_list, question):

        def tokenize_with_mapping(token, tokenizer):
            bert_tokens = []
            orig_to_tok_map = []

            for orig_token in token:
                orig_to_tok_map.append(len(bert_tokens))
                bert_tokens.extend(tokenizer.tokenize(orig_token))

            assert len(token) == len(orig_to_tok_map)
            return bert_tokens, orig_to_tok_map

        bert_tokenizer = cls.bert_tokenizer
        node_list = [bert_tokenizer.tokenize(node) for node in node_list] # split
        tokens_que, que_tok_map = tokenize_with_mapping(question, bert_tokenizer)

        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        max_seq_length = 512

        tokens_a = ["[CLS]", "[none]", "[SEP]"]
        for node in node_list:
            tokens_a += node
            tokens_a += ["[SEP]"]

        if len(tokens_a) + len(tokens_que) + 1 > max_seq_length:
            return [], [], [], [], []

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector".

        len_a = len(tokens_a)
        segment_ids = [0] * len_a
        que_tok_map = [i + len_a for i in que_tok_map]

        tokens = tokens_a + tokens_que + ["[SEP]"]
        segment_ids += [1] * (len(tokens_que) + 1)

        indicator_ids = [int(i == "[SEP]") for i in tokens]
        indicator_ids[-1] = 0 #标记sep符号

        assert sum(indicator_ids) == len(node_list) + 1

        input_ids = bert_tokenizer.convert_tokens_to_ids(tokens) #???

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        # padding_item = [0] * (max_seq_length - len(input_ids))
        # input_ids += padding_item
        # input_mask += padding_item
        # segment_ids += padding_item
        #
        # assert len(input_ids) == max_seq_length
        # assert len(input_mask) == max_seq_length
        # assert len(segment_ids) == max_seq_length

        return input_ids, input_mask, segment_ids, indicator_ids, que_tok_map

    # @classmethod
    # def convert_sql_kws(cls, kws):
    #     return [SqlKeyWords.get_index(kw) for kw in kws]
    #
    # @classmethod
    # def convert_db_node(cls, entity_node, db):
    #     entity = [0 for _ in range(db.node_num)]
    #     for nodes in entity_node:
    #         for node in nodes:
    #             node_id = db.get_global_node_id(node)
    #             entity[node_id] = 1
    #     return entity

    @classmethod
    def convert_slot(cls, slot, db): # 返回一个list 长度和question一样的list e_i 对应了第i个节点index
        keys = ["tables", "columns", "values"]
        type_map = {
            "tables": NodeType.TBL,
            "columns": NodeType.COL,
            "values": NodeType.VAL
        }
        tok_num = len(slot["columns"])
        none_node_idx = 0
        mapping = [none_node_idx for _ in range(tok_num)]

        for key in keys:
            for tok_idx in range(tok_num):
                tok = slot[key][tok_idx]
                tok_node, tok_str = tok #???

                if tok_node is None:
                    continue

                node_type = type_map[key]
                node_id = tok_node[0]
                node_id = db.get_global_id(node_id, node_type)
                mapping[tok_idx] = node_id + 1  # one bias caused by none_node

        return mapping


class SpiderDataLoader:
    spider_path_trn = spider_dir + "train_spider.json"
    spider_path_dev = spider_dir + "dev.json"

    # db_id, query, query_toks, query_toks_no_value, question, question_toks, sql

    spider_annotation_trn = spider_dir + "annotated_train_spider.json"
    spider_annotation_dev = spider_dir + "annotated_dev_spider.json"

    # cmps, count_tbl, db_id, deps, group, having, id, lemma, ner, nests ???
    # norm_query, order, pos, query, question, sels, sen_pos, slots, tbls, toks ???

    @classmethod
    def load_spider_question(cls):
        questions_trn = cls.__load_attr_trn("question")
        questions_dev = cls.__load_attr_dev("question")
        return questions_trn, questions_dev

    @classmethod
    def load_spider_query(cls):
        query_trn = cls.__load_attr_trn("query")
        query_dev = cls.__load_attr_dev("query")
        return query_trn, query_dev

    @classmethod
    def load_spider_sql_json(cls):
        sql_trn = cls.__load_attr_trn("sql")
        sql_dev = cls.__load_attr_dev("sql")
        return sql_trn, sql_dev

    @classmethod
    def load_spider_sql_db_id(cls):
        db_id_trn = cls.__load_attr_trn("db_id")
        db_id_dev = cls.__load_attr_dev("db_id")
        return db_id_trn, db_id_dev

    @classmethod
    def load_slot_filling(cls):
        slot_trn = cls.__load_annotation_attr_trn("slots")
        slot_dev = cls.__load_annotation_attr_dev("slots")
        return slot_trn, slot_dev

    @classmethod
    def load_lemma(cls):
        lemma_trn = cls.__load_annotation_attr_trn("lemma")
        lemma_dev = cls.__load_annotation_attr_dev("lemma")
        return lemma_trn, lemma_dev

    @classmethod
    def load_spider_dbs(cls):
        file_path = spider_dir + "tables.json"

        database_tables = pd.read_json(file_path)
        keys = database_tables.columns
        databases = {
            db_item[3]: Database(
                {
                    k: v for k, v in zip(keys, db_item)
                }
            ) for db_item in database_tables.values
        }
        return databases

    @classmethod
    def __load_attr_trn(cls, attr):
        file_path = cls.spider_path_trn
        attrs = cls.__load_json_item(file_path, attr)
        return attrs

    @classmethod
    def __load_attr_dev(cls, attr):
        file_path = cls.spider_path_dev
        attrs = cls.__load_json_item(file_path, attr)
        return attrs

    @classmethod
    def __load_annotation_attr_trn(cls, attr):
        file_path = cls.spider_annotation_trn
        attrs = cls.__load_json_item(file_path, attr)
        return attrs

    @classmethod
    def __load_annotation_attr_dev(cls, attr):
        file_path = cls.spider_annotation_dev
        attrs = cls.__load_json_item(file_path, attr)
        return attrs

    @classmethod
    def __load_json_item(cls, file_path, attr):
        with open(file_path, 'r') as f:
            data = json.load(f)

        attrs = []
        for sample in data:
            item = sample[attr]
            if type(item) == str and attr != "db_id":
                item = item.lower()
            attrs.append(item)

        return attrs


class Stack(object):
    def __init__(self):
        self.__stack__ = []

    def pop(self):
        return self.__stack__.pop(-1)

    def push(self, e):
        self.__stack__.append(e)

    def size(self):
        return len(self.__stack__)

    def top(self):
        return self.__stack__[-1]

    def is_empty(self):
        return len(self.__stack__) == 0

    def __str__(self):
        return " ".join([str(element) for element in self.__stack__])


class Queue(object):
    def __init__(self):
        self.__queue__ = []

    def pop(self):
        return self.__queue__.pop(0)

    def push(self, e):
        self.__queue__.append(e)

    def size(self):
        return len(self.__queue__)

    def is_empty(self):
        return len(self.__queue__) == 0

    def __str__(self):
        return " ".join([str(element) for element in self.__queue__])

