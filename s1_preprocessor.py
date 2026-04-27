#coding=utf-8
import codecs
import json
import os
import pickle
import re
from collections import Counter
import numpy as np
from tqdm import tqdm
from config import *

def simple_tokenize_code(code_str):
    """简化的正则分词，保留英文字母、数字和独立的标点符号"""
    tokens = re.findall(r'[A-Za-z_]+|[0-9]+|[^\w\s]', code_str)
    return tokens

def tokenize_raw_data(raw_data_path, token_data_path, max_code_len=max_code_len, max_text_len=max_text_len):
    logging.info('########### Start tokenize 1D Sequence data ##########')
    token_data_dir = os.path.dirname(token_data_path)
    if not os.path.exists(token_data_dir): 
        os.makedirs(token_data_dir)
        
    with codecs.open(raw_data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    pbar = tqdm(raw_data)
    for item in pbar:
        # 1. 提取代码纯一维序列
        code_str = item['code'].replace('\\n', ' ').strip()
        code_tokens = simple_tokenize_code(code_str)[:max_code_len]
        
        # 将数字替换为特殊符号（泛化）
        code_tokens = ['<number>' if token.isdigit() else token for token in code_tokens]
        
        # 只存 nodes（序列），放弃任何边
        item['graph'] = {'nodes': str(code_tokens)}

        # 2. 处理自然语言文本
        text_tokens = simple_tokenize_code(item['text'].lower())[:max_text_len]
        if len(text_tokens) == max_text_len or text_tokens[-1] == '.':
            text_tokens[-1] = '..'
        elif text_tokens[-1] != '.':
            text_tokens.append('..')
        item['text'] = ' '.join(text_tokens)

    with codecs.open(token_data_path, 'w', encoding='utf-8') as f:
        json.dump(raw_data, f, indent=4, ensure_ascii=False)

def build_w2i2w(train_token_data_path, io_token_w2i_path, io_token_i2w_path, io_min_token_count=3, unk_aliased=True):
    logging.info('########### Start building the dictionary of the training set ##########')
    dic_paths = [io_token_w2i_path, io_token_i2w_path]
    for dic_path in dic_paths:
        dic_dir = os.path.dirname(dic_path)
        if not os.path.exists(dic_dir):
            os.makedirs(dic_dir)

    with codecs.open(train_token_data_path, 'r', encoding='utf-8') as f:
        token_data = json.load(f)

    code_token_counter = Counter()
    text_token_counter = Counter()
    max_code_len = 0
    
    for item in tqdm(token_data):
        code_nodes = list(eval(item['graph']['nodes']))
        code_token_counter += Counter(code_nodes)
        text_token_counter += Counter(item['text'].split())
        max_code_len = max(max_code_len, len(code_nodes))
        
    general_vocabs = [PAD_TOKEN, UNK_TOKEN]
    io_token_counter = code_token_counter + text_token_counter
    io_tokens = list(filter(lambda x: io_token_counter[x] >= io_min_token_count, io_token_counter.keys()))
    
    text_tokens = set(io_tokens) & set(text_token_counter.keys())
    other_tokens = set(io_tokens) - text_tokens
    
    graph_unk_aliases = []
    if unk_aliased:
        max_alias_num = 0
        for item in token_data:
            aliases = list(filter(lambda x: x not in io_tokens, set(list(eval(item['graph']['nodes'])))))
            max_alias_num = max(max_alias_num, len(aliases))
        graph_unk_aliases = ['<unk-alias-{}>'.format(i) for i in range(max_alias_num)]

    copy_tokens = []
    for i in range(max_code_len):
        COPY_TOKEN = '<copy_{}>'.format(i)
        copy_tokens.append(COPY_TOKEN)
        
    io_tokens = general_vocabs + list(text_tokens) + [OUT_END_TOKEN, OUT_BEGIN_TOKEN] + copy_tokens + list(other_tokens) + graph_unk_aliases

    io_token_indices = list(range(len(io_tokens)))
    io_token_w2i = dict(zip(io_tokens, io_token_indices))
    io_token_i2w = dict(zip(io_token_indices, io_tokens))

    dics = [io_token_w2i, io_token_i2w]
    for dic, dic_path in zip(dics, dic_paths):
        with open(dic_path, 'wb') as f:
            pickle.dump(dic, f)
        with codecs.open(dic_path + '.json', 'w') as f:
            json.dump(dic, f, indent=4, ensure_ascii=False)
    logging.info('########### Finish building the dictionary ##########')

def get_ex_tgt_dict(src_tokens, tgt_w2i):
    ex_src_tokens = list(filter(lambda x: x not in tgt_w2i.keys(), src_tokens))
    ex_src_tokens = sorted(list(set(ex_src_tokens)), key=ex_src_tokens.index)
    ex_src_token_indices = list(range(len(tgt_w2i), len(tgt_w2i) + len(ex_src_tokens)))
    ex_tgt_w2i = dict(zip(ex_src_tokens, ex_src_token_indices))
    ex_tgt_i2w = dict(zip(ex_src_token_indices, ex_src_tokens))
    return ex_tgt_w2i, ex_tgt_i2w

def get_src2tgt_map_ids(src_tokens, tgt_w2i, ex_tgt_w2i):
    all_tgt_w2i = {**tgt_w2i, **ex_tgt_w2i}
    src_map = [all_tgt_w2i[token] for token in src_tokens]
    return src_map

def get_align_tgt_ids(tgt_tokens, tgt_w2i, ex_tgt_w2i):
    all_tgt_w2i = {**tgt_w2i, **ex_tgt_w2i}
    unk_idx = tgt_w2i[UNK_TOKEN]
    tgt_token_ids = [all_tgt_w2i.get(token, unk_idx) for token in tgt_tokens]
    return tgt_token_ids

def build_avail_data(token_data_path, avail_data_path, io_token_w2i_path, io_token_i2w_path, unk_aliased=True):
    logging.info('########### Start building the train dataset available for the model ##########')
    avail_data_dir = os.path.dirname(avail_data_path)
    if not os.path.exists(avail_data_dir):
        os.makedirs(avail_data_dir)

    with open(io_token_w2i_path, 'rb') as f: io_token_w2i = pickle.load(f)
    with open(io_token_i2w_path, 'rb') as f: io_token_i2w = pickle.load(f)
    with codecs.open(token_data_path, 'r') as f: token_data = json.load(f)

    io_token_unk_idx = io_token_w2i[UNK_TOKEN]
    out_begin_idx = io_token_w2i[OUT_BEGIN_TOKEN]

    text_tokens, text_token_ids = zip(*[(io_token_i2w[idx], idx) for idx in range(out_begin_idx + 1)]) 
    text_i2w = dict(zip(text_token_ids, text_tokens))
    text_w2i = dict(zip(text_tokens, text_token_ids))
    
    avail_data = {'code_seqs': [], 'texts': [], 'ids': [], 'text_dic': {'text_i2w': text_i2w, 'ex_text_i2ws': []}}

    for i, item in enumerate(tqdm(token_data)):
        code_tokens = list(eval(item['graph']['nodes']))
        text_tokens = item['text'].split()

        ex_text_w2i, ex_text_i2w = get_ex_tgt_dict(code_tokens, text_w2i)
        code_seq2text_map_ids = get_src2tgt_map_ids(code_tokens, text_w2i, ex_text_w2i)
        text_token_ids = get_align_tgt_ids(text_tokens, text_w2i, ex_text_w2i)

        if unk_aliased:
            all_unk_aliases = filter(lambda x: x not in io_token_w2i.keys(), code_tokens)
            unk_aliases = []
            for unk_alias in all_unk_aliases:
                if unk_alias not in unk_aliases: unk_aliases.append(unk_alias)
            code_tokens = [node if node not in unk_aliases else '<unk-alias-{}>'.format(unk_aliases.index(node)) for node in code_tokens]

        code_seq_ids = [io_token_w2i.get(token, io_token_unk_idx) for token in code_tokens]

        seq_data = {
            'code_ids': code_seq_ids,
            'map_ids': code_seq2text_map_ids,
            'code_len': len(code_seq_ids)
        }
        
        avail_data['code_seqs'].append(seq_data)
        avail_data['texts'].append(text_token_ids)
        avail_data['ids'].append(i)
        avail_data['text_dic']['ex_text_i2ws'].append(ex_text_i2w)

    with open(avail_data_path, 'wb') as f:
        pickle.dump(avail_data, f)
    logging.info('########### Finish building the train dataset available for the model ##########')

if __name__=='__main__':
    tokenize_raw_data(raw_data_path=train_raw_data_path, token_data_path=train_token_data_path)
    tokenize_raw_data(raw_data_path=valid_raw_data_path, token_data_path=valid_token_data_path)
    tokenize_raw_data(raw_data_path=test_raw_data_path, token_data_path=test_token_data_path)
    
    build_w2i2w(train_token_data_path=train_token_data_path, io_token_w2i_path=io_token_w2i_path, io_token_i2w_path=io_token_i2w_path, io_min_token_count=io_min_token_count, unk_aliased=unk_aliased)

    build_avail_data(token_data_path=train_token_data_path, avail_data_path=train_avail_data_path, io_token_w2i_path=io_token_w2i_path, io_token_i2w_path=io_token_i2w_path, unk_aliased=unk_aliased)
    build_avail_data(token_data_path=valid_token_data_path, avail_data_path=valid_avail_data_path, io_token_w2i_path=io_token_w2i_path, io_token_i2w_path=io_token_i2w_path, unk_aliased=unk_aliased)
    build_avail_data(token_data_path=test_token_data_path, avail_data_path=test_avail_data_path, io_token_w2i_path=io_token_w2i_path, io_token_i2w_path=io_token_i2w_path, unk_aliased=unk_aliased)