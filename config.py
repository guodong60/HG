# #coding=utf-8
# import logging
# import os
# import sys
# sys.path.append('../../../')
# from lib.util.eval.translate_metric import get_nltk33_sent_bleu1 as get_sent_bleu1, \
#                                               get_nltk33_sent_bleu2 as get_sent_bleu2,  \
#                                             get_nltk33_sent_bleu3 as get_sent_bleu3,  \
#                                             get_nltk33_sent_bleu4 as get_sent_bleu4,  \
#                                             get_nltk33_sent_bleu as get_sent_bleu
# from lib.util.eval.translate_metric import get_corp_bleu1,get_corp_bleu2,get_corp_bleu3,get_corp_bleu4,get_corp_bleu
# from lib.util.eval.translate_metric import get_meteor,get_rouge,get_cider
# import math

# train_data_name='train_data'
# valid_data_name='valid_data'
# test_data_name='test_data'

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# # 顶级数据目录
# top_data_dir= '../../../data/java'

# raw_data_dir=os.path.join(top_data_dir,'raw_data/')
# train_raw_data_path=os.path.join(raw_data_dir,'{}.json'.format(train_data_name))
# valid_raw_data_path=os.path.join(raw_data_dir,'{}.json'.format(valid_data_name))
# test_raw_data_path=os.path.join(raw_data_dir,'{}.json'.format(test_data_name))
# tech_term_path=os.path.join(raw_data_dir,'tech_term.txt')
# keep_test_data_id_path=os.path.join(raw_data_dir,'keep_test_data_ids.txt')

# max_code_len=321     
# max_graph_size=437
# max_text_len=38

# token_data_dir=os.path.join(top_data_dir,'token_data/')
# train_token_data_path=os.path.join(token_data_dir,'{}.json'.format(train_data_name))
# valid_token_data_path=os.path.join(token_data_dir,'{}.json'.format(valid_data_name))
# test_token_data_path=os.path.join(token_data_dir,'{}.json'.format(test_data_name))

# USER_WORDS=[('\\','n'),('e','.','g','.'),('i','.','e','.'),('-','>')]

# basic_info_dir=os.path.join(top_data_dir,'basic_info/')
# size_info_path=os.path.join(basic_info_dir,'size_info.pkl')
# rev_dic_path=os.path.join(basic_info_dir,'rev_dic.json')
# noise_token_path=os.path.join(basic_info_dir,'noise_token.json')

# w2i2w_dir=os.path.join(top_data_dir,'w2i2w/')
# io_token_w2i_path=os.path.join(w2i2w_dir,'io_token_w2i.pkl')
# io_token_i2w_path=os.path.join(w2i2w_dir,'io_token_i2w.pkl')

# io_min_token_count=3
# unk_aliased=True  

# avail_data_dir=os.path.join(top_data_dir,'avail_data/')
# train_avail_data_path=os.path.join(avail_data_dir,'{}.pkl'.format(train_data_name))
# valid_avail_data_path=os.path.join(avail_data_dir,'{}.pkl'.format(valid_data_name))
# test_avail_data_path=os.path.join(avail_data_dir,'{}.pkl'.format(test_data_name))

# OUT_BEGIN_TOKEN='</s>'
# OUT_END_TOKEN='</e>'
# PAD_TOKEN='<pad>'
# UNK_TOKEN='<unk>'

# model_dir=os.path.join(top_data_dir,'model/')
# os.environ["CUDA_VISIBLE_DEVICES"] ="1,2" 
# import os
# from torch_geometric.nn import SAGEConv

# # ================= 核心超参数配置 ================= 
# emb_dims = 512  
# graph_gnn_layers = 6
# text_att_layers = 8    
# # 物理显存减半，代码中通过 accumulation_steps=2 恢复等价 64 的效果
# train_batch_size = 100  
# version = '8_AttnRes_Plus_CopyHighway'  
# model_name = 'codescriber_v{}_{}_{}_{}'.format(version, graph_gnn_layers, text_att_layers, emb_dims)

# params = dict(
#     model_dir=model_dir,
#     model_name=model_name,
#     model_id=None,
#     emb_dims=emb_dims,
#     graph_gnn_layers=graph_gnn_layers, 
#     graph_GNN=SAGEConv,
#     graph_gnn_aggr='mean',
#     text_att_layers=text_att_layers,
#     text_att_heads=8,
#     text_att_head_dims=None,
#     text_ff_hid_dims=4 * emb_dims,
#     drop_rate=0.15,  
#     copy=True,
#     pad_idx=0,
#     train_batch_size=train_batch_size,
#     pred_batch_size=math.ceil(train_batch_size * 1.5), 
#     max_train_size=-1,  
#     max_valid_size=-1,  
#     max_big_epochs=100,  
#     early_stop=10,
#     regular_rate=1e-5,
#     lr_base=5e-4,
#     lr_decay=0.95,
#     min_lr_rate=0.01,
#     warm_big_epochs=3,
#     beam_width=5,
#     start_valid_epoch=60,
#     gpu_ids=os.environ["CUDA_VISIBLE_DEVICES"],
#     train_mode=True,

#     # ================= 【终极解耦架构：创新开关】 ================= 
    
#     # 1. 双端注意力残差 (主干网络：负责提取深层复杂的语义和结构逻辑)
#     use_enc_attn_res=True,  
#     use_dec_attn_res=True,  
    
#     # 2. 全局指针高速公路 (复制旁路：负责无损传递底层纯净词法给 Copy 机制)
#     use_copy_highway=True, 
    
#     # 3. 有向超图注意力 (DHGAT)
#     use_directed_hyperedges=True,
    
#     # 4. 位置编码分流
#     use_hyperedge_pos_emb=True,  

#     # 5. 动态语义超边
#     use_dynamic_edges=True,
#     dynamic_threshold=0.85, 

#     # 6. 图级对比学习
#     use_cl=True,
#     cl_weight=0.05,        
#     cl_temp=0.1,           
#     edge_drop_rate=0.15    
# )

# train_metrics = [get_sent_bleu]
# valid_metric = get_sent_bleu
# test_metrics = [get_rouge, get_cider,get_meteor,
#                 get_sent_bleu1,get_sent_bleu2,get_sent_bleu3,get_sent_bleu4,get_sent_bleu,
#                 get_corp_bleu1,get_corp_bleu2,get_corp_bleu3,get_corp_bleu4,get_corp_bleu] 

# res_dir=os.path.join(top_data_dir,'result/')
# res_path=os.path.join(res_dir,model_name+'.json')

# import random
# import torch
# import numpy as np
# def seed_torch(seed=0):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed) 
#     torch.cuda.manual_seed(seed)    
#     torch.cuda.manual_seed_all(seed)    
#     # 极致防弹：关闭动态寻优，防止跑了几十个Epoch后 cuDNN 崩溃
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
# seeds=[0,42,7,23,124,1084,87]
# seed_torch(seeds[0])

# 跳跃知识网络
# coding=utf-8
# import logging
# import os
# import sys
# sys.path.append('../../../')
# from lib.util.eval.translate_metric import get_nltk33_sent_bleu1 as get_sent_bleu1, \
#                                               get_nltk33_sent_bleu2 as get_sent_bleu2,  \
#                                             get_nltk33_sent_bleu3 as get_sent_bleu3,  \
#                                             get_nltk33_sent_bleu4 as get_sent_bleu4,  \
#                                             get_nltk33_sent_bleu as get_sent_bleu
# from lib.util.eval.translate_metric import get_corp_bleu1,get_corp_bleu2,get_corp_bleu3,get_corp_bleu4,get_corp_bleu
# from lib.util.eval.translate_metric import get_meteor,get_rouge,get_cider
# import math

# train_data_name='train_data'
# valid_data_name='valid_data'
# test_data_name='test_data'

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# # 顶级数据目录
# top_data_dir= '../../../data/java'

# raw_data_dir=os.path.join(top_data_dir,'raw_data/')
# train_raw_data_path=os.path.join(raw_data_dir,'{}.json'.format(train_data_name))
# valid_raw_data_path=os.path.join(raw_data_dir,'{}.json'.format(valid_data_name))
# test_raw_data_path=os.path.join(raw_data_dir,'{}.json'.format(test_data_name))
# tech_term_path=os.path.join(raw_data_dir,'tech_term.txt')
# keep_test_data_id_path=os.path.join(raw_data_dir,'keep_test_data_ids.txt')

# max_code_len=321     
# max_graph_size=437
# max_text_len=38

# token_data_dir=os.path.join(top_data_dir,'token_data/')
# train_token_data_path=os.path.join(token_data_dir,'{}.json'.format(train_data_name))
# valid_token_data_path=os.path.join(token_data_dir,'{}.json'.format(valid_data_name))
# test_token_data_path=os.path.join(token_data_dir,'{}.json'.format(test_data_name))

# USER_WORDS=[('\\','n'),('e','.','g','.'),('i','.','e','.'),('-','>')]

# basic_info_dir=os.path.join(top_data_dir,'basic_info/')
# size_info_path=os.path.join(basic_info_dir,'size_info.pkl')
# rev_dic_path=os.path.join(basic_info_dir,'rev_dic.json')
# noise_token_path=os.path.join(basic_info_dir,'noise_token.json')

# w2i2w_dir=os.path.join(top_data_dir,'w2i2w/')
# io_token_w2i_path=os.path.join(w2i2w_dir,'io_token_w2i.pkl')
# io_token_i2w_path=os.path.join(w2i2w_dir,'io_token_i2w.pkl')

# io_min_token_count=3
# unk_aliased=True  

# avail_data_dir=os.path.join(top_data_dir,'avail_data/')
# train_avail_data_path=os.path.join(avail_data_dir,'{}.pkl'.format(train_data_name))
# valid_avail_data_path=os.path.join(avail_data_dir,'{}.pkl'.format(valid_data_name))
# test_avail_data_path=os.path.join(avail_data_dir,'{}.pkl'.format(test_data_name))

# OUT_BEGIN_TOKEN='</s>'
# OUT_END_TOKEN='</e>'
# PAD_TOKEN='<pad>'
# UNK_TOKEN='<unk>'

# model_dir=os.path.join(top_data_dir,'model/')
# os.environ["CUDA_VISIBLE_DEVICES"] ="1,2" 
# import os
# from torch_geometric.nn import SAGEConv

# # ================= 核心超参数配置 ================= 
# emb_dims = 512  
# graph_gnn_layers = 6
# text_att_layers = 8    
# # 物理显存减半为32，代码中通过 accumulation_steps=2 恢复等价 64 的效果
# train_batch_size = 100  
# version = '10_JK_Readout_Ultimate'  
# model_name = 'codescriber_v{}_{}_{}_{}'.format(version, graph_gnn_layers, text_att_layers, emb_dims)

# params = dict(
#     model_dir=model_dir,
#     model_name=model_name,
#     model_id=None,
#     emb_dims=emb_dims,
#     graph_gnn_layers=graph_gnn_layers, 
#     graph_GNN=SAGEConv,
#     graph_gnn_aggr='mean',
#     text_att_layers=text_att_layers,
#     text_att_heads=8,
#     text_att_head_dims=None,
#     text_ff_hid_dims=4 * emb_dims,
#     drop_rate=0.15,  
#     copy=True,
#     pad_idx=0,
#     train_batch_size=train_batch_size,
#     pred_batch_size=math.ceil(train_batch_size * 1.5), 
#     max_train_size=-1,  
#     max_valid_size=-1,  
#     max_big_epochs=100,  
#     early_stop=10,
#     regular_rate=1e-5,
#     lr_base=5e-4,
#     lr_decay=0.95,
#     min_lr_rate=0.01,
#     warm_big_epochs=3,
#     beam_width=5,
#     start_valid_epoch=60,
#     gpu_ids=os.environ["CUDA_VISIBLE_DEVICES"],
#     train_mode=True,

#     # ================= 【打破 SOTA 的终极解耦大招】 ================= 
    
#     # 1. [首创] 跳跃知识网络 (Jumping Knowledge Readout)
#     # 完美融合第0层的纯净词法与第6层的深层语义，且绝对不撕裂特征对齐空间！
#     use_jk_readout=True, 
    
#     # 2. 有向超图注意力 (DHGAT)
#     use_directed_hyperedges=True,
    
#     # 3. 位置编码分流
#     use_hyperedge_pos_emb=True,  

#     # 4. 动态语义超边
#     use_dynamic_edges=True,
#     dynamic_threshold=0.85, 

#     # 5. 图级对比学习
#     use_cl=True,
#     cl_weight=0.05,        
#     cl_temp=0.1,           
#     edge_drop_rate=0.15    
# )

# train_metrics = [get_sent_bleu]
# valid_metric = get_sent_bleu
# test_metrics = [get_rouge, get_cider,get_meteor,
#                 get_sent_bleu1,get_sent_bleu2,get_sent_bleu3,get_sent_bleu4,get_sent_bleu,
#                 get_corp_bleu1,get_corp_bleu2,get_corp_bleu3,get_corp_bleu4,get_corp_bleu] 

# res_dir=os.path.join(top_data_dir,'result/')
# res_path=os.path.join(res_dir,model_name+'.json')

# import random
# import torch
# import numpy as np
# def seed_torch(seed=0):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed) 
#     torch.cuda.manual_seed(seed)    
#     torch.cuda.manual_seed_all(seed)    
#     # 【极致防弹】：关闭动态寻优，防止跑了几十个Epoch后 cuDNN 崩溃
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
# seeds=[0,42,7,23,124,1084,87]
# seed_torch(seeds[0])

#重整化群理论
#coding=utf-8
import logging
import os
import sys
from my_lib.util.eval.translate_metric import get_nltk33_sent_bleu1 as get_sent_bleu1, \
                                              get_nltk33_sent_bleu2 as get_sent_bleu2,  \
                                            get_nltk33_sent_bleu3 as get_sent_bleu3,  \
                                            get_nltk33_sent_bleu4 as get_sent_bleu4,  \
                                            get_nltk33_sent_bleu as get_sent_bleu
from my_lib.util.eval.translate_metric import get_corp_bleu1,get_corp_bleu2,get_corp_bleu3,get_corp_bleu4,get_corp_bleu
from my_lib.util.eval.translate_metric import get_meteor,get_rouge,get_cider
import math

train_data_name='train_data'
valid_data_name='valid_data'
test_data_name='test_data'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

top_data_dir= '../../../data/java'
raw_data_dir=os.path.join(top_data_dir,'raw_data/')
train_raw_data_path=os.path.join(raw_data_dir,'{}.json'.format(train_data_name))
valid_raw_data_path=os.path.join(raw_data_dir,'{}.json'.format(valid_data_name))
test_raw_data_path=os.path.join(raw_data_dir,'{}.json'.format(test_data_name))
tech_term_path=os.path.join(raw_data_dir,'tech_term.txt')
keep_test_data_id_path=os.path.join(raw_data_dir,'keep_test_data_ids.txt')

# 1D-Seq-RGNet 核心长度参数
max_code_len=300     
max_text_len=38

token_data_dir=os.path.join(top_data_dir,'token_data/')
train_token_data_path=os.path.join(token_data_dir,'{}.json'.format(train_data_name))
valid_token_data_path=os.path.join(token_data_dir,'{}.json'.format(valid_data_name))
test_token_data_path=os.path.join(token_data_dir,'{}.json'.format(test_data_name))

USER_WORDS=[('\\','n'),('e','.','g','.'),('i','.','e','.'),('-','>')]

basic_info_dir=os.path.join(top_data_dir,'basic_info/')
rev_dic_path=os.path.join(basic_info_dir,'rev_dic.json')
noise_token_path=os.path.join(basic_info_dir,'noise_token.json')

w2i2w_dir=os.path.join(top_data_dir,'w2i2w/')
io_token_w2i_path=os.path.join(w2i2w_dir,'io_token_w2i.pkl')
io_token_i2w_path=os.path.join(w2i2w_dir,'io_token_i2w.pkl')

io_min_token_count=3
unk_aliased=True  

avail_data_dir=os.path.join(top_data_dir,'avail_data/')
train_avail_data_path=os.path.join(avail_data_dir,'{}.pkl'.format(train_data_name))
valid_avail_data_path=os.path.join(avail_data_dir,'{}.pkl'.format(valid_data_name))
test_avail_data_path=os.path.join(avail_data_dir,'{}.pkl'.format(test_data_name))

OUT_BEGIN_TOKEN='</s>'
OUT_END_TOKEN='</e>'
PAD_TOKEN='<pad>'
UNK_TOKEN='<unk>'

model_dir=os.path.join(top_data_dir,'model/')
os.environ["CUDA_VISIBLE_DEVICES"] ="1" 

emb_dims = 512
text_att_layers = 6

# ==========================================
# 重整化群 (RG) 理论物理参数
# ==========================================
rg_steps = 3           # 经历 3 次粗粒化相变 (长度将缩短 2^3=8 倍)
pool_size = 2          # 卡达诺夫块自旋打包大小 (相邻 2 个 Token 融合成 1 个)

train_batch_size = 192 
version = '1D_RGNet'

model_name='codescriber_v{}_{}'.format(version,emb_dims)
params = dict(model_dir=model_dir,
              model_name=model_name,
              model_id=None,
              emb_dims=emb_dims,
              rg_steps=rg_steps,     
              pool_size=pool_size,   
              text_att_layers=text_att_layers,
              text_att_heads=8,
              text_att_head_dims=None,
              text_ff_hid_dims=4 * emb_dims,
              drop_rate=0.2,
              copy=True,
              pad_idx=0,
              train_batch_size=train_batch_size,
              pred_batch_size=math.ceil(train_batch_size * 1.25), 
              max_train_size=-1,  
              max_valid_size=-1,  
              max_big_epochs=100,  
              early_stop=10,
              regular_rate=1e-5,
              lr_base=5e-4,
              lr_decay=0.95,
              min_lr_rate=0.01,
              warm_big_epochs=3,
              beam_width=5,
              start_valid_epoch=60, 
              gpu_ids=os.environ["CUDA_VISIBLE_DEVICES"],
              train_mode=True)

train_metrics = [get_sent_bleu]
valid_metric = get_sent_bleu
test_metrics = [get_rouge, get_cider,get_meteor,
                get_sent_bleu1,get_sent_bleu2,get_sent_bleu3,get_sent_bleu4,get_sent_bleu,
                get_corp_bleu1,get_corp_bleu2,get_corp_bleu3,get_corp_bleu4,get_corp_bleu]

res_dir=os.path.join(top_data_dir,'result/')
res_path=os.path.join(res_dir,model_name+'.json')

import random
import torch
import numpy as np
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
seed_torch(0)