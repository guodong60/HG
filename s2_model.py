# # coding=utf-8
# import os
# import re
# import sys
# sys.path.append('../../../')
# from lib.neural_module.learn_strategy import LrWarmUp
# from lib.neural_module.transformer import TranEnc, TranDec, DualTranDec,ResFF,ResMHA
# from lib.neural_module.embedding import PosEnc,SinusoidalPositionalEncoding
# from lib.neural_module.loss import LabelSmoothSoftmaxCEV2, CriterionNet
# from lib.neural_module.balanced_data_parallel import BalancedDataParallel
# from lib.neural_module.copy_attention import DualMultiCopyGenerator,MultiCopyGenerator,DualCopyGenerator
# from lib.neural_module.beam_search import trans_beam_search
# from lib.neural_model.seq_to_seq_model import TransSeq2Seq
# from lib.neural_model.base_model import BaseNet
# from lib.neural_module.transformer import ResFF
    
# # from torch_geometric.nn import HypergraphConv
# from typing import Any,Optional,Union

# from config import *

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler
# from torch.nn.utils import clip_grad_norm_
# from torch.utils.data import Dataset
# from torch_geometric.data import HeteroData
# from torch_geometric.loader.data_list_loader import DataListLoader
# from torch_geometric.utils import to_dense_batch
# from torch_geometric.data.storage import (BaseStorage, NodeStorage,EdgeStorage)
# from torch_geometric.nn.data_parallel import DataParallel
# from torch_geometric.nn import HeteroConv,GraphNorm,HypergraphConv,GATConv
# from lib.neural_module.DHGNN import HyperedgeDiffusionConv,KStepHypergraphConv
# import random
# import numpy as np
# import os
# import logging
# import pickle
# import json
# import codecs
# from tqdm import tqdm
# import pickle
# import numpy as np
# import pandas as pd
# import math
# from copy import deepcopy

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# NodeOrEdgeStorage = Union[NodeStorage, EdgeStorage]

# class Datax(HeteroData):
#     # def __init__(self,
#     #              graph_func_node=dict(x=None),
#     #              graph_attr_node=dict(x=None),
#     #              graph_glob_node=dict(x=None),
#     #              ):
#     #     super().__init__()
#     #     self.graph_func_node=graph_func_node
#     def __cat_dim__(self, key: str, value: Any,
#                     store: Optional[NodeOrEdgeStorage] = None, *args,
#                     **kwargs) -> Any:
#         if bool(re.search('(token)', key)): #|map
#             return None  # generate a new 0 dimension
#         if bool(re.search('(pos)', key)):
#             return -1
#         return super().__cat_dim__(key, value,store)    #return不能漏了！！！

# class Datasetx(Dataset):
#     '''
#     文本对数据集对象（根据具体数据再修改）
#     '''
#     def __init__(self,
#                  code_graphs,
#                  texts=None,
#                  ids=None,
#                  text_max_len=None,
#                  text_begin_idx=1,
#                  text_end_idx=2,
#                  pad_idx=0):
#         self.len = len(code_graphs)  # 样本个数
#         self.text_max_len = text_max_len
#         self.text_begin_idx = text_begin_idx
#         self.text_end_idx = text_end_idx

#         if text_max_len is None and texts is not None:
#             self.text_max_len = max([len(text) for text in texts])  # 每个输出只是一个序列
#         self.code_graphs = code_graphs
#         self.texts = texts
#         self.ids = ids
#         self.pad_idx = pad_idx

#     def __getitem__(self, index):
#         if self.texts is None:
#             pad_text_in = np.zeros((self.text_max_len + 1,), dtype=np.int64)  # decoder端的输入
#             pad_text_in[0] = self.text_begin_idx
#             pad_text_out = None
#         else:
#             tru_text = self.texts[index][:self.text_max_len]  # 先做截断
#             pad_text_in = np.lib.pad(tru_text,
#                                     (1, self.text_max_len - len(tru_text)),
#                                     'constant',
#                                     constant_values=(self.text_begin_idx, self.pad_idx))
#             tru_text_out = np.lib.pad(tru_text,
#                                      (0, 1),
#                                      'constant',
#                                      constant_values=(0, self.text_end_idx))  # padding
#             pad_text_out = np.lib.pad(tru_text_out,
#                                      (0, self.text_max_len + 1 - len(tru_text_out)),
#                                      'constant',
#                                      constant_values=(self.pad_idx, self.pad_idx))  # padding
#             # pad_out_input=np.lib.pad(pad_out[:-1],(1,0),'constant',constant_values=(self.text_begin_idx, 0))
#         data=Datax()
#         data['node'].x=torch.tensor(self.code_graphs[index]['nodes'])
#         data['node'].src_map=torch.tensor(self.code_graphs[index]['node2text_map_ids']).long()
#         data['node'].code_mask=torch.tensor(self.code_graphs[index]['code_node_mask']).bool()
#         data['node','parent_child_hyperedges','node'].edge_index=torch.tensor(self.code_graphs[index]['parent_child_hyperedges']).long()
#         data['node','line_hyperedges','node'].edge_index=torch.tensor(self.code_graphs[index]['line_hyperedges']).long()
#         data['node','dfg_hyperedges','node'].edge_index=torch.tensor(self.code_graphs[index]['dfg_hyperedges']).long()
#         data['node','block_hyperedges','node'].edge_index=torch.tensor(self.code_graphs[index]['block_hyperedges']).long()
#         data['node','layout_sibling_hyperedges','node'].edge_index=torch.tensor(self.code_graphs[index]['layout_sibling_hyperedges']).long()
#         # data['node','base_child','node'].edge_index=torch.tensor(self.code_graphs[index]['base_father2child_edges']).long()
#         # data['node','base_father','node'].edge_index=torch.tensor(self.code_graphs[index]['base_child2father_edges']).long()
#         # data['node','sibling_next','node'].edge_index=torch.tensor(self.code_graphs[index]['sibling_prev2next_edges']).long()
#         # data['node','sibling_prev','node'].edge_index=torch.tensor(self.code_graphs[index]['sibling_next2prev_edges']).long()
#         # data['node','dfg_next','node'].edge_index=torch.tensor(self.code_graphs[index]['dfg_prev2next_edges']).long()
#         # data['node','dfg_prev','node'].edge_index=torch.tensor(self.code_graphs[index]['dfg_next2prev_edges']).long()
#         # data['node','code_next','node'].edge_index=torch.tensor(self.code_graphs[index]['code_prev2next_edges']).long()
#         # data['node','code_prev','node'].edge_index=torch.tensor(self.code_graphs[index]['code_next2prev_edges']).long()
#         # data['node','cfg_next','node'].edge_index=torch.tensor(self.code_graphs[index]['cfg_prev2next_edges']).long()
#         # data['node','cfg_prev','node'].edge_index=torch.tensor(self.code_graphs[index]['cfg_next2prev_edges']).long()


#         data['text'].text_token_input=torch.tensor(pad_text_in).long()
#         if self.texts is not None:
#             data['text'].text_token_output = torch.tensor(pad_text_out).long()
#         data['text'].num_nodes = pad_text_in.shape[0]
#         if self.ids is not None:
#             data['idx'].idx=torch.tensor(self.ids[index])
#             data['idx'].num_nodes=1
#         # print(data.num_nodes)
#         return data

#     def __len__(self):
#         return self.len

# class CodeGraphEnc(nn.Module):
#     def __init__(self,
#                  emb_dims,
#                  graph_max_size,
#                  code_max_len,
#                  graph_node_emb_op,
#                 #  code_mpos_voc_size,
#                 #  code_npos_voc_size,
#                 #  code_att_layers=2,
#                 #  code_att_heads=8,
#                 #  code_att_head_dims=None,
#                 #  code_ff_hid_dims=2048,
#                  graph_gnn_layers=6,
#                  graph_GNN=SAGEConv,
#                  graph_gnn_aggr='mean',
#                  drop_rate=0.,
#                  **kwargs,
#                  ):
#         super().__init__()
#         kwargs.setdefault('pad_idx', 0)
#         self.pad_idx = kwargs['pad_idx']
#         self.graph_max_size = graph_max_size
#         self.code_max_len=code_max_len
#         self.emb_dims=emb_dims

#         # assert len(graph_sim_node_ids.shape)==1
#         # graph_sim_node_voc_size=np.unique(graph_sim_node_ids).shape[0]
#         # self.graph_node_to_sim_token_map_op=nn.Embedding.from_pretrained(torch.tensor(graph_sim_node_ids).view([-1,1]).float(),freeze=True,padding_idx=kwargs['pad_idx'])
#         self.graph_node_emb_op = graph_node_emb_op
#         # self.graph_node_to_sim_token_map_op=graph_node_to_sim_token_map_op
#         # self.graph_node_emb_op = nn.Embedding(graph_node_voc_size, emb_dims, padding_idx=kwargs['pad_idx'])
#         # self.sim_node_emb_op = nn.Embedding(graph_sim_node_voc_size, emb_dims, padding_idx=kwargs['pad_idx'])
        
#         # self.code_mpos_emb_op = nn.Embedding(code_mpos_voc_size, emb_dims, padding_idx=kwargs['pad_idx'])
#         # self.code_npos_emb_op = nn.Embedding(code_npos_voc_size, emb_dims, padding_idx=kwargs['pad_idx'])
        
#         # nn.init.xavier_uniform_(self.graph_node_emb_op.weight[1:, ])
#         # nn.init.xavier_uniform_(self.graph_sim_node_emb_op.weight[1:, ])
#         # nn.init.xavier_uniform_(self.code_mpos_emb_op.weight[1:, ])
#         # nn.init.xavier_uniform_(self.code_npos_emb_op.weight[1:, ])

#         # self.graph_emb_norm_op = GraphNorm(emb_dims)
        
#         # self.pos_encoding = SinusoidalPositionalEncoding(
#         #         emb_dims=emb_dims,
#         #         max_len=graph_max_size,
#         #         pad_idx=kwargs['pad_idx']
#         #     )
        
#             # 方式1: 为图节点添加可学习的位置编码
#         max_position = graph_max_size * 2
#         self.graph_pos_encoding = nn.Embedding(
#                 max_position + 1,  # +1 for padding
#                 emb_dims, 
#                 padding_idx=kwargs['pad_idx']
#             )
#         nn.init.xavier_uniform_(self.graph_pos_encoding.weight[1:, ])
            
#             # 方式2: 为代码序列添加位置编码
#         # self.code_pos_encoding = PosEnc(
#         #         max_len=code_max_len,
#         #         emb_dims=emb_dims,
#         #         train=True,
#         #         pad=True,
#         #         pad_idx=kwargs['pad_idx']
#         #     )
        
#         self.emb_drop_op = nn.Dropout(p=drop_rate)
#         # self.code_emb_norm_op = nn.LayerNorm(emb_dims)
#         # self.graph_emb_norm_op = nn.LayerNorm(emb_dims)

#         # self.code_enc_op = TranEnc(query_dims=emb_dims,
#         #                             head_num=code_att_heads,
#         #                             ff_hid_dims=code_ff_hid_dims,
#         #                             head_dims=code_att_head_dims,
#         #                             layer_num=code_att_layers,
#         #                             drop_rate=drop_rate,
#         #                             pad_idx=kwargs['pad_idx'])

#         self.gnn_layers = graph_gnn_layers
#         self.gnn_ops=nn.ModuleList()
#         self.gnorm_ops=nn.ModuleList()
#         self.grelu_ops=nn.ModuleList()
#         for _ in range(graph_gnn_layers):
#             if graph_GNN==SAGEConv:
#                 gnn=HeteroConv({

#                     # ('node', 'base_child', 'node'): graph_GNN((emb_dims,emb_dims), emb_dims, aggr=graph_gnn_aggr,root_weight=True),
#                     # ('node', 'base_father', 'node'): graph_GNN((emb_dims,emb_dims), emb_dims, aggr=graph_gnn_aggr,root_weight=False),
#                     # ('node', 'sibling_next', 'node'): graph_GNN((emb_dims,emb_dims), emb_dims, aggr=graph_gnn_aggr,root_weight=False),
#                     # ('node', 'sibling_prev', 'node'): graph_GNN((emb_dims,emb_dims), emb_dims, aggr=graph_gnn_aggr,root_weight=False),
#                     # ('node', 'dfg_next', 'node'): graph_GNN((emb_dims,emb_dims), emb_dims, aggr=graph_gnn_aggr,root_weight=False),
#                     # ('node', 'dfg_prev', 'node'): graph_GNN((emb_dims,emb_dims), emb_dims, aggr=graph_gnn_aggr,root_weight=False),
#                     # ('node', 'code_next', 'node'): graph_GNN((emb_dims,emb_dims), emb_dims, aggr=graph_gnn_aggr,root_weight=False),
#                     # ('node', 'code_prev', 'node'): graph_GNN((emb_dims,emb_dims), emb_dims, aggr=graph_gnn_aggr,root_weight=False),
#                     # ('node', 'cfg_next', 'node'): graph_GNN((emb_dims,emb_dims), emb_dims, aggr=graph_gnn_aggr,root_weight=False),
#                     # ('node', 'cfg_prev', 'node'): graph_GNN((emb_dims,emb_dims), emb_dims, aggr=graph_gnn_aggr,root_weight=False),
#                     # ('node', 'parent_child_hyperedges', 'node'): HypergraphConv(emb_dims, emb_dims, aggr=graph_gnn_aggr),
#                     # ('node', 'line_hyperedges', 'node'): HypergraphConv(emb_dims, emb_dims, aggr=graph_gnn_aggr),
#                     # ('node','dfg_hyperedges','node'):HypergraphConv(emb_dims, emb_dims, aggr=graph_gnn_aggr),
#                     # ('node','block_hyperedges','node'):HypergraphConv(emb_dims, emb_dims, aggr=graph_gnn_aggr),
                    
#                     ('node', 'parent_child_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims,emb_dims,1),
#                     ('node', 'line_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims,emb_dims,1),
#                     ('node','dfg_hyperedges','node'):HyperedgeDiffusionConv(emb_dims,emb_dims,1),
#                     ('node','block_hyperedges','node'):HyperedgeDiffusionConv(emb_dims,emb_dims,1),
#                     ('node','layout_sibling_hyperedges','node'):HyperedgeDiffusionConv(emb_dims,emb_dims,1),
#                 },aggr='sum')
                    
#             else:
#                 gnn=HeteroConv({
#                     ('node', 'base_child', 'node'): graph_GNN(emb_dims, emb_dims, aggr=graph_gnn_aggr),
#                     ('node', 'base_father', 'node'): graph_GNN(emb_dims, emb_dims, aggr=graph_gnn_aggr),
                    
#                     ('node', 'sibling_next', 'node'): graph_GNN(emb_dims, emb_dims, aggr=graph_gnn_aggr),
#                     ('node', 'sibling_prev', 'node'): graph_GNN(emb_dims, emb_dims, aggr=graph_gnn_aggr),
#                     ('node', 'dfg_next', 'node'): graph_GNN(emb_dims, emb_dims, aggr=graph_gnn_aggr),
#                     ('node', 'dfg_prev', 'node'): graph_GNN(emb_dims, emb_dims, aggr=graph_gnn_aggr),
#                     ('node', 'code_next', 'node'): graph_GNN(emb_dims, emb_dims, aggr=graph_gnn_aggr),
#                     ('node', 'code_prev', 'node'): graph_GNN(emb_dims, emb_dims, aggr=graph_gnn_aggr),
#                     ('node', 'parent_child_hyperedges', 'node'): HypergraphConv((emb_dims,emb_dims), emb_dims, aggr=graph_gnn_aggr,root_weight=False),
#                     ('node', 'line_hyperedges', 'node'): HypergraphConv((emb_dims,emb_dims), emb_dims, aggr=graph_gnn_aggr,root_weight=False),
#                 },aggr='sum')
#             self.gnn_ops.append(gnn)
#             self.grelu_ops.append(nn.Sequential(nn.ReLU(), nn.Dropout(p=drop_rate)))
#             self.gnorm_ops.append(GraphNorm(emb_dims))

#     def forward(self, data):
#         assert len(data['node'].x.size()) == 1  #[batch_graph_node_num,]
#         assert len(data['node'].src_map.size())==1 #[batch_graph_node_num,]
#         # assert len(data['node'].code_pos.size())==2 # [2(m,n),batch_graph_node_num]
#         assert len(data['node'].code_mask.size())==1 #[batch_graph_node_num,]
#         # assert len(data.edge_index_dict[('node','base_child','node')].size()) == 2  # 点是一堆节点序号[2,batch_xx_edge_num]
#         # assert len(data.edge_index_dict[('node','base_father','node')].size()) == 2  # 点是一堆节点序号[2,batch_xx_edge_num]
#         # assert len(data.edge_index_dict[('node','sibling_prev','node')].size()) == 2  # 点是一堆节点序号[2,batch_xx_edge_num]
#         # assert len(data.edge_index_dict[('node','sibling_next','node')].size()) == 2  # 点是一堆节点序号[2,batch_xx_edge_num]
#         # assert len(data.edge_index_dict[('node','dfg_prev','node')].size()) == 2  # 点是一堆节点序号[2,batch_xx_edge_num]
#         # assert len(data.edge_index_dict[('node','dfg_next','node')].size()) == 2  # 点是一堆节点序号[2,batch_xx_edge_num]
#         # assert len(data.edge_index_dict[('node','code_prev','node')].size()) == 2  # 点是一堆节点序号[2,batch_xx_edge_num]
#         # assert len(data.edge_index_dict[('node','code_next','node')].size()) == 2  # 点是一堆节点序号[2,batch_xx_edge_num]
        

#         graph_node_emb=self.graph_node_emb_op(data.x_dict['node'])  ##[batch_graph_node_num,emb_dims]
#         # graph_node_emb[sim_node_mask==True,:]=graph_node_emb[sim_node_mask==True,:].add(sim_node_emb)*0.5
        
        
#         #  方式1: 基于节点在batch中的索引添加位置编码
#         batch_size = data.x_batch_dict['node'].max().item() + 1
#         device = graph_node_emb.device
        
#         # 为每个batch中的节点创建位置索引
#         pos_indices_list = []
#         for b in range(batch_size):
#             mask = data.x_batch_dict['node'] == b
#             num_nodes_in_batch = mask.sum().item()
            
#             # 关键修复：确保不超出 embedding 表的范围
#             max_allowed_pos = self.graph_pos_encoding.num_embeddings - 1
#             if num_nodes_in_batch > max_allowed_pos:
#                 # 如果超出范围，截断或使用模运算
#                 # 方法A: 截断（简单但可能丢失信息）
#                 # positions = torch.arange(1, min(num_nodes_in_batch, max_allowed_pos) + 1, device=device)
#                 # positions = torch.cat([positions, torch.full((num_nodes_in_batch - len(positions),), max_allowed_pos, device=device)])
                
#                 # 方法B: 模运算（循环使用位置编码，推荐）
#                 positions = torch.arange(1, num_nodes_in_batch + 1, device=device)
#                 positions = (positions % max_allowed_pos) + 1  # 确保在 [1, max_allowed_pos] 范围内
#             else:
#                 positions = torch.arange(1, num_nodes_in_batch + 1, device=device)
            
#             pos_indices_list.append(positions)
        
#         pos_indices = torch.cat(pos_indices_list)
        
#         # 直接通过Embedding获取位置编码
#         pos_emb = self.graph_pos_encoding(pos_indices)  # [batch_graph_node_num, emb_dims]
        
#         # 添加位置编码
#         graph_node_emb = graph_node_emb * np.sqrt(self.emb_dims)
#         graph_node_emb = graph_node_emb + pos_emb
#         # graph_node_emb = self.emb_layer_norm(graph_node_emb)
        
#         # batch_size = data.x_batch_dict['node'].max().item() + 1
#         # device = graph_node_emb.device
        
#         # # 为每个batch中的节点创建位置索引
#         # pos_indices_list = []
#         # for b in range(batch_size):
#         #     mask = data.x_batch_dict['node'] == b
#         #     num_nodes_in_batch = mask.sum().item()
#         #     # 位置从1开始 (0保留给padding)
#         #     pos_indices_list.append(
#         #         torch.arange(1, num_nodes_in_batch + 1, device=device)
#         #     )
#         # pos_indices = torch.cat(pos_indices_list)  # [batch_graph_node_num]
        
#         # # 获取位置编码
#         # pos_emb = self.pos_encoding(pos_indices)  # [batch_graph_node_num, emb_dims]
        
#         # # 添加位置编码 (Transformer标准做法)
#         # graph_node_emb = graph_node_emb * np.sqrt(self.emb_dims)
#         # graph_node_emb = graph_node_emb + pos_emb
        
#         # # 方式2: 为代码节点(叶子节点)单独添加序列位置编码
#         # code_mask = data['node'].code_mask
#         # if code_mask.any():
#         #     code_x_batch = data.x_batch_dict['node'][code_mask]
#         #     code_positions = torch.zeros(code_mask.sum(),device=device).long()
            
#         #     for i in range(batch_size):
#         #         mask = code_x_batch == i
#         #         if mask.any():
#         #             code_count = mask.sum().item()
#         #             code_positions[mask] = torch.arange(code_count)
            
#         #     code_pos_emb = self.code_pos_encoding.weight(code_positions)
#         #     graph_node_emb[code_mask] = graph_node_emb[code_mask] + code_pos_emb
#     # =========================================
        
#         # 归一化和dropout
#         # graph_node_emb = self.emb_layer_norm(graph_node_emb)
#         data['node'].x=self.emb_drop_op(graph_node_emb) ##[batch_graph_node_num,emb_dims]
#         # data['node'].x=self.graph_emb_norm_op(data['node'].x)
#         # graph_node_emb2=data['node'].x.clone()
#         # code_emb=data['node'].x[data['node'].code_mask==True,:]* np.sqrt(self.emb_dims) ##[batch_leaf_node_num,emb_dims]
#         # data['node'].x=self.graph_emb_norm_op(data['node'].x) ##[batch_graph_node_num,emb_dims]
#         # code_mpos_emb=self.code_mpos_emb_op(data['node'].code_pos[0,:][data['node'].code_mask==True])     #[batch_leaf_node_num,emb_dims]
#         # code_npos_emb=self.code_npos_emb_op(data['node'].code_pos[1,:][data['node'].code_mask==True])     #[batch_leaf_node_num,emb_dims]
#         # code_pos_emb=self.emb_drop_op(code_mpos_emb.add(code_npos_emb)) #[batch_leaf_node_num,emb_dims]

#         code_x_batch=data.x_batch_dict['node'][data['node'].code_mask==True]    #[batch_leaf_node_num,]
        
#         # code_emb,code_mask=to_dense_batch(code_emb,
#         #                                 batch=code_x_batch,
#         #                                 fill_value=self.pad_idx,
#         #                                 max_num_nodes=self.code_max_len)    #[batch_size,code_max_len,emb_dims],[batch_size,code_max_len]
#         # code_pos_emb,_=to_dense_batch(code_pos_emb,
#         #                                 batch=code_x_batch,
#         #                                 fill_value=self.pad_idx,
#         #                                 max_num_nodes=self.code_max_len)    #[batch_size,code_max_len,emb_dims],[batch_size,code_max_len]
#         # code_emb=self.code_emb_norm_op(code_emb.add(code_pos_emb))   #[batch_size,code_max_len,emb_dims]
#         # code_enc=self.code_enc_op(query=code_emb,query_mask=code_mask)  # [batch_data_num,code_max_len,emb_dims]
#         # sparse_code_enc=code_enc.contiguous().view(-1,code_enc.size(-1))[code_mask.view(-1)==True,:] ###[batch_leaf_node_num,emb_dims] convert dense batch into sparse batch
#         # data['node'].x[data['node'].code_mask==True,:]=data['node'].x[data['node'].code_mask==True,:].add(sparse_code_enc)  #[batch_leaf_node_num,emb_dims]
        
        
#         # =code_emb
#         # graph_node_emb=data['node'].x.clone()
#         for gnn,relu,norm in zip(self.gnn_ops,self.grelu_ops,self.gnorm_ops):
#             x_dict=gnn(x_dict=data.x_dict,edge_index_dict=data.edge_index_dict)   # dict(xx_node:[batch_xx_node_num,hid_dims])
#             data['node'].x=norm(data['node'].x.add(relu(x_dict['node']))) #data[key].x residual connection ,batch=data.x_batch_dict['node']
#             # data['node'].x=norm(graph_node_emb.add(relu(x_dict['node']))) #data[key].x residual connection
        
#         # data['node'].x=graph_node_emb2.add(data['node'].x)
        
#         graph_enc,_=to_dense_batch(data.x_dict['node'],
#                                   batch=data.x_batch_dict['node'], #data['leaf'].x_batch也可以
#                                   fill_value=self.pad_idx,
#                                   max_num_nodes=self.graph_max_size)  #[batch_size,graph_max_size,emb_dims],[batch_size,graph_max_size]

#         code_src_map,_=to_dense_batch(data.src_map_dict['node'][data['node'].code_mask==True],
#                                         batch=code_x_batch,  # data['leaf'].x_batch也可以
#                                         fill_value=self.pad_idx,
#                                         max_num_nodes=self.code_max_len)    # [batch_data_num,code_max_len]
#         graph_code_enc,_=to_dense_batch(data.x_dict['node'][data['node'].code_mask==True],
#                                         batch=code_x_batch,  # data['leaf'].x_batch也可以
#                                         fill_value=self.pad_idx,
#                                         max_num_nodes=self.code_max_len)    # [batch_data_num,code_max_len]

#         return graph_enc,graph_code_enc,code_src_map

# class Dec(nn.Module):
#     def __init__(self,
#                  emb_dims,
#                  text_voc_size,
#                  text_emb_op,
#                  text_max_len,
#                  enc_out_dims,
#                  att_layers,
#                  att_heads,
#                  att_head_dims=None,
#                  ff_hid_dims=2048,
#                  drop_rate=0.,
#                  **kwargs
#                  ):
#         super().__init__()
#         kwargs.setdefault('pad_idx', 0)
#         kwargs.setdefault('copy', True)
#         self._copy = kwargs['copy']
#         self.emb_dims = emb_dims
#         self.text_voc_size = text_voc_size
#         # embedding dims为text_voc_size+2*code_max_len

#         # assert len(text_sim_token_ids.shape)==1
#         # text_sim_token_voc_size=np.unique(text_sim_token_ids).shape[0]
#         # self.text_token_to_sim_token_map_op=nn.Embedding.from_pretrained(torch.tensor(text_sim_token_ids).view([-1,1]).float(),freeze=True,padding_idx=kwargs['pad_idx'])
#         # self.text_token_to_sim_token_map_op=text_token_to_sim_token_map_op
#         self.text_emb_op = text_emb_op
#         # self.text_emb_op = nn.Embedding(text_voc_size + code_max_len, emb_dims, padding_idx=kwargs['pad_idx'])
#         # self.sim_token_emb_op = nn.Embedding(text_sim_token_voc_size, emb_dims, padding_idx=kwargs['pad_idx'])
#         # nn.init.xavier_uniform_(self.text_emb_op.weight[1:, ])
#         # nn.init.xavier_uniform_(self.sim_token_emb_op.weight[1:, ])
#         self.pos_encoding = PosEnc(max_len=text_max_len+1, emb_dims=emb_dims, train=True, pad=True,pad_idx=kwargs['pad_idx'])  #不要忘了+1,因为输入前加了begin_id
#         # nn.init.xavier_uniform_(self.pos_encoding.weight[1:, ])
#         self.emb_layer_norm = nn.LayerNorm(emb_dims)
#         self.text_dec_op = TranDec(query_dims=emb_dims,
#                                    key_dims=enc_out_dims,
#                                    head_nums=att_heads,
#                                    head_dims=att_head_dims,
#                                    layer_num=att_layers,
#                                    ff_hid_dims=ff_hid_dims,
#                                    drop_rate=drop_rate,
#                                    pad_idx=kwargs['pad_idx'],
#                                    self_causality=True)
#         # self.text_dec_op = DualTranDec(query_dims=emb_dims,
#         #                             key_dims=enc_out_dims,
#         #                             head_num=att_heads,
#         #                             ff_hid_dims=ff_hid_dims,
#         #                             head_dims=att_head_dims,
#         #                             layer_num=att_layers,
#         #                             drop_rate=drop_rate,
#         #                             pad_idx=kwargs['pad_idx'],
#         #                             mode='sequential',
#         #                             self_causality=True)
#         self.dropout = nn.Dropout(p=drop_rate)
#         self.out_fc = nn.Linear(emb_dims, text_voc_size)
#         self.copy_generator = MultiCopyGenerator(tgt_dims=emb_dims,
#                                                      tgt_voc_size=text_voc_size,
#                                                      src_dims=enc_out_dims,
#                                                      att_heads=att_heads,
#                                                      att_head_dims=att_head_dims,
#                                                      drop_rate=drop_rate,
#                                                      pad_idx=kwargs['pad_idx'])

#     def forward(self,graph_enc,graph_code_enc,code_src_map,text_input):
#         # sim_text_token=self.text_token_to_sim_token_map_op(text_input).squeeze(-1).long()  #[batch_text,L_text]
#         # sim_token_mask=sim_text_token.abs().sign()   #[batch_text,L_text]
#         # sim_token_emb=self.text_sim_token_emb_op(sim_text_token[sim_token_mask==True])   # (B*L_text,D_text_emb)  .view(sim_text_token.size())
        
#         text_emb = self.text_emb_op(text_input)   # (B,L_text,D_text_emb)
#         # text_emb[sim_token_mask==True,:]=text_emb[sim_token_mask==True,:].add(sim_token_emb)*0.5
#         text_emb=text_emb* np.sqrt(self.emb_dims)
#         pos_emb = self.pos_encoding(text_input)  # # (B,L_text,D_emb)
#         text_dec = self.dropout(text_emb.add(pos_emb))  # (B,L_text,D_emb)
#         text_dec = self.emb_layer_norm(text_dec)  # (B,L_text,D_emb)

#         graph_mask = graph_enc.abs().sum(-1).sign()  # (batch_size,graph_max_size)
#         # code_mask=code_enc.abs().sum(-1).sign() # (batch_size,code_max_len)
#         text_mask = text_input.abs().sign()  # (B,L_text)
#         text_dec = self.text_dec_op(query=text_dec,
#                                     key=graph_enc,
#                                     query_mask=text_mask,
#                                     key_mask=graph_mask
#                                     )  # (B,L_text,D_text_emb)

#         if not self._copy:
#             text_output = self.out_fc(text_dec)  # (B,L_text,text_voc_size)包含begin_idx和end_idx
#             # text_output = F.softmax(text_output, dim=-1)
#             # text_output[:,:,-1]=0.    #不生成begin_idx，默认该位在text_voc_size最后一个，置0
#         else:
#             # text_output=F.pad(text_output,(0,2*self.text_max_len)) #pad last dim
#             text_output = self.copy_generator(text_dec,
#                                              graph_code_enc,code_src_map)
#         # text_output[:, :, self.text_voc_size - 1] = 0.  # 不生成begin_idx，默认该位在text_voc_size最后一个，置0
#         # text_output[:, :, 0] = 0.  # pad位不生成
#         return text_output.transpose(1, 2)

# class TNet(BaseNet):
#     def __init__(self,
#                  emb_dims,
#                  graph_max_size,
#                  code_max_len,
#                  text_max_len,
#                 #  sim_token_ids,
#                  io_voc_size,
#                 #  code_mpos_voc_size,
#                 #  code_npos_voc_size,
#                  text_voc_size,
#                 #  code_att_layers=2,
#                 #  code_att_heads=8,
#                 #  code_att_head_dims=None,
#                 #  code_ff_hid_dims=2048,
#                  graph_gnn_layers=6,
#                  graph_GNN=SAGEConv,
#                  graph_gnn_aggr='add',
#                  text_att_layers=3,
#                  text_att_heads=8,
#                  text_att_head_dims=None,
#                  text_ff_hid_dims=2048,
#                  drop_rate=0.,
#                  **kwargs,
#                  ):
#         super().__init__()
#         kwargs.setdefault('copy', True)
#         kwargs.setdefault('pad_idx', 0)  # GraphData.batch to_dense_data用的
#         self.init_params = locals()
#         io_token_emb_op=nn.Embedding(io_voc_size, emb_dims, padding_idx=kwargs['pad_idx'])
#         nn.init.xavier_uniform_(io_token_emb_op.weight[1:, ])
#         # assert len(sim_token_ids.shape)==1
#         # print(np.unique(sim_token_ids).shape[0],np.unique(sim_token_ids).max()+1)
#         # sim_token_voc_size=np.unique(sim_token_ids).shape[0]
#         # assert np.unique(sim_token_ids).shape[0]==np.unique(sim_token_ids).max()+1
#         # io_token_to_sim_token_map_op=nn.Embedding.from_pretrained(torch.tensor(sim_token_ids).view([-1,1]).float(),freeze=True,padding_idx=kwargs['pad_idx'])
#         # sim_token_emb_op = nn.Embedding(np.unique(sim_token_ids).shape[0], emb_dims, padding_idx=kwargs['pad_idx'])
#         # nn.init.xavier_uniform_(sim_token_emb_op.weight[1:, ])
#         self.enc_op = CodeGraphEnc(emb_dims=emb_dims,
#                                 graph_max_size=graph_max_size,
#                                 code_max_len=code_max_len,
#                                 # graph_node_voc_size=graph_node_voc_size,
#                                 graph_node_emb_op=io_token_emb_op,
#                                 # graph_node_to_sim_token_map_op=io_token_to_sim_token_map_op,
#                                 # code_mpos_voc_size=code_mpos_voc_size,
#                                 # code_npos_voc_size=code_npos_voc_size,
#                                 # code_att_layers=code_att_layers,
#                                 # code_att_heads=code_att_heads,
#                                 # code_att_head_dims=code_att_head_dims,
#                                 # code_ff_hid_dims=code_ff_hid_dims,
#                                 graph_gnn_layers=graph_gnn_layers,
#                                 graph_GNN=graph_GNN,
#                                 graph_gnn_aggr=graph_gnn_aggr,
#                                 drop_rate=drop_rate,
#                                 pad_idx=kwargs['pad_idx'])
#         self.dec_op = Dec(emb_dims=emb_dims,
#                             text_voc_size=text_voc_size,
#                             text_max_len=text_max_len,
#                             # code_max_len=code_max_len,
#                             text_emb_op=io_token_emb_op,
#                             # text_token_to_sim_token_map_op=io_token_to_sim_token_map_op,
#                             enc_out_dims=emb_dims,
#                             att_layers=text_att_layers,
#                             att_heads=text_att_heads,
#                             att_head_dims=text_att_head_dims,
#                             ff_hid_dims=text_ff_hid_dims,
#                             drop_rate=drop_rate,
#                             copy=kwargs['copy'],
#                             pad_idx=kwargs['pad_idx'])

#     def forward(self, code_graph):
#         text_input=code_graph['text'].text_token_input.clone()
#         del code_graph['text']
#         graph_enc,graph_code_enc,code_src_map = self.enc_op(data=code_graph)
#         text_output = self.dec_op(graph_enc=graph_enc,graph_code_enc=graph_code_enc,
#                                     code_src_map=code_src_map,
#                                     text_input=text_input)
#         return text_output

# class TModel(TransSeq2Seq):
#     def __init__(self,
#                 #  sim_token_ids,
#                  model_dir,
#                  model_name='Transformer_based_model',
#                  model_id=None,
#                  emb_dims=512,
#                 #  code_att_layers=3,
#                 #  code_att_heads=8,
#                 #  code_att_head_dims=None,
#                 #  code_ff_hid_dims=2048,
#                  graph_gnn_layers=3,
#                  graph_GNN=SAGEConv,
#                  graph_gnn_aggr='add',
#                  text_att_layers=3,
#                  text_att_heads=8,
#                  text_att_head_dims=None,
#                  text_ff_hid_dims=2048,
#                  drop_rate=0.,
#                  copy=True,
#                  pad_idx=0,
#                  train_batch_size=32,
#                  pred_batch_size=32,
#                  max_train_size=-1,
#                  max_valid_size=32 * 10,
#                  max_big_epochs=20,
#                  regular_rate=1e-5,
#                  lr_base=0.001,
#                  lr_decay=0.9,
#                  min_lr_rate=0.01,
#                  warm_big_epochs=2,
#                  start_valid_epoch=20,
#                  early_stop=20,
#                  Net=TNet,
#                  Dataset=Datasetx,
#                  beam_width=1,
#                  train_metrics=[get_sent_bleu],
#                  valid_metric=get_sent_bleu,
#                  test_metrics=[get_sent_bleu],
#                  train_mode=True,
#                  **kwargs
#                  ):
#         logging.info('Construct %s' % model_name)
#         super().__init__(model_name=model_name,
#                          model_dir=model_dir,
#                          model_id=model_id)
#         self.init_params = locals()
#         # self.sim_token_ids=sim_token_ids
#         self.emb_dims = emb_dims
#         # self.code_att_layers = code_att_layers
#         # self.code_att_heads = code_att_heads
#         # self.code_att_head_dims = code_att_head_dims
#         # self.code_ff_hid_dims = code_ff_hid_dims
#         self.graph_gnn_layers = graph_gnn_layers
#         self.graph_GNN = graph_GNN
#         self.graph_gnn_aggr = graph_gnn_aggr
#         self.text_att_layers = text_att_layers
#         self.text_att_heads = text_att_heads
#         self.text_att_head_dims = text_att_head_dims
#         self.text_ff_hid_dims = text_ff_hid_dims
#         self.drop_rate = drop_rate
#         self.pad_idx = pad_idx
#         self.copy = copy
#         self.train_batch_size = train_batch_size
#         self.pred_batch_size = pred_batch_size
#         self.max_train_size = max_train_size
#         self.max_valid_size = max_valid_size
#         self.max_big_epochs = max_big_epochs
#         self.regular_rate = regular_rate
#         self.lr_base = lr_base
#         self.lr_decay = lr_decay
#         self.min_lr_rate = min_lr_rate
#         self.warm_big_epochs = warm_big_epochs
#         self.start_valid_epoch=start_valid_epoch
#         self.early_stop=early_stop
#         self.Net = Net
#         self.Dataset = Dataset
#         self.beam_width = beam_width
#         self.train_metrics = train_metrics
#         self.valid_metric = valid_metric
#         self.test_metrics = test_metrics
#         self.train_mode = train_mode

#     def _logging_paramerter_num(self):
#         logging.info("{} have {} paramerters in total".format(self.model_name, sum(
#             x.numel() for x in self.net.parameters() if x.requires_grad)))
#         # 计算enc+dec的parameter总数
#         code_graph_enc_param_num = sum(x.numel() for x in self.net.module.enc_op.gnn_ops.parameters() if x.requires_grad) + \
#                                     sum(x.numel() for x in self.net.module.enc_op.gnorm_ops.parameters() if x.requires_grad) + \
#                                     sum(x.numel() for x in self.net.module.enc_op.grelu_ops.parameters() if x.requires_grad)

#         text_dec_param_num = sum(x.numel() for x in self.net.module.dec_op.text_dec_op.parameters() if x.requires_grad)
#                             # sum(x.numel() for x in self.net.module.dec_op.copy_generator.parameters() if x.requires_grad)
#         enc_dec_param_num = code_graph_enc_param_num + text_dec_param_num
#         logging.info("{} have {} paramerters in encoder and decoder".format(self.model_name, enc_dec_param_num))

#     def fit(self,
#             train_data,
#             valid_data,
#             **kwargs
#             ):
#         self.graph_max_size=0
#         self.code_max_len = 0
#         self.io_voc_size = 0
#         # self.code_mpos_voc_size = 0
#         # self.code_npos_voc_size = 0
#         self.text_max_len=0
#         for code_graph,text in zip(train_data['code_graphs'],train_data['texts']):
#             self.graph_max_size = max(self.graph_max_size,len(code_graph['nodes']))
#             self.code_max_len = max(self.code_max_len,code_graph['code_node_mask'].sum())
#             self.io_voc_size = max(self.io_voc_size,max(code_graph['nodes']))
#             # self.code_mpos_voc_size = max(self.code_mpos_voc_size,np.max(code_graph['node_in_code_poses'][0,:]))
#             # self.code_npos_voc_size = max(self.code_npos_voc_size,np.max(code_graph['node_in_code_poses'][1,:]))
#             self.text_max_len=max(self.text_max_len,len(text))
#         self.io_voc_size+=1
#         # self.code_mpos_voc_size+=1
#         # self.code_npos_voc_size+=1

#         self.text_voc_size = len(train_data['text_dic']['text_i2w'])  # 包含了begin_idx和end_idx
#         self.io_voc_size=max(self.io_voc_size,self.text_voc_size+2*self.code_max_len)
#         # print(self.graph_max_size, self.code_max_len,self.text_max_len,
#         #       self.io_voc_size, self.text_voc_size,
#         #       self.code_mpos_voc_size,self.code_npos_voc_size)

#         net = self.Net(
#                         # sim_token_ids=self.sim_token_ids,
#                         emb_dims=self.emb_dims,
#                        graph_max_size=self.graph_max_size,
#                        code_max_len=self.code_max_len,
#                        text_max_len=self.text_max_len,
#                        io_voc_size=self.io_voc_size,
#                     #    code_mpos_voc_size=self.code_mpos_voc_size,
#                     #    code_npos_voc_size=self.code_npos_voc_size,
#                        text_voc_size=self.text_voc_size,
#                     #    code_att_layers=self.code_att_layers,
#                     #    code_att_heads=self.code_att_heads,
#                     #    code_att_head_dims=self.code_att_head_dims,
#                     #    code_ff_hid_dims=self.code_ff_hid_dims,
#                        graph_gnn_layers=self.graph_gnn_layers,
#                        graph_GNN=self.graph_GNN,
#                        graph_gnn_aggr=self.graph_gnn_aggr,
#                        text_att_layers=self.text_att_layers,
#                        text_att_heads=self.text_att_heads,
#                        text_att_head_dims=self.text_att_head_dims,
#                        text_ff_hid_dims=self.text_ff_hid_dims,
#                        drop_rate=self.drop_rate,
#                        pad_idx=self.pad_idx,
#                        copy=self.copy
#                        )

#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择GPU优先

#         self.net =DataParallel(net.to(device),follow_batch=['x'])  # 并行使用多GPU
#         # self.net = BalancedDataParallel(0, net.to(device), dim=0)  # 并行使用多GPU
#         # self.net = net.to(device)  # 数据转移到设备
#         self._logging_paramerter_num()  # 需要有并行的self.net和self.model_name
#         self.net.train()  # 设置网络为训练模式

#         self.optimizer = optim.Adam(self.net.parameters(),
#                                     lr=self.lr_base,
#                                     weight_decay=self.regular_rate)

#         # graph_enc_params=self.net.module.enc.graph_enc.parameters()
#         # graph_enc_param_ids=list(map(id,graph_enc_params))
#         # ex_params=filter(lambda p: id(p) not in graph_enc_param_ids,self.net.parameters())
#         # optim_cfg = [{'params': graph_enc_params, 'lr': 0.001,'weight_decay': self.regular_rate* 10.},
#         #              {'params': ex_params, 'lr': self.lr_base, 'weight_decay': self.regular_rate}]
#         # self.optimizer=optim.Adam(optim_cfg)

#         self.criterion = LabelSmoothSoftmaxCEV2(reduction='mean', ignore_index=self.pad_idx, label_smooth=0.0)
#         # self.criterion = nn.NLLLoss(ignore_index=self.pad_idx)

#         self.text_begin_idx = self.text_voc_size - 1
#         self.text_end_idx = self.text_voc_size - 2
#         self.tgt_begin_idx,self.tgt_end_idx=self.text_begin_idx,self.text_end_idx
#         assert train_data['text_dic']['text_i2w'][self.text_end_idx] == OUT_END_TOKEN
#         assert train_data['text_dic']['text_i2w'][self.text_begin_idx] == OUT_BEGIN_TOKEN  # 最后两个是end_idx 和begin_idx

#         self.max_train_size = len(train_data['code_graphs']) if self.max_train_size == -1 else self.max_train_size
#         train_code_graphs, train_texts,train_ids = zip(*random.sample(list(zip(train_data['code_graphs'], train_data['texts'],train_data['ids'])),
#                                                      min(self.max_train_size,
#                                                          len(train_data['code_graphs']))
#                                                      )
#                                       )

#         train_set = self.Dataset(code_graphs=train_code_graphs,
#                                  texts=train_texts,
#                                  ids=train_ids,
#                                  text_max_len=self.text_max_len,
#                                  text_begin_idx=self.text_begin_idx,
#                                  text_end_idx=self.text_end_idx,
#                                  pad_idx=self.pad_idx)
#         # train_loader = DataLoader(dataset=train_set,
#         #                           train_batch_size=self.train_batch_size,
#         #                           shuffle=True,
#         #                           follow_batch=['graph_node', 'graph_node_after'])
#         train_loader=DataListLoader(dataset=train_set,
#                                     batch_size=self.train_batch_size,
#                                     shuffle=True,
#                                     drop_last=True) 

#         if self.warm_big_epochs is None:
#             self.warm_big_epochs = max(self.max_big_epochs // 10, 2)
#         self.scheduler = LrWarmUp(self.optimizer,
#                                   min_rate=self.min_lr_rate,
#                                   lr_decay=self.lr_decay,
#                                   warm_steps=self.warm_big_epochs * len(train_loader),
#                                   # max(self.max_big_epochs//10,2)*train_loader.__len__()
#                                   reduce_steps=len(train_loader))  # 预热次数 train_loader.__len__()
#         if self.train_mode:  # 如果进行训练
#             # best_net_path = os.path.join(self.model_dir, '{}_best_net.net'.format(self.model_name))
#             # self.net.load_state_dict(torch.load(best_net_path))
#             # self.net.train()
#             # torch.cuda.empty_cache()
#             for i in range(0,self.max_big_epochs):
#                 # logging.info('---------Train big epoch %d/%d' % (i + 1, self.max_big_epochs))
#                 pbar = tqdm(train_loader)
#                 for j, batch_data in enumerate(pbar):
#                     batch_text_output = []
#                     ids=[]
#                     for data in batch_data:
#                         batch_text_output.append(data['text'].text_token_output.unsqueeze(0))
#                         del data['text'].text_token_output
#                         ids.append(data['idx'].idx.item())
#                         del data['idx']

#                     batch_text_output = torch.cat(batch_text_output, dim=0).to(device)
#                     # print(batch_text_output[:2,:])
#                     pred_text_output = self.net(batch_data)

#                     loss = self.criterion(pred_text_output, batch_text_output)  # 计算loss
#                     self.optimizer.zero_grad()  # 梯度置0
#                     loss.backward()  # 反向传播
#                     # clip_grad_norm_(self.net.parameters(),1e-2)  #减弱梯度爆炸
#                     self.optimizer.step()  # 优化
#                     self.scheduler.step()  # 衰减

#                     # log_info = '[Big epoch:{}/{}]'.format(i + 1, self.max_big_epochs)
#                     # if i+1>=self.start_valid_epoch:
#                     text_dic = {'text_i2w': train_data['text_dic']['text_i2w'],
#                                'ex_text_i2ws': [train_data['text_dic']['ex_text_i2ws'][k] for k in ids]}
#                     log_info=self._get_log_fit_eval(loss=loss,
#                                                     pred_tgt=pred_text_output,
#                                                     gold_tgt=batch_text_output,
#                                                     tgt_i2w=text_dic
#                                                     )
#                     log_info = '[Big epoch:{}/{},{}]'.format(i + 1, self.max_big_epochs, log_info)
#                     pbar.set_description(log_info)
#                     del pred_text_output,batch_text_output,batch_data

#                 del pbar
#                 if i+1 >= self.start_valid_epoch:
#                     self.max_valid_size = len(valid_data['code_graphs']) if self.max_valid_size == -1 else self.max_valid_size
#                     valid_srcs, valid_tgts, ex_text_i2ws = zip(*random.sample(list(zip(valid_data['code_graphs'],
#                                                                                        valid_data['texts'],
#                                                                                        valid_data['text_dic']['ex_text_i2ws'])),
#                                                                               min(self.max_valid_size,
#                                                                                   len(valid_data['code_graphs']))
#                                                                               )
#                                                                )
#                     text_dic = {'text_i2w': train_data['text_dic']['text_i2w'],
#                                 'ex_text_i2ws': ex_text_i2ws}
#                     # torch.cuda.empty_cache()
#                     worse_epochs = self._do_validation(valid_srcs=valid_srcs,  # valid_data['code_graphs']
#                                                        valid_tgts=valid_tgts,  # valid_data['texts']
#                                                        tgt_i2w=text_dic,  # valid_data['text_dic']
#                                                        increase_better=True,
#                                                        last=False)  # 根据验证集loss选择best_net
#                     # worse_epochs = self._do_validation(valid_srcs=valid_data['code_graphs'],  #
#                     #                                    valid_tgts=valid_data['texts'],  #
#                     #                                    tgt_i2w=valid_data['text_dic'],  #
#                     #                                    increase_better=True,
#                     #                                    last=False)  # 根据验证集loss选择best_net
#                     if worse_epochs>=self.early_stop:
#                         break
#         # torch.cuda.empty_cache()
#         self._do_validation(valid_srcs=valid_data['code_graphs'],
#                             valid_tgts=valid_data['texts'],
#                             tgt_i2w=valid_data['text_dic'],
#                             increase_better=True,
#                             last=True)  # 根据验证集loss选择best_net
#         self._logging_paramerter_num()  # 需要有并行的self.net和self.model_name

#     def predict(self,
#                 code_graphs,
#                 text_dic):
#         logging.info('Predict outputs of %s' % self.model_name)
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择GPU优先
#         # self.net = self.net.to(device)  # 数据转移到设备,不重新赋值不行
#         self.net.eval()  # 切换测试模式
#         enc_op=DataParallel(self.net.module.enc_op,follow_batch=['x'])
#         dec_op=torch.nn.DataParallel(self.net.module.dec_op)
#         # enc.eval()
#         # dec.eval()
#         data_set = self.Dataset(code_graphs=code_graphs,
#                                 texts=None,
#                                 ids=None,
#                                 text_max_len=self.text_max_len,
#                                 text_begin_idx=self.text_begin_idx,
#                                 text_end_idx=self.text_end_idx,
#                                 pad_idx=self.pad_idx)  # 数据集，没有out，不需要id

#         data_loader = DataListLoader(dataset=data_set,
#                                      batch_size=self.pred_batch_size,   #1.5,2.5
#                                      shuffle=False)
#                                  # follow_batch=['graph_node', 'graph_node_after'])  # data loader
#         pred_text_id_np_batches = []  # 所有batch的预测出的id np
#         with torch.no_grad():  # 取消梯度
#             pbar = tqdm(data_loader)
#             for batch_data in pbar:
#                 # 从batch_data图里把解码器输入输出端数据调出来
#                 batch_text_input = []
#                 for data in batch_data:
#                     batch_text_input.append(data['text'].text_token_input.unsqueeze(0))
#                     del data['text']
#                 batch_text_input = torch.cat(batch_text_input, dim=0).to(device)

#                 # 先跑encoder，生成编码
#                 batch_graph_enc,batch_graph_code_enc,batch_code_src_map=enc_op(batch_data)
#                 batch_text_output: list = []  # 每步的output tensor
#                 if self.beam_width == 1:
#                     for i in range(self.text_max_len + 1):  # 每步开启
#                         pred_out = dec_op(graph_enc=batch_graph_enc,graph_code_enc=batch_graph_code_enc,code_src_map=batch_code_src_map,text_input=batch_text_input)  # 预测该步输出 (B,text_voc_size,L_text)
#                         batch_text_output.append(pred_out[:, :, i].unsqueeze(-1).to('cpu').data.numpy())  # 将该步输出加入msg output
#                         if i < self.text_max_len:  # 如果没到最后，将id加入input
#                             batch_text_input[:, i + 1] = torch.argmax(pred_out[:, :, i], dim=1)
#                     batch_pred_text = np.concatenate(batch_text_output, axis=-1)[:, :, :-1]  # (B,D_tgt,L_tgt)
#                     batch_pred_text[:, self.tgt_begin_idx, :] = -np.inf  # (B,D_tgt,L_tgt)
#                     batch_pred_text[:, self.pad_idx, :] = -np.inf  # (B,D_tgt,L_tgt)
#                     batch_pred_text_np = np.argmax(batch_pred_text, axis=1)  # (B,L_tgt) 要除去pad id和begin id
#                     pred_text_id_np_batches.append(batch_pred_text_np)  # [(B,L_tgt)]
#                 else:
#                     batch_pred_text=trans_beam_search(net=dec_op,
#                                                       beam_width=self.beam_width,
#                                                       dec_input_arg_name='text_input',
#                                                       length_penalty=1,
#                                                       begin_idx=self.tgt_begin_idx,
#                                                       pad_idx=self.pad_idx,
#                                                       end_idx=self.tgt_end_idx,
#                                                       graph_enc=batch_graph_enc,
#                                                       graph_code_enc=batch_graph_code_enc,
#                                                       code_src_map=batch_code_src_map,
#                                                       text_input=batch_text_input
#                                                       )     # (B,L_tgt)

#                     pred_text_id_np_batches.append(batch_pred_text.to('cpu').data.numpy()[:,:-1])  # [(B,L_tgt)]

#         pred_text_id_np = np.concatenate(pred_text_id_np_batches,axis=0)  # (AB,tgt_voc_size,L_tgy)
#         self.net.train()  # 切换回训练模式
#         # pred_texts=[[{**text_dic['text_i2w'],**text_dic['ex_text_i2ws'][j]}[i] for ]]
#         # 利用字典将msg转为token
#         pred_texts = self._tgt_ids2tokens(pred_text_id_np, text_dic, self.text_end_idx)

#         return pred_texts  # 序列概率输出形状为（A,D)
    
#     def generate_texts(self,code_graphs,text_dic,res_path,gold_texts,raw_data,token_data,**kwargs):
#         '''
#         生成src对应的tgt并保存
#         :param code_graphs:
#         :param text_dic:
#         :param res_path:
#         :param kwargs:
#         :return:
#         '''
#         logging.info('>>>>>>>Generate the targets according to sources and save the result to {}'.format(res_path))
#         kwargs.setdefault('beam_width',1)
#         res_dir=os.path.dirname(res_path)
#         if not os.path.exists(res_dir):
#             os.makedirs(res_dir)
#         pred_texts=self.predict(code_graphs=code_graphs,
#                                 text_dic=text_dic
#                                 )
#         # codes=map(lambda x:x['code']['tokens'],code_graphs)
#         # codes=self._code_ids2tokens(codes,code_i2w,self.pad_idx)
#         gold_texts=self._tgt_ids2tokens(gold_texts,text_dic,self.pad_idx)
#         res_data = []
#         for i,(pred_text,gold_text,raw_item,token_item) in \
#                 enumerate(zip(pred_texts,gold_texts,raw_data,token_data)):
#             sent_bleu=self.valid_metric([pred_text],[gold_text])
#             res_data.append(dict(pred_text=' '.join(pred_text),
#                                  gold_text=' '.join(gold_text),
#                                  sent_bleu=sent_bleu,
#                                  raw_code=raw_item['code'],
#                                  raw_text=raw_item['text'],
#                                  id=raw_item['id'],
#                                  token_text=token_item['text'],
#                                  ))
#         # res_df=pd.DataFrame(res_dic).T
#         # # print(res_df)
#         # excel_writer = pd.ExcelWriter(res_path)  # 根据路径savePath打开一个excel写文件
#         # res_df.to_excel(excel_writer,header=True,index=True)
#         # excel_writer.save()
#         with codecs.open(res_path,'w',encoding='utf-8') as f:
#             json.dump(res_data,f,indent=4, ensure_ascii=False)
#         self._logging_paramerter_num()  # 需要有并行的self.net和self.model_name
#         logging.info('>>>>>>>The result has been saved to {}'.format(res_path))

#     def _code_ids2tokens(self,code_idss, code_i2w, end_idx):
#         return [[code_i2w[idx] for idx in (code_ids[:code_ids.tolist().index(end_idx)]
#                                                     if end_idx in code_ids else code_ids)]
#                           for code_ids in code_idss]
    
#     def _tgt_ids2tokens(self, text_id_np, text_dic, end_idx=0, **kwargs):
#         if self.copy:
#             text_tokens: list = []
#             for j, text_ids in enumerate(text_id_np):
#                 text_i2w = {**text_dic['text_i2w'], **text_dic['ex_text_i2ws'][j]}
#                 end_i = text_ids.tolist().index(end_idx) if end_idx in text_ids else len(text_ids)
#                 text_tokens.append([text_i2w[text_idx] for text_idx in text_ids[:end_i]])
#                 # if end_i == 0:
#                 #     print()
#         else:
#             text_i2w=text_dic['text_i2w']
#             text_tokens = [[text_i2w[idx] for idx in (text_ids[:text_ids.tolist().index(end_idx)]
#                                                       if end_idx in text_ids else text_ids)]
#                           for text_ids in text_id_np]

#         return text_tokens
#     # def _tgt_ids2tokens(self, text_id_np, text_dic, end_idx=0, **kwargs):
#     #     """将目标序列的 ID 转换为 token"""
        
#     #     def to_list(arr):
#     #         """统一转换为 Python list"""
#     #         if isinstance(arr, torch.Tensor):
#     #             return arr.cpu().numpy().tolist()
#     #         elif isinstance(arr, np.ndarray):
#     #             return arr.tolist()
#     #         else:
#     #             return list(arr)
        
#     #     def find_end(ids_list, end_idx):
#     #         """找到结束位置"""
#     #         try:
#     #             return ids_list.index(end_idx)
#     #         except ValueError:
#     #             return len(ids_list)
        
#     #     if self.copy:
#     #         text_tokens = []
#     #         for j, text_ids in enumerate(text_id_np):
#     #             # 合并基础词汇表和扩展词汇表
#     #             text_i2w = {**text_dic['text_i2w'], **text_dic['ex_text_i2ws'][j]}
                
#     #             text_ids_list = to_list(text_ids)
#     #             end_i = find_end(text_ids_list, end_idx)
                
#     #             # 转换 token，使用 get 避免 KeyError
#     #             tokens = [text_i2w.get(int(idx), '<UNK>') for idx in text_ids_list[:end_i]]
#     #             text_tokens.append(tokens)
#     #     else:
#     #         text_i2w = text_dic['text_i2w']
#     #         text_tokens = []
#     #         for text_ids in text_id_np:
#     #             text_ids_list = to_list(text_ids)
#     #             end_i = find_end(text_ids_list, end_idx)
                
#     #             tokens = [text_i2w.get(int(idx), '<UNK>') for idx in text_ids_list[:end_i]]
#     #             text_tokens.append(tokens)

#     #     return text_tokens

# if __name__ == '__main__':

#     logging.info('Parameters are listed below: \n'+'\n'.join(['{}: {}'.format(key,value) for key,value in params.items()]))

#     model = TModel(
#                     # sim_token_ids=np.load(io_token_sim_id_path),
#                     model_dir=params['model_dir'],
#                    model_name=params['model_name'],
#                    model_id=params['model_id'],
#                    emb_dims=params['emb_dims'],
#                 #    code_att_layers=params['code_att_layers'],
#                 #    code_att_heads=params['code_att_heads'],
#                 #    code_att_head_dims=params['code_att_head_dims'],
#                 #    code_ff_hid_dims=params['code_ff_hid_dims'],
#                    graph_gnn_layers=params['graph_gnn_layers'],
#                    graph_GNN=params['graph_GNN'],
#                    graph_gnn_aggr=params['graph_gnn_aggr'],
#                    text_att_layers=params['text_att_layers'],
#                    text_att_heads=params['text_att_heads'],
#                    text_att_head_dims=params['text_att_head_dims'],
#                    text_ff_hid_dims=params['text_ff_hid_dims'],
#                    drop_rate=params['drop_rate'],
#                    copy=params['copy'],
#                    pad_idx=params['pad_idx'],
#                    train_batch_size=params['train_batch_size'],
#                    pred_batch_size=params['pred_batch_size'],
#                    max_train_size=params['max_train_size'],  # -1 means all
#                    max_valid_size=params['max_valid_size'],  ####################10
#                    max_big_epochs=params['max_big_epochs'],
#                    regular_rate=params['regular_rate'],
#                    lr_base=params['lr_base'],
#                    lr_decay=params['lr_decay'],
#                    min_lr_rate=params['min_lr_rate'],
#                    warm_big_epochs=params['warm_big_epochs'],
#                    early_stop=params['early_stop'],
#                    start_valid_epoch=params['start_valid_epoch'],
#                    Net=TNet,
#                    Dataset=Datasetx,
#                    beam_width=params['beam_width'],
#                    train_metrics=train_metrics,
#                    valid_metric=valid_metric,
#                    test_metrics=test_metrics,
#                    train_mode=params['train_mode'])

#     logging.info('Load data ...')
#     # print(train_avail_data_path)
#     with codecs.open(train_avail_data_path, 'rb') as f:
#         train_data = pickle.load(f)
#     with codecs.open(valid_avail_data_path, 'rb') as f:
#         valid_data = pickle.load(f)
#     with codecs.open(test_avail_data_path, 'rb') as f:
#         test_data = pickle.load(f)
#     # io_token_sim_ids=np.load(io_token_sim_id_path)

#     # with codecs.open(code_node_i2w_path, 'rb') as f:
#     #     code_i2w = pickle.load(f)

#     with codecs.open(test_token_data_path,'r') as f:
#         test_token_data=json.load(f)

#     with codecs.open(test_raw_data_path,'r') as f:
#         test_raw_data=json.load(f)

#     # train_data['code_graphs']=train_data['code_graphs'][:1000]
#     # train_data['texts']=train_data['texts'][:1000]
#     # train_data['ids']=train_data['ids'][:1000]

#     # print(len(train_data['texts']), len(valid_data['texts']), len(test_data['texts']))
#     model.fit(train_data=train_data,
#               valid_data=valid_data)

#     for key, value in params.items():
#         logging.info('{}: {}'.format(key, value))
#     logging.info('Parameters are listed below: \n'+'\n'.join(['{}: {}'.format(key,value) for key,value in params.items()]))

#     # test_data['code_graphs']=test_data['code_graphs'][14246:]
#     # test_data['texts']=test_data['texts'][14246:]
#     # test_data['ids']=test_data['ids'][14246:]

#     # valid_data['code_graphs']=valid_data['code_graphs'][12762:]
#     # valid_data['texts']=valid_data['texts'][12762:]
#     # valid_data['ids']=valid_data['ids'][12762:]

#     test_eval_df=model.eval(test_srcs=test_data['code_graphs'],
#                             test_tgts=test_data['texts'],
#                             tgt_i2w=test_data['text_dic'])
#     logging.info('Model performance on test dataset:\n')
#     for i in range(0,len(test_eval_df.columns),4):
#         print(test_eval_df.iloc[:, i:i+4])

#     model.generate_texts(code_graphs=test_data['code_graphs'],
#                          text_dic=test_data['text_dic'],
#                          res_path=res_path,
#                          # code_i2w=code_i2w, d
#                          gold_texts=test_data['texts'],
#                          raw_data=test_raw_data,
#                          token_data=test_token_data)


# # coding=utf-8
# import os
# import re
# import sys
# sys.path.append('../../../')
# from lib.neural_module.learn_strategy import LrWarmUp
# from lib.neural_module.transformer import TranEnc, TranDec, DualTranDec,ResFF,ResMHA
# from lib.neural_module.embedding import PosEnc,SinusoidalPositionalEncoding
# from lib.neural_module.loss import LabelSmoothSoftmaxCEV2, CriterionNet
# from lib.neural_module.balanced_data_parallel import BalancedDataParallel
# from lib.neural_module.copy_attention import DualMultiCopyGenerator,MultiCopyGenerator,DualCopyGenerator
# from lib.neural_module.beam_search import trans_beam_search
# from lib.neural_model.seq_to_seq_model import TransSeq2Seq
# from lib.neural_model.base_model import BaseNet
# from lib.neural_module.transformer import ResFF
    
# # from torch_geometric.nn import HypergraphConv
# from typing import Any,Optional,Union

# from config import *

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler
# from torch.nn.utils import clip_grad_norm_
# from torch.utils.data import Dataset
# from torch_geometric.data import HeteroData
# from torch_geometric.loader.data_list_loader import DataListLoader
# from torch_geometric.utils import to_dense_batch
# from torch_geometric.data.storage import (BaseStorage, NodeStorage,EdgeStorage)
# from torch_geometric.nn.data_parallel import DataParallel
# from torch_geometric.nn import HeteroConv,GraphNorm,HypergraphConv,GATConv
# from lib.neural_module.DHGNN import HyperedgeDiffusionConv,KStepHypergraphConv
# import random
# import numpy as np
# import os
# import logging
# import pickle
# import json
# import codecs
# from tqdm import tqdm
# import pickle
# import numpy as np
# import pandas as pd
# import math
# from copy import deepcopy

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# NodeOrEdgeStorage = Union[NodeStorage, EdgeStorage]

# class Datax(HeteroData):
#     # def __init__(self,
#     #              graph_func_node=dict(x=None),
#     #              graph_attr_node=dict(x=None),
#     #              graph_glob_node=dict(x=None),
#     #              ):
#     #     super().__init__()
#     #     self.graph_func_node=graph_func_node
#     def __cat_dim__(self, key: str, value: Any,
#                     store: Optional[NodeOrEdgeStorage] = None, *args,
#                     **kwargs) -> Any:
#         if bool(re.search('(token)', key)): #|map
#             return None  # generate a new 0 dimension
#         if bool(re.search('(pos)', key)):
#             return -1
#         return super().__cat_dim__(key, value,store)    #return不能漏了！！！


# class _ProjectedEmbedding(nn.Module):
#     """
#     将 nn.Embedding（CodeBERT维度，如768）+ nn.Linear（→emb_dims，如512）
#     封装为与 nn.Embedding 接口一致的模块，使 enc_op/dec_op 对投影透明。
#     仅在 CodeBERT_dim != emb_dims 时使用。
#     """
#     def __init__(self, embedding: nn.Embedding, proj: nn.Linear):
#         super().__init__()
#         self.embedding = embedding
#         self.proj      = proj

#     def forward(self, indices: torch.Tensor) -> torch.Tensor:
#         return self.proj(self.embedding(indices))

#     @property
#     def weight(self):
#         return self.embedding.weight
#     # def __init__(self,
#     #              graph_func_node=dict(x=None),
#     #              graph_attr_node=dict(x=None),
#     #              graph_glob_node=dict(x=None),
#     #              ):
#     #     super().__init__()
#     #     self.graph_func_node=graph_func_node
#     def __cat_dim__(self, key: str, value: Any,
#                     store: Optional[NodeOrEdgeStorage] = None, *args,
#                     **kwargs) -> Any:
#         if bool(re.search('(token)', key)): #|map
#             return None  # generate a new 0 dimension
#         if bool(re.search('(pos)', key)):
#             return -1
#         return super().__cat_dim__(key, value,store)    #return不能漏了！！！

# class Datasetx(Dataset):
#     '''
#     文本对数据集对象（根据具体数据再修改）
#     '''
#     def __init__(self,
#                  code_graphs,
#                  texts=None,
#                  ids=None,
#                  text_max_len=None,
#                  text_begin_idx=1,
#                  text_end_idx=2,
#                  pad_idx=0):
#         self.len = len(code_graphs)  # 样本个数
#         self.text_max_len = text_max_len
#         self.text_begin_idx = text_begin_idx
#         self.text_end_idx = text_end_idx

#         if text_max_len is None and texts is not None:
#             self.text_max_len = max([len(text) for text in texts])  # 每个输出只是一个序列
#         self.code_graphs = code_graphs
#         self.texts = texts
#         self.ids = ids
#         self.pad_idx = pad_idx

#     def __getitem__(self, index):
#         if self.texts is None:
#             pad_text_in = np.zeros((self.text_max_len + 1,), dtype=np.int64)  # decoder端的输入
#             pad_text_in[0] = self.text_begin_idx
#             pad_text_out = None
#         else:
#             tru_text = self.texts[index][:self.text_max_len]  # 先做截断
#             pad_text_in = np.lib.pad(tru_text,
#                                     (1, self.text_max_len - len(tru_text)),
#                                     'constant',
#                                     constant_values=(self.text_begin_idx, self.pad_idx))
#             tru_text_out = np.lib.pad(tru_text,
#                                      (0, 1),
#                                      'constant',
#                                      constant_values=(0, self.text_end_idx))  # padding
#             pad_text_out = np.lib.pad(tru_text_out,
#                                      (0, self.text_max_len + 1 - len(tru_text_out)),
#                                      'constant',
#                                      constant_values=(self.pad_idx, self.pad_idx))  # padding
#             # pad_out_input=np.lib.pad(pad_out[:-1],(1,0),'constant',constant_values=(self.text_begin_idx, 0))
#         data=Datax()
#         data['node'].x=torch.tensor(self.code_graphs[index]['nodes'])
#         data['node'].src_map=torch.tensor(self.code_graphs[index]['node2text_map_ids']).long()
#         data['node'].code_mask=torch.tensor(self.code_graphs[index]['code_node_mask']).bool()
#         data['node','parent_child_hyperedges','node'].edge_index=torch.tensor(self.code_graphs[index]['parent_child_hyperedges']).long()
#         data['node','line_hyperedges','node'].edge_index=torch.tensor(self.code_graphs[index]['line_hyperedges']).long()
#         data['node','dfg_hyperedges','node'].edge_index=torch.tensor(self.code_graphs[index]['dfg_hyperedges']).long()
#         data['node','block_hyperedges','node'].edge_index=torch.tensor(self.code_graphs[index]['block_hyperedges']).long()
#         data['node','layout_sibling_hyperedges','node'].edge_index=torch.tensor(self.code_graphs[index]['layout_sibling_hyperedges']).long()
#         data['node','base_child','node'].edge_index=torch.tensor(self.code_graphs[index]['base_father2child_edges']).long()
#         data['node','base_father','node'].edge_index=torch.tensor(self.code_graphs[index]['base_child2father_edges']).long()
#         data['node','sibling_next','node'].edge_index=torch.tensor(self.code_graphs[index]['sibling_prev2next_edges']).long()
#         data['node','sibling_prev','node'].edge_index=torch.tensor(self.code_graphs[index]['sibling_next2prev_edges']).long()
#         data['node','dfg_next','node'].edge_index=torch.tensor(self.code_graphs[index]['dfg_prev2next_edges']).long()
#         data['node','dfg_prev','node'].edge_index=torch.tensor(self.code_graphs[index]['dfg_next2prev_edges']).long()
#         data['node','code_next','node'].edge_index=torch.tensor(self.code_graphs[index]['code_prev2next_edges']).long()
#         data['node','code_prev','node'].edge_index=torch.tensor(self.code_graphs[index]['code_next2prev_edges']).long()
#         data['node','cfg_next','node'].edge_index=torch.tensor(self.code_graphs[index]['cfg_prev2next_edges']).long()
#         data['node','cfg_prev','node'].edge_index=torch.tensor(self.code_graphs[index]['cfg_next2prev_edges']).long()


#         data['text'].text_token_input=torch.tensor(pad_text_in).long()
#         if self.texts is not None:
#             data['text'].text_token_output = torch.tensor(pad_text_out).long()
#         data['text'].num_nodes = pad_text_in.shape[0]
#         if self.ids is not None:
#             data['idx'].idx=torch.tensor(self.ids[index])
#             data['idx'].num_nodes=1
#         # print(data.num_nodes)
#         return data

#     def __len__(self):
#         return self.len

# class CodeGraphEnc(nn.Module):
#     def __init__(self,
#                  emb_dims,
#                  graph_max_size,
#                  code_max_len,
#                  graph_node_emb_op,
#                  graph_gnn_layers=6,
#                  graph_GNN=SAGEConv,
#                  graph_gnn_aggr='mean',
#                  drop_rate=0.,
#                  use_pos_encoding=True,       # 可选：是否使用图节点位置编码
#                  hyper_aggr_mode='hetero_conv', # 可选：'hetero_conv' 或 'weighted_sum'
#                  **kwargs,
#                  ):
#         super().__init__()
#         kwargs.setdefault('pad_idx', 0)
#         self.pad_idx = kwargs['pad_idx']
#         self.graph_max_size = graph_max_size
#         self.code_max_len = code_max_len
#         self.emb_dims = emb_dims
#         self.use_pos_encoding = use_pos_encoding
#         self.hyper_aggr_mode = hyper_aggr_mode

#         self.graph_node_emb_op = graph_node_emb_op

#         # ── 可选位置编码 ──────────────────────────────────────────────────────
#         # use_pos_encoding=True  : 为图节点添加可学习位置编码 + √emb_dims 缩放
#         # use_pos_encoding=False : 直接使用嵌入向量，无缩放（与HetCoS基线一致）
#         if self.use_pos_encoding:
#             max_position = graph_max_size * 2
#             self.graph_pos_encoding = nn.Embedding(
#                 max_position + 1,   # +1 for padding idx=0
#                 emb_dims,
#                 padding_idx=kwargs['pad_idx']
#             )
#             nn.init.xavier_uniform_(self.graph_pos_encoding.weight[1:, ])

#         self.emb_drop_op = nn.Dropout(p=drop_rate)

#         # ── 超图边类型定义（B12修复：parent_child，不再是parentparent_child）─
#         self.hyper_edge_types = [
#             ('node', 'parent_child_hyperedges',     'node'),
#             ('node', 'line_hyperedges',              'node'),
#             ('node', 'dfg_hyperedges',               'node'),
#             ('node', 'block_hyperedges',             'node'),
#             ('node', 'layout_sibling_hyperedges',    'node'),
#         ]

#         self.gnn_layers = graph_gnn_layers
#         self.gnorm_ops  = nn.ModuleList()
#         self.grelu_ops  = nn.ModuleList()

#         # ── 两种超图聚合模式 ──────────────────────────────────────────────────
#         # 'hetero_conv'  : 使用 HeteroConv(aggr='sum')，所有超边平等求和
#         # 'weighted_sum' : 每种超边有独立可学习权重，动态 softmax 加权（B2修复）
#         if self.hyper_aggr_mode == 'hetero_conv':
#             # 原始 HeteroConv 方式
#             self.gnn_ops = nn.ModuleList()
#             for _ in range(graph_gnn_layers):
#                 if graph_GNN == SAGEConv:
#                     gnn = HeteroConv({
#                         ('node', 'parent_child_hyperedges',  'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, 1),
#                         ('node', 'line_hyperedges',           'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, 1),
#                         ('node', 'dfg_hyperedges',            'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, 1),
#                         ('node', 'block_hyperedges',          'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, 1),
#                         ('node', 'layout_sibling_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, 1),
#                         # ('node', 'parent_child_hyperedges',  'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, 0),
#                         # ('node', 'line_hyperedges',           'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, 0),
#                         # ('node', 'dfg_hyperedges',            'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, 0),
#                         # ('node', 'block_hyperedges',          'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, 0),
#                         # ('node', 'layout_sibling_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, 0),
#                         # ('node', 'parent_child_hyperedges',  'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, 2),
#                         # ('node', 'line_hyperedges',           'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, 2),
#                         # ('node', 'dfg_hyperedges',            'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, 2),
#                         # ('node', 'block_hyperedges',          'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, 2),
#                         # ('node', 'layout_sibling_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, 2),
#                     # ('node', 'base_child', 'node'): graph_GNN((emb_dims,emb_dims), emb_dims, aggr=graph_gnn_aggr,root_weight=True),
#                     # ('node', 'base_father', 'node'): graph_GNN((emb_dims,emb_dims), emb_dims, aggr=graph_gnn_aggr,root_weight=False),
#                     # ('node', 'sibling_next', 'node'): graph_GNN((emb_dims,emb_dims), emb_dims, aggr=graph_gnn_aggr,root_weight=False),
#                     # ('node', 'sibling_prev', 'node'): graph_GNN((emb_dims,emb_dims), emb_dims, aggr=graph_gnn_aggr,root_weight=False),
#                     # ('node', 'dfg_next', 'node'): graph_GNN((emb_dims,emb_dims), emb_dims, aggr=graph_gnn_aggr,root_weight=False),
#                     # ('node', 'dfg_prev', 'node'): graph_GNN((emb_dims,emb_dims), emb_dims, aggr=graph_gnn_aggr,root_weight=False),
#                     # ('node', 'code_next', 'node'): graph_GNN((emb_dims,emb_dims), emb_dims, aggr=graph_gnn_aggr,root_weight=False),
#                     # ('node', 'code_prev', 'node'): graph_GNN((emb_dims,emb_dims), emb_dims, aggr=graph_gnn_aggr,root_weight=False),
#                     }, aggr='sum')
#                 else:
#                     gnn = HeteroConv({
#                         ('node', 'parent_child_hyperedges',  'node'): HypergraphConv(emb_dims, emb_dims),
#                         ('node', 'line_hyperedges',           'node'): HypergraphConv(emb_dims, emb_dims),
#                         ('node', 'dfg_hyperedges',            'node'): HypergraphConv(emb_dims, emb_dims),
#                         ('node', 'block_hyperedges',          'node'): HypergraphConv(emb_dims, emb_dims),
#                         ('node', 'layout_sibling_hyperedges', 'node'): HypergraphConv(emb_dims, emb_dims),
#                     }, aggr='sum')
#                 self.gnn_ops.append(gnn)
#                 self.grelu_ops.append(nn.Sequential(nn.ReLU(), nn.Dropout(p=drop_rate)))
#                 self.gnorm_ops.append(GraphNorm(emb_dims))

#         elif self.hyper_aggr_mode == 'weighted_sum':
#             # 可学习权重方式：每层每种超边有独立权重，动态 softmax
#             # 可学习权重矩阵 [graph_gnn_layers, num_hyper_types]
#             self.hyperedge_weights = nn.Parameter(
#                 torch.zeros(graph_gnn_layers, len(self.hyper_edge_types))
#             )
#             # 每层用 ModuleDict 存放各超边的独立卷积模块
#             self.hyper_gnn_ops = nn.ModuleList()
#             for _ in range(graph_gnn_layers):
#                 hyper_layer = nn.ModuleDict({
#                     et[1]: HyperedgeDiffusionConv(emb_dims, emb_dims, K=1)
#                     for et in self.hyper_edge_types
#                 })
#                 self.hyper_gnn_ops.append(hyper_layer)
#                 self.grelu_ops.append(nn.Sequential(nn.ReLU(), nn.Dropout(p=drop_rate)))
#                 self.gnorm_ops.append(GraphNorm(emb_dims))
#         else:
#             raise ValueError(
#                 f"hyper_aggr_mode must be 'hetero_conv' or 'weighted_sum', got '{hyper_aggr_mode}'"
#             )

#     def forward(self, data):
#         assert len(data['node'].x.size()) == 1       # [batch_graph_node_num,]
#         assert len(data['node'].src_map.size()) == 1 # [batch_graph_node_num,]
#         assert len(data['node'].code_mask.size()) == 1

#         graph_node_emb = self.graph_node_emb_op(data.x_dict['node'])  # [N, emb_dims]

#         # ── 可选位置编码 ──────────────────────────────────────────────────────
#         # use_pos_encoding=True : 添加可学习位置编码 + √emb_dims 缩放
#         # use_pos_encoding=False: 直接 dropout，与 HetCoS 基线对齐
#         if self.use_pos_encoding:
#             batch_size = data.x_batch_dict['node'].max().item() + 1
#             device = graph_node_emb.device
#             pos_indices_list = []
#             for b in range(batch_size):
#                 mask = data.x_batch_dict['node'] == b
#                 num_nodes = mask.sum().item()
#                 max_allowed = self.graph_pos_encoding.num_embeddings - 1
#                 if num_nodes > max_allowed:
#                     # 模运算，循环复用位置编码
#                     positions = (torch.arange(1, num_nodes + 1, device=device)
#                                  % max_allowed) + 1
#                 else:
#                     positions = torch.arange(1, num_nodes + 1, device=device)
#                 pos_indices_list.append(positions)
#             pos_indices = torch.cat(pos_indices_list)
#             pos_emb = self.graph_pos_encoding(pos_indices)  # [N, emb_dims]
#             graph_node_emb = graph_node_emb * np.sqrt(self.emb_dims) + pos_emb

#         data['node'].x = self.emb_drop_op(graph_node_emb)  # [N, emb_dims]

#         code_x_batch = data.x_batch_dict['node'][data['node'].code_mask == True]

#         # ── GNN 聚合（两种模式）──────────────────────────────────────────────
#         if self.hyper_aggr_mode == 'hetero_conv':
#             # 模式1：HeteroConv 直接 sum 聚合，所有超边平等对待
#             for gnn, relu, norm in zip(self.gnn_ops, self.grelu_ops, self.gnorm_ops):
#                 x_dict = gnn(x_dict=data.x_dict, edge_index_dict=data.edge_index_dict)
#                 data['node'].x = norm(data['node'].x.add(relu(x_dict['node'])))

#         elif self.hyper_aggr_mode == 'weighted_sum':
#             # 模式2：可学习权重加权，只对当前 batch 中存在的超边做 softmax
#             for i, (hyper_layer, relu, norm) in enumerate(
#                     zip(self.hyper_gnn_ops, self.grelu_ops, self.gnorm_ops)):
#                 # 找出当前 batch 中实际存在且非空的超边
#                 valid_idx = [
#                     j for j, et in enumerate(self.hyper_edge_types)
#                     if et in data.edge_index_dict
#                     and data.edge_index_dict[et].numel() > 0
#                 ]
#                 if valid_idx:
#                     # 仅对存在的超边做 softmax，避免权重稀释
#                     valid_w = torch.softmax(
#                         self.hyperedge_weights[i][valid_idx], dim=0)
#                     hyper_out = torch.zeros_like(data['node'].x)
#                     for k, j in enumerate(valid_idx):
#                         et = self.hyper_edge_types[j]
#                         out = hyper_layer[et[1]](
#                             data['node'].x, data.edge_index_dict[et])
#                         hyper_out = hyper_out + valid_w[k] * out
#                 else:
#                     hyper_out = torch.zeros_like(data['node'].x)
#                 data['node'].x = norm(data['node'].x + relu(hyper_out))
        
#         graph_enc,_=to_dense_batch(data.x_dict['node'],
#                                   batch=data.x_batch_dict['node'], #data['leaf'].x_batch也可以
#                                   fill_value=self.pad_idx,
#                                   max_num_nodes=self.graph_max_size)  #[batch_size,graph_max_size,emb_dims],[batch_size,graph_max_size]

#         code_src_map,_=to_dense_batch(data.src_map_dict['node'][data['node'].code_mask==True],
#                                         batch=code_x_batch,  # data['leaf'].x_batch也可以
#                                         fill_value=self.pad_idx,
#                                         max_num_nodes=self.code_max_len)    # [batch_data_num,code_max_len]
#         graph_code_enc,_=to_dense_batch(data.x_dict['node'][data['node'].code_mask==True],
#                                         batch=code_x_batch,  # data['leaf'].x_batch也可以
#                                         fill_value=self.pad_idx,
#                                         max_num_nodes=self.code_max_len)    # [batch_data_num,code_max_len]

#         return graph_enc,graph_code_enc,code_src_map

# class Dec(nn.Module):
#     def __init__(self,
#                  emb_dims,
#                  text_voc_size,
#                  text_emb_op,
#                  text_max_len,
#                  enc_out_dims,
#                  att_layers,
#                  att_heads,
#                  att_head_dims=None,
#                  ff_hid_dims=2048,
#                  drop_rate=0.,
#                  **kwargs
#                  ):
#         super().__init__()
#         kwargs.setdefault('pad_idx', 0)
#         kwargs.setdefault('copy', True)
#         self._copy = kwargs['copy']
#         self.emb_dims = emb_dims
#         self.text_voc_size = text_voc_size
#         # embedding dims为text_voc_size+2*code_max_len

#         # assert len(text_sim_token_ids.shape)==1
#         # text_sim_token_voc_size=np.unique(text_sim_token_ids).shape[0]
#         # self.text_token_to_sim_token_map_op=nn.Embedding.from_pretrained(torch.tensor(text_sim_token_ids).view([-1,1]).float(),freeze=True,padding_idx=kwargs['pad_idx'])
#         # self.text_token_to_sim_token_map_op=text_token_to_sim_token_map_op
#         self.text_emb_op = text_emb_op
#         # self.text_emb_op = nn.Embedding(text_voc_size + code_max_len, emb_dims, padding_idx=kwargs['pad_idx'])
#         # self.sim_token_emb_op = nn.Embedding(text_sim_token_voc_size, emb_dims, padding_idx=kwargs['pad_idx'])
#         # nn.init.xavier_uniform_(self.text_emb_op.weight[1:, ])
#         # nn.init.xavier_uniform_(self.sim_token_emb_op.weight[1:, ])
#         self.pos_encoding = PosEnc(max_len=text_max_len+1, emb_dims=emb_dims, train=True, pad=True,pad_idx=kwargs['pad_idx'])  #不要忘了+1,因为输入前加了begin_id
#         # nn.init.xavier_uniform_(self.pos_encoding.weight[1:, ])
#         self.emb_layer_norm = nn.LayerNorm(emb_dims)
#         self.text_dec_op = TranDec(query_dims=emb_dims,
#                                    key_dims=enc_out_dims,
#                                    head_nums=att_heads,
#                                    head_dims=att_head_dims,
#                                    layer_num=att_layers,
#                                    ff_hid_dims=ff_hid_dims,
#                                    drop_rate=drop_rate,
#                                    pad_idx=kwargs['pad_idx'],
#                                    self_causality=True)
#         # self.text_dec_op = DualTranDec(query_dims=emb_dims,
#         #                             key_dims=enc_out_dims,
#         #                             head_num=att_heads,
#         #                             ff_hid_dims=ff_hid_dims,
#         #                             head_dims=att_head_dims,
#         #                             layer_num=att_layers,
#         #                             drop_rate=drop_rate,
#         #                             pad_idx=kwargs['pad_idx'],
#         #                             mode='sequential',
#         #                             self_causality=True)
#         self.dropout = nn.Dropout(p=drop_rate)
#         self.out_fc = nn.Linear(emb_dims, text_voc_size)
#         self.copy_generator = MultiCopyGenerator(tgt_dims=emb_dims,
#                                                      tgt_voc_size=text_voc_size,
#                                                      src_dims=enc_out_dims,
#                                                      att_heads=att_heads,
#                                                      att_head_dims=att_head_dims,
#                                                      drop_rate=drop_rate,
#                                                      pad_idx=kwargs['pad_idx'])

#     def forward(self,graph_enc,graph_code_enc,code_src_map,text_input):
#         # sim_text_token=self.text_token_to_sim_token_map_op(text_input).squeeze(-1).long()  #[batch_text,L_text]
#         # sim_token_mask=sim_text_token.abs().sign()   #[batch_text,L_text]
#         # sim_token_emb=self.text_sim_token_emb_op(sim_text_token[sim_token_mask==True])   # (B*L_text,D_text_emb)  .view(sim_text_token.size())
        
#         text_emb = self.text_emb_op(text_input)   # (B,L_text,D_text_emb)
#         # text_emb[sim_token_mask==True,:]=text_emb[sim_token_mask==True,:].add(sim_token_emb)*0.5
#         text_emb=text_emb* np.sqrt(self.emb_dims)
#         pos_emb = self.pos_encoding(text_input)  # # (B,L_text,D_emb)
#         text_dec = self.dropout(text_emb.add(pos_emb))  # (B,L_text,D_emb)
#         text_dec = self.emb_layer_norm(text_dec)  # (B,L_text,D_emb)

#         graph_mask = graph_enc.abs().sum(-1).sign()  # (batch_size,graph_max_size)
#         # code_mask=code_enc.abs().sum(-1).sign() # (batch_size,code_max_len)
#         text_mask = text_input.abs().sign()  # (B,L_text)
#         text_dec = self.text_dec_op(query=text_dec,
#                                     key=graph_enc,
#                                     query_mask=text_mask,
#                                     key_mask=graph_mask
#                                     )  # (B,L_text,D_text_emb)

#         if not self._copy:
#             text_output = self.out_fc(text_dec)  # (B,L_text,text_voc_size)包含begin_idx和end_idx
#             # text_output = F.softmax(text_output, dim=-1)
#             # text_output[:,:,-1]=0.    #不生成begin_idx，默认该位在text_voc_size最后一个，置0
#         else:
#             # text_output=F.pad(text_output,(0,2*self.text_max_len)) #pad last dim
#             text_output = self.copy_generator(text_dec,
#                                              graph_code_enc,code_src_map)
#         # text_output[:, :, self.text_voc_size - 1] = 0.  # 不生成begin_idx，默认该位在text_voc_size最后一个，置0
#         # text_output[:, :, 0] = 0.  # pad位不生成
#         return text_output.transpose(1, 2)

# class TNet(BaseNet):
#     def __init__(self,
#                  emb_dims,
#                  graph_max_size,
#                  code_max_len,
#                  text_max_len,
#                  io_voc_size,
#                  text_voc_size,
#                  graph_gnn_layers=6,
#                  graph_GNN=SAGEConv,
#                  graph_gnn_aggr='add',
#                  text_att_layers=3,
#                  text_att_heads=8,
#                  text_att_head_dims=None,
#                  text_ff_hid_dims=2048,
#                  drop_rate=0.,
#                  use_pos_encoding=True,        # 可选：是否使用图节点位置编码
#                  hyper_aggr_mode='hetero_conv', # 可选：'hetero_conv' 或 'weighted_sum'
#                  pretrained_emb=None,           # 可选：CodeBERT预训练嵌入矩阵Tensor，None则随机初始化
#                  **kwargs,
#                  ):
#         super().__init__()
#         kwargs.setdefault('copy', True)
#         kwargs.setdefault('pad_idx', 0)
#         self.init_params = locals()

#         # ── 嵌入层初始化（可选CodeBERT或随机初始化）──────────────────────────
#         # pretrained_emb=None     : 随机Xavier初始化（原始方式，训练稳定）
#         # pretrained_emb=Tensor   : 从CodeBERT预训练矩阵初始化，freeze=False允许微调
#         #   词表结构：
#         #     [0]              PAD          → 全零
#         #     [1..text_voc-3]  正常词       → CodeBERT子词均值
#         #     [text_voc-2]     END token    → 随机
#         #     [text_voc-1]     BEGIN token  → 随机
#         #     [text_voc..]     copy tokens  → 随机（样本级动态指针）
#         #     [..]             unk-alias-N  → 随机（样本级占位）
#         self._use_emb_proj = False
#         if pretrained_emb is not None:
#             codebert_dim = pretrained_emb.shape[1]   # 通常768
#             if codebert_dim == emb_dims:
#                 # 维度匹配，直接使用
#                 io_token_emb_op = nn.Embedding.from_pretrained(
#                     pretrained_emb, freeze=False, padding_idx=kwargs['pad_idx'])
#                 logging.info(f'[Emb] CodeBERT直接使用，维度={codebert_dim}')
#             else:
#                 # 维度不匹配（如CodeBERT=768, emb_dims=512），加投影层
#                 io_token_emb_op = nn.Embedding.from_pretrained(
#                     pretrained_emb, freeze=False, padding_idx=kwargs['pad_idx'])
#                 self.emb_proj = nn.Linear(codebert_dim, emb_dims, bias=False)
#                 nn.init.xavier_uniform_(self.emb_proj.weight)
#                 self._use_emb_proj = True
#                 # 用ProjectedEmbedding包装，对enc/dec透明
#                 io_token_emb_op = _ProjectedEmbedding(io_token_emb_op, self.emb_proj)
#                 del self.emb_proj
#                 logging.info(f'[Emb] CodeBERT带投影层，{codebert_dim} → {emb_dims}')
#         else:
#             # 随机Xavier初始化（原始方式）
#             io_token_emb_op = nn.Embedding(
#                 io_voc_size, emb_dims, padding_idx=kwargs['pad_idx'])
#             nn.init.xavier_uniform_(io_token_emb_op.weight[1:, ])
#             logging.info(f'[Emb] 随机Xavier初始化，维度={emb_dims}')

#         self.enc_op = CodeGraphEnc(
#             emb_dims=emb_dims,
#             graph_max_size=graph_max_size,
#             code_max_len=code_max_len,
#             graph_node_emb_op=io_token_emb_op,
#             graph_gnn_layers=graph_gnn_layers,
#             graph_GNN=graph_GNN,
#             graph_gnn_aggr=graph_gnn_aggr,
#             drop_rate=drop_rate,
#             use_pos_encoding=use_pos_encoding,
#             hyper_aggr_mode=hyper_aggr_mode,
#             pad_idx=kwargs['pad_idx'])
#         self.dec_op = Dec(
#             emb_dims=emb_dims,
#             text_voc_size=text_voc_size,
#             text_max_len=text_max_len,
#             text_emb_op=io_token_emb_op,
#             enc_out_dims=emb_dims,
#             att_layers=text_att_layers,
#             att_heads=text_att_heads,
#             att_head_dims=text_att_head_dims,
#             ff_hid_dims=text_ff_hid_dims,
#             drop_rate=drop_rate,
#             copy=kwargs['copy'],
#             pad_idx=kwargs['pad_idx'])

#     def forward(self, code_graph):
#         text_input=code_graph['text'].text_token_input.clone()
#         del code_graph['text']
#         graph_enc,graph_code_enc,code_src_map = self.enc_op(data=code_graph)
#         text_output = self.dec_op(graph_enc=graph_enc,graph_code_enc=graph_code_enc,
#                                     code_src_map=code_src_map,
#                                     text_input=text_input)
#         return text_output

# class TModel(TransSeq2Seq):
#     def __init__(self,
#                 #  sim_token_ids,
#                  model_dir,
#                  model_name='Transformer_based_model',
#                  model_id=None,
#                  emb_dims=512,
#                 #  code_att_layers=3,
#                 #  code_att_heads=8,
#                 #  code_att_head_dims=None,
#                 #  code_ff_hid_dims=2048,
#                  graph_gnn_layers=3,
#                  graph_GNN=SAGEConv,
#                  graph_gnn_aggr='add',
#                  text_att_layers=3,
#                  text_att_heads=8,
#                  text_att_head_dims=None,
#                  text_ff_hid_dims=2048,
#                  drop_rate=0.,
#                  copy=True,
#                  pad_idx=0,
#                  train_batch_size=32,
#                  pred_batch_size=32,
#                  max_train_size=-1,
#                  max_valid_size=32 * 10,
#                  max_big_epochs=20,
#                  regular_rate=1e-5,
#                  lr_base=0.001,
#                  lr_decay=0.9,
#                  min_lr_rate=0.01,
#                  warm_big_epochs=2,
#                  start_valid_epoch=20,
#                  early_stop=20,
#                  Net=TNet,
#                  Dataset=Datasetx,
#                  beam_width=1,
#                  train_metrics=[get_sent_bleu],
#                  valid_metric=get_sent_bleu,
#                  test_metrics=[get_sent_bleu],
#                  train_mode=True,
#                  use_pos_encoding=True,         # 可选：是否使用图节点位置编码
#                  hyper_aggr_mode='hetero_conv',  # 可选：'hetero_conv' 或 'weighted_sum'
#                  use_pretrained_emb=False,        # 可选：是否加载CodeBERT嵌入矩阵
#                  **kwargs
#                  ):
#         logging.info('Construct %s' % model_name)
#         super().__init__(model_name=model_name,
#                          model_dir=model_dir,
#                          model_id=model_id)
#         self.init_params = locals()
#         # self.sim_token_ids=sim_token_ids
#         self.emb_dims = emb_dims
#         # self.code_att_layers = code_att_layers
#         # self.code_att_heads = code_att_heads
#         # self.code_att_head_dims = code_att_head_dims
#         # self.code_ff_hid_dims = code_ff_hid_dims
#         self.graph_gnn_layers = graph_gnn_layers
#         self.graph_GNN = graph_GNN
#         self.graph_gnn_aggr = graph_gnn_aggr
#         self.text_att_layers = text_att_layers
#         self.text_att_heads = text_att_heads
#         self.text_att_head_dims = text_att_head_dims
#         self.text_ff_hid_dims = text_ff_hid_dims
#         self.drop_rate = drop_rate
#         self.pad_idx = pad_idx
#         self.copy = copy
#         self.train_batch_size = train_batch_size
#         self.pred_batch_size = pred_batch_size
#         self.max_train_size = max_train_size
#         self.max_valid_size = max_valid_size
#         self.max_big_epochs = max_big_epochs
#         self.regular_rate = regular_rate
#         self.lr_base = lr_base
#         self.lr_decay = lr_decay
#         self.min_lr_rate = min_lr_rate
#         self.warm_big_epochs = warm_big_epochs
#         self.start_valid_epoch=start_valid_epoch
#         self.early_stop=early_stop
#         self.Net = Net
#         self.Dataset = Dataset
#         self.beam_width = beam_width
#         self.train_metrics = train_metrics
#         self.valid_metric = valid_metric
#         self.test_metrics = test_metrics
#         self.train_mode = train_mode
#         self.use_pos_encoding    = use_pos_encoding
#         self.hyper_aggr_mode     = hyper_aggr_mode
#         self.use_pretrained_emb  = use_pretrained_emb

#     def _logging_paramerter_num(self):
#         logging.info("{} have {} paramerters in total".format(self.model_name, sum(
#             x.numel() for x in self.net.parameters() if x.requires_grad)))
#         enc_op = self.net.module.enc_op
#         # 兼容两种GNN聚合模式
#         if hasattr(enc_op, 'gnn_ops'):
#             gnn_param = sum(x.numel() for x in enc_op.gnn_ops.parameters() if x.requires_grad)
#         elif hasattr(enc_op, 'hyper_gnn_ops'):
#             gnn_param = sum(x.numel() for x in enc_op.hyper_gnn_ops.parameters() if x.requires_grad)
#         else:
#             gnn_param = 0
#         code_graph_enc_param_num = (
#             gnn_param +
#             sum(x.numel() for x in enc_op.gnorm_ops.parameters() if x.requires_grad) +
#             sum(x.numel() for x in enc_op.grelu_ops.parameters() if x.requires_grad)
#         )
#         text_dec_param_num = sum(
#             x.numel() for x in self.net.module.dec_op.text_dec_op.parameters()
#             if x.requires_grad)
#         enc_dec_param_num = code_graph_enc_param_num + text_dec_param_num
#         logging.info("{} have {} paramerters in encoder and decoder".format(
#             self.model_name, enc_dec_param_num))

#     def fit(self,
#             train_data,
#             valid_data,
#             **kwargs
#             ):
#         self.graph_max_size=0
#         self.code_max_len = 0
#         self.io_voc_size = 0
#         # self.code_mpos_voc_size = 0
#         # self.code_npos_voc_size = 0
#         self.text_max_len=0
#         for code_graph,text in zip(train_data['code_graphs'],train_data['texts']):
#             self.graph_max_size = max(self.graph_max_size,len(code_graph['nodes']))
#             self.code_max_len = max(self.code_max_len,code_graph['code_node_mask'].sum())
#             self.io_voc_size = max(self.io_voc_size,max(code_graph['nodes']))
#             # self.code_mpos_voc_size = max(self.code_mpos_voc_size,np.max(code_graph['node_in_code_poses'][0,:]))
#             # self.code_npos_voc_size = max(self.code_npos_voc_size,np.max(code_graph['node_in_code_poses'][1,:]))
#             self.text_max_len=max(self.text_max_len,len(text))
#         self.io_voc_size+=1
#         # self.code_mpos_voc_size+=1
#         # self.code_npos_voc_size+=1

#         self.text_voc_size = len(train_data['text_dic']['text_i2w'])  # 包含了begin_idx和end_idx
#         self.io_voc_size=max(self.io_voc_size,self.text_voc_size+2*self.code_max_len)

#         # ── 可选：加载 CodeBERT 预训练嵌入矩阵 ──────────────────────────────
#         # use_pretrained_emb=True  : 从 codebert_emb_matrix.npy 加载，需先运行
#         #                            build_codebert_emb.py 生成该文件
#         # use_pretrained_emb=False : 跳过，使用随机 Xavier 初始化（原始稳定方式）
#         pretrained_emb_tensor = None
#         if self.use_pretrained_emb:
#             from config import top_data_dir as _top_data_dir
#             import os as _os
#             codebert_path = _os.path.join(
#                 _top_data_dir, 'basic_info', 'codebert_emb_matrix.npy')
#             if _os.path.exists(codebert_path):
#                 logging.info(f'加载 CodeBERT 嵌入矩阵: {codebert_path}')
#                 emb_np = np.load(codebert_path)          # [saved_voc, 768]
#                 saved_size, codebert_dim = emb_np.shape
#                 if self.io_voc_size > saved_size:
#                     # copy/unk-alias token 扩展部分用随机初始化
#                     extra = np.random.normal(
#                         0, 0.02,
#                         (self.io_voc_size - saved_size, codebert_dim)
#                     ).astype(np.float32)
#                     emb_np = np.concatenate([emb_np, extra], axis=0)
#                 pretrained_emb_tensor = torch.tensor(emb_np, dtype=torch.float32)
#                 logging.info(
#                     f'CodeBERT 嵌入矩阵加载完成，shape={pretrained_emb_tensor.shape}')
#             else:
#                 logging.warning(
#                     f'未找到 CodeBERT 嵌入矩阵: {codebert_path}\n'
#                     '请先运行 build_codebert_emb.py。本次回退到随机初始化。'
#                 )
#         # print(self.graph_max_size, self.code_max_len,self.text_max_len,
#         #       self.io_voc_size, self.text_voc_size,
#         #       self.code_mpos_voc_size,self.code_npos_voc_size)

#         net = self.Net(
#                         emb_dims=self.emb_dims,
#                        graph_max_size=self.graph_max_size,
#                        code_max_len=self.code_max_len,
#                        text_max_len=self.text_max_len,
#                        io_voc_size=self.io_voc_size,
#                        text_voc_size=self.text_voc_size,
#                        graph_gnn_layers=self.graph_gnn_layers,
#                        graph_GNN=self.graph_GNN,
#                        graph_gnn_aggr=self.graph_gnn_aggr,
#                        text_att_layers=self.text_att_layers,
#                        text_att_heads=self.text_att_heads,
#                        text_att_head_dims=self.text_att_head_dims,
#                        text_ff_hid_dims=self.text_ff_hid_dims,
#                        drop_rate=self.drop_rate,
#                        pad_idx=self.pad_idx,
#                        copy=self.copy,
#                        use_pos_encoding=self.use_pos_encoding,
#                        hyper_aggr_mode=self.hyper_aggr_mode,
#                        pretrained_emb=pretrained_emb_tensor,
#                        )

#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择GPU优先

#         self.net =DataParallel(net.to(device),follow_batch=['x'])  # 并行使用多GPU
#         # self.net = BalancedDataParallel(0, net.to(device), dim=0)  # 并行使用多GPU
#         # self.net = net.to(device)  # 数据转移到设备
#         self._logging_paramerter_num()  # 需要有并行的self.net和self.model_name
#         self.net.train()  # 设置网络为训练模式

#         self.optimizer = optim.Adam(self.net.parameters(),
#                                     lr=self.lr_base,
#                                     weight_decay=self.regular_rate)

#         # graph_enc_params=self.net.module.enc.graph_enc.parameters()
#         # graph_enc_param_ids=list(map(id,graph_enc_params))
#         # ex_params=filter(lambda p: id(p) not in graph_enc_param_ids,self.net.parameters())
#         # optim_cfg = [{'params': graph_enc_params, 'lr': 0.001,'weight_decay': self.regular_rate* 10.},
#         #              {'params': ex_params, 'lr': self.lr_base, 'weight_decay': self.regular_rate}]
#         # self.optimizer=optim.Adam(optim_cfg)

#         self.criterion = LabelSmoothSoftmaxCEV2(reduction='mean', ignore_index=self.pad_idx, label_smooth=0.0)
#         # self.criterion = nn.NLLLoss(ignore_index=self.pad_idx)

#         self.text_begin_idx = self.text_voc_size - 1
#         self.text_end_idx = self.text_voc_size - 2
#         self.tgt_begin_idx,self.tgt_end_idx=self.text_begin_idx,self.text_end_idx
#         assert train_data['text_dic']['text_i2w'][self.text_end_idx] == OUT_END_TOKEN
#         assert train_data['text_dic']['text_i2w'][self.text_begin_idx] == OUT_BEGIN_TOKEN  # 最后两个是end_idx 和begin_idx

#         self.max_train_size = len(train_data['code_graphs']) if self.max_train_size == -1 else self.max_train_size
#         train_code_graphs, train_texts,train_ids = zip(*random.sample(list(zip(train_data['code_graphs'], train_data['texts'],train_data['ids'])),
#                                                      min(self.max_train_size,
#                                                          len(train_data['code_graphs']))
#                                                      )
#                                       )

#         train_set = self.Dataset(code_graphs=train_code_graphs,
#                                  texts=train_texts,
#                                  ids=train_ids,
#                                  text_max_len=self.text_max_len,
#                                  text_begin_idx=self.text_begin_idx,
#                                  text_end_idx=self.text_end_idx,
#                                  pad_idx=self.pad_idx)
#         # train_loader = DataLoader(dataset=train_set,
#         #                           train_batch_size=self.train_batch_size,
#         #                           shuffle=True,
#         #                           follow_batch=['graph_node', 'graph_node_after'])
#         train_loader=DataListLoader(dataset=train_set,
#                                     batch_size=self.train_batch_size,
#                                     shuffle=True,
#                                     drop_last=True) 

#         if self.warm_big_epochs is None:
#             self.warm_big_epochs = max(self.max_big_epochs // 10, 2)
#         self.scheduler = LrWarmUp(self.optimizer,
#                                   min_rate=self.min_lr_rate,
#                                   lr_decay=self.lr_decay,
#                                   warm_steps=self.warm_big_epochs * len(train_loader),
#                                   # max(self.max_big_epochs//10,2)*train_loader.__len__()
#                                   reduce_steps=len(train_loader))  # 预热次数 train_loader.__len__()
#         if self.train_mode:  # 如果进行训练
#             # best_net_path = os.path.join(self.model_dir, '{}_best_net.net'.format(self.model_name))
#             # self.net.load_state_dict(torch.load(best_net_path))
#             # self.net.train()
#             # torch.cuda.empty_cache()
#             for i in range(0,self.max_big_epochs):
#                 # logging.info('---------Train big epoch %d/%d' % (i + 1, self.max_big_epochs))
#                 pbar = tqdm(train_loader)
#                 for j, batch_data in enumerate(pbar):
#                     batch_text_output = []
#                     ids=[]
#                     for data in batch_data:
#                         batch_text_output.append(data['text'].text_token_output.unsqueeze(0))
#                         del data['text'].text_token_output
#                         ids.append(data['idx'].idx.item())
#                         del data['idx']

#                     batch_text_output = torch.cat(batch_text_output, dim=0).to(device)
#                     # print(batch_text_output[:2,:])
#                     pred_text_output = self.net(batch_data)

#                     loss = self.criterion(pred_text_output, batch_text_output)  # 计算loss
#                     self.optimizer.zero_grad()  # 梯度置0
#                     loss.backward()  # 反向传播
#                     # clip_grad_norm_(self.net.parameters(),1e-2)  #减弱梯度爆炸
#                     self.optimizer.step()  # 优化
#                     self.scheduler.step()  # 衰减

#                     # log_info = '[Big epoch:{}/{}]'.format(i + 1, self.max_big_epochs)
#                     # if i+1>=self.start_valid_epoch:
#                     text_dic = {'text_i2w': train_data['text_dic']['text_i2w'],
#                                'ex_text_i2ws': [train_data['text_dic']['ex_text_i2ws'][k] for k in ids]}
#                     log_info=self._get_log_fit_eval(loss=loss,
#                                                     pred_tgt=pred_text_output,
#                                                     gold_tgt=batch_text_output,
#                                                     tgt_i2w=text_dic
#                                                     )
#                     log_info = '[Big epoch:{}/{},{}]'.format(i + 1, self.max_big_epochs, log_info)
#                     pbar.set_description(log_info)
#                     del pred_text_output,batch_text_output,batch_data

#                 del pbar
#                 if i+1 >= self.start_valid_epoch:
#                     self.max_valid_size = len(valid_data['code_graphs']) if self.max_valid_size == -1 else self.max_valid_size
#                     valid_srcs, valid_tgts, ex_text_i2ws = zip(*random.sample(list(zip(valid_data['code_graphs'],
#                                                                                        valid_data['texts'],
#                                                                                        valid_data['text_dic']['ex_text_i2ws'])),
#                                                                               min(self.max_valid_size,
#                                                                                   len(valid_data['code_graphs']))
#                                                                               )
#                                                                )
#                     text_dic = {'text_i2w': train_data['text_dic']['text_i2w'],
#                                 'ex_text_i2ws': ex_text_i2ws}
#                     # torch.cuda.empty_cache()
#                     worse_epochs = self._do_validation(valid_srcs=valid_srcs,  # valid_data['code_graphs']
#                                                        valid_tgts=valid_tgts,  # valid_data['texts']
#                                                        tgt_i2w=text_dic,  # valid_data['text_dic']
#                                                        increase_better=True,
#                                                        last=False)  # 根据验证集loss选择best_net
#                     # worse_epochs = self._do_validation(valid_srcs=valid_data['code_graphs'],  #
#                     #                                    valid_tgts=valid_data['texts'],  #
#                     #                                    tgt_i2w=valid_data['text_dic'],  #
#                     #                                    increase_better=True,
#                     #                                    last=False)  # 根据验证集loss选择best_net
#                     if worse_epochs>=self.early_stop:
#                         break
#         # torch.cuda.empty_cache()
#         self._do_validation(valid_srcs=valid_data['code_graphs'],
#                             valid_tgts=valid_data['texts'],
#                             tgt_i2w=valid_data['text_dic'],
#                             increase_better=True,
#                             last=True)  # 根据验证集loss选择best_net
#         self._logging_paramerter_num()  # 需要有并行的self.net和self.model_name

#     def predict(self,
#                 code_graphs,
#                 text_dic):
#         logging.info('Predict outputs of %s' % self.model_name)
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择GPU优先
#         # self.net = self.net.to(device)  # 数据转移到设备,不重新赋值不行
#         self.net.eval()  # 切换测试模式
#         enc_op=DataParallel(self.net.module.enc_op,follow_batch=['x'])
#         dec_op=torch.nn.DataParallel(self.net.module.dec_op)
#         # enc.eval()
#         # dec.eval()
#         data_set = self.Dataset(code_graphs=code_graphs,
#                                 texts=None,
#                                 ids=None,
#                                 text_max_len=self.text_max_len,
#                                 text_begin_idx=self.text_begin_idx,
#                                 text_end_idx=self.text_end_idx,
#                                 pad_idx=self.pad_idx)  # 数据集，没有out，不需要id

#         data_loader = DataListLoader(dataset=data_set,
#                                      batch_size=self.pred_batch_size,   #1.5,2.5
#                                      shuffle=False)
#                                  # follow_batch=['graph_node', 'graph_node_after'])  # data loader
#         pred_text_id_np_batches = []  # 所有batch的预测出的id np
#         with torch.no_grad():  # 取消梯度
#             pbar = tqdm(data_loader)
#             for batch_data in pbar:
#                 # 从batch_data图里把解码器输入输出端数据调出来
#                 batch_text_input = []
#                 for data in batch_data:
#                     batch_text_input.append(data['text'].text_token_input.unsqueeze(0))
#                     del data['text']
#                 batch_text_input = torch.cat(batch_text_input, dim=0).to(device)

#                 # 先跑encoder，生成编码
#                 batch_graph_enc,batch_graph_code_enc,batch_code_src_map=enc_op(batch_data)
#                 batch_text_output: list = []  # 每步的output tensor
#                 if self.beam_width == 1:
#                     for i in range(self.text_max_len + 1):  # 每步开启
#                         pred_out = dec_op(graph_enc=batch_graph_enc,graph_code_enc=batch_graph_code_enc,code_src_map=batch_code_src_map,text_input=batch_text_input)  # 预测该步输出 (B,text_voc_size,L_text)
#                         batch_text_output.append(pred_out[:, :, i].unsqueeze(-1).to('cpu').data.numpy())  # 将该步输出加入msg output
#                         if i < self.text_max_len:  # 如果没到最后，将id加入input
#                             batch_text_input[:, i + 1] = torch.argmax(pred_out[:, :, i], dim=1)
#                     batch_pred_text = np.concatenate(batch_text_output, axis=-1)[:, :, :-1]  # (B,D_tgt,L_tgt)
#                     batch_pred_text[:, self.tgt_begin_idx, :] = -np.inf  # (B,D_tgt,L_tgt)
#                     batch_pred_text[:, self.pad_idx, :] = -np.inf  # (B,D_tgt,L_tgt)
#                     batch_pred_text_np = np.argmax(batch_pred_text, axis=1)  # (B,L_tgt) 要除去pad id和begin id
#                     pred_text_id_np_batches.append(batch_pred_text_np)  # [(B,L_tgt)]
#                 else:
#                     batch_pred_text=trans_beam_search(net=dec_op,
#                                                       beam_width=self.beam_width,
#                                                       dec_input_arg_name='text_input',
#                                                       length_penalty=1,
#                                                       begin_idx=self.tgt_begin_idx,
#                                                       pad_idx=self.pad_idx,
#                                                       end_idx=self.tgt_end_idx,
#                                                       graph_enc=batch_graph_enc,
#                                                       graph_code_enc=batch_graph_code_enc,
#                                                       code_src_map=batch_code_src_map,
#                                                       text_input=batch_text_input
#                                                       )     # (B,L_tgt)

#                     pred_text_id_np_batches.append(batch_pred_text.to('cpu').data.numpy()[:,:-1])  # [(B,L_tgt)]

#         pred_text_id_np = np.concatenate(pred_text_id_np_batches,axis=0)  # (AB,tgt_voc_size,L_tgy)
#         self.net.train()  # 切换回训练模式
#         # pred_texts=[[{**text_dic['text_i2w'],**text_dic['ex_text_i2ws'][j]}[i] for ]]
#         # 利用字典将msg转为token
#         pred_texts = self._tgt_ids2tokens(pred_text_id_np, text_dic, self.text_end_idx)

#         return pred_texts  # 序列概率输出形状为（A,D)
    
#     def generate_texts(self,code_graphs,text_dic,res_path,gold_texts,raw_data,token_data,**kwargs):
#         '''
#         生成src对应的tgt并保存
#         :param code_graphs:
#         :param text_dic:
#         :param res_path:
#         :param kwargs:
#         :return:
#         '''
#         logging.info('>>>>>>>Generate the targets according to sources and save the result to {}'.format(res_path))
#         kwargs.setdefault('beam_width',1)
#         res_dir=os.path.dirname(res_path)
#         if not os.path.exists(res_dir):
#             os.makedirs(res_dir)
#         pred_texts=self.predict(code_graphs=code_graphs,
#                                 text_dic=text_dic
#                                 )
#         # codes=map(lambda x:x['code']['tokens'],code_graphs)
#         # codes=self._code_ids2tokens(codes,code_i2w,self.pad_idx)
#         gold_texts=self._tgt_ids2tokens(gold_texts,text_dic,self.pad_idx)
#         res_data = []
#         for i,(pred_text,gold_text,raw_item,token_item) in \
#                 enumerate(zip(pred_texts,gold_texts,raw_data,token_data)):
#             sent_bleu=self.valid_metric([pred_text],[gold_text])
#             res_data.append(dict(pred_text=' '.join(pred_text),
#                                  gold_text=' '.join(gold_text),
#                                  sent_bleu=sent_bleu,
#                                  raw_code=raw_item['code'],
#                                  raw_text=raw_item['text'],
#                                  id=raw_item['id'],
#                                  token_text=token_item['text'],
#                                  ))
#         # res_df=pd.DataFrame(res_dic).T
#         # # print(res_df)
#         # excel_writer = pd.ExcelWriter(res_path)  # 根据路径savePath打开一个excel写文件
#         # res_df.to_excel(excel_writer,header=True,index=True)
#         # excel_writer.save()
#         with codecs.open(res_path,'w',encoding='utf-8') as f:
#             json.dump(res_data,f,indent=4, ensure_ascii=False)
#         self._logging_paramerter_num()  # 需要有并行的self.net和self.model_name
#         logging.info('>>>>>>>The result has been saved to {}'.format(res_path))

#     def _code_ids2tokens(self,code_idss, code_i2w, end_idx):
#         return [[code_i2w[idx] for idx in (code_ids[:code_ids.tolist().index(end_idx)]
#                                                     if end_idx in code_ids else code_ids)]
#                           for code_ids in code_idss]
    
#     def _tgt_ids2tokens(self, text_id_np, text_dic, end_idx=0, **kwargs):
#         if self.copy:
#             text_tokens: list = []
#             for j, text_ids in enumerate(text_id_np):
#                 text_i2w = {**text_dic['text_i2w'], **text_dic['ex_text_i2ws'][j]}
#                 end_i = text_ids.tolist().index(end_idx) if end_idx in text_ids else len(text_ids)
#                 text_tokens.append([text_i2w[text_idx] for text_idx in text_ids[:end_i]])
#                 # if end_i == 0:
#                 #     print()
#         else:
#             text_i2w=text_dic['text_i2w']
#             text_tokens = [[text_i2w[idx] for idx in (text_ids[:text_ids.tolist().index(end_idx)]
#                                                       if end_idx in text_ids else text_ids)]
#                           for text_ids in text_id_np]

#         return text_tokens
#     # def _tgt_ids2tokens(self, text_id_np, text_dic, end_idx=0, **kwargs):
#     #     """将目标序列的 ID 转换为 token"""
        
#     #     def to_list(arr):
#     #         """统一转换为 Python list"""
#     #         if isinstance(arr, torch.Tensor):
#     #             return arr.cpu().numpy().tolist()
#     #         elif isinstance(arr, np.ndarray):
#     #             return arr.tolist()
#     #         else:
#     #             return list(arr)
        
#     #     def find_end(ids_list, end_idx):
#     #         """找到结束位置"""
#     #         try:
#     #             return ids_list.index(end_idx)
#     #         except ValueError:
#     #             return len(ids_list)
        
#     #     if self.copy:
#     #         text_tokens = []
#     #         for j, text_ids in enumerate(text_id_np):
#     #             # 合并基础词汇表和扩展词汇表
#     #             text_i2w = {**text_dic['text_i2w'], **text_dic['ex_text_i2ws'][j]}
                
#     #             text_ids_list = to_list(text_ids)
#     #             end_i = find_end(text_ids_list, end_idx)
                
#     #             # 转换 token，使用 get 避免 KeyError
#     #             tokens = [text_i2w.get(int(idx), '<UNK>') for idx in text_ids_list[:end_i]]
#     #             text_tokens.append(tokens)
#     #     else:
#     #         text_i2w = text_dic['text_i2w']
#     #         text_tokens = []
#     #         for text_ids in text_id_np:
#     #             text_ids_list = to_list(text_ids)
#     #             end_i = find_end(text_ids_list, end_idx)
                
#     #             tokens = [text_i2w.get(int(idx), '<UNK>') for idx in text_ids_list[:end_i]]
#     #             text_tokens.append(tokens)

#     #     return text_tokens

# if __name__ == '__main__':

#     logging.info('Parameters are listed below: \n'+'\n'.join(['{}: {}'.format(key,value) for key,value in params.items()]))

#     model = TModel(
#                     model_dir=params['model_dir'],
#                    model_name=params['model_name'],
#                    model_id=params['model_id'],
#                    emb_dims=params['emb_dims'],
#                    graph_gnn_layers=params['graph_gnn_layers'],
#                    graph_GNN=params['graph_GNN'],
#                    graph_gnn_aggr=params['graph_gnn_aggr'],
#                    text_att_layers=params['text_att_layers'],
#                    text_att_heads=params['text_att_heads'],
#                    text_att_head_dims=params['text_att_head_dims'],
#                    text_ff_hid_dims=params['text_ff_hid_dims'],
#                    drop_rate=params['drop_rate'],
#                    copy=params['copy'],
#                    pad_idx=params['pad_idx'],
#                    train_batch_size=params['train_batch_size'],
#                    pred_batch_size=params['pred_batch_size'],
#                    max_train_size=params['max_train_size'],
#                    max_valid_size=params['max_valid_size'],
#                    max_big_epochs=params['max_big_epochs'],
#                    regular_rate=params['regular_rate'],
#                    lr_base=params['lr_base'],
#                    lr_decay=params['lr_decay'],
#                    min_lr_rate=params['min_lr_rate'],
#                    warm_big_epochs=params['warm_big_epochs'],
#                    early_stop=params['early_stop'],
#                    start_valid_epoch=params['start_valid_epoch'],
#                    Net=TNet,
#                    Dataset=Datasetx,
#                    beam_width=params['beam_width'],
#                    train_metrics=train_metrics,
#                    valid_metric=valid_metric,
#                    test_metrics=test_metrics,
#                    train_mode=params['train_mode'],
#                    # ── 四个可选控制开关 ──────────────────────────────────────
#                    use_pos_encoding=params['use_pos_encoding'],
#                    hyper_aggr_mode=params['hyper_aggr_mode'],
#                    use_pretrained_emb=params['use_pretrained_emb'],
#                    )

#     logging.info('Load data ...')
#     # print(train_avail_data_path)
#     with codecs.open(train_avail_data_path, 'rb') as f:
#         train_data = pickle.load(f)
#     with codecs.open(valid_avail_data_path, 'rb') as f:
#         valid_data = pickle.load(f)
#     with codecs.open(test_avail_data_path, 'rb') as f:
#         test_data = pickle.load(f)
#     # io_token_sim_ids=np.load(io_token_sim_id_path)

#     # with codecs.open(code_node_i2w_path, 'rb') as f:
#     #     code_i2w = pickle.load(f)

#     with codecs.open(test_token_data_path,'r') as f:
#         test_token_data=json.load(f)

#     with codecs.open(test_raw_data_path,'r') as f:
#         test_raw_data=json.load(f)

#     # train_data['code_graphs']=train_data['code_graphs'][:1000]
#     # train_data['texts']=train_data['texts'][:1000]
#     # train_data['ids']=train_data['ids'][:1000]

#     # print(len(train_data['texts']), len(valid_data['texts']), len(test_data['texts']))
#     model.fit(train_data=train_data,
#               valid_data=valid_data)

#     for key, value in params.items():
#         logging.info('{}: {}'.format(key, value))
#     logging.info('Parameters are listed below: \n'+'\n'.join(['{}: {}'.format(key,value) for key,value in params.items()]))

#     # test_data['code_graphs']=test_data['code_graphs'][14246:]
#     # test_data['texts']=test_data['texts'][14246:]
#     # test_data['ids']=test_data['ids'][14246:]

#     # valid_data['code_graphs']=valid_data['code_graphs'][12762:]
#     # valid_data['texts']=valid_data['texts'][12762:]
#     # valid_data['ids']=valid_data['ids'][12762:]

#     test_eval_df=model.eval(test_srcs=test_data['code_graphs'],
#                             test_tgts=test_data['texts'],
#                             tgt_i2w=test_data['text_dic'])
#     logging.info('Model performance on test dataset:\n')
#     for i in range(0,len(test_eval_df.columns),4):
#         print(test_eval_df.iloc[:, i:i+4])

#     model.generate_texts(code_graphs=test_data['code_graphs'],
#                          text_dic=test_data['text_dic'],
#                          res_path=res_path,
#                          # code_i2w=code_i2w, d
#                          gold_texts=test_data['texts'],
#                          raw_data=test_raw_data,
#                          token_data=test_token_data)


# #改进decoder
# # coding=utf-8
# import os
# import re
# import sys
# sys.path.append('../../../')
# from lib.neural_module.learn_strategy import LrWarmUp
# from lib.neural_module.transformer import TranEnc, TranDec, DualTranDec,ResFF,ResMHA
# from lib.neural_module.embedding import PosEnc,SinusoidalPositionalEncoding
# from lib.neural_module.loss import LabelSmoothSoftmaxCEV2, CriterionNet
# from lib.neural_module.balanced_data_parallel import BalancedDataParallel
# from lib.neural_module.copy_attention import (DualMultiCopyGenerator, MultiCopyGenerator,
#                                                DualCopyGenerator, ImprovedMultiCopyGenerator)
# from lib.neural_module.beam_search import trans_beam_search
# from lib.neural_model.seq_to_seq_model import TransSeq2Seq
# from lib.neural_model.base_model import BaseNet
# from lib.neural_module.transformer import ResFF

# from typing import Any,Optional,Union
# from config import *

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler
# from torch.nn.utils import clip_grad_norm_
# from torch.utils.data import Dataset
# from torch_geometric.data import HeteroData
# from torch_geometric.loader.data_list_loader import DataListLoader
# from torch_geometric.utils import to_dense_batch
# from torch_geometric.data.storage import (BaseStorage, NodeStorage,EdgeStorage)
# from torch_geometric.nn.data_parallel import DataParallel
# from torch_geometric.nn import HeteroConv,GraphNorm,HypergraphConv,GATConv
# from lib.neural_module.DHGNN import HyperedgeDiffusionConv,KStepHypergraphConv
# import random
# import numpy as np
# import os
# import logging
# import pickle
# import json
# import codecs
# from tqdm import tqdm
# import numpy as np
# import pandas as pd
# import math
# from copy import deepcopy

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# NodeOrEdgeStorage = Union[NodeStorage, EdgeStorage]


# class _ProjectedEmbedding(nn.Module):
#     """
#     将 nn.Embedding(CodeBERT维度768) + nn.Linear(768->emb_dims) 封装为透明模块。
#     仅在 use_pretrained_emb=True 且 CodeBERT维度 != emb_dims 时使用。
#     """
#     def __init__(self, embedding: nn.Embedding, proj: nn.Linear):
#         super().__init__()
#         self.embedding = embedding
#         self.proj      = proj

#     def forward(self, indices: torch.Tensor) -> torch.Tensor:
#         return self.proj(self.embedding(indices))

#     @property
#     def weight(self):
#         return self.embedding.weight


# class Datax(HeteroData):
#     def __cat_dim__(self, key: str, value: Any,
#                     store: Optional[NodeOrEdgeStorage] = None, *args,
#                     **kwargs) -> Any:
#         if bool(re.search('(token)', key)):
#             return None
#         if bool(re.search('(pos)', key)):
#             return -1
#         return super().__cat_dim__(key, value, store)


# class Datasetx(Dataset):
#     def __init__(self,
#                  code_graphs,
#                  texts=None,
#                  ids=None,
#                  text_max_len=None,
#                  text_begin_idx=1,
#                  text_end_idx=2,
#                  pad_idx=0):
#         self.len = len(code_graphs)
#         self.text_max_len = text_max_len
#         self.text_begin_idx = text_begin_idx
#         self.text_end_idx = text_end_idx
#         if text_max_len is None and texts is not None:
#             self.text_max_len = max([len(text) for text in texts])
#         self.code_graphs = code_graphs
#         self.texts = texts
#         self.ids = ids
#         self.pad_idx = pad_idx

#     def __getitem__(self, index):
#         if self.texts is None:
#             pad_text_in = np.zeros((self.text_max_len + 1,), dtype=np.int64)
#             pad_text_in[0] = self.text_begin_idx
#             pad_text_out = None
#         else:
#             tru_text = self.texts[index][:self.text_max_len]
#             pad_text_in = np.lib.pad(tru_text,
#                                      (1, self.text_max_len - len(tru_text)),
#                                      'constant',
#                                      constant_values=(self.text_begin_idx, self.pad_idx))
#             tru_text_out = np.lib.pad(tru_text,
#                                       (0, 1),
#                                       'constant',
#                                       constant_values=(0, self.text_end_idx))
#             pad_text_out = np.lib.pad(tru_text_out,
#                                       (0, self.text_max_len + 1 - len(tru_text_out)),
#                                       'constant',
#                                       constant_values=(self.pad_idx, self.pad_idx))
#         data = Datax()
#         data['node'].x = torch.tensor(self.code_graphs[index]['nodes'])
#         data['node'].src_map = torch.tensor(self.code_graphs[index]['node2text_map_ids']).long()
#         data['node'].code_mask = torch.tensor(self.code_graphs[index]['code_node_mask']).bool()
#         # B12修复：parent_child_hyperedges（原来误写为parentparent_child_hyperedges）
#         data['node', 'parent_child_hyperedges',     'node'].edge_index = torch.tensor(self.code_graphs[index]['parent_child_hyperedges']).long()
#         data['node', 'line_hyperedges',              'node'].edge_index = torch.tensor(self.code_graphs[index]['line_hyperedges']).long()
#         data['node', 'dfg_hyperedges',               'node'].edge_index = torch.tensor(self.code_graphs[index]['dfg_hyperedges']).long()
#         data['node', 'block_hyperedges',             'node'].edge_index = torch.tensor(self.code_graphs[index]['block_hyperedges']).long()
#         data['node', 'layout_sibling_hyperedges',    'node'].edge_index = torch.tensor(self.code_graphs[index]['layout_sibling_hyperedges']).long()

#         data['text'].text_token_input = torch.tensor(pad_text_in).long()
#         if self.texts is not None:
#             data['text'].text_token_output = torch.tensor(pad_text_out).long()
#         data['text'].num_nodes = pad_text_in.shape[0]
#         if self.ids is not None:
#             data['idx'].idx = torch.tensor(self.ids[index])
#             data['idx'].num_nodes = 1
#         return data

#     def __len__(self):
#         return self.len


# class CodeGraphEnc(nn.Module):
#     def __init__(self,
#                  emb_dims,
#                  graph_max_size,
#                  code_max_len,
#                  graph_node_emb_op,
#                  graph_gnn_layers=6,
#                  graph_GNN=SAGEConv,
#                  graph_gnn_aggr='mean',
#                  drop_rate=0.,
#                  use_pos_encoding=False,        # 可选：是否使用图节点位置编码（默认False）
#                  hyper_aggr_mode='hetero_conv', # 可选：'hetero_conv' 或 'weighted_sum'
#                  **kwargs):
#         super().__init__()
#         kwargs.setdefault('pad_idx', 0)
#         self.pad_idx          = kwargs['pad_idx']
#         self.graph_max_size   = graph_max_size
#         self.code_max_len     = code_max_len
#         self.emb_dims         = emb_dims
#         self.use_pos_encoding = use_pos_encoding
#         self.hyper_aggr_mode  = hyper_aggr_mode

#         self.graph_node_emb_op = graph_node_emb_op

#         # ── 可选位置编码 ──────────────────────────────────────────────────────
#         # use_pos_encoding=False（推荐，默认）:
#         #   直接dropout，无缩放，与HetCoS基线对齐，训练稳定。
#         #   图节点顺序无语义意义，位置编码会引入错误归纳偏置；
#         #   sqrt(512)≈22.6的缩放会破坏Xavier初始化的数值分布。
#         # use_pos_encoding=True:
#         #   为图节点添加可学习位置编码 + sqrt(emb_dims)缩放（实验用）
#         if self.use_pos_encoding:
#             max_position = graph_max_size * 2
#             self.graph_pos_encoding = nn.Embedding(
#                 max_position + 1,
#                 emb_dims,
#                 padding_idx=kwargs['pad_idx']
#             )
#             nn.init.xavier_uniform_(self.graph_pos_encoding.weight[1:, ])

#         self.emb_drop_op = nn.Dropout(p=drop_rate)

#         # ── 超图边类型（B12修复）────────────────────────────────────────────
#         self.hyper_edge_types = [
#             ('node', 'parent_child_hyperedges',     'node'),
#             ('node', 'line_hyperedges',              'node'),
#             ('node', 'dfg_hyperedges',               'node'),
#             ('node', 'block_hyperedges',             'node'),
#             ('node', 'layout_sibling_hyperedges',    'node'),
#         ]

#         self.gnn_layers = graph_gnn_layers
#         self.gnorm_ops  = nn.ModuleList()
#         self.grelu_ops  = nn.ModuleList()

#         # ── 两种超图聚合模式 ──────────────────────────────────────────────────
#         # 'hetero_conv' （推荐，默认）: HeteroConv aggr='sum'，参数少，训练稳定
#         # 'weighted_sum': 可学习权重+动态softmax，更灵活但显存更大
#         if self.hyper_aggr_mode == 'hetero_conv':
#             self.gnn_ops = nn.ModuleList()
#             for _ in range(graph_gnn_layers):
#                 gnn = HeteroConv({
#                     ('node', 'parent_child_hyperedges',  'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, 1),
#                     ('node', 'line_hyperedges',           'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, 1),
#                     ('node', 'dfg_hyperedges',            'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, 1),
#                     ('node', 'block_hyperedges',          'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, 1),
#                     ('node', 'layout_sibling_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, 1),
#                 }, aggr='sum')
#                 self.gnn_ops.append(gnn)
#                 self.grelu_ops.append(nn.Sequential(nn.ReLU(), nn.Dropout(p=drop_rate)))
#                 self.gnorm_ops.append(GraphNorm(emb_dims))

#         elif self.hyper_aggr_mode == 'weighted_sum':
#             self.hyperedge_weights = nn.Parameter(
#                 torch.zeros(graph_gnn_layers, len(self.hyper_edge_types)))
#             self.hyper_gnn_ops = nn.ModuleList()
#             for _ in range(graph_gnn_layers):
#                 hyper_layer = nn.ModuleDict({
#                     et[1]: HyperedgeDiffusionConv(emb_dims, emb_dims, K=1)
#                     for et in self.hyper_edge_types
#                 })
#                 self.hyper_gnn_ops.append(hyper_layer)
#                 self.grelu_ops.append(nn.Sequential(nn.ReLU(), nn.Dropout(p=drop_rate)))
#                 self.gnorm_ops.append(GraphNorm(emb_dims))
#         else:
#             raise ValueError(
#                 f"hyper_aggr_mode must be 'hetero_conv' or 'weighted_sum', got '{hyper_aggr_mode}'"
#             )

#     def forward(self, data):
#         assert len(data['node'].x.size()) == 1
#         assert len(data['node'].src_map.size()) == 1
#         assert len(data['node'].code_mask.size()) == 1

#         graph_node_emb = self.graph_node_emb_op(data.x_dict['node'])  # [N, D]

#         # ── 可选位置编码 ──────────────────────────────────────────────────────
#         if self.use_pos_encoding:
#             batch_size = data.x_batch_dict['node'].max().item() + 1
#             device     = graph_node_emb.device
#             pos_indices_list = []
#             for b in range(batch_size):
#                 mask      = data.x_batch_dict['node'] == b
#                 num_nodes = mask.sum().item()
#                 max_allowed = self.graph_pos_encoding.num_embeddings - 1
#                 if num_nodes > max_allowed:
#                     positions = (torch.arange(1, num_nodes + 1, device=device) % max_allowed) + 1
#                 else:
#                     positions = torch.arange(1, num_nodes + 1, device=device)
#                 pos_indices_list.append(positions)
#             pos_indices    = torch.cat(pos_indices_list)
#             pos_emb        = self.graph_pos_encoding(pos_indices)
#             graph_node_emb = graph_node_emb * np.sqrt(self.emb_dims) + pos_emb

#         data['node'].x = self.emb_drop_op(graph_node_emb)

#         code_x_batch = data.x_batch_dict['node'][data['node'].code_mask == True]

#         # ── GNN聚合（两种模式）────────────────────────────────────────────────
#         if self.hyper_aggr_mode == 'hetero_conv':
#             for gnn, relu, norm in zip(self.gnn_ops, self.grelu_ops, self.gnorm_ops):
#                 x_dict = gnn(x_dict=data.x_dict, edge_index_dict=data.edge_index_dict)
#                 data['node'].x = norm(data['node'].x.add(relu(x_dict['node'])))

#         elif self.hyper_aggr_mode == 'weighted_sum':
#             for i, (hyper_layer, relu, norm) in enumerate(
#                     zip(self.hyper_gnn_ops, self.grelu_ops, self.gnorm_ops)):
#                 valid_idx = [
#                     j for j, et in enumerate(self.hyper_edge_types)
#                     if et in data.edge_index_dict and data.edge_index_dict[et].numel() > 0
#                 ]
#                 if valid_idx:
#                     valid_w   = torch.softmax(self.hyperedge_weights[i][valid_idx], dim=0)
#                     hyper_out = torch.zeros_like(data['node'].x)
#                     for k, j in enumerate(valid_idx):
#                         et  = self.hyper_edge_types[j]
#                         out = hyper_layer[et[1]](data['node'].x, data.edge_index_dict[et])
#                         hyper_out = hyper_out + valid_w[k] * out
#                 else:
#                     hyper_out = torch.zeros_like(data['node'].x)
#                 data['node'].x = norm(data['node'].x + relu(hyper_out))

#         graph_enc, _ = to_dense_batch(
#             data.x_dict['node'], batch=data.x_batch_dict['node'],
#             fill_value=self.pad_idx, max_num_nodes=self.graph_max_size)   # [B, G, D]

#         code_src_map, _ = to_dense_batch(
#             data.src_map_dict['node'][data['node'].code_mask == True],
#             batch=code_x_batch, fill_value=self.pad_idx,
#             max_num_nodes=self.code_max_len)   # [B, L_src]

#         graph_code_enc, _ = to_dense_batch(
#             data.x_dict['node'][data['node'].code_mask == True],
#             batch=code_x_batch, fill_value=self.pad_idx,
#             max_num_nodes=self.code_max_len)   # [B, L_src, D]

#         # 构建dense格式代码节点mask，供Dec节点重要性偏置使用
#         graph_code_mask, _ = to_dense_batch(
#             data['node'].code_mask.float(), batch=data.x_batch_dict['node'],
#             fill_value=0.0, max_num_nodes=self.graph_max_size)  # [B, G]

#         return graph_enc, graph_code_enc, code_src_map, graph_code_mask


# class Dec(nn.Module):
#     def __init__(self,
#                  emb_dims,
#                  text_voc_size,
#                  text_emb_op,
#                  text_max_len,
#                  enc_out_dims,
#                  att_layers,
#                  att_heads,
#                  att_head_dims=None,
#                  ff_hid_dims=2048,
#                  drop_rate=0.,
#                  **kwargs):
#         super().__init__()
#         kwargs.setdefault('pad_idx', 0)
#         kwargs.setdefault('copy', True)
#         self._copy = kwargs['copy']
#         self.emb_dims = emb_dims
#         self.text_voc_size = text_voc_size

#         self.text_emb_op = text_emb_op
#         self.pos_encoding = PosEnc(max_len=text_max_len+1, emb_dims=emb_dims,
#                                    train=True, pad=True, pad_idx=kwargs['pad_idx'])
#         self.emb_layer_norm = nn.LayerNorm(emb_dims)
#         self.text_dec_op = TranDec(query_dims=emb_dims,
#                                    key_dims=enc_out_dims,
#                                    head_nums=att_heads,
#                                    head_dims=att_head_dims,
#                                    layer_num=att_layers,
#                                    ff_hid_dims=ff_hid_dims,
#                                    drop_rate=drop_rate,
#                                    pad_idx=kwargs['pad_idx'],
#                                    self_causality=True)
#         self.dropout = nn.Dropout(p=drop_rate)
#         self.out_fc   = nn.Linear(emb_dims, text_voc_size)

#         # ── 思路一：节点重要性先验偏置（新增，参数量极少）────────────────────
#         # 代码token节点（叶子节点）对摘要生成更重要
#         # node_importance_bias: 可学习标量，初始为0（无偏置），模型自己学习偏置方向
#         #   > 0 → 代码节点在cross-attention中更受关注（期望方向）
#         #   < 0 → 结构节点更受关注
#         # node_importance_proj: Linear，恒等初始化，从无偏置出发逐步学习
#         self.node_importance_bias = nn.Parameter(torch.tensor(0.0))
#         self.node_importance_proj = nn.Linear(enc_out_dims, enc_out_dims, bias=False)
#         nn.init.eye_(self.node_importance_proj.weight)

#         # ── 思路二：词频感知复制生成器（新增，仅1个额外标量参数）────────────
#         # 用 ImprovedMultiCopyGenerator 替换 MultiCopyGenerator
#         # freq_gate 初始为0时完全退化为原始行为，训练中自动学习最优平衡点
#         self.copy_generator = ImprovedMultiCopyGenerator(
#             tgt_dims=emb_dims,
#             tgt_voc_size=text_voc_size,
#             src_dims=enc_out_dims,
#             att_heads=att_heads,
#             att_head_dims=att_head_dims,
#             drop_rate=drop_rate,
#             pad_idx=kwargs['pad_idx'])

#     def forward(self, graph_enc, graph_code_enc, code_src_map, text_input,
#                 graph_code_mask=None):
#         """
#         Args:
#             graph_enc      : [B, G, D]   图编码器输出（全部节点）
#             graph_code_enc : [B, L, D]   代码节点编码
#             code_src_map   : [B, L]      代码节点到词表的映射
#             text_input     : [B, T]      解码器输入
#             graph_code_mask: [B, G]      代码节点位置为1，结构节点为0（可为None）
#         """
#         text_emb = self.text_emb_op(text_input)           # (B, T, D)
#         text_emb = text_emb * np.sqrt(self.emb_dims)
#         pos_emb  = self.pos_encoding(text_input)           # (B, T, D)
#         text_dec = self.dropout(text_emb.add(pos_emb))    # (B, T, D)
#         text_dec = self.emb_layer_norm(text_dec)           # (B, T, D)

#         graph_mask = graph_enc.abs().sum(-1).sign()        # (B, G)
#         text_mask  = text_input.abs().sign()               # (B, T)

#         # ── 思路一：节点重要性先验偏置 ────────────────────────────────────────
#         # 在cross-attention前对代码节点施加可学习偏置
#         # graph_code_mask为None时跳过（向后兼容）
#         if graph_code_mask is not None:
#             code_mask_3d = graph_code_mask.unsqueeze(-1)       # (B, G, 1)
#             importance   = (self.node_importance_bias
#                             * code_mask_3d
#                             * self.node_importance_proj(graph_enc))
#             graph_enc_biased = graph_enc + importance
#         else:
#             graph_enc_biased = graph_enc

#         text_dec = self.text_dec_op(
#             query=text_dec, key=graph_enc_biased,
#             query_mask=text_mask, key_mask=graph_mask)     # (B, T, D)

#         if not self._copy:
#             text_output = self.out_fc(text_dec)
#         else:
#             # ── 思路二：词频感知复制（在ImprovedMultiCopyGenerator内部处理）──
#             text_output = self.copy_generator(text_dec, graph_code_enc, code_src_map)

#         return text_output.transpose(1, 2)


# class TNet(BaseNet):
#     def __init__(self,
#                  emb_dims,
#                  graph_max_size,
#                  code_max_len,
#                  text_max_len,
#                  io_voc_size,
#                  text_voc_size,
#                  graph_gnn_layers=6,
#                  graph_GNN=SAGEConv,
#                  graph_gnn_aggr='add',
#                  text_att_layers=3,
#                  text_att_heads=8,
#                  text_att_head_dims=None,
#                  text_ff_hid_dims=2048,
#                  drop_rate=0.,
#                  use_pos_encoding=False,         # 可选：位置编码（默认False）
#                  hyper_aggr_mode='hetero_conv',  # 可选：聚合模式
#                  pretrained_emb=None,            # 可选：CodeBERT预训练嵌入矩阵Tensor
#                  **kwargs):
#         super().__init__()
#         kwargs.setdefault('copy', True)
#         kwargs.setdefault('pad_idx', 0)
#         self.init_params = locals()

#         # ── 嵌入层初始化 ──────────────────────────────────────────────────────
#         # pretrained_emb=None    : 随机Xavier初始化（原始稳定方式，推荐）
#         # pretrained_emb=Tensor  : CodeBERT预训练初始化（freeze=False允许微调）
#         #   维度匹配时直接用；不匹配时加Linear投影层（_ProjectedEmbedding）
#         if pretrained_emb is not None:
#             codebert_dim = pretrained_emb.shape[1]
#             if codebert_dim == emb_dims:
#                 io_token_emb_op = nn.Embedding.from_pretrained(
#                     pretrained_emb, freeze=False, padding_idx=kwargs['pad_idx'])
#                 logging.info(f'[Emb] CodeBERT直接使用，维度={codebert_dim}')
#             else:
#                 raw_emb = nn.Embedding.from_pretrained(
#                     pretrained_emb, freeze=False, padding_idx=kwargs['pad_idx'])
#                 proj = nn.Linear(codebert_dim, emb_dims, bias=False)
#                 nn.init.xavier_uniform_(proj.weight)
#                 io_token_emb_op = _ProjectedEmbedding(raw_emb, proj)
#                 logging.info(f'[Emb] CodeBERT+投影层，{codebert_dim}->{emb_dims}')
#         else:
#             io_token_emb_op = nn.Embedding(
#                 io_voc_size, emb_dims, padding_idx=kwargs['pad_idx'])
#             nn.init.xavier_uniform_(io_token_emb_op.weight[1:, ])
#             logging.info(f'[Emb] 随机Xavier初始化，维度={emb_dims}')

#         self.enc_op = CodeGraphEnc(
#             emb_dims=emb_dims,
#             graph_max_size=graph_max_size,
#             code_max_len=code_max_len,
#             graph_node_emb_op=io_token_emb_op,
#             graph_gnn_layers=graph_gnn_layers,
#             graph_GNN=graph_GNN,
#             graph_gnn_aggr=graph_gnn_aggr,
#             drop_rate=drop_rate,
#             use_pos_encoding=use_pos_encoding,
#             hyper_aggr_mode=hyper_aggr_mode,
#             pad_idx=kwargs['pad_idx'])

#         self.dec_op = Dec(
#             emb_dims=emb_dims,
#             text_voc_size=text_voc_size,
#             text_max_len=text_max_len,
#             text_emb_op=io_token_emb_op,
#             enc_out_dims=emb_dims,
#             att_layers=text_att_layers,
#             att_heads=text_att_heads,
#             att_head_dims=text_att_head_dims,
#             ff_hid_dims=text_ff_hid_dims,
#             drop_rate=drop_rate,
#             copy=kwargs['copy'],
#             pad_idx=kwargs['pad_idx'])

#     def forward(self, code_graph):
#         text_input = code_graph['text'].text_token_input.clone()
#         del code_graph['text']
#         graph_enc, graph_code_enc, code_src_map, graph_code_mask = \
#             self.enc_op(data=code_graph)
#         text_output = self.dec_op(
#             graph_enc=graph_enc,
#             graph_code_enc=graph_code_enc,
#             code_src_map=code_src_map,
#             text_input=text_input,
#             graph_code_mask=graph_code_mask)
#         return text_output


# class TModel(TransSeq2Seq):
#     def __init__(self,
#                  model_dir,
#                  model_name='Transformer_based_model',
#                  model_id=None,
#                  emb_dims=512,
#                  graph_gnn_layers=3,
#                  graph_GNN=SAGEConv,
#                  graph_gnn_aggr='add',
#                  text_att_layers=3,
#                  text_att_heads=8,
#                  text_att_head_dims=None,
#                  text_ff_hid_dims=2048,
#                  drop_rate=0.,
#                  copy=True,
#                  pad_idx=0,
#                  train_batch_size=32,
#                  pred_batch_size=32,
#                  max_train_size=-1,
#                  max_valid_size=32 * 10,
#                  max_big_epochs=20,
#                  regular_rate=1e-5,
#                  lr_base=0.001,
#                  lr_decay=0.9,
#                  min_lr_rate=0.01,
#                  warm_big_epochs=2,
#                  start_valid_epoch=20,
#                  early_stop=20,
#                  Net=TNet,
#                  Dataset=Datasetx,
#                  beam_width=1,
#                  train_metrics=[get_sent_bleu],
#                  valid_metric=get_sent_bleu,
#                  test_metrics=[get_sent_bleu],
#                  train_mode=True,
#                  use_pos_encoding=False,         # 可选：位置编码（默认False，推荐）
#                  hyper_aggr_mode='hetero_conv',  # 可选：聚合模式（默认hetero_conv）
#                  use_pretrained_emb=False,        # 可选：CodeBERT嵌入（默认False）
#                  **kwargs):
#         logging.info('Construct %s' % model_name)
#         super().__init__(model_name=model_name, model_dir=model_dir, model_id=model_id)
#         self.init_params         = locals()
#         self.emb_dims            = emb_dims
#         self.graph_gnn_layers    = graph_gnn_layers
#         self.graph_GNN           = graph_GNN
#         self.graph_gnn_aggr      = graph_gnn_aggr
#         self.text_att_layers     = text_att_layers
#         self.text_att_heads      = text_att_heads
#         self.text_att_head_dims  = text_att_head_dims
#         self.text_ff_hid_dims    = text_ff_hid_dims
#         self.drop_rate           = drop_rate
#         self.pad_idx             = pad_idx
#         self.copy                = copy
#         self.train_batch_size    = train_batch_size
#         self.pred_batch_size     = pred_batch_size
#         self.max_train_size      = max_train_size
#         self.max_valid_size      = max_valid_size
#         self.max_big_epochs      = max_big_epochs
#         self.regular_rate        = regular_rate
#         self.lr_base             = lr_base
#         self.lr_decay            = lr_decay
#         self.min_lr_rate         = min_lr_rate
#         self.warm_big_epochs     = warm_big_epochs
#         self.start_valid_epoch   = start_valid_epoch
#         self.early_stop          = early_stop
#         self.Net                 = Net
#         self.Dataset             = Dataset
#         self.beam_width          = beam_width
#         self.train_metrics       = train_metrics
#         self.valid_metric        = valid_metric
#         self.test_metrics        = test_metrics
#         self.train_mode          = train_mode
#         self.use_pos_encoding    = use_pos_encoding
#         self.hyper_aggr_mode     = hyper_aggr_mode
#         self.use_pretrained_emb  = use_pretrained_emb

#     def _logging_paramerter_num(self):
#         logging.info("{} have {} paramerters in total".format(
#             self.model_name, sum(x.numel() for x in self.net.parameters() if x.requires_grad)))
#         enc_op = self.net.module.enc_op
#         if hasattr(enc_op, 'gnn_ops'):
#             gnn_param = sum(x.numel() for x in enc_op.gnn_ops.parameters() if x.requires_grad)
#         elif hasattr(enc_op, 'hyper_gnn_ops'):
#             gnn_param = sum(x.numel() for x in enc_op.hyper_gnn_ops.parameters() if x.requires_grad)
#         else:
#             gnn_param = 0
#         code_graph_enc_param_num = (
#             gnn_param
#             + sum(x.numel() for x in enc_op.gnorm_ops.parameters() if x.requires_grad)
#             + sum(x.numel() for x in enc_op.grelu_ops.parameters() if x.requires_grad))
#         text_dec_param_num = sum(
#             x.numel() for x in self.net.module.dec_op.text_dec_op.parameters()
#             if x.requires_grad)
#         logging.info("{} have {} paramerters in encoder and decoder".format(
#             self.model_name, code_graph_enc_param_num + text_dec_param_num))

#     def fit(self, train_data, valid_data, **kwargs):
#         self.graph_max_size = 0
#         self.code_max_len   = 0
#         self.io_voc_size    = 0
#         self.text_max_len   = 0
#         for code_graph, text in zip(train_data['code_graphs'], train_data['texts']):
#             self.graph_max_size = max(self.graph_max_size, len(code_graph['nodes']))
#             self.code_max_len   = max(self.code_max_len,   code_graph['code_node_mask'].sum())
#             self.io_voc_size    = max(self.io_voc_size,    max(code_graph['nodes']))
#             self.text_max_len   = max(self.text_max_len,   len(text))
#         self.io_voc_size += 1

#         self.text_voc_size = len(train_data['text_dic']['text_i2w'])
#         self.io_voc_size   = max(self.io_voc_size, self.text_voc_size + 2 * self.code_max_len)

#         # ── 可选：加载CodeBERT预训练嵌入矩阵 ─────────────────────────────────
#         # use_pretrained_emb=True  : 从 codebert_emb_matrix.npy 加载
#         #                            需先运行 build_codebert_emb.py 生成该文件
#         # use_pretrained_emb=False : 跳过，使用随机Xavier初始化（推荐）
#         pretrained_emb_tensor = None
#         if self.use_pretrained_emb:
#             from config import top_data_dir as _top_data_dir
#             codebert_path = os.path.join(_top_data_dir, 'basic_info', 'codebert_emb_matrix.npy')
#             if os.path.exists(codebert_path):
#                 logging.info(f'加载CodeBERT嵌入矩阵: {codebert_path}')
#                 emb_np = np.load(codebert_path)
#                 saved_size, codebert_dim = emb_np.shape
#                 if self.io_voc_size > saved_size:
#                     extra = np.random.normal(
#                         0, 0.02, (self.io_voc_size - saved_size, codebert_dim)
#                     ).astype(np.float32)
#                     emb_np = np.concatenate([emb_np, extra], axis=0)
#                 pretrained_emb_tensor = torch.tensor(emb_np, dtype=torch.float32)
#                 logging.info(f'CodeBERT嵌入加载完成，shape={pretrained_emb_tensor.shape}')
#             else:
#                 logging.warning(f'未找到CodeBERT嵌入矩阵: {codebert_path}，回退到随机初始化')

#         net = self.Net(
#             emb_dims=self.emb_dims,
#             graph_max_size=self.graph_max_size,
#             code_max_len=self.code_max_len,
#             text_max_len=self.text_max_len,
#             io_voc_size=self.io_voc_size,
#             text_voc_size=self.text_voc_size,
#             graph_gnn_layers=self.graph_gnn_layers,
#             graph_GNN=self.graph_GNN,
#             graph_gnn_aggr=self.graph_gnn_aggr,
#             text_att_layers=self.text_att_layers,
#             text_att_heads=self.text_att_heads,
#             text_att_head_dims=self.text_att_head_dims,
#             text_ff_hid_dims=self.text_ff_hid_dims,
#             drop_rate=self.drop_rate,
#             pad_idx=self.pad_idx,
#             copy=self.copy,
#             use_pos_encoding=self.use_pos_encoding,
#             hyper_aggr_mode=self.hyper_aggr_mode,
#             pretrained_emb=pretrained_emb_tensor,
#         )

#         device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.net = DataParallel(net.to(device), follow_batch=['x'])
#         self._logging_paramerter_num()
#         self.net.train()

#         self.optimizer = optim.Adam(self.net.parameters(),
#                                     lr=self.lr_base, weight_decay=self.regular_rate)
#         self.criterion = LabelSmoothSoftmaxCEV2(
#             reduction='mean', ignore_index=self.pad_idx, label_smooth=0.0)

#         self.text_begin_idx = self.text_voc_size - 1
#         self.text_end_idx   = self.text_voc_size - 2
#         self.tgt_begin_idx, self.tgt_end_idx = self.text_begin_idx, self.text_end_idx
#         assert train_data['text_dic']['text_i2w'][self.text_end_idx]   == OUT_END_TOKEN
#         assert train_data['text_dic']['text_i2w'][self.text_begin_idx] == OUT_BEGIN_TOKEN

#         self.max_train_size = (len(train_data['code_graphs'])
#                                if self.max_train_size == -1 else self.max_train_size)
#         train_code_graphs, train_texts, train_ids = zip(*random.sample(
#             list(zip(train_data['code_graphs'], train_data['texts'], train_data['ids'])),
#             min(self.max_train_size, len(train_data['code_graphs']))))

#         train_set = self.Dataset(
#             code_graphs=train_code_graphs, texts=train_texts, ids=train_ids,
#             text_max_len=self.text_max_len, text_begin_idx=self.text_begin_idx,
#             text_end_idx=self.text_end_idx, pad_idx=self.pad_idx)
#         train_loader = DataListLoader(
#             dataset=train_set, batch_size=self.train_batch_size,
#             shuffle=True, drop_last=True)

#         if self.warm_big_epochs is None:
#             self.warm_big_epochs = max(self.max_big_epochs // 10, 2)
#         self.scheduler = LrWarmUp(
#             self.optimizer, min_rate=self.min_lr_rate, lr_decay=self.lr_decay,
#             warm_steps=self.warm_big_epochs * len(train_loader),
#             reduce_steps=len(train_loader))

#         if self.train_mode:
#             for i in range(0, self.max_big_epochs):
#                 pbar = tqdm(train_loader)
#                 for j, batch_data in enumerate(pbar):
#                     batch_text_output = []
#                     ids = []
#                     for data in batch_data:
#                         batch_text_output.append(data['text'].text_token_output.unsqueeze(0))
#                         del data['text'].text_token_output
#                         ids.append(data['idx'].idx.item())
#                         del data['idx']

#                     batch_text_output = torch.cat(batch_text_output, dim=0).to(device)
#                     pred_text_output  = self.net(batch_data)

#                     loss = self.criterion(pred_text_output, batch_text_output)
#                     self.optimizer.zero_grad()
#                     loss.backward()
#                     self.optimizer.step()
#                     self.scheduler.step()

#                     text_dic = {'text_i2w': train_data['text_dic']['text_i2w'],
#                                 'ex_text_i2ws': [train_data['text_dic']['ex_text_i2ws'][k] for k in ids]}
#                     log_info = self._get_log_fit_eval(
#                         loss=loss, pred_tgt=pred_text_output,
#                         gold_tgt=batch_text_output, tgt_i2w=text_dic)
#                     pbar.set_description('[Big epoch:{}/{},{}]'.format(i+1, self.max_big_epochs, log_info))
#                     del pred_text_output, batch_text_output, batch_data

#                 del pbar
#                 if i + 1 >= self.start_valid_epoch:
#                     self.max_valid_size = (len(valid_data['code_graphs'])
#                                           if self.max_valid_size == -1 else self.max_valid_size)
#                     valid_srcs, valid_tgts, ex_text_i2ws = zip(*random.sample(
#                         list(zip(valid_data['code_graphs'], valid_data['texts'],
#                                  valid_data['text_dic']['ex_text_i2ws'])),
#                         min(self.max_valid_size, len(valid_data['code_graphs']))))
#                     text_dic = {'text_i2w': train_data['text_dic']['text_i2w'],
#                                 'ex_text_i2ws': ex_text_i2ws}
#                     worse_epochs = self._do_validation(
#                         valid_srcs=valid_srcs, valid_tgts=valid_tgts,
#                         tgt_i2w=text_dic, increase_better=True, last=False)
#                     if worse_epochs >= self.early_stop:
#                         break

#         self._do_validation(valid_srcs=valid_data['code_graphs'],
#                             valid_tgts=valid_data['texts'],
#                             tgt_i2w=valid_data['text_dic'],
#                             increase_better=True, last=True)
#         self._logging_paramerter_num()

#     def predict(self, code_graphs, text_dic):
#         logging.info('Predict outputs of %s' % self.model_name)
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.net.eval()
#         enc_op = DataParallel(self.net.module.enc_op, follow_batch=['x'])
#         dec_op = torch.nn.DataParallel(self.net.module.dec_op)

#         data_set = self.Dataset(
#             code_graphs=code_graphs, texts=None, ids=None,
#             text_max_len=self.text_max_len, text_begin_idx=self.text_begin_idx,
#             text_end_idx=self.text_end_idx, pad_idx=self.pad_idx)
#         data_loader = DataListLoader(dataset=data_set, batch_size=self.pred_batch_size, shuffle=False)

#         pred_text_id_np_batches = []
#         with torch.no_grad():
#             pbar = tqdm(data_loader)
#             for batch_data in pbar:
#                 batch_text_input = []
#                 for data in batch_data:
#                     batch_text_input.append(data['text'].text_token_input.unsqueeze(0))
#                     del data['text']
#                 batch_text_input = torch.cat(batch_text_input, dim=0).to(device)

#                 batch_graph_enc, batch_graph_code_enc, batch_code_src_map, batch_graph_code_mask = \
#                     enc_op(batch_data)

#                 batch_text_output: list = []
#                 if self.beam_width == 1:
#                     for i in range(self.text_max_len + 1):
#                         pred_out = dec_op(
#                             graph_enc=batch_graph_enc,
#                             graph_code_enc=batch_graph_code_enc,
#                             code_src_map=batch_code_src_map,
#                             text_input=batch_text_input,
#                             graph_code_mask=batch_graph_code_mask)
#                         batch_text_output.append(pred_out[:, :, i].unsqueeze(-1).to('cpu').data.numpy())
#                         if i < self.text_max_len:
#                             batch_text_input[:, i+1] = torch.argmax(pred_out[:, :, i], dim=1)
#                     batch_pred_text = np.concatenate(batch_text_output, axis=-1)[:, :, :-1]
#                     batch_pred_text[:, self.tgt_begin_idx, :] = -np.inf
#                     batch_pred_text[:, self.pad_idx, :]       = -np.inf
#                     pred_text_id_np_batches.append(np.argmax(batch_pred_text, axis=1))
#                 else:
#                     batch_pred_text = trans_beam_search(
#                         net=dec_op,
#                         beam_width=self.beam_width,
#                         dec_input_arg_name='text_input',
#                         length_penalty=1,
#                         begin_idx=self.tgt_begin_idx,
#                         pad_idx=self.pad_idx,
#                         end_idx=self.tgt_end_idx,
#                         graph_enc=batch_graph_enc,
#                         graph_code_enc=batch_graph_code_enc,
#                         code_src_map=batch_code_src_map,
#                         text_input=batch_text_input,
#                         graph_code_mask=batch_graph_code_mask)
#                     pred_text_id_np_batches.append(batch_pred_text.to('cpu').data.numpy()[:, :-1])

#         pred_text_id_np = np.concatenate(pred_text_id_np_batches, axis=0)
#         self.net.train()
#         return self._tgt_ids2tokens(pred_text_id_np, text_dic, self.text_end_idx)

#     def generate_texts(self, code_graphs, text_dic, res_path, gold_texts, raw_data, token_data, **kwargs):
#         logging.info('>>>>>>>Generate the targets according to sources and save the result to {}'.format(res_path))
#         res_dir = os.path.dirname(res_path)
#         if not os.path.exists(res_dir):
#             os.makedirs(res_dir)
#         pred_texts = self.predict(code_graphs=code_graphs, text_dic=text_dic)
#         gold_texts = self._tgt_ids2tokens(gold_texts, text_dic, self.pad_idx)
#         res_data = []
#         for pred_text, gold_text, raw_item, token_item in \
#                 zip(pred_texts, gold_texts, raw_data, token_data):
#             sent_bleu = self.valid_metric([pred_text], [gold_text])
#             res_data.append(dict(pred_text=' '.join(pred_text),
#                                  gold_text=' '.join(gold_text),
#                                  sent_bleu=sent_bleu,
#                                  raw_code=raw_item['code'],
#                                  raw_text=raw_item['text'],
#                                  id=raw_item['id'],
#                                  token_text=token_item['text']))
#         with codecs.open(res_path, 'w', encoding='utf-8') as f:
#             json.dump(res_data, f, indent=4, ensure_ascii=False)
#         self._logging_paramerter_num()
#         logging.info('>>>>>>>The result has been saved to {}'.format(res_path))

#     def _code_ids2tokens(self, code_idss, code_i2w, end_idx):
#         return [[code_i2w[idx] for idx in (
#             code_ids[:code_ids.tolist().index(end_idx)] if end_idx in code_ids else code_ids)]
#                 for code_ids in code_idss]

#     def _tgt_ids2tokens(self, text_id_np, text_dic, end_idx=0, **kwargs):
#         if self.copy:
#             text_tokens: list = []
#             for j, text_ids in enumerate(text_id_np):
#                 text_i2w = {**text_dic['text_i2w'], **text_dic['ex_text_i2ws'][j]}
#                 end_i    = text_ids.tolist().index(end_idx) if end_idx in text_ids else len(text_ids)
#                 text_tokens.append([text_i2w[text_idx] for text_idx in text_ids[:end_i]])
#         else:
#             text_i2w    = text_dic['text_i2w']
#             text_tokens = [[text_i2w[idx] for idx in (
#                 text_ids[:text_ids.tolist().index(end_idx)] if end_idx in text_ids else text_ids)]
#                            for text_ids in text_id_np]
#         return text_tokens


# if __name__ == '__main__':
#     logging.info('Parameters are listed below: \n' +
#                  '\n'.join(['{}: {}'.format(k, v) for k, v in params.items()]))

#     model = TModel(
#         model_dir=params['model_dir'],
#         model_name=params['model_name'],
#         model_id=params['model_id'],
#         emb_dims=params['emb_dims'],
#         graph_gnn_layers=params['graph_gnn_layers'],
#         graph_GNN=params['graph_GNN'],
#         graph_gnn_aggr=params['graph_gnn_aggr'],
#         text_att_layers=params['text_att_layers'],
#         text_att_heads=params['text_att_heads'],
#         text_att_head_dims=params['text_att_head_dims'],
#         text_ff_hid_dims=params['text_ff_hid_dims'],
#         drop_rate=params['drop_rate'],
#         copy=params['copy'],
#         pad_idx=params['pad_idx'],
#         train_batch_size=params['train_batch_size'],
#         pred_batch_size=params['pred_batch_size'],
#         max_train_size=params['max_train_size'],
#         max_valid_size=params['max_valid_size'],
#         max_big_epochs=params['max_big_epochs'],
#         regular_rate=params['regular_rate'],
#         lr_base=params['lr_base'],
#         lr_decay=params['lr_decay'],
#         min_lr_rate=params['min_lr_rate'],
#         warm_big_epochs=params['warm_big_epochs'],
#         early_stop=params['early_stop'],
#         start_valid_epoch=params['start_valid_epoch'],
#         Net=TNet,
#         Dataset=Datasetx,
#         beam_width=params['beam_width'],
#         train_metrics=train_metrics,
#         valid_metric=valid_metric,
#         test_metrics=test_metrics,
#         train_mode=params['train_mode'],
#         # ── 三个可选控制开关（从config.py的params中读取）────────────────
#         use_pos_encoding=params['use_pos_encoding'],
#         hyper_aggr_mode=params['hyper_aggr_mode'],
#         use_pretrained_emb=params['use_pretrained_emb'],
#     )

#     logging.info('Load data ...')
#     with codecs.open(train_avail_data_path, 'rb') as f:
#         train_data = pickle.load(f)
#     with codecs.open(valid_avail_data_path, 'rb') as f:
#         valid_data = pickle.load(f)
#     with codecs.open(test_avail_data_path, 'rb') as f:
#         test_data = pickle.load(f)
#     with codecs.open(test_token_data_path, 'r') as f:
#         test_token_data = json.load(f)
#     with codecs.open(test_raw_data_path, 'r') as f:
#         test_raw_data = json.load(f)

#     model.fit(train_data=train_data, valid_data=valid_data)

#     for key, value in params.items():
#         logging.info('{}: {}'.format(key, value))

#     test_eval_df = model.eval(test_srcs=test_data['code_graphs'],
#                               test_tgts=test_data['texts'],
#                               tgt_i2w=test_data['text_dic'])
#     logging.info('Model performance on test dataset:\n')
#     for i in range(0, len(test_eval_df.columns), 4):
#         print(test_eval_df.iloc[:, i:i+4])

#     model.generate_texts(code_graphs=test_data['code_graphs'],
#                          text_dic=test_data['text_dic'],
#                          res_path=res_path,
#                          gold_texts=test_data['texts'],
#                          raw_data=test_raw_data,
#                          token_data=test_token_data)

# # coding=utf-8
# # coding=utf-8
# import os
# import re
# import sys
# import copy as copy_module   # 用于 deepcopy，避免与变量名冲突
# sys.path.append('../../../')
# from lib.neural_module.learn_strategy import LrWarmUp
# from lib.neural_module.transformer import TranEnc, TranDec, DualTranDec, ResFF, ResMHA
# from lib.neural_module.embedding import PosEnc, SinusoidalPositionalEncoding
# from lib.neural_module.loss import LabelSmoothSoftmaxCEV2, CriterionNet
# from lib.neural_module.balanced_data_parallel import BalancedDataParallel
# from lib.neural_module.copy_attention import (DualMultiCopyGenerator, MultiCopyGenerator,
#                                                DualCopyGenerator, ImprovedMultiCopyGenerator)
# from lib.neural_module.beam_search import trans_beam_search
# from lib.neural_model.seq_to_seq_model import TransSeq2Seq
# from lib.neural_model.base_model import BaseNet
# from lib.neural_module.transformer import ResFF

# from typing import Any, Optional, Union
# from config import *

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler
# from torch.nn.utils import clip_grad_norm_
# from torch.utils.data import Dataset
# from torch_geometric.data import HeteroData
# from torch_geometric.loader.data_list_loader import DataListLoader
# from torch_geometric.utils import to_dense_batch
# from torch_geometric.data.storage import (BaseStorage, NodeStorage, EdgeStorage)
# from torch_geometric.nn.data_parallel import DataParallel
# from torch_geometric.nn import HeteroConv, GraphNorm, HypergraphConv, GATConv
# from lib.neural_module.DHGNN import HyperedgeDiffusionConv, KStepHypergraphConv
# import random
# import numpy as np
# import logging
# import pickle
# import json
# import codecs
# from tqdm import tqdm
# import math

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# NodeOrEdgeStorage = Union[NodeStorage, EdgeStorage]


# # ─────────────────────────────────────────────────────────────────────────────
# # 辅助类
# # ─────────────────────────────────────────────────────────────────────────────

# class _ProjectedEmbedding(nn.Module):
#     """
#     将 nn.Embedding(CodeBERT维度) + nn.Linear 封装为透明模块。
#     仅在 use_pretrained_emb=True 且 CodeBERT维度 != emb_dims 时使用。
#     """
#     def __init__(self, embedding: nn.Embedding, proj: nn.Linear):
#         super().__init__()
#         self.embedding = embedding
#         self.proj      = proj

#     def forward(self, indices: torch.Tensor) -> torch.Tensor:
#         return self.proj(self.embedding(indices))

#     @property
#     def weight(self):
#         return self.embedding.weight


# # ─────────────────────────────────────────────────────────────────────────────
# # 数据类
# # ─────────────────────────────────────────────────────────────────────────────

# class Datax(HeteroData):
#     def __cat_dim__(self, key: str, value: Any,
#                     store: Optional[NodeOrEdgeStorage] = None, *args,
#                     **kwargs) -> Any:
#         if bool(re.search('(token)', key)):
#             return None
#         if bool(re.search('(pos)', key)):
#             return -1
#         return super().__cat_dim__(key, value, store)


# class Datasetx(Dataset):
#     """
#     数据集：只加载5种超图边（普通图边由于GNN未使用，不加载以节省内存）
#     """
#     def __init__(self,
#                  code_graphs,
#                  texts=None,
#                  ids=None,
#                  text_max_len=None,
#                  text_begin_idx=1,
#                  text_end_idx=2,
#                  pad_idx=0):
#         self.len            = len(code_graphs)
#         self.text_max_len   = text_max_len
#         self.text_begin_idx = text_begin_idx
#         self.text_end_idx   = text_end_idx
#         if text_max_len is None and texts is not None:
#             self.text_max_len = max([len(text) for text in texts])
#         self.code_graphs = code_graphs
#         self.texts       = texts
#         self.ids         = ids
#         self.pad_idx     = pad_idx

#     def __getitem__(self, index):
#         if self.texts is None:
#             pad_text_in  = np.zeros((self.text_max_len + 1,), dtype=np.int64)
#             pad_text_in[0] = self.text_begin_idx
#             pad_text_out = None
#         else:
#             tru_text     = self.texts[index][:self.text_max_len]
#             pad_text_in  = np.lib.pad(tru_text,
#                                       (1, self.text_max_len - len(tru_text)),
#                                       'constant',
#                                       constant_values=(self.text_begin_idx, self.pad_idx))
#             tru_text_out = np.lib.pad(tru_text, (0, 1), 'constant',
#                                       constant_values=(0, self.text_end_idx))
#             pad_text_out = np.lib.pad(tru_text_out,
#                                       (0, self.text_max_len + 1 - len(tru_text_out)),
#                                       'constant',
#                                       constant_values=(self.pad_idx, self.pad_idx))

#         data = Datax()
#         data['node'].x         = torch.tensor(self.code_graphs[index]['nodes'])
#         data['node'].src_map   = torch.tensor(self.code_graphs[index]['node2text_map_ids']).long()
#         data['node'].code_mask = torch.tensor(self.code_graphs[index]['code_node_mask']).bool()

#         # ── 5种超图边（B12修复：parent_child，不再是parentparent_child）────────
#         data['node', 'parent_child_hyperedges',     'node'].edge_index = \
#             torch.tensor(self.code_graphs[index]['parent_child_hyperedges']).long()
#         data['node', 'line_hyperedges',              'node'].edge_index = \
#             torch.tensor(self.code_graphs[index]['line_hyperedges']).long()
#         data['node', 'dfg_hyperedges',               'node'].edge_index = \
#             torch.tensor(self.code_graphs[index]['dfg_hyperedges']).long()
#         data['node', 'block_hyperedges',             'node'].edge_index = \
#             torch.tensor(self.code_graphs[index]['block_hyperedges']).long()
#         data['node', 'layout_sibling_hyperedges',    'node'].edge_index = \
#             torch.tensor(self.code_graphs[index]['layout_sibling_hyperedges']).long()

#         # ── 普通图边（GNN中已注释未使用，此处同样注释，保持一致，节省内存）──
#         # data['node','base_child','node'].edge_index = ...
#         # data['node','base_father','node'].edge_index = ...
#         # data['node','sibling_next','node'].edge_index = ...
#         # data['node','sibling_prev','node'].edge_index = ...
#         # data['node','dfg_next','node'].edge_index = ...
#         # data['node','dfg_prev','node'].edge_index = ...
#         # data['node','code_next','node'].edge_index = ...
#         # data['node','code_prev','node'].edge_index = ...
#         # data['node','cfg_next','node'].edge_index = ...
#         # data['node','cfg_prev','node'].edge_index = ...

#         data['text'].text_token_input = torch.tensor(pad_text_in).long()
#         if self.texts is not None:
#             data['text'].text_token_output = torch.tensor(pad_text_out).long()
#         data['text'].num_nodes = pad_text_in.shape[0]
#         if self.ids is not None:
#             data['idx'].idx       = torch.tensor(self.ids[index])
#             data['idx'].num_nodes = 1
#         return data

#     def __len__(self):
#         return self.len


# # ─────────────────────────────────────────────────────────────────────────────
# # Encoder
# # ─────────────────────────────────────────────────────────────────────────────

# class CodeGraphEnc(nn.Module):
#     def __init__(self,
#                  emb_dims,
#                  graph_max_size,
#                  code_max_len,
#                  graph_node_emb_op,
#                  graph_gnn_layers=6,
#                  graph_GNN=SAGEConv,
#                  graph_gnn_aggr='mean',
#                  drop_rate=0.,
#                  use_pos_encoding=False,         # 是否使用图节点位置编码（默认False）
#                  hyper_aggr_mode='hetero_conv',  # 'hetero_conv' 或 'weighted_sum'
#                  use_dynamic_edges=True,          # 是否使用动态语义超边
#                  dynamic_threshold=0.85,          # 动态超边余弦相似度阈值
#                  unk_token_id=1,                  # 用于feature masking的替换token id
#                  **kwargs):
#         super().__init__()
#         kwargs.setdefault('pad_idx', 0)
#         self.pad_idx           = kwargs['pad_idx']
#         self.graph_max_size    = graph_max_size
#         self.code_max_len      = code_max_len
#         self.emb_dims          = emb_dims
#         self.use_pos_encoding  = use_pos_encoding
#         self.hyper_aggr_mode   = hyper_aggr_mode
#         self.use_dynamic_edges = use_dynamic_edges
#         self.dynamic_threshold = dynamic_threshold
#         self.unk_token_id      = unk_token_id

#         self.graph_node_emb_op = graph_node_emb_op

#         # ── 可选位置编码 ──────────────────────────────────────────────────────
#         # use_pos_encoding=False（推荐默认）:
#         #   直接dropout，无√emb_dims缩放，与HetCoS基线对齐，训练稳定
#         # use_pos_encoding=True:
#         #   加可学习位置编码 + √emb_dims缩放（实验用）
#         if self.use_pos_encoding:
#             max_position = graph_max_size * 2
#             self.graph_pos_encoding = nn.Embedding(
#                 max_position + 1, emb_dims, padding_idx=kwargs['pad_idx'])
#             nn.init.xavier_uniform_(self.graph_pos_encoding.weight[1:, ])

#         self.emb_drop_op = nn.Dropout(p=drop_rate)

#         # ── 超图边类型（B12修复）────────────────────────────────────────────
#         self.hyper_edge_types = [
#             ('node', 'parent_child_hyperedges',     'node'),
#             ('node', 'line_hyperedges',              'node'),
#             ('node', 'dfg_hyperedges',               'node'),
#             ('node', 'block_hyperedges',             'node'),
#             ('node', 'layout_sibling_hyperedges',    'node'),
#         ]
#         # 动态超边在运行时生成，不在静态列表里预注册
#         # 但需要在GNN中为它预留一个卷积模块
#         if self.use_dynamic_edges:
#             self.hyper_edge_types.append(
#                 ('node', 'dynamic_semantic_hyperedges', 'node'))

#         self.gnn_layers = graph_gnn_layers
#         self.gnorm_ops  = nn.ModuleList()
#         self.grelu_ops  = nn.ModuleList()

#         # ── 两种超图聚合模式 ──────────────────────────────────────────────────
#         if self.hyper_aggr_mode == 'hetero_conv':
#             self.gnn_ops = nn.ModuleList()
#             for _ in range(graph_gnn_layers):
#                 conv_dict = {
#                     ('node', 'parent_child_hyperedges',  'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, 2),
#                     ('node', 'line_hyperedges',           'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, 1),
#                     ('node', 'dfg_hyperedges',            'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, 1),
#                     ('node', 'block_hyperedges',          'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, 1),
#                     # ('node', 'layout_sibling_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, 1),
#                 }
#                 if self.use_dynamic_edges:
#                     conv_dict[('node', 'dynamic_semantic_hyperedges', 'node')] = \
#                         HyperedgeDiffusionConv(emb_dims, emb_dims, 1)
#                 gnn = HeteroConv(conv_dict, aggr='sum')
#                 self.gnn_ops.append(gnn)
#                 self.grelu_ops.append(nn.Sequential(nn.ReLU(), nn.Dropout(p=drop_rate)))
#                 self.gnorm_ops.append(GraphNorm(emb_dims))

#         elif self.hyper_aggr_mode == 'weighted_sum':
#             self.hyperedge_weights = nn.Parameter(
#                 torch.zeros(graph_gnn_layers, len(self.hyper_edge_types)))
#             self.hyper_gnn_ops = nn.ModuleList()
#             for _ in range(graph_gnn_layers):
#                 hyper_layer = nn.ModuleDict({
#                     et[1]: HyperedgeDiffusionConv(emb_dims, emb_dims, K=1)
#                     for et in self.hyper_edge_types
#                 })
#                 self.hyper_gnn_ops.append(hyper_layer)
#                 self.grelu_ops.append(nn.Sequential(nn.ReLU(), nn.Dropout(p=drop_rate)))
#                 self.gnorm_ops.append(GraphNorm(emb_dims))
#         else:
#             raise ValueError(
#                 f"hyper_aggr_mode must be 'hetero_conv' or 'weighted_sum', "
#                 f"got '{hyper_aggr_mode}'"
#             )

#     def _add_dynamic_edges(self, data):
#         """
#         在当前节点特征空间中计算余弦相似度，动态生成语义超边。

#         修复点：
#         1. 添加valid mask过滤，确保padding位置不会产生虚假超边
#         2. flat_indices正确处理只有valid节点才分配全局id
#         """
#         batch_idx    = data.x_batch_dict['node']
#         node_feats   = data['node'].x                           # [N, D]，此时已经是float特征

#         # 转为dense batch方便做bmm
#         dense_x, mask = to_dense_batch(node_feats, batch_idx, fill_value=0.0)
#         # dense_x: [B, N_max, D],  mask: [B, N_max]

#         norm_x     = F.normalize(dense_x, p=2, dim=-1)         # [B, N_max, D]
#         sim_matrix = torch.bmm(norm_x, norm_x.transpose(1, 2)) # [B, N_max, N_max]

#         # 把padding位置和自连接位置的相似度设为-1，不会触发阈值
#         valid_mask_2d = mask.unsqueeze(1) & mask.unsqueeze(2)   # [B, N_max, N_max]
#         sim_matrix.masked_fill_(~valid_mask_2d, -1.0)
#         sim_matrix.diagonal(dim1=1, dim2=2).fill_(-1.0)         # 去掉自连接

#         # 找出超过阈值的节点对
#         adj = sim_matrix > self.dynamic_threshold               # [B, N_max, N_max]
#         b_idx, row_idx, col_idx = adj.nonzero(as_tuple=True)

#         if b_idx.numel() == 0:
#             # 没有超过阈值的节点对，不添加动态超边
#             return

#         # ── 修复：只对valid节点做全局索引映射 ──────────────────────────────
#         # flat_indices[b, n] = 该节点在全局节点数组里的下标（仅valid节点有意义）
#         flat_indices = torch.full(mask.shape, fill_value=-1,
#                                   dtype=torch.long, device=node_feats.device)
#         flat_indices[mask] = torch.arange(mask.sum(), device=node_feats.device)

#         # 确保找到的行列都是valid节点（padding位置不会触发，但双重保险）
#         valid_row_mask = mask[b_idx, row_idx]
#         valid_col_mask = mask[b_idx, col_idx]
#         valid          = valid_row_mask & valid_col_mask
#         b_idx, row_idx, col_idx = b_idx[valid], row_idx[valid], col_idx[valid]

#         if b_idx.numel() == 0:
#             return

#         global_row = flat_indices[b_idx, row_idx]               # [E]
#         global_col = flat_indices[b_idx, col_idx]               # [E]

#         # 作为无向边（超边中的节点→超边id，HyperedgeDiffusionConv的格式）
#         # edge_index格式：[node_id, hyperedge_id]，这里把每对节点当成一个独立超边
#         # 简化处理：把动态超边当成普通边（节点对），用hyperedge_id = col作为边标识
#         # 实际上HyperedgeDiffusionConv需要[node_id, hyperedge_id]格式
#         # 这里用全局col作为hyperedge_id（相当于以高相似度节点为中心的超边）
#         dynamic_edge_index = torch.stack([global_row, global_col], dim=0)  # [2, E]
#         data.edge_index_dict[('node', 'dynamic_semantic_hyperedges', 'node')] = \
#             dynamic_edge_index
            
#         # logging.info(f'dynamic edges: {b_idx.numel()} pairs')

#     def forward(self, data):
#         assert len(data['node'].x.size()) == 1
#         assert len(data['node'].src_map.size()) == 1
#         assert len(data['node'].code_mask.size()) == 1

#         graph_node_emb = self.graph_node_emb_op(data.x_dict['node'])   # [N, D]

#         # ── 修复1：use_pos_encoding=False时不做任何缩放 ──────────────────────
#         # 原代码无论是否用位置编码都乘了√emb_dims，已修复
#         if self.use_pos_encoding:
#             batch_size = data.x_batch_dict['node'].max().item() + 1
#             device     = graph_node_emb.device
#             pos_indices_list = []
#             for b in range(batch_size):
#                 mask      = data.x_batch_dict['node'] == b
#                 num_nodes = mask.sum().item()
#                 max_allowed = self.graph_pos_encoding.num_embeddings - 1
#                 if num_nodes > max_allowed:
#                     positions = (torch.arange(1, num_nodes + 1, device=device)
#                                  % max_allowed) + 1
#                 else:
#                     positions = torch.arange(1, num_nodes + 1, device=device)
#                 pos_indices_list.append(positions)
#             pos_indices    = torch.cat(pos_indices_list)
#             pos_emb        = self.graph_pos_encoding(pos_indices)
#             # 有位置编码时才做√emb_dims缩放（Transformer标准做法）
#             graph_node_emb = graph_node_emb * np.sqrt(self.emb_dims) + pos_emb
#         # use_pos_encoding=False：直接dropout，无缩放，与HetCoS基线对齐

#         data['node'].x = self.emb_drop_op(graph_node_emb)   # [N, D]

#         code_x_batch = data.x_batch_dict['node'][data['node'].code_mask == True]

#         # ── GNN聚合 ───────────────────────────────────────────────────────────
#         if self.hyper_aggr_mode == 'hetero_conv':
#             for i, (gnn, relu, norm) in enumerate(
#                     zip(self.gnn_ops, self.grelu_ops, self.gnorm_ops)):

#                 # 动态超边：在第gnn_layers//2层之后（特征有一定质量时）触发
#                 if self.use_dynamic_edges and i == (self.gnn_layers // 2):
#                     self._add_dynamic_edges(data)

#                 x_dict = gnn(x_dict=data.x_dict, edge_index_dict=data.edge_index_dict)
#                 data['node'].x = norm(data['node'].x.add(relu(x_dict['node'])))

#         elif self.hyper_aggr_mode == 'weighted_sum':
#             for i, (hyper_layer, relu, norm) in enumerate(
#                     zip(self.hyper_gnn_ops, self.grelu_ops, self.gnorm_ops)):

#                 if self.use_dynamic_edges and i == (self.gnn_layers // 2):
#                     self._add_dynamic_edges(data)

#                 valid_idx = [
#                     j for j, et in enumerate(self.hyper_edge_types)
#                     if et in data.edge_index_dict
#                     and data.edge_index_dict[et].numel() > 0
#                 ]
#                 if valid_idx:
#                     valid_w   = torch.softmax(
#                         self.hyperedge_weights[i][valid_idx], dim=0)
#                     hyper_out = torch.zeros_like(data['node'].x)
#                     for k, j in enumerate(valid_idx):
#                         et  = self.hyper_edge_types[j]
#                         out = hyper_layer[et[1]](
#                             data['node'].x, data.edge_index_dict[et])
#                         hyper_out = hyper_out + valid_w[k] * out
#                 else:
#                     hyper_out = torch.zeros_like(data['node'].x)
#                 data['node'].x = norm(data['node'].x + relu(hyper_out))

#         graph_enc, _ = to_dense_batch(
#             data.x_dict['node'], batch=data.x_batch_dict['node'],
#             fill_value=self.pad_idx, max_num_nodes=self.graph_max_size)       # [B, G, D]

#         code_src_map, _ = to_dense_batch(
#             data.src_map_dict['node'][data['node'].code_mask == True],
#             batch=code_x_batch, fill_value=self.pad_idx,
#             max_num_nodes=self.code_max_len)                                   # [B, L]

#         graph_code_enc, _ = to_dense_batch(
#             data.x_dict['node'][data['node'].code_mask == True],
#             batch=code_x_batch, fill_value=self.pad_idx,
#             max_num_nodes=self.code_max_len)                                   # [B, L, D]

#         # 构建dense格式代码节点mask，供Dec节点重要性偏置使用
#         graph_code_mask, _ = to_dense_batch(
#             data['node'].code_mask.float(), batch=data.x_batch_dict['node'],
#             fill_value=0.0, max_num_nodes=self.graph_max_size)                # [B, G]

#         return graph_enc, graph_code_enc, code_src_map, graph_code_mask


# # ─────────────────────────────────────────────────────────────────────────────
# # Decoder
# # ─────────────────────────────────────────────────────────────────────────────

# class Dec(nn.Module):
#     def __init__(self,
#                  emb_dims,
#                  text_voc_size,
#                  text_emb_op,
#                  text_max_len,
#                  enc_out_dims,
#                  att_layers,
#                  att_heads,
#                  att_head_dims=None,
#                  ff_hid_dims=2048,
#                  drop_rate=0.,
#                  **kwargs):
#         super().__init__()
#         kwargs.setdefault('pad_idx', 0)
#         kwargs.setdefault('copy', True)
#         self._copy         = kwargs['copy']
#         self.emb_dims      = emb_dims
#         self.text_voc_size = text_voc_size

#         self.text_emb_op    = text_emb_op
#         self.pos_encoding   = PosEnc(max_len=text_max_len + 1, emb_dims=emb_dims,
#                                      train=True, pad=True, pad_idx=kwargs['pad_idx'])
#         self.emb_layer_norm = nn.LayerNorm(emb_dims)
#         self.text_dec_op    = TranDec(query_dims=emb_dims,
#                                       key_dims=enc_out_dims,
#                                       head_nums=att_heads,
#                                       head_dims=att_head_dims,
#                                       layer_num=att_layers,
#                                       ff_hid_dims=ff_hid_dims,
#                                       drop_rate=drop_rate,
#                                       pad_idx=kwargs['pad_idx'],
#                                       self_causality=True)
#         self.dropout = nn.Dropout(p=drop_rate)
#         self.out_fc  = nn.Linear(emb_dims, text_voc_size)

#         # ── 节点重要性先验偏置（思路一）──────────────────────────────────────
#         self.node_importance_bias = nn.Parameter(torch.tensor(0.0))
#         self.node_importance_proj = nn.Linear(enc_out_dims, enc_out_dims, bias=False)
#         nn.init.eye_(self.node_importance_proj.weight)

#         # ── 词频感知复制生成器（思路二）──────────────────────────────────────
#         self.copy_generator = ImprovedMultiCopyGenerator(
#             tgt_dims=emb_dims,
#             tgt_voc_size=text_voc_size,
#             src_dims=enc_out_dims,
#             att_heads=att_heads,
#             att_head_dims=att_head_dims,
#             drop_rate=drop_rate,
#             pad_idx=kwargs['pad_idx'])

#     def forward(self, graph_enc, graph_code_enc, code_src_map, text_input,
#                 graph_code_mask=None):
#         """
#         Args:
#             graph_enc      : [B, G, D]
#             graph_code_enc : [B, L, D]
#             code_src_map   : [B, L]
#             text_input     : [B, T]
#             graph_code_mask: [B, G]，代码节点位置为1（可为None）
#         """
#         text_emb = self.text_emb_op(text_input)           # [B, T, D]
#         text_emb = text_emb * np.sqrt(self.emb_dims)
#         pos_emb  = self.pos_encoding(text_input)           # [B, T, D]
#         text_dec = self.dropout(text_emb.add(pos_emb))    # [B, T, D]
#         text_dec = self.emb_layer_norm(text_dec)

#         graph_mask = graph_enc.abs().sum(-1).sign()        # [B, G]
#         text_mask  = text_input.abs().sign()               # [B, T]

#         # ── 节点重要性偏置 ────────────────────────────────────────────────────
#         if graph_code_mask is not None:
#             code_mask_3d     = graph_code_mask.unsqueeze(-1)   # [B, G, 1]
#             importance       = (self.node_importance_bias
#                                 * code_mask_3d
#                                 * self.node_importance_proj(graph_enc))
#             graph_enc_biased = graph_enc + importance
#         else:
#             graph_enc_biased = graph_enc

#         text_dec = self.text_dec_op(
#             query=text_dec, key=graph_enc_biased,
#             query_mask=text_mask, key_mask=graph_mask)     # [B, T, D]

#         if not self._copy:
#             text_output = self.out_fc(text_dec)
#         else:
#             text_output = self.copy_generator(
#                 text_dec, graph_code_enc, code_src_map)

#         return text_output.transpose(1, 2)


# # ─────────────────────────────────────────────────────────────────────────────
# # TNet（含对比学习）
# # ─────────────────────────────────────────────────────────────────────────────

# class TNet(BaseNet):
#     def __init__(self,
#                  emb_dims,
#                  graph_max_size,
#                  code_max_len,
#                  text_max_len,
#                  io_voc_size,
#                  text_voc_size,
#                  graph_gnn_layers=6,
#                  graph_GNN=SAGEConv,
#                  graph_gnn_aggr='add',
#                  text_att_layers=3,
#                  text_att_heads=8,
#                  text_att_head_dims=None,
#                  text_ff_hid_dims=2048,
#                  drop_rate=0.,
#                  use_pos_encoding=False,
#                  hyper_aggr_mode='hetero_conv',
#                  pretrained_emb=None,
#                  use_dynamic_edges=True,
#                  dynamic_threshold=0.85,
#                  use_cl=True,
#                  cl_temp=0.1,
#                  feature_drop_rate=0.1,
#                  edge_drop_rate=0.2,
#                  **kwargs):
#         super().__init__()
#         kwargs.setdefault('copy', True)
#         kwargs.setdefault('pad_idx', 0)
#         self.init_params      = locals()
#         self.use_cl           = use_cl
#         self.cl_temp          = cl_temp
#         self.feature_drop_rate = feature_drop_rate
#         self.edge_drop_rate    = edge_drop_rate

#         # ── 嵌入层 ────────────────────────────────────────────────────────────
#         if pretrained_emb is not None:
#             codebert_dim = pretrained_emb.shape[1]
#             if codebert_dim == emb_dims:
#                 io_token_emb_op = nn.Embedding.from_pretrained(
#                     pretrained_emb, freeze=False, padding_idx=kwargs['pad_idx'])
#                 logging.info(f'[Emb] CodeBERT直接使用，维度={codebert_dim}')
#             else:
#                 raw_emb = nn.Embedding.from_pretrained(
#                     pretrained_emb, freeze=False, padding_idx=kwargs['pad_idx'])
#                 proj = nn.Linear(codebert_dim, emb_dims, bias=False)
#                 nn.init.xavier_uniform_(proj.weight)
#                 io_token_emb_op = _ProjectedEmbedding(raw_emb, proj)
#                 logging.info(f'[Emb] CodeBERT+投影层，{codebert_dim}->{emb_dims}')
#         else:
#             io_token_emb_op = nn.Embedding(
#                 io_voc_size, emb_dims, padding_idx=kwargs['pad_idx'])
#             nn.init.xavier_uniform_(io_token_emb_op.weight[1:, ])
#             logging.info(f'[Emb] 随机Xavier初始化，维度={emb_dims}')

#         # UNK token id（用于feature masking替换，比0更合理）
#         # io_token_w2i里UNK通常是index=1，保持与预处理一致
#         self.unk_token_id = 1

#         self.enc_op = CodeGraphEnc(
#             emb_dims=emb_dims,
#             graph_max_size=graph_max_size,
#             code_max_len=code_max_len,
#             graph_node_emb_op=io_token_emb_op,
#             graph_gnn_layers=graph_gnn_layers,
#             graph_GNN=graph_GNN,
#             graph_gnn_aggr=graph_gnn_aggr,
#             drop_rate=drop_rate,
#             use_pos_encoding=use_pos_encoding,
#             hyper_aggr_mode=hyper_aggr_mode,
#             use_dynamic_edges=use_dynamic_edges,
#             dynamic_threshold=dynamic_threshold,
#             unk_token_id=self.unk_token_id,
#             pad_idx=kwargs['pad_idx'])

#         self.dec_op = Dec(
#             emb_dims=emb_dims,
#             text_voc_size=text_voc_size,
#             text_max_len=text_max_len,
#             text_emb_op=io_token_emb_op,
#             enc_out_dims=emb_dims,
#             att_layers=text_att_layers,
#             att_heads=text_att_heads,
#             att_head_dims=text_att_head_dims,
#             ff_hid_dims=text_ff_hid_dims,
#             drop_rate=drop_rate,
#             copy=kwargs['copy'],
#             pad_idx=kwargs['pad_idx'])

#     def augment_hetero_data(self, aug_data):
#         """
#         对图数据进行两种增强：
#         1. Feature Masking：把10%的节点token ID替换为UNK（id=1），而非0（pad）
#            修复点：原代码用x*mask会把节点变成pad_idx=0，误导后续mask计算
#         2. Edge Dropout：随机丢弃20%的超边连接
#         """
#         x = aug_data['node'].x.clone()   # [N]，token ID（整数）

#         # ── 修复2：用unk_token_id(=1)替换，不用0(=pad_idx) ──────────────────
#         mask_positions = torch.rand(x.size(0), device=x.device) < self.feature_drop_rate
#         x[mask_positions] = self.unk_token_id
#         aug_data['node'].x = x

#         # Edge Dropout：随机保留(1-edge_drop_rate)比例的边
#         for edge_type in list(aug_data.edge_index_dict.keys()):
#             edge_index = aug_data.edge_index_dict[edge_type]
#             num_edges  = edge_index.size(1)
#             if num_edges > 0:
#                 keep_mask = (torch.rand(num_edges, device=edge_index.device)
#                              > self.edge_drop_rate)
#                 # 直接赋值新张量，不修改原始引用
#                 aug_data.edge_index_dict[edge_type] = edge_index[:, keep_mask]
#         return aug_data

#     def forward(self, code_graph):
#         text_input = code_graph['text'].text_token_input.clone()
#         del code_graph['text']

#         if self.training and self.use_cl:
#             # ── 修复3：使用deepcopy确保两份数据完全独立 ─────────────────────
#             # 原代码只clone了node.x，edge_index_dict是浅拷贝，
#             # augment_hetero_data对边的修改会污染original的数据
#             cg_original = copy_module.deepcopy(code_graph)
#             cg_aug      = copy_module.deepcopy(code_graph)
#             cg_aug      = self.augment_hetero_data(cg_aug)

#             # 分别过encoder
#             graph_enc,     graph_code_enc, code_src_map, graph_code_mask = \
#                 self.enc_op(data=cg_original)
#             graph_enc_aug, _,              _,            _                = \
#                 self.enc_op(data=cg_aug)

#             # 正常生成摘要
#             text_output = self.dec_op(
#                 graph_enc=graph_enc,
#                 graph_code_enc=graph_code_enc,
#                 code_src_map=code_src_map,
#                 text_input=text_input,
#                 graph_code_mask=graph_code_mask)

#             # ── 修复4：用代码节点mean而非全图mean ────────────────────────────
#             # 原代码用 graph_enc.mean(dim=1) 包含了括号、分号等结构节点
#             # 改为只对代码token节点取加权平均，语义信号更纯净
#             # graph_code_mask: [B, G]，代码节点为1
#             code_mask_3d = graph_code_mask.unsqueeze(-1)          # [B, G, 1]
#             code_count   = code_mask_3d.sum(dim=1).clamp(min=1)   # [B, 1]

#             z1 = (graph_enc     * code_mask_3d).sum(dim=1) / code_count  # [B, D]
#             z2 = (graph_enc_aug * code_mask_3d).sum(dim=1) / code_count  # [B, D]

#             z1 = F.normalize(z1, p=2, dim=-1)
#             z2 = F.normalize(z2, p=2, dim=-1)

#             # InfoNCE（双向对称）
#             sim_matrix = torch.matmul(z1, z2.transpose(0, 1)) / self.cl_temp
#             labels     = torch.arange(z1.size(0), device=z1.device)
#             loss_cl    = (F.cross_entropy(sim_matrix, labels)
#                           + F.cross_entropy(sim_matrix.transpose(0, 1), labels)) / 2

#             return text_output, loss_cl

#         else:
#             graph_enc, graph_code_enc, code_src_map, graph_code_mask = \
#                 self.enc_op(data=code_graph)
#             text_output = self.dec_op(
#                 graph_enc=graph_enc,
#                 graph_code_enc=graph_code_enc,
#                 code_src_map=code_src_map,
#                 text_input=text_input,
#                 graph_code_mask=graph_code_mask)
#             return text_output


# # ─────────────────────────────────────────────────────────────────────────────
# # TModel
# # ─────────────────────────────────────────────────────────────────────────────

# class TModel(TransSeq2Seq):
#     def __init__(self,
#                  model_dir,
#                  model_name='Transformer_based_model',
#                  model_id=None,
#                  emb_dims=512,
#                  graph_gnn_layers=3,
#                  graph_GNN=SAGEConv,
#                  graph_gnn_aggr='add',
#                  text_att_layers=3,
#                  text_att_heads=8,
#                  text_att_head_dims=None,
#                  text_ff_hid_dims=2048,
#                  drop_rate=0.,
#                  copy=True,
#                  pad_idx=0,
#                  train_batch_size=32,
#                  pred_batch_size=32,
#                  max_train_size=-1,
#                  max_valid_size=32 * 10,
#                  max_big_epochs=20,
#                  regular_rate=1e-5,
#                  lr_base=0.001,
#                  lr_decay=0.9,
#                  min_lr_rate=0.01,
#                  warm_big_epochs=2,
#                  start_valid_epoch=20,
#                  early_stop=20,
#                  Net=TNet,
#                  Dataset=Datasetx,
#                  beam_width=1,
#                  train_metrics=[get_sent_bleu],
#                  valid_metric=get_sent_bleu,
#                  test_metrics=[get_sent_bleu],
#                  train_mode=True,
#                  # ── 三个原有控制开关 ─────────────────────────────────────────
#                  use_pos_encoding=False,
#                  hyper_aggr_mode='hetero_conv',
#                  use_pretrained_emb=False,
#                  # ── 三个新增功能开关 ─────────────────────────────────────────
#                  use_dynamic_edges=True,    # 动态语义超边
#                  dynamic_threshold=0.85,    # 动态超边相似度阈值
#                  use_cl=True,              # 自监督对比学习
#                  cl_weight=0.1,            # 对比损失权重
#                  cl_temp=0.1,              # InfoNCE温度
#                  feature_drop_rate=0.1,    # 节点feature masking比例
#                  edge_drop_rate=0.2,       # 超边dropout比例
#                  **kwargs):
#         logging.info('Construct %s' % model_name)
#         super().__init__(model_name=model_name, model_dir=model_dir, model_id=model_id)
#         self.init_params        = locals()
#         self.emb_dims           = emb_dims
#         self.graph_gnn_layers   = graph_gnn_layers
#         self.graph_GNN          = graph_GNN
#         self.graph_gnn_aggr     = graph_gnn_aggr
#         self.text_att_layers    = text_att_layers
#         self.text_att_heads     = text_att_heads
#         self.text_att_head_dims = text_att_head_dims
#         self.text_ff_hid_dims   = text_ff_hid_dims
#         self.drop_rate          = drop_rate
#         self.pad_idx            = pad_idx
#         self.copy               = copy
#         self.train_batch_size   = train_batch_size
#         self.pred_batch_size    = pred_batch_size
#         self.max_train_size     = max_train_size
#         self.max_valid_size     = max_valid_size
#         self.max_big_epochs     = max_big_epochs
#         self.regular_rate       = regular_rate
#         self.lr_base            = lr_base
#         self.lr_decay           = lr_decay
#         self.min_lr_rate        = min_lr_rate
#         self.warm_big_epochs    = warm_big_epochs
#         self.start_valid_epoch  = start_valid_epoch
#         self.early_stop         = early_stop
#         self.Net                = Net
#         self.Dataset            = Dataset
#         self.beam_width         = beam_width
#         self.train_metrics      = train_metrics
#         self.valid_metric       = valid_metric
#         self.test_metrics       = test_metrics
#         self.train_mode         = train_mode
#         self.use_pos_encoding   = use_pos_encoding
#         self.hyper_aggr_mode    = hyper_aggr_mode
#         self.use_pretrained_emb = use_pretrained_emb
#         self.use_dynamic_edges  = use_dynamic_edges
#         self.dynamic_threshold  = dynamic_threshold
#         self.use_cl             = use_cl
#         self.cl_weight          = cl_weight
#         self.cl_temp            = cl_temp
#         self.feature_drop_rate  = feature_drop_rate
#         self.edge_drop_rate     = edge_drop_rate

#     def _logging_paramerter_num(self):
#         logging.info("{} have {} paramerters in total".format(
#             self.model_name,
#             sum(x.numel() for x in self.net.parameters() if x.requires_grad)))
#         enc_op = self.net.module.enc_op
#         if hasattr(enc_op, 'gnn_ops'):
#             gnn_param = sum(x.numel() for x in enc_op.gnn_ops.parameters()
#                             if x.requires_grad)
#         elif hasattr(enc_op, 'hyper_gnn_ops'):
#             gnn_param = sum(x.numel() for x in enc_op.hyper_gnn_ops.parameters()
#                             if x.requires_grad)
#         else:
#             gnn_param = 0
#         code_graph_enc_param_num = (
#             gnn_param
#             + sum(x.numel() for x in enc_op.gnorm_ops.parameters() if x.requires_grad)
#             + sum(x.numel() for x in enc_op.grelu_ops.parameters() if x.requires_grad))
#         text_dec_param_num = sum(
#             x.numel() for x in self.net.module.dec_op.text_dec_op.parameters()
#             if x.requires_grad)
#         logging.info("{} have {} paramerters in encoder and decoder".format(
#             self.model_name, code_graph_enc_param_num + text_dec_param_num))

#     def fit(self, train_data, valid_data, **kwargs):
#         self.graph_max_size = 0
#         self.code_max_len   = 0
#         self.io_voc_size    = 0
#         self.text_max_len   = 0
#         for code_graph, text in zip(train_data['code_graphs'], train_data['texts']):
#             self.graph_max_size = max(self.graph_max_size, len(code_graph['nodes']))
#             self.code_max_len   = max(self.code_max_len,   code_graph['code_node_mask'].sum())
#             self.io_voc_size    = max(self.io_voc_size,    max(code_graph['nodes']))
#             self.text_max_len   = max(self.text_max_len,   len(text))
#         self.io_voc_size += 1
#         self.text_voc_size = len(train_data['text_dic']['text_i2w'])
#         self.io_voc_size   = max(self.io_voc_size, self.text_voc_size + 2 * self.code_max_len)

#         # ── 可选CodeBERT嵌入矩阵 ─────────────────────────────────────────────
#         pretrained_emb_tensor = None
#         if self.use_pretrained_emb:
#             from config import top_data_dir as _top_data_dir
#             codebert_path = os.path.join(
#                 _top_data_dir, 'basic_info', 'codebert_emb_matrix.npy')
#             if os.path.exists(codebert_path):
#                 logging.info(f'加载CodeBERT嵌入矩阵: {codebert_path}')
#                 emb_np = np.load(codebert_path)
#                 saved_size, codebert_dim = emb_np.shape
#                 if self.io_voc_size > saved_size:
#                     extra = np.random.normal(
#                         0, 0.02,
#                         (self.io_voc_size - saved_size, codebert_dim)
#                     ).astype(np.float32)
#                     emb_np = np.concatenate([emb_np, extra], axis=0)
#                 pretrained_emb_tensor = torch.tensor(emb_np, dtype=torch.float32)
#                 logging.info(f'加载完成，shape={pretrained_emb_tensor.shape}')
#             else:
#                 logging.warning(f'未找到{codebert_path}，回退到随机初始化')

#         net = self.Net(
#             emb_dims=self.emb_dims,
#             graph_max_size=self.graph_max_size,
#             code_max_len=self.code_max_len,
#             text_max_len=self.text_max_len,
#             io_voc_size=self.io_voc_size,
#             text_voc_size=self.text_voc_size,
#             graph_gnn_layers=self.graph_gnn_layers,
#             graph_GNN=self.graph_GNN,
#             graph_gnn_aggr=self.graph_gnn_aggr,
#             text_att_layers=self.text_att_layers,
#             text_att_heads=self.text_att_heads,
#             text_att_head_dims=self.text_att_head_dims,
#             text_ff_hid_dims=self.text_ff_hid_dims,
#             drop_rate=self.drop_rate,
#             pad_idx=self.pad_idx,
#             copy=self.copy,
#             use_pos_encoding=self.use_pos_encoding,
#             hyper_aggr_mode=self.hyper_aggr_mode,
#             pretrained_emb=pretrained_emb_tensor,
#             use_dynamic_edges=self.use_dynamic_edges,
#             dynamic_threshold=self.dynamic_threshold,
#             use_cl=self.use_cl,
#             cl_temp=self.cl_temp,
#             feature_drop_rate=self.feature_drop_rate,
#             edge_drop_rate=self.edge_drop_rate,
#         )

#         device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.net = DataParallel(net.to(device), follow_batch=['x'])
#         self._logging_paramerter_num()
#         self.net.train()

#         self.optimizer = optim.Adam(
#             self.net.parameters(), lr=self.lr_base, weight_decay=self.regular_rate)
#         self.criterion = LabelSmoothSoftmaxCEV2(
#             reduction='mean', ignore_index=self.pad_idx, label_smooth=0.0)

#         self.text_begin_idx = self.text_voc_size - 1
#         self.text_end_idx   = self.text_voc_size - 2
#         self.tgt_begin_idx, self.tgt_end_idx = self.text_begin_idx, self.text_end_idx
#         assert train_data['text_dic']['text_i2w'][self.text_end_idx]   == OUT_END_TOKEN
#         assert train_data['text_dic']['text_i2w'][self.text_begin_idx] == OUT_BEGIN_TOKEN

#         self.max_train_size = (len(train_data['code_graphs'])
#                                if self.max_train_size == -1 else self.max_train_size)
#         train_code_graphs, train_texts, train_ids = zip(*random.sample(
#             list(zip(train_data['code_graphs'],
#                      train_data['texts'],
#                      train_data['ids'])),
#             min(self.max_train_size, len(train_data['code_graphs']))))

#         train_set = self.Dataset(
#             code_graphs=train_code_graphs,
#             texts=train_texts,
#             ids=train_ids,
#             text_max_len=self.text_max_len,
#             text_begin_idx=self.text_begin_idx,
#             text_end_idx=self.text_end_idx,
#             pad_idx=self.pad_idx)
#         train_loader = DataListLoader(
#             dataset=train_set, batch_size=self.train_batch_size,
#             shuffle=True, drop_last=True)

#         if self.warm_big_epochs is None:
#             self.warm_big_epochs = max(self.max_big_epochs // 10, 2)
#         self.scheduler = LrWarmUp(
#             self.optimizer,
#             min_rate=self.min_lr_rate,
#             lr_decay=self.lr_decay,
#             warm_steps=self.warm_big_epochs * len(train_loader),
#             reduce_steps=len(train_loader))

#         if self.train_mode:
#             for i in range(0, self.max_big_epochs):
#                 pbar = tqdm(train_loader)
#                 for j, batch_data in enumerate(pbar):
#                     batch_text_output = []
#                     ids = []
#                     for data in batch_data:
#                         batch_text_output.append(
#                             data['text'].text_token_output.unsqueeze(0))
#                         del data['text'].text_token_output
#                         ids.append(data['idx'].idx.item())
#                         del data['idx']

#                     batch_text_output = torch.cat(batch_text_output, dim=0).to(device)

#                     # ── 对比学习：合并CE loss和CL loss ───────────────────────
#                     if self.use_cl:
#                         result  = self.net(batch_data)
#                         # DataParallel 返回时可能是list，也可能直接是tuple
#                         if isinstance(result, (list, tuple)) and len(result) == 2:
#                             pred_text_output, loss_cl = result
#                             if isinstance(loss_cl, torch.Tensor) and loss_cl.dim() > 0:
#                                 loss_cl = loss_cl.mean()  # 多GPU取平均
#                         else:
#                             # 降级：DataParallel合并后直接返回了文本输出
#                             pred_text_output = result
#                             loss_cl = torch.tensor(0.0, device=device)

#                         loss_ce = self.criterion(pred_text_output, batch_text_output)
#                         loss    = loss_ce + self.cl_weight * loss_cl
#                     else:
#                         pred_text_output = self.net(batch_data)
#                         loss    = self.criterion(pred_text_output, batch_text_output)
#                         loss_ce = loss
#                         loss_cl = torch.tensor(0.0)

#                     self.optimizer.zero_grad()
#                     loss.backward()
#                     clip_grad_norm_(self.net.parameters(), max_norm=5.0)
#                     self.optimizer.step()
#                     self.scheduler.step()

#                     text_dic = {
#                         'text_i2w': train_data['text_dic']['text_i2w'],
#                         'ex_text_i2ws': [train_data['text_dic']['ex_text_i2ws'][k]
#                                          for k in ids]}
#                     log_info = self._get_log_fit_eval(
#                         loss=loss_ce,
#                         pred_tgt=pred_text_output,
#                         gold_tgt=batch_text_output,
#                         tgt_i2w=text_dic)

#                     if self.use_cl:
#                         pbar.set_description(
#                             '[Epoch:{}/{}, CE:{:.3f}, CL:{:.3f}, {}]'.format(
#                                 i+1, self.max_big_epochs,
#                                 loss_ce.item(), loss_cl.item(), log_info))
#                     else:
#                         pbar.set_description(
#                             '[Big epoch:{}/{},{}]'.format(
#                                 i+1, self.max_big_epochs, log_info))

#                     del pred_text_output, batch_text_output, batch_data

#                 del pbar
#                 if i + 1 >= self.start_valid_epoch:
#                     self.max_valid_size = (len(valid_data['code_graphs'])
#                                           if self.max_valid_size == -1
#                                           else self.max_valid_size)
#                     valid_srcs, valid_tgts, ex_text_i2ws = zip(*random.sample(
#                         list(zip(valid_data['code_graphs'],
#                                  valid_data['texts'],
#                                  valid_data['text_dic']['ex_text_i2ws'])),
#                         min(self.max_valid_size, len(valid_data['code_graphs']))))
#                     text_dic = {'text_i2w': train_data['text_dic']['text_i2w'],
#                                 'ex_text_i2ws': ex_text_i2ws}
#                     worse_epochs = self._do_validation(
#                         valid_srcs=valid_srcs, valid_tgts=valid_tgts,
#                         tgt_i2w=text_dic, increase_better=True, last=False)
#                     if worse_epochs >= self.early_stop:
#                         break

#         self._do_validation(
#             valid_srcs=valid_data['code_graphs'],
#             valid_tgts=valid_data['texts'],
#             tgt_i2w=valid_data['text_dic'],
#             increase_better=True, last=True)
#         self._logging_paramerter_num()

#     def predict(self, code_graphs, text_dic):
#         logging.info('Predict outputs of %s' % self.model_name)
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.net.eval()
#         enc_op = DataParallel(self.net.module.enc_op, follow_batch=['x'])
#         dec_op = torch.nn.DataParallel(self.net.module.dec_op)

#         data_set = self.Dataset(
#             code_graphs=code_graphs, texts=None, ids=None,
#             text_max_len=self.text_max_len,
#             text_begin_idx=self.text_begin_idx,
#             text_end_idx=self.text_end_idx,
#             pad_idx=self.pad_idx)
#         data_loader = DataListLoader(
#             dataset=data_set, batch_size=self.pred_batch_size, shuffle=False)

#         pred_text_id_np_batches = []
#         with torch.no_grad():
#             pbar = tqdm(data_loader)
#             for batch_data in pbar:
#                 batch_text_input = []
#                 for data in batch_data:
#                     batch_text_input.append(data['text'].text_token_input.unsqueeze(0))
#                     del data['text']
#                 batch_text_input = torch.cat(batch_text_input, dim=0).to(device)

#                 batch_graph_enc, batch_graph_code_enc, batch_code_src_map, \
#                     batch_graph_code_mask = enc_op(batch_data)

#                 batch_text_output: list = []
#                 if self.beam_width == 1:
#                     for i in range(self.text_max_len + 1):
#                         pred_out = dec_op(
#                             graph_enc=batch_graph_enc,
#                             graph_code_enc=batch_graph_code_enc,
#                             code_src_map=batch_code_src_map,
#                             text_input=batch_text_input,
#                             graph_code_mask=batch_graph_code_mask)
#                         batch_text_output.append(
#                             pred_out[:, :, i].unsqueeze(-1).to('cpu').data.numpy())
#                         if i < self.text_max_len:
#                             batch_text_input[:, i+1] = torch.argmax(
#                                 pred_out[:, :, i], dim=1)
#                     batch_pred_text = np.concatenate(batch_text_output, axis=-1)[:, :, :-1]
#                     batch_pred_text[:, self.tgt_begin_idx, :] = -np.inf
#                     batch_pred_text[:, self.pad_idx, :]       = -np.inf
#                     pred_text_id_np_batches.append(np.argmax(batch_pred_text, axis=1))
#                 else:
#                     batch_pred_text = trans_beam_search(
#                         net=dec_op,
#                         beam_width=self.beam_width,
#                         dec_input_arg_name='text_input',
#                         length_penalty=1,
#                         begin_idx=self.tgt_begin_idx,
#                         pad_idx=self.pad_idx,
#                         end_idx=self.tgt_end_idx,
#                         graph_enc=batch_graph_enc,
#                         graph_code_enc=batch_graph_code_enc,
#                         code_src_map=batch_code_src_map,
#                         text_input=batch_text_input,
#                         graph_code_mask=batch_graph_code_mask)
#                     pred_text_id_np_batches.append(
#                         batch_pred_text.to('cpu').data.numpy()[:, :-1])

#         pred_text_id_np = np.concatenate(pred_text_id_np_batches, axis=0)
#         self.net.train()
#         return self._tgt_ids2tokens(pred_text_id_np, text_dic, self.text_end_idx)

#     def generate_texts(self, code_graphs, text_dic, res_path, gold_texts,
#                        raw_data, token_data, **kwargs):
#         logging.info('>>>>>>>Generate results and save to {}'.format(res_path))
#         res_dir = os.path.dirname(res_path)
#         if not os.path.exists(res_dir):
#             os.makedirs(res_dir)
#         pred_texts = self.predict(code_graphs=code_graphs, text_dic=text_dic)
#         gold_texts = self._tgt_ids2tokens(gold_texts, text_dic, self.pad_idx)
#         res_data   = []
#         for pred_text, gold_text, raw_item, token_item in \
#                 zip(pred_texts, gold_texts, raw_data, token_data):
#             sent_bleu = self.valid_metric([pred_text], [gold_text])
#             res_data.append(dict(
#                 pred_text=' '.join(pred_text),
#                 gold_text=' '.join(gold_text),
#                 sent_bleu=sent_bleu,
#                 raw_code=raw_item['code'],
#                 raw_text=raw_item['text'],
#                 id=raw_item['id'],
#                 token_text=token_item['text']))
#         with codecs.open(res_path, 'w', encoding='utf-8') as f:
#             json.dump(res_data, f, indent=4, ensure_ascii=False)
#         self._logging_paramerter_num()
#         logging.info('>>>>>>>The result has been saved to {}'.format(res_path))

#     def _tgt_ids2tokens(self, text_id_np, text_dic, end_idx=0, **kwargs):
#         if self.copy:
#             text_tokens: list = []
#             for j, text_ids in enumerate(text_id_np):
#                 text_i2w = {**text_dic['text_i2w'], **text_dic['ex_text_i2ws'][j]}
#                 end_i    = (text_ids.tolist().index(end_idx)
#                             if end_idx in text_ids else len(text_ids))
#                 text_tokens.append(
#                     [text_i2w[text_idx] for text_idx in text_ids[:end_i]])
#         else:
#             text_i2w    = text_dic['text_i2w']
#             text_tokens = [
#                 [text_i2w[idx] for idx in (
#                     text_ids[:text_ids.tolist().index(end_idx)]
#                     if end_idx in text_ids else text_ids)]
#                 for text_ids in text_id_np]
#         return text_tokens


# # ─────────────────────────────────────────────────────────────────────────────
# # 入口
# # ─────────────────────────────────────────────────────────────────────────────

# if __name__ == '__main__':
#     # 新增参数的默认值（在config.py的params里加入后会自动读取）
#     params.setdefault('use_dynamic_edges',  True)
#     params.setdefault('dynamic_threshold',  0.85)
#     params.setdefault('use_cl',             True)
#     params.setdefault('cl_weight',          0.1)
#     params.setdefault('cl_temp',            0.1)
#     params.setdefault('feature_drop_rate',  0.1)
#     params.setdefault('edge_drop_rate',     0.2)

#     logging.info('Parameters are listed below: \n' +
#                  '\n'.join(['{}: {}'.format(k, v) for k, v in params.items()]))

#     model = TModel(
#         model_dir=params['model_dir'],
#         model_name=params['model_name'],
#         model_id=params['model_id'],
#         emb_dims=params['emb_dims'],
#         graph_gnn_layers=params['graph_gnn_layers'],
#         graph_GNN=params['graph_GNN'],
#         graph_gnn_aggr=params['graph_gnn_aggr'],
#         text_att_layers=params['text_att_layers'],
#         text_att_heads=params['text_att_heads'],
#         text_att_head_dims=params['text_att_head_dims'],
#         text_ff_hid_dims=params['text_ff_hid_dims'],
#         drop_rate=params['drop_rate'],
#         copy=params['copy'],
#         pad_idx=params['pad_idx'],
#         train_batch_size=params['train_batch_size'],
#         pred_batch_size=params['pred_batch_size'],
#         max_train_size=params['max_train_size'],
#         max_valid_size=params['max_valid_size'],
#         max_big_epochs=params['max_big_epochs'],
#         regular_rate=params['regular_rate'],
#         lr_base=params['lr_base'],
#         lr_decay=params['lr_decay'],
#         min_lr_rate=params['min_lr_rate'],
#         warm_big_epochs=params['warm_big_epochs'],
#         early_stop=params['early_stop'],
#         start_valid_epoch=params['start_valid_epoch'],
#         Net=TNet,
#         Dataset=Datasetx,
#         beam_width=params['beam_width'],
#         train_metrics=train_metrics,
#         valid_metric=valid_metric,
#         test_metrics=test_metrics,
#         train_mode=params['train_mode'],
#         # ── 三个原有控制开关 ─────────────────────────────────────────────────
#         use_pos_encoding=params['use_pos_encoding'],
#         hyper_aggr_mode=params['hyper_aggr_mode'],
#         use_pretrained_emb=params['use_pretrained_emb'],
#         # ── 三个新增功能开关 ─────────────────────────────────────────────────
#         use_dynamic_edges=params['use_dynamic_edges'],
#         dynamic_threshold=params['dynamic_threshold'],
#         use_cl=params['use_cl'],
#         cl_weight=params['cl_weight'],
#         cl_temp=params['cl_temp'],
#         feature_drop_rate=params['feature_drop_rate'],
#         edge_drop_rate=params['edge_drop_rate'],
#     )

#     logging.info('Load data ...')
#     with codecs.open(train_avail_data_path, 'rb') as f:
#         train_data = pickle.load(f)
#     with codecs.open(valid_avail_data_path, 'rb') as f:
#         valid_data = pickle.load(f)
#     with codecs.open(test_avail_data_path, 'rb') as f:
#         test_data = pickle.load(f)
#     with codecs.open(test_token_data_path, 'r') as f:
#         test_token_data = json.load(f)
#     with codecs.open(test_raw_data_path, 'r') as f:
#         test_raw_data = json.load(f)

#     model.fit(train_data=train_data, valid_data=valid_data)

#     for key, value in params.items():
#         logging.info('{}: {}'.format(key, value))

#     test_eval_df = model.eval(
#         test_srcs=test_data['code_graphs'],
#         test_tgts=test_data['texts'],
#         tgt_i2w=test_data['text_dic'])
#     logging.info('Model performance on test dataset:\n')
#     for i in range(0, len(test_eval_df.columns), 4):
#         print(test_eval_df.iloc[:, i:i+4])

#     model.generate_texts(
#         code_graphs=test_data['code_graphs'],
#         text_dic=test_data['text_dic'],
#         res_path=res_path,
#         gold_texts=test_data['texts'],
#         raw_data=test_raw_data,
#         token_data=test_token_data)




# coding=utf-8
# import os
# import re
# import sys
# sys.path.append('../../../')
# from lib.neural_module.learn_strategy import LrWarmUp
# from lib.neural_module.transformer import TranEnc, TranDec, DualTranDec,ResFF,ResMHA
# from lib.neural_module.embedding import PosEnc,SinusoidalPositionalEncoding
# from lib.neural_module.loss import LabelSmoothSoftmaxCEV2, CriterionNet
# from lib.neural_module.balanced_data_parallel import BalancedDataParallel
# from lib.neural_module.copy_attention import DualMultiCopyGenerator,MultiCopyGenerator,DualCopyGenerator
# from lib.neural_module.beam_search import trans_beam_search
# from lib.neural_model.seq_to_seq_model import TransSeq2Seq
# from lib.neural_model.base_model import BaseNet
# from lib.neural_module.transformer import ResFF
    
# from typing import Any,Optional,Union
# from config import *

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler
# from torch.nn.utils import clip_grad_norm_
# from torch.utils.data import Dataset
# from torch_geometric.data import HeteroData
# from torch_geometric.loader.data_list_loader import DataListLoader
# from torch_geometric.utils import to_dense_batch
# from torch_geometric.data.storage import (BaseStorage, NodeStorage,EdgeStorage)
# from torch_geometric.nn.data_parallel import DataParallel
# from torch_geometric.nn import HeteroConv,GraphNorm,HypergraphConv,GATConv

# # ===== 新增引用 =====
# from torch_scatter import scatter, scatter_add
# from torch_geometric.utils import softmax as scatter_softmax
# from torch_geometric.utils import degree
# import random
# import numpy as np
# import logging
# import pickle
# import json
# import codecs
# from tqdm import tqdm
# import math
# from copy import deepcopy 

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# NodeOrEdgeStorage = Union[NodeStorage, EdgeStorage]

# # ================= 防弹版 K 阶无向超图算子 =================
# class HyperedgeDiffusionConv(nn.Module):
#     def __init__(self, in_channels, out_channels, K=1, alpha=0.5, bias=True):
#         super(HyperedgeDiffusionConv, self).__init__()
#         self.K = K
#         self._alpha_init = alpha
#         self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
#         self.lin = nn.Linear(in_channels, out_channels, bias=bias)
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.lin.reset_parameters()
#         if hasattr(self, 'alpha'):
#             self.alpha.data.fill_(self._alpha_init)

#     def forward(self, x, hyperedge_index, num_nodes=None, num_edges=None):
#         if num_nodes is None: num_nodes = x.size(0)
#         if num_edges is None: num_edges = int(hyperedge_index[1].max()) + 1
#         node_idx, edge_idx = hyperedge_index

#         d_v = degree(node_idx, num_nodes, dtype=x.dtype)
#         d_v[d_v == 0] = 1.0
#         d_v_inv_sqrt = d_v.pow(-0.5)
        
#         d_v_inv = d_v.pow(-1.0) 
#         d_v_norm_term = d_v_inv[node_idx] 

#         d_e = degree(edge_idx, num_edges, dtype=x.dtype)
#         d_e[d_e == 0] = 1.0
#         d_e_inv = d_e.pow(-1.0)

#         x = self.lin(x)

#         x_temp = x * d_v_inv_sqrt.unsqueeze(-1)
#         H_e = scatter_add(x_temp[node_idx], edge_idx, dim=0, dim_size=num_edges)
#         H_e = H_e * d_e_inv.unsqueeze(-1)
        
#         H_e_0 = H_e 

#         # 防止 alpha 爆炸导致 NaN
#         cur_alpha = torch.clamp(self.alpha, min=0.0, max=1.0)
        
#         for k in range(self.K):
#             H_node_temp = H_e[edge_idx] 
#             H_node_temp = H_node_temp * d_v_norm_term.unsqueeze(-1)
#             H_e_diffused = scatter_add(H_node_temp, edge_idx, dim=0, dim_size=num_edges)
#             H_e_diffused = H_e_diffused * d_e_inv.unsqueeze(-1)
#             H_e = (1 - cur_alpha) * H_e_diffused + cur_alpha * H_e_0

#         out = H_e[edge_idx]
#         out = scatter_add(out, node_idx, dim=0, dim_size=num_nodes)
#         out = out * d_v_inv_sqrt.unsqueeze(-1)

#         return out

# # ================= 防弹版有向超图注意力算子 (DHGAT) =================
# def build_directed_hyperedges_from_simple(edges, group_by='src'):
#     if edges is None or np.size(edges) == 0:
#         return np.empty((2, 0), dtype=np.int64), np.empty((2, 0), dtype=np.int64)
#     src_nodes, dst_nodes = np.array(edges[0]), np.array(edges[1])
#     src_hyper_edges, dst_hyper_edges = [], []
#     if group_by == 'src':
#         unique_srcs = np.unique(src_nodes)
#         for h_id, src in enumerate(unique_srcs):
#             src_hyper_edges.append([src, h_id])
#             for child in dst_nodes[src_nodes == src]:
#                 dst_hyper_edges.append([child, h_id])  
#     else:
#         unique_dsts = np.unique(dst_nodes)
#         for h_id, dst in enumerate(unique_dsts):
#             dst_hyper_edges.append([dst, h_id])
#             for parent in src_nodes[dst_nodes == dst]:
#                 src_hyper_edges.append([parent, h_id])
#     src_arr = np.array(src_hyper_edges, dtype=np.int64).T if src_hyper_edges else np.empty((2, 0), dtype=np.int64)
#     dst_arr = np.array(dst_hyper_edges, dtype=np.int64).T if dst_hyper_edges else np.empty((2, 0), dtype=np.int64)
#     return src_arr, dst_arr

# class DirectedHypergraphAttention(nn.Module):
#     def __init__(self, emb_dims):
#         super().__init__()
#         self.emb_dims = emb_dims
#         self.att_src = nn.Linear(emb_dims, 1, bias=False)
#         self.att_dst = nn.Linear(emb_dims, 1, bias=False)
#         self.leaky_relu = nn.LeakyReLU(0.2)
#         self.out_proj = nn.Linear(emb_dims, emb_dims, bias=False)

#     def forward(self, x, edge_index_src, edge_index_dst):
#         if edge_index_src is None or edge_index_dst is None or edge_index_src.numel() == 0 or edge_index_dst.numel() == 0:
#             return torch.zeros_like(x)

#         src_nodes, src_edges = edge_index_src[0], edge_index_src[1]
#         dst_nodes, dst_edges = edge_index_dst[0], edge_index_dst[1]

#         max_src = src_edges.max().item() if src_edges.numel() > 0 else -1
#         max_dst = dst_edges.max().item() if dst_edges.numel() > 0 else -1
#         num_hyperedges = max(max_src, max_dst) + 1
#         if num_hyperedges <= 0: return torch.zeros_like(x)

#         x_src = x[src_nodes]
#         alpha_src = self.leaky_relu(self.att_src(x_src))
#         alpha_src = scatter_softmax(alpha_src, src_edges, dim=0) 
#         hyperedge_x = scatter(x_src * alpha_src, src_edges, dim=0, dim_size=num_hyperedges, reduce='sum')

#         x_edge_broadcast = hyperedge_x[dst_edges]
#         x_dst = x[dst_nodes]
#         alpha_dst = self.leaky_relu(self.att_dst(x_edge_broadcast + x_dst))
#         alpha_dst = scatter_softmax(alpha_dst, dst_nodes, dim=0)
#         out = scatter(x_edge_broadcast * alpha_dst, dst_nodes, dim=0, dim_size=x.size(0), reduce='sum')

#         return self.out_proj(out)

# # ================= 数据预处理 =================
# class Datax(HeteroData):
#     def __cat_dim__(self, key: str, value: Any, store: Optional[NodeOrEdgeStorage] = None, *args, **kwargs) -> Any:
#         if bool(re.search('(token)', key)): return None  
#         if bool(re.search('(pos)', key)): return -1
#         return super().__cat_dim__(key, value,store)    

# class Datasetx(Dataset):
#     def __init__(self, code_graphs, texts=None, ids=None, text_max_len=None, text_begin_idx=1, text_end_idx=2, pad_idx=0):
#         self.len = len(code_graphs)  
#         self.text_max_len = max([len(t) for t in texts]) if text_max_len is None and texts is not None else text_max_len
#         self.text_begin_idx, self.text_end_idx, self.pad_idx = text_begin_idx, text_end_idx, pad_idx
#         self.code_graphs, self.texts, self.ids = code_graphs, texts, ids

#     def __getitem__(self, index):
#         if self.texts is None:
#             pad_text_in = np.zeros((self.text_max_len + 1,), dtype=np.int64)  
#             pad_text_in[0] = self.text_begin_idx
#             pad_text_out = None
#         else:
#             tru_text = self.texts[index][:self.text_max_len]  
#             pad_text_in = np.lib.pad(tru_text, (1, self.text_max_len - len(tru_text)), 'constant', constant_values=(self.text_begin_idx, self.pad_idx))
#             tru_text_out = np.lib.pad(tru_text, (0, 1), 'constant', constant_values=(0, self.text_end_idx))  
#             pad_text_out = np.lib.pad(tru_text_out, (0, self.text_max_len + 1 - len(tru_text_out)), 'constant', constant_values=(self.pad_idx, self.pad_idx))  
            
#         data=Datax()
#         data['node'].x=torch.tensor(self.code_graphs[index]['nodes'])
#         data['node'].src_map=torch.tensor(self.code_graphs[index]['node2text_map_ids']).long()
#         data['node'].code_mask=torch.tensor(self.code_graphs[index]['code_node_mask']).bool()
        
#         # 提取所有的无向超边 (拼写修正为 parent_child_hyperedges, 并包含 layout_sibling)
#         data['node','parent_child_hyperedges','node'].edge_index=torch.tensor(self.code_graphs[index].get('parent_child_hyperedges', [])).long()
#         data['node','line_hyperedges','node'].edge_index=torch.tensor(self.code_graphs[index].get('line_hyperedges', [])).long()
#         data['node','block_hyperedges','node'].edge_index=torch.tensor(self.code_graphs[index].get('block_hyperedges', [])).long()
#         data['node','dfg_hyperedges','node'].edge_index=torch.tensor(self.code_graphs[index].get('dfg_hyperedges', [])).long()
#         data['node','layout_sibling_hyperedges','node'].edge_index=torch.tensor(self.code_graphs[index].get('layout_sibling_hyperedges', [])).long()

#         # 提取转换为有向超边的数据
#         base_f2c = self.code_graphs[index].get('base_father2child_edges', [])
#         ast_src, ast_dst = build_directed_hyperedges_from_simple(base_f2c, group_by='src')
#         data['node', 'ast_directed_src', 'node'].edge_index = torch.tensor(ast_src).long()
#         data['node', 'ast_directed_dst', 'node'].edge_index = torch.tensor(ast_dst).long()

#         dfg_p2n = self.code_graphs[index].get('dfg_prev2next_edges', [])
#         dfg_src, dfg_dst = build_directed_hyperedges_from_simple(dfg_p2n, group_by='dst')
#         data['node', 'dfg_directed_src', 'node'].edge_index = torch.tensor(dfg_src).long()
#         data['node', 'dfg_directed_dst', 'node'].edge_index = torch.tensor(dfg_dst).long()

#         data['text'].text_token_input=torch.tensor(pad_text_in).long()
#         if self.texts is not None: data['text'].text_token_output = torch.tensor(pad_text_out).long()
#         data['text'].num_nodes = pad_text_in.shape[0]
#         if self.ids is not None:
#             data['idx'].idx = torch.tensor(self.ids[index])
#             data['idx'].num_nodes = 1
#         return data

#     def __len__(self): return self.len

# # ================= 纯超图架构编码器 =================
# # ================= 纯超图架构编码器 (真正回归 HeteroConv) =================
# class CodeGraphEnc(nn.Module):
#     def __init__(self, emb_dims, graph_max_size, code_max_len, graph_node_emb_op, graph_gnn_layers=6, drop_rate=0., use_hyperedge_pos_emb=True, use_directed_hyperedges=True, use_dynamic_edges=True, dynamic_threshold=0.85, **kwargs):
#         super().__init__()
#         self.pad_idx = kwargs.get('pad_idx', 0)
#         self.graph_max_size = graph_max_size
#         self.code_max_len=code_max_len
#         self.emb_dims=emb_dims
        
#         self.use_hyperedge_pos_emb = use_hyperedge_pos_emb
#         self.use_directed_hyperedges = use_directed_hyperedges
#         self.use_dynamic_edges = use_dynamic_edges
#         self.dynamic_threshold = dynamic_threshold

#         self.graph_node_emb_op = graph_node_emb_op
#         self.graph_pos_encoding = nn.Embedding(graph_max_size * 2 + 1, emb_dims, padding_idx=self.pad_idx)
#         nn.init.xavier_uniform_(self.graph_pos_encoding.weight[1:, ])
        
#         self.emb_drop_op = nn.Dropout(p=drop_rate)
#         self.gnn_layers = graph_gnn_layers
        
#         # 分离出两大阵营：官方规范的 GNN 和 我们创新的 DHGAT
#         self.gnn_ops = nn.ModuleList()
#         self.dhgat_ops = nn.ModuleList()
        
#         self.gnorm_ops = nn.ModuleList()
#         self.grelu_ops = nn.ModuleList()
        
#         for _ in range(graph_gnn_layers):
#             # =========================================================
#             # 阵营 1：纯正的 HeteroConv (享受底层 C++ aggr='sum' 加速)
#             # =========================================================
#             gnn = HeteroConv({
#                 ('node', 'block_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, K=1),
#                 ('node', 'line_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, K=1),
#                 ('node', 'layout_sibling_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, K=1),
#                 ('node', 'parent_child_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, K=4), # 你的心血 K=4 扩散
#                 ('node', 'dfg_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, K=1), # 无向DFG兜底
#                 ('node', 'dynamic_semantic_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, K=1)
#             }, aggr='sum')
#             self.gnn_ops.append(gnn)
            
#             # =========================================================
#             # 阵营 2：独立的 DHGAT 字典 (处理需要双矩阵的双向特征)
#             # =========================================================
#             dhgat_dict = nn.ModuleDict({
#                 'ast_dir': DirectedHypergraphAttention(emb_dims),
#                 'dfg_dir': DirectedHypergraphAttention(emb_dims)
#             })
#             self.dhgat_ops.append(dhgat_dict)
            
#             self.grelu_ops.append(nn.Sequential(nn.ReLU(), nn.Dropout(p=drop_rate)))
#             self.gnorm_ops.append(GraphNorm(emb_dims))

#     def _add_dynamic_edges(self, data):
#         batch_idx = data.x_batch_dict['node']
#         dense_x, mask = to_dense_batch(data['node'].x, batch_idx, fill_value=0.0) 
#         norm_x = F.normalize(dense_x, p=2, dim=-1) 
#         sim_matrix = torch.bmm(norm_x, norm_x.transpose(1, 2)) 
#         valid_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
#         sim_matrix.masked_fill_(~valid_mask, -1.0)
#         sim_matrix.diagonal(dim1=1, dim2=2).fill_(-1.0)
        
#         adj = sim_matrix > self.dynamic_threshold
#         b_idx, row_idx, col_idx = adj.nonzero(as_tuple=True)
        
#         flat_indices = torch.zeros_like(mask, dtype=torch.long)
#         flat_indices[mask] = torch.arange(mask.sum(), device=mask.device)
#         global_row, global_col = flat_indices[b_idx, row_idx], flat_indices[b_idx, col_idx]
#         data.edge_index_dict[('node', 'dynamic_semantic_hyperedges', 'node')] = torch.stack([global_row, global_col], dim=0)

#     def forward(self, data):
#         graph_node_emb = self.graph_node_emb_op(data.x_dict['node'])  
#         batch_size = data.x_batch_dict['node'].max().item() + 1
#         device = graph_node_emb.device
        
#         pos_indices_list = []
#         for b in range(batch_size):
#             mask = data.x_batch_dict['node'] == b
#             num_nodes = mask.sum().item()
#             max_allowed = self.graph_pos_encoding.num_embeddings - 1
#             positions = (torch.arange(1, num_nodes + 1, device=device) % max_allowed) + 1 if num_nodes > max_allowed else torch.arange(1, num_nodes + 1, device=device)
#             pos_indices_list.append(positions)
        
#         pos_emb = self.graph_pos_encoding(torch.cat(pos_indices_list)) 
#         graph_node_emb = graph_node_emb * np.sqrt(self.emb_dims)
        
#         # 将位置编码深深烙印进特征中
#         if self.use_hyperedge_pos_emb:
#             graph_node_emb = graph_node_emb + pos_emb
            
#         data['node'].x = self.emb_drop_op(graph_node_emb) 
#         code_x_batch = data.x_batch_dict['node'][data['node'].code_mask==True]   
        
#         # 极度清晰、高效的计算流
#         for i, (gnn, dhgat_dict, relu, norm) in enumerate(zip(self.gnn_ops, self.dhgat_ops, self.grelu_ops, self.gnorm_ops)):
            
#             if getattr(self, 'use_dynamic_edges', True) and i == (self.gnn_layers // 2):
#                 self._add_dynamic_edges(data)
                
#             x_current = data['node'].x  
            
#             # 1. 运行官方的 HeteroConv (它会自动跳过不存在的边，并进行 aggr='sum')
#             # 这里的执行效率和信号强度，与你跑出 50.3% 的原版代码完全一致！
#             x_dict = gnn(x_dict=data.x_dict, edge_index_dict=data.edge_index_dict)
#             out_x = x_dict['node'] # 取出聚合后的特征
            
#             # 2. 如果开启了有向超图创新，我们手动合并它的注意力特征
#             if getattr(self, 'use_directed_hyperedges', True):
#                 a_s, a_d = data.edge_index_dict.get(('node', 'ast_directed_src', 'node')), data.edge_index_dict.get(('node', 'ast_directed_dst', 'node'))
#                 if a_s is not None and a_d is not None and a_s.numel() > 0: 
#                     out_x = out_x + dhgat_dict['ast_dir'](x_current, a_s, a_d)
                
#                 d_s, d_d = data.edge_index_dict.get(('node', 'dfg_directed_src', 'node')), data.edge_index_dict.get(('node', 'dfg_directed_dst', 'node'))
#                 if d_s is not None and d_d is not None and d_s.numel() > 0: 
#                     out_x = out_x + dhgat_dict['dfg_dir'](x_current, d_s, d_d)

#             # 3. 完美的残差与 Norm 收尾
#             data['node'].x = norm(x_current + relu(out_x))
            
#         graph_enc,_=to_dense_batch(data.x_dict['node'], batch=data.x_batch_dict['node'], fill_value=self.pad_idx, max_num_nodes=self.graph_max_size)  
#         code_src_map,_=to_dense_batch(data.src_map_dict['node'][data['node'].code_mask==True], batch=code_x_batch, fill_value=self.pad_idx, max_num_nodes=self.code_max_len)    
#         graph_code_enc,_=to_dense_batch(data.x_dict['node'][data['node'].code_mask==True], batch=code_x_batch, fill_value=self.pad_idx, max_num_nodes=self.code_max_len)    

#         return graph_enc,graph_code_enc,code_src_map

# class Dec(nn.Module):
#     def __init__(self, emb_dims, text_voc_size, text_emb_op, text_max_len, enc_out_dims, att_layers, att_heads, att_head_dims=None, ff_hid_dims=2048, drop_rate=0., **kwargs):
#         super().__init__()
#         kwargs.setdefault('pad_idx', 0)
#         kwargs.setdefault('copy', True)
#         self._copy = kwargs['copy']
#         self.emb_dims = emb_dims
#         self.text_voc_size = text_voc_size
#         self.text_emb_op = text_emb_op
#         self.pos_encoding = PosEnc(max_len=text_max_len+1, emb_dims=emb_dims, train=True, pad=True,pad_idx=kwargs['pad_idx']) 
#         self.emb_layer_norm = nn.LayerNorm(emb_dims)
#         self.text_dec_op = TranDec(query_dims=emb_dims, key_dims=enc_out_dims, head_nums=att_heads, head_dims=att_head_dims, layer_num=att_layers, ff_hid_dims=ff_hid_dims, drop_rate=drop_rate, pad_idx=kwargs['pad_idx'], self_causality=True)
#         self.dropout = nn.Dropout(p=drop_rate)
#         self.out_fc = nn.Linear(emb_dims, text_voc_size)
#         self.copy_generator = MultiCopyGenerator(tgt_dims=emb_dims, tgt_voc_size=text_voc_size, src_dims=enc_out_dims, att_heads=att_heads, att_head_dims=att_head_dims, drop_rate=drop_rate, pad_idx=kwargs['pad_idx'])

#     def forward(self,graph_enc,graph_code_enc,code_src_map,text_input):
#         text_emb = self.text_emb_op(text_input)   
#         text_emb = text_emb * np.sqrt(self.emb_dims)
#         pos_emb = self.pos_encoding(text_input)  
#         text_dec = self.dropout(text_emb.add(pos_emb))  
#         text_dec = self.emb_layer_norm(text_dec)  
#         graph_mask = graph_enc.abs().sum(-1).sign()  
#         text_mask = text_input.abs().sign()  
#         text_dec = self.text_dec_op(query=text_dec, key=graph_enc, query_mask=text_mask, key_mask=graph_mask)  

#         if not self._copy:
#             text_output = self.out_fc(text_dec)  
#         else:
#             text_output = self.copy_generator(text_dec, graph_code_enc,code_src_map)
#         return text_output.transpose(1, 2)

# # ================= 引入对比学习的总网络 =================
# class TNet(BaseNet):
#     def __init__(self, emb_dims, graph_max_size, code_max_len, text_max_len, io_voc_size, text_voc_size, graph_gnn_layers=6, graph_GNN=SAGEConv, graph_gnn_aggr='add', text_att_layers=3, text_att_heads=8, text_att_head_dims=None, text_ff_hid_dims=2048, drop_rate=0., use_hyperedge_pos_emb=True, 
#                  use_cl=True, cl_temp=0.1, edge_drop_rate=0.15, use_directed_hyperedges=True, use_dynamic_edges=True, **kwargs):
#         super().__init__()
#         kwargs.setdefault('copy', True)
#         kwargs.setdefault('pad_idx', 0)  
#         self.init_params = locals()
        
#         self.use_cl = use_cl
#         self.cl_temp = cl_temp
#         self.edge_drop_rate = edge_drop_rate

#         io_token_emb_op=nn.Embedding(io_voc_size, emb_dims, padding_idx=kwargs['pad_idx'])
#         nn.init.xavier_uniform_(io_token_emb_op.weight[1:, ])
        
#         self.enc_op = CodeGraphEnc(emb_dims=emb_dims, graph_max_size=graph_max_size, code_max_len=code_max_len, graph_node_emb_op=io_token_emb_op, graph_gnn_layers=graph_gnn_layers, drop_rate=drop_rate, pad_idx=kwargs['pad_idx'], use_hyperedge_pos_emb=use_hyperedge_pos_emb, use_directed_hyperedges=use_directed_hyperedges, use_dynamic_edges=use_dynamic_edges)
#         self.dec_op = Dec(emb_dims=emb_dims, text_voc_size=text_voc_size, text_max_len=text_max_len, text_emb_op=io_token_emb_op, enc_out_dims=emb_dims, att_layers=text_att_layers, att_heads=text_att_heads, att_head_dims=text_att_head_dims, ff_hid_dims=text_ff_hid_dims, drop_rate=drop_rate, copy=kwargs['copy'], pad_idx=kwargs['pad_idx'])

#     def augment_edges_only(self, aug_data):
#         for edge_type in list(aug_data.edge_index_dict.keys()):
#             edge_index = aug_data.edge_index_dict[edge_type]
#             num_edges = edge_index.size(1)
#             if num_edges > 0:
#                 keep_mask = torch.rand(num_edges, device=edge_index.device) > self.edge_drop_rate
#                 aug_data[edge_type].edge_index = edge_index[:, keep_mask]
#         return aug_data

#     def forward(self, code_graph):
#         text_input = code_graph['text'].text_token_input.clone()
#         del code_graph['text']
        
#         if self.training and getattr(self, 'use_cl', True):
#             # 深拷贝，确保视图安全隔离
#             cg_original = deepcopy(code_graph)
#             cg_aug = self.augment_edges_only(deepcopy(code_graph))
            
#             graph_enc, graph_code_enc, code_src_map = self.enc_op(data=cg_original)
#             graph_enc_aug, _, _ = self.enc_op(data=cg_aug)
            
#             text_output = self.dec_op(graph_enc=graph_enc, graph_code_enc=graph_code_enc, code_src_map=code_src_map, text_input=text_input)

#             z1, z2 = F.normalize(graph_enc.mean(dim=1), p=2, dim=-1), F.normalize(graph_enc_aug.mean(dim=1), p=2, dim=-1)
#             sim_matrix = torch.matmul(z1, z2.transpose(0, 1)) / self.cl_temp
            
#             labels = torch.arange(z1.size(0), device=z1.device)
#             loss_cl = (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.transpose(0, 1), labels)) / 2
            
#             return text_output, loss_cl
#         else:
#             graph_enc,graph_code_enc,code_src_map = self.enc_op(data=code_graph)
#             return self.dec_op(graph_enc=graph_enc,graph_code_enc=graph_code_enc, code_src_map=code_src_map, text_input=text_input)

# class TModel(TransSeq2Seq):
#     def __init__(self, model_dir, model_name='Transformer_based_model', model_id=None, emb_dims=512, graph_gnn_layers=3, graph_GNN=SAGEConv, graph_gnn_aggr='add', text_att_layers=3, text_att_heads=8, text_att_head_dims=None, text_ff_hid_dims=2048, drop_rate=0., copy=True, pad_idx=0, train_batch_size=32, pred_batch_size=32, max_train_size=-1, max_valid_size=32 * 10, max_big_epochs=20, regular_rate=1e-5, lr_base=0.001, lr_decay=0.9, min_lr_rate=0.01, warm_big_epochs=2, start_valid_epoch=20, early_stop=20, Net=TNet, Dataset=Datasetx, beam_width=1, train_metrics=[get_sent_bleu], valid_metric=get_sent_bleu, test_metrics=[get_sent_bleu], train_mode=True, 
#                  use_hyperedge_pos_emb=True, use_directed_hyperedges=True, use_dynamic_edges=True, use_cl=True, cl_weight=0.05, **kwargs):
#         logging.info('Construct %s' % model_name)
#         super().__init__(model_name=model_name, model_dir=model_dir, model_id=model_id)
#         self.init_params = locals()
#         self.emb_dims = emb_dims
#         self.graph_gnn_layers = graph_gnn_layers
#         self.graph_GNN = graph_GNN
#         self.graph_gnn_aggr = graph_gnn_aggr
#         self.text_att_layers = text_att_layers
#         self.text_att_heads = text_att_heads
#         self.text_att_head_dims = text_att_head_dims
#         self.text_ff_hid_dims = text_ff_hid_dims
#         self.drop_rate = drop_rate
#         self.pad_idx = pad_idx
#         self.copy = copy
#         self.train_batch_size = train_batch_size
#         self.pred_batch_size = pred_batch_size
#         self.max_train_size = max_train_size
#         self.max_valid_size = max_valid_size
#         self.max_big_epochs = max_big_epochs
#         self.regular_rate = regular_rate
#         self.lr_base = lr_base
#         self.lr_decay = lr_decay
#         self.min_lr_rate = min_lr_rate
#         self.warm_big_epochs = warm_big_epochs
#         self.start_valid_epoch=start_valid_epoch
#         self.early_stop=early_stop
#         self.Net = Net
#         self.Dataset = Dataset
#         self.beam_width = beam_width
#         self.train_metrics = train_metrics
#         self.valid_metric = valid_metric
#         self.test_metrics = test_metrics
#         self.train_mode = train_mode
#         self.use_hyperedge_pos_emb = use_hyperedge_pos_emb
#         self.use_directed_hyperedges = use_directed_hyperedges
#         self.use_dynamic_edges = use_dynamic_edges
#         self.use_cl = use_cl
#         self.cl_weight = cl_weight

#     def _logging_paramerter_num(self):
#         logging.info("{} have {} paramerters in total".format(self.model_name, sum( x.numel() for x in self.net.parameters() if x.requires_grad)))
#         code_graph_enc_param_num = sum(x.numel() for x in self.net.module.enc_op.gnn_ops.parameters() if x.requires_grad) + sum(x.numel() for x in self.net.module.enc_op.gnorm_ops.parameters() if x.requires_grad) + sum(x.numel() for x in self.net.module.enc_op.grelu_ops.parameters() if x.requires_grad)
#         text_dec_param_num = sum(x.numel() for x in self.net.module.dec_op.text_dec_op.parameters() if x.requires_grad)
#         enc_dec_param_num = code_graph_enc_param_num + text_dec_param_num
#         logging.info("{} have {} paramerters in encoder and decoder".format(self.model_name, enc_dec_param_num))

#     def fit(self, train_data, valid_data, **kwargs):
#         self.graph_max_size, self.code_max_len, self.io_voc_size, self.text_max_len=0, 0, 0, 0
#         for code_graph,text in zip(train_data['code_graphs'],train_data['texts']):
#             self.graph_max_size = max(self.graph_max_size,len(code_graph['nodes']))
#             self.code_max_len = max(self.code_max_len,code_graph['code_node_mask'].sum())
#             self.io_voc_size = max(self.io_voc_size,max(code_graph['nodes']))
#             self.text_max_len=max(self.text_max_len,len(text))
#         self.io_voc_size+=1

#         self.text_voc_size = len(train_data['text_dic']['text_i2w']) 
#         self.io_voc_size=max(self.io_voc_size,self.text_voc_size+2*self.code_max_len)
#         net = self.Net(
#             emb_dims=self.emb_dims, graph_max_size=self.graph_max_size, code_max_len=self.code_max_len, text_max_len=self.text_max_len, io_voc_size=self.io_voc_size, text_voc_size=self.text_voc_size, graph_gnn_layers=self.graph_gnn_layers, graph_GNN=self.graph_GNN, graph_gnn_aggr=self.graph_gnn_aggr,
#             text_att_layers=self.text_att_layers, text_att_heads=self.text_att_heads, text_att_head_dims=self.text_att_head_dims, text_ff_hid_dims=self.text_ff_hid_dims, 
#             drop_rate=self.drop_rate, pad_idx=self.pad_idx, copy=self.copy, 
#             use_hyperedge_pos_emb=self.use_hyperedge_pos_emb, use_directed_hyperedges=self.use_directed_hyperedges, use_dynamic_edges=self.use_dynamic_edges, use_cl=self.use_cl
#         )
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
#         self.net =DataParallel(net.to(device),follow_batch=['x'])  
#         self._logging_paramerter_num()  
#         self.net.train()  

#         self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr_base, weight_decay=self.regular_rate)
#         self.criterion = LabelSmoothSoftmaxCEV2(reduction='mean', ignore_index=self.pad_idx, label_smooth=0.0)

#         self.text_begin_idx = self.text_voc_size - 1
#         self.text_end_idx = self.text_voc_size - 2
#         self.tgt_begin_idx,self.tgt_end_idx=self.text_begin_idx,self.text_end_idx

#         self.max_train_size = len(train_data['code_graphs']) if self.max_train_size == -1 else self.max_train_size
#         train_code_graphs, train_texts,train_ids = zip(*random.sample(list(zip(train_data['code_graphs'], train_data['texts'],train_data['ids'])), min(self.max_train_size, len(train_data['code_graphs']))))
#         train_set = self.Dataset(code_graphs=train_code_graphs, texts=train_texts, ids=train_ids, text_max_len=self.text_max_len, text_begin_idx=self.text_begin_idx, text_end_idx=self.text_end_idx, pad_idx=self.pad_idx)
#         train_loader=DataListLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True, drop_last=True) 

#         if self.warm_big_epochs is None: self.warm_big_epochs = max(self.max_big_epochs // 10, 2)
#         self.scheduler = LrWarmUp(self.optimizer, min_rate=self.min_lr_rate, lr_decay=self.lr_decay, warm_steps=self.warm_big_epochs * len(train_loader), reduce_steps=len(train_loader))  
        
#         if self.train_mode:  
#             for i in range(0,self.max_big_epochs):
#                 pbar = tqdm(train_loader)
#                 for j, batch_data in enumerate(pbar):
#                     batch_text_output = []
#                     ids=[]
#                     for data in batch_data:
#                         batch_text_output.append(data['text'].text_token_output.unsqueeze(0))
#                         del data['text'].text_token_output
#                         ids.append(data['idx'].idx.item())
#                         del data['idx']

#                     batch_text_output = torch.cat(batch_text_output, dim=0).to(device)
                    
#                     if self.use_cl:
#                         pred_text_output, loss_cl = self.net(batch_data)
#                         loss_cl = loss_cl.mean()
#                         loss_ce = self.criterion(pred_text_output, batch_text_output)
#                         loss = loss_ce + self.cl_weight * loss_cl
#                     else:
#                         pred_text_output = self.net(batch_data)
#                         loss_ce = self.criterion(pred_text_output, batch_text_output)
#                         loss = loss_ce

#                     self.optimizer.zero_grad()  
#                     loss.backward()  
#                     clip_grad_norm_(self.net.parameters(), 2.0)
#                     self.optimizer.step()  
#                     self.scheduler.step()  

#                     text_dic = {'text_i2w': train_data['text_dic']['text_i2w'], 'ex_text_i2ws': [train_data['text_dic']['ex_text_i2ws'][k] for k in ids]}
#                     log_info=self._get_log_fit_eval(loss=loss_ce, pred_tgt=pred_text_output, gold_tgt=batch_text_output, tgt_i2w=text_dic)
                    
#                     if self.use_cl:
#                         log_info = '[Big epoch:{}/{}, CE:{:.3f}, CL:{:.3f}, {}]'.format(i + 1, self.max_big_epochs, loss_ce.item(), loss_cl.item(), log_info)
#                     else:
#                         log_info = '[Big epoch:{}/{},{}]'.format(i + 1, self.max_big_epochs, log_info)
#                     pbar.set_description(log_info)
#                     del pred_text_output,batch_text_output,batch_data

#                 del pbar
#                 if i+1 >= self.start_valid_epoch:
#                     self.max_valid_size = len(valid_data['code_graphs']) if self.max_valid_size == -1 else self.max_valid_size
#                     valid_srcs, valid_tgts, ex_text_i2ws = zip(*random.sample(list(zip(valid_data['code_graphs'], valid_data['texts'], valid_data['text_dic']['ex_text_i2ws'])), min(self.max_valid_size, len(valid_data['code_graphs']))))
#                     text_dic = {'text_i2w': train_data['text_dic']['text_i2w'], 'ex_text_i2ws': ex_text_i2ws}
#                     worse_epochs = self._do_validation(valid_srcs=valid_srcs, valid_tgts=valid_tgts, tgt_i2w=text_dic, increase_better=True, last=False)  
#                     if worse_epochs>=self.early_stop: break
#         self._do_validation(valid_srcs=valid_data['code_graphs'], valid_tgts=valid_data['texts'], tgt_i2w=valid_data['text_dic'], increase_better=True, last=True)  
#         self._logging_paramerter_num()  

#     def predict(self, code_graphs, text_dic):
#         logging.info('Predict outputs of %s' % self.model_name)
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
#         self.net.eval()  
#         enc_op=DataParallel(self.net.module.enc_op,follow_batch=['x'])
#         dec_op=torch.nn.DataParallel(self.net.module.dec_op)
#         data_set = self.Dataset(code_graphs=code_graphs, texts=None, ids=None, text_max_len=self.text_max_len, text_begin_idx=self.text_begin_idx, text_end_idx=self.text_end_idx, pad_idx=self.pad_idx)  

#         data_loader = DataListLoader(dataset=data_set, batch_size=self.pred_batch_size, shuffle=False)
#         pred_text_id_np_batches = []  
#         with torch.no_grad():  
#             pbar = tqdm(data_loader)
#             for batch_data in pbar:
#                 batch_text_input = []
#                 for data in batch_data:
#                     batch_text_input.append(data['text'].text_token_input.unsqueeze(0))
#                     del data['text']
#                 batch_text_input = torch.cat(batch_text_input, dim=0).to(device)

#                 batch_graph_enc,batch_graph_code_enc,batch_code_src_map=enc_op(batch_data)
#                 batch_text_output: list = []  
#                 if self.beam_width == 1:
#                     for i in range(self.text_max_len + 1):  
#                         pred_out = dec_op(graph_enc=batch_graph_enc,graph_code_enc=batch_graph_code_enc,code_src_map=batch_code_src_map,text_input=batch_text_input)  
#                         batch_text_output.append(pred_out[:, :, i].unsqueeze(-1).to('cpu').data.numpy())  
#                         if i < self.text_max_len:  
#                             batch_text_input[:, i + 1] = torch.argmax(pred_out[:, :, i], dim=1)
#                     batch_pred_text = np.concatenate(batch_text_output, axis=-1)[:, :, :-1]  
#                     batch_pred_text[:, self.tgt_begin_idx, :] = -np.inf  
#                     batch_pred_text[:, self.pad_idx, :] = -np.inf  
#                     batch_pred_text_np = np.argmax(batch_pred_text, axis=1)  
#                     pred_text_id_np_batches.append(batch_pred_text_np)  
#                 else:
#                     batch_pred_text=trans_beam_search(net=dec_op, beam_width=self.beam_width, dec_input_arg_name='text_input', length_penalty=1, begin_idx=self.tgt_begin_idx, pad_idx=self.pad_idx, end_idx=self.tgt_end_idx, graph_enc=batch_graph_enc, graph_code_enc=batch_graph_code_enc, code_src_map=batch_code_src_map, text_input=batch_text_input)     
#                     pred_text_id_np_batches.append(batch_pred_text.to('cpu').data.numpy()[:,:-1])  

#         pred_text_id_np = np.concatenate(pred_text_id_np_batches,axis=0)  
#         self.net.train()  
#         return self._tgt_ids2tokens(pred_text_id_np, text_dic, self.text_end_idx)
    
#     def generate_texts(self,code_graphs,text_dic,res_path,gold_texts,raw_data,token_data,**kwargs):
#         logging.info('>>>>>>>Generate the targets according to sources and save the result to {}'.format(res_path))
#         kwargs.setdefault('beam_width',1)
#         res_dir=os.path.dirname(res_path)
#         if not os.path.exists(res_dir): os.makedirs(res_dir)
#         pred_texts=self.predict(code_graphs=code_graphs, text_dic=text_dic)
#         gold_texts=self._tgt_ids2tokens(gold_texts,text_dic,self.pad_idx)
#         res_data = []
#         for i,(pred_text,gold_text,raw_item,token_item) in enumerate(zip(pred_texts,gold_texts,raw_data,token_data)):
#             sent_bleu=self.valid_metric([pred_text],[gold_text])
#             res_data.append(dict(pred_text=' '.join(pred_text), gold_text=' '.join(gold_text), sent_bleu=sent_bleu, raw_code=raw_item['code'], raw_text=raw_item['text'], id=raw_item['id'], token_text=token_item['text'],))
#         with codecs.open(res_path,'w',encoding='utf-8') as f:
#             json.dump(res_data,f,indent=4, ensure_ascii=False)
#         self._logging_paramerter_num()  
#         logging.info('>>>>>>>The result has been saved to {}'.format(res_path))

#     def _code_ids2tokens(self,code_idss, code_i2w, end_idx):
#         return [[code_i2w[idx] for idx in (code_ids[:code_ids.tolist().index(end_idx)] if end_idx in code_ids else code_ids)] for code_ids in code_idss]
    
#     def _tgt_ids2tokens(self, text_id_np, text_dic, end_idx=0, **kwargs):
#         if self.copy:
#             text_tokens: list = []
#             for j, text_ids in enumerate(text_id_np):
#                 text_i2w = {**text_dic['text_i2w'], **text_dic['ex_text_i2ws'][j]}
#                 end_i = text_ids.tolist().index(end_idx) if end_idx in text_ids else len(text_ids)
#                 text_tokens.append([text_i2w[text_idx] for text_idx in text_ids[:end_i]])
#         else:
#             text_i2w=text_dic['text_i2w']
#             text_tokens = [[text_i2w[idx] for idx in (text_ids[:text_ids.tolist().index(end_idx)] if end_idx in text_ids else text_ids)] for text_ids in text_id_np]
#         return text_tokens

# if __name__ == '__main__':
#     params.setdefault('use_directed_hyperedges', False) # 默认退回纯无向超图
#     params.setdefault('use_hyperedge_pos_emb', True)
#     params.setdefault('use_dynamic_edges', True)
#     params.setdefault('use_cl', True)
#     params.setdefault('cl_weight', 0.05)

#     logging.info('Parameters are listed below: \n'+'\n'.join(['{}: {}'.format(key,value) for key,value in params.items()]))

#     model = TModel(
#                    model_dir=params['model_dir'],
#                    model_name=params['model_name'],
#                    model_id=params['model_id'],
#                    emb_dims=params['emb_dims'],
#                    graph_gnn_layers=params['graph_gnn_layers'],
#                    graph_GNN=params['graph_GNN'],
#                    graph_gnn_aggr=params['graph_gnn_aggr'],
#                    text_att_layers=params['text_att_layers'],
#                    text_att_heads=params['text_att_heads'],
#                    text_att_head_dims=params['text_att_head_dims'],
#                    text_ff_hid_dims=params['text_ff_hid_dims'],
#                    drop_rate=params['drop_rate'],
#                    copy=params['copy'],
#                    pad_idx=params['pad_idx'],
#                    train_batch_size=params['train_batch_size'],
#                    pred_batch_size=params['pred_batch_size'],
#                    max_train_size=params['max_train_size'],  
#                    max_valid_size=params['max_valid_size'],  
#                    max_big_epochs=params['max_big_epochs'],
#                    regular_rate=params['regular_rate'],
#                    lr_base=params['lr_base'],
#                    lr_decay=params['lr_decay'],
#                    min_lr_rate=params['min_lr_rate'],
#                    warm_big_epochs=params['warm_big_epochs'],
#                    early_stop=params['early_stop'],
#                    start_valid_epoch=params['start_valid_epoch'],
#                    use_directed_hyperedges=params['use_directed_hyperedges'],
#                    use_hyperedge_pos_emb=params['use_hyperedge_pos_emb'],
#                    use_dynamic_edges=params['use_dynamic_edges'],
#                    use_cl=params['use_cl'],
#                    cl_weight=params['cl_weight'],
#                    Net=TNet,
#                    Dataset=Datasetx,
#                    beam_width=params['beam_width'],
#                    train_metrics=train_metrics,
#                    valid_metric=valid_metric,
#                    test_metrics=test_metrics,
#                    train_mode=params['train_mode'])

#     logging.info('Load data ...')
#     with codecs.open(train_avail_data_path, 'rb') as f: train_data = pickle.load(f)
#     with codecs.open(valid_avail_data_path, 'rb') as f: valid_data = pickle.load(f)
#     with codecs.open(test_avail_data_path, 'rb') as f: test_data = pickle.load(f)
#     with codecs.open(test_token_data_path,'r') as f: test_token_data=json.load(f)
#     with codecs.open(test_raw_data_path,'r') as f: test_raw_data=json.load(f)

#     model.fit(train_data=train_data, valid_data=valid_data)

#     test_eval_df=model.eval(test_srcs=test_data['code_graphs'], test_tgts=test_data['texts'], tgt_i2w=test_data['text_dic'])
#     logging.info('Model performance on test dataset:\n')
#     for i in range(0,len(test_eval_df.columns),4): print(test_eval_df.iloc[:, i:i+4])

#     model.generate_texts(code_graphs=test_data['code_graphs'], text_dic=test_data['text_dic'], res_path=res_path, gold_texts=test_data['texts'], raw_data=test_raw_data, token_data=test_token_data)




##encoder注意力残差
# coding=utf-8
# import os
# import re
# import sys
# sys.path.append('../../../')
# from lib.neural_module.learn_strategy import LrWarmUp
# from lib.neural_module.transformer import TranEnc, TranDec, DualTranDec,ResFF,ResMHA
# from lib.neural_module.embedding import PosEnc,SinusoidalPositionalEncoding
# from lib.neural_module.loss import LabelSmoothSoftmaxCEV2, CriterionNet
# from lib.neural_module.balanced_data_parallel import BalancedDataParallel
# from lib.neural_module.copy_attention import DualMultiCopyGenerator,MultiCopyGenerator,DualCopyGenerator
# from lib.neural_module.beam_search import trans_beam_search
# from lib.neural_model.seq_to_seq_model import TransSeq2Seq
# from lib.neural_model.base_model import BaseNet
# from lib.neural_module.transformer import ResFF
    
# from typing import Any,Optional,Union
# from config import *

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler
# from torch.nn.utils import clip_grad_norm_
# from torch.utils.data import Dataset
# from torch_geometric.data import HeteroData
# from torch_geometric.loader.data_list_loader import DataListLoader
# from torch_geometric.utils import to_dense_batch
# from torch_geometric.data.storage import (BaseStorage, NodeStorage,EdgeStorage)
# from torch_geometric.nn.data_parallel import DataParallel
# from torch_geometric.nn import HeteroConv,GraphNorm,HypergraphConv,GATConv

# # ===== 核心组件引用 =====
# from torch_scatter import scatter, scatter_add
# from torch_geometric.utils import softmax as scatter_softmax
# from torch_geometric.utils import degree
# import random
# import numpy as np
# import logging
# import pickle
# import json
# import codecs
# from tqdm import tqdm
# import math
# from copy import deepcopy 

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# NodeOrEdgeStorage = Union[NodeStorage, EdgeStorage]

# # ================= 1. 深度架构组件：RMSNorm & NAG-AttnRes =================
# class RMSNorm(nn.Module):
#     """防止深度增加导致的数值膨胀，Kimi论文关键组件"""
#     def __init__(self, d, eps=1e-8):
#         super(RMSNorm, self).__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(d))

#     def forward(self, x):
#         normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
#         return normed * self.weight

# class NodeAwareAttentionResidual(nn.Module):
#     """首创：节点级图注意力残差。让每个节点独立决定检索哪一层的历史特征。"""
#     def __init__(self, emb_dims):
#         super().__init__()
#         # 初始化为0，确保训练初期等价于均匀平均，保持稳定
#         self.query = nn.Parameter(torch.zeros(emb_dims))
#         self.norm = RMSNorm(emb_dims)

#     def forward(self, history_x):
#         # history_x: [Layer_num, Total_Nodes, Emb_dims]
#         if len(history_x) == 1:
#             return history_x[0]

#         V = torch.stack(history_x, dim=0) 
#         K = self.norm(V) # 对Key进行归一化防止大数值主导

#         # 计算每个节点对历史每一层的注意力分数
#         # query: [D], K: [L, N, D] -> scores: [L, N]
#         scores = torch.einsum('d, lnd -> ln', self.query, K)
#         alpha = F.softmax(scores, dim=0)

#         # 聚合历史特征
#         out = torch.einsum('ln, lnd -> nd', alpha, V)
#         return out

# # ================= 2. 图算子：防弹版 K阶扩散 & DHGAT =================
# class HyperedgeDiffusionConv(nn.Module):
#     def __init__(self, in_channels, out_channels, K=1, alpha=0.5, bias=True):
#         super(HyperedgeDiffusionConv, self).__init__()
#         self.K = K
#         self._alpha_init = alpha
#         self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
#         self.lin = nn.Linear(in_channels, out_channels, bias=bias)
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.lin.reset_parameters()
#         if hasattr(self, 'alpha'): 
#             self.alpha.data.fill_(self._alpha_init)

#     def forward(self, x, hyperedge_index, num_nodes=None, num_edges=None):
#         if num_nodes is None: num_nodes = x.size(0)
#         if num_edges is None: num_edges = int(hyperedge_index[1].max()) + 1
#         node_idx, edge_idx = hyperedge_index

#         d_v = degree(node_idx, num_nodes, dtype=x.dtype)
#         d_v[d_v == 0] = 1.0
#         d_v_inv_sqrt = d_v.pow(-0.5)
#         d_v_inv = d_v.pow(-1.0) 
#         d_v_norm_term = d_v_inv[node_idx] 

#         d_e = degree(edge_idx, num_edges, dtype=x.dtype)
#         d_e[d_e == 0] = 1.0
#         d_e_inv = d_e.pow(-1.0)

#         x = self.lin(x)
#         x_temp = x * d_v_inv_sqrt.unsqueeze(-1)
#         H_e = scatter_add(x_temp[node_idx], edge_idx, dim=0, dim_size=num_edges) * d_e_inv.unsqueeze(-1)
#         H_e_0 = H_e 

#         # 安全限位
#         cur_alpha = torch.clamp(self.alpha, min=0.0, max=1.0)
        
#         for k in range(self.K):
#             H_node_temp = H_e[edge_idx] * d_v_norm_term.unsqueeze(-1)
#             H_e_diffused = scatter_add(H_node_temp, edge_idx, dim=0, dim_size=num_edges) * d_e_inv.unsqueeze(-1)
#             H_e = (1 - cur_alpha) * H_e_diffused + cur_alpha * H_e_0

#         out = scatter_add(H_e[edge_idx], node_idx, dim=0, dim_size=num_nodes) * d_v_inv_sqrt.unsqueeze(-1)
#         return out

# def build_directed_hyperedges_from_simple(edges, group_by='src'):
#     if edges is None or np.size(edges) == 0:
#         return np.empty((2, 0), dtype=np.int64), np.empty((2, 0), dtype=np.int64)
#     src_nodes, dst_nodes = np.array(edges[0]), np.array(edges[1])
#     src_hyper_edges, dst_hyper_edges = [], []
#     if group_by == 'src': # 父传子
#         unique_srcs = np.unique(src_nodes)
#         for h_id, src in enumerate(unique_srcs):
#             src_hyper_edges.append([src, h_id])
#             for child in dst_nodes[src_nodes == src]: dst_hyper_edges.append([child, h_id])  
#     else: # DFG 汇聚
#         unique_dsts = np.unique(dst_nodes)
#         for h_id, dst in enumerate(unique_dsts):
#             dst_hyper_edges.append([dst, h_id])
#             for parent in src_nodes[dst_nodes == dst]: src_hyper_edges.append([parent, h_id])
#     return np.array(src_hyper_edges, dtype=np.int64).T, np.array(dst_hyper_edges, dtype=np.int64).T

# class DirectedHypergraphAttention(nn.Module):
#     def __init__(self, emb_dims):
#         super().__init__()
#         self.att_src = nn.Linear(emb_dims, 1, bias=False)
#         self.att_dst = nn.Linear(emb_dims, 1, bias=False)
#         self.leaky_relu = nn.LeakyReLU(0.2)
#         self.out_proj = nn.Linear(emb_dims, emb_dims, bias=False)

#     def forward(self, x, edge_index_src, edge_index_dst):
#         if edge_index_src.numel() == 0: return torch.zeros_like(x)
#         src_nodes, src_edges = edge_index_src[0], edge_index_src[1]
#         dst_nodes, dst_edges = edge_index_dst[0], edge_index_dst[1]
#         num_he = max(src_edges.max().item(), dst_edges.max().item()) + 1
        
#         alpha_src = scatter_softmax(self.leaky_relu(self.att_src(x[src_nodes])), src_edges, dim=0)
#         he_x = scatter(x[src_nodes] * alpha_src, src_edges, dim=0, dim_size=num_he, reduce='sum')
        
#         alpha_dst = scatter_softmax(self.leaky_relu(self.att_dst(he_x[dst_edges] + x[dst_nodes])), dst_nodes, dim=0)
#         out = scatter(he_x[dst_edges] * alpha_dst, dst_nodes, dim=0, dim_size=x.size(0), reduce='sum')
#         return self.out_proj(out)

# # ================= 3. 数据预处理 =================
# class Datax(HeteroData):
#     def __cat_dim__(self, key: str, value: Any, store: Optional[NodeOrEdgeStorage] = None, *args, **kwargs) -> Any:
#         if bool(re.search('(token)', key)): return None  
#         if bool(re.search('(pos)', key)): return -1
#         return super().__cat_dim__(key, value,store)    

# class Datasetx(Dataset):
#     def __init__(self, code_graphs, texts=None, ids=None, text_max_len=None, text_begin_idx=1, text_end_idx=2, pad_idx=0):
#         self.len = len(code_graphs)  
#         self.text_max_len = max([len(t) for t in texts]) if text_max_len is None and texts is not None else text_max_len
#         self.text_begin_idx, self.text_end_idx, self.pad_idx = text_begin_idx, text_end_idx, pad_idx
#         self.code_graphs, self.texts, self.ids = code_graphs, texts, ids

#     def __getitem__(self, index):
#         if self.texts is None:
#             pad_text_in = np.zeros((self.text_max_len + 1,), dtype=np.int64)
#             pad_text_in[0] = self.text_begin_idx
#             pad_text_out = None
#         else:
#             tru_text = self.texts[index][:self.text_max_len]
#             pad_text_in = np.lib.pad(tru_text, (1, self.text_max_len - len(tru_text)), 'constant', constant_values=(self.text_begin_idx, self.pad_idx))
#             tru_text_out = np.lib.pad(np.lib.pad(tru_text, (0, 1), 'constant', constant_values=(0, self.text_end_idx)), (0, self.text_max_len + 1 - (len(tru_text)+1)), 'constant', constant_values=(self.pad_idx, self.pad_idx))
#             pad_text_out = tru_text_out
            
#         data=Datax()
#         cg = self.code_graphs[index]
#         data['node'].x = torch.tensor(cg['nodes'])
#         data['node'].src_map = torch.tensor(cg['node2text_map_ids']).long()
#         data['node'].code_mask = torch.tensor(cg['code_node_mask']).bool()
        
#         # 修正拼写：parent_child_hyperedges
#         for key in ['parent_child_hyperedges', 'line_hyperedges', 'block_hyperedges', 'dfg_hyperedges', 'layout_sibling_hyperedges']:
#             data['node', key, 'node'].edge_index = torch.tensor(cg.get(key, [])).long()

#         # 提取有向超边数据
#         ast_s, ast_d = build_directed_hyperedges_from_simple(cg.get('base_father2child_edges', []), 'src')
#         data['node', 'ast_dir_s', 'node'].edge_index, data['node', 'ast_dir_d', 'node'].edge_index = torch.tensor(ast_s).long(), torch.tensor(ast_d).long()
#         dfg_s, dfg_d = build_directed_hyperedges_from_simple(cg.get('dfg_prev2next_edges', []), 'dst')
#         data['node', 'dfg_dir_s', 'node'].edge_index, data['node', 'dfg_dir_d', 'node'].edge_index = torch.tensor(dfg_s).long(), torch.tensor(dfg_d).long()

#         data['text'].text_token_input = torch.tensor(pad_text_in).long()
#         if self.texts is not None: data['text'].text_token_output = torch.tensor(pad_text_out).long()
#         data['text'].num_nodes = pad_text_in.shape[0]
#         if self.ids is not None:
#             data['idx'].idx = torch.tensor(self.ids[index]); data['idx'].num_nodes = 1
#         return data

#     def __len__(self): return self.len

# # ================= 4. 核心网络：CodeGraphEnc & TNet =================
# class CodeGraphEnc(nn.Module):
#     def __init__(self, emb_dims, graph_max_size, code_max_len, graph_node_emb_op, graph_gnn_layers=6, drop_rate=0., **kwargs):
#         super().__init__()
#         self.graph_max_size, self.code_max_len, self.emb_dims = graph_max_size, code_max_len, emb_dims
#         self.pad_idx = kwargs.get('pad_idx', 0)
#         self.use_hyperedge_pos_emb = kwargs.get('use_hyperedge_pos_emb', True)
#         self.use_directed_hyperedges = kwargs.get('use_directed_hyperedges', True)
#         self.use_dynamic_edges = kwargs.get('use_dynamic_edges', True)
#         self.dynamic_threshold = kwargs.get('dynamic_threshold', 0.85)
#         self.gnn_layers = graph_gnn_layers

#         self.graph_node_emb_op = graph_node_emb_op
#         self.graph_pos_encoding = nn.Embedding(graph_max_size * 2 + 1, emb_dims, padding_idx=self.pad_idx)
#         nn.init.xavier_uniform_(self.graph_pos_encoding.weight[1:, ])
#         self.emb_drop_op = nn.Dropout(p=drop_rate)
        
#         # NAG-AttnRes 模块
#         self.use_attn_res = kwargs.get('use_attn_res', True)
#         if self.use_attn_res:
#             self.attn_res_ops = nn.ModuleList([NodeAwareAttentionResidual(emb_dims) for _ in range(graph_gnn_layers)])

#         self.gnn_ops, self.dhgat_ops, self.gnorm_ops, self.grelu_ops = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
#         # 取消 Softmax，直接用可学习标量维持信号强度
#         self.hetero_alpha = nn.Parameter(torch.ones(graph_gnn_layers, 8))
        
#         for _ in range(graph_gnn_layers):
#             self.gnn_ops.append(HeteroConv({
#                 ('node', 'block_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, K=1),
#                 ('node', 'line_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, K=1),
#                 ('node', 'layout_sibling_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, K=1),
#                 ('node', 'parent_child_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, K=4),
#                 ('node', 'dfg_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, K=1),
#                 ('node', 'dynamic_semantic_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, K=1)
#             }, aggr='sum'))
#             self.dhgat_ops.append(nn.ModuleDict({
#                 'ast_dir': DirectedHypergraphAttention(emb_dims),
#                 'dfg_dir': DirectedHypergraphAttention(emb_dims)
#             }))
#             self.grelu_ops.append(nn.Sequential(nn.ReLU(), nn.Dropout(p=drop_rate)))
#             self.gnorm_ops.append(GraphNorm(emb_dims))

#     def _add_dynamic_edges(self, data):
#         dense_x, mask = to_dense_batch(data['node'].x, data.x_batch_dict['node'], fill_value=0.0) 
#         sim_matrix = torch.bmm(F.normalize(dense_x, p=2, dim=-1), F.normalize(dense_x, p=2, dim=-1).transpose(1, 2)) 
#         sim_matrix.diagonal(dim1=1, dim2=2).fill_(-1.0)
#         adj = (sim_matrix > self.dynamic_threshold) & (mask.unsqueeze(1) & mask.unsqueeze(2))
#         b_idx, r, c = adj.nonzero(as_tuple=True)
#         flat = torch.zeros_like(mask, dtype=torch.long); flat[mask] = torch.arange(mask.sum(), device=mask.device)
#         data.edge_index_dict[('node', 'dynamic_semantic_hyperedges', 'node')] = torch.stack([flat[b_idx, r], flat[b_idx, c]], dim=0)

#     def forward(self, data):
#         graph_node_emb = self.graph_node_emb_op(data.x_dict['node']) * np.sqrt(self.emb_dims)
#         batch_size = data.x_batch_dict['node'].max().item() + 1
#         pos_indices_list = []
#         for b in range(batch_size):
#             num = (data.x_batch_dict['node'] == b).sum().item()
#             max_p = self.graph_pos_encoding.num_embeddings - 1
#             pos_indices_list.append((torch.arange(1, num + 1, device=graph_node_emb.device) % max_p) + 1)
#         pos_emb = self.graph_pos_encoding(torch.cat(pos_indices_list)) 
        
#         # 烙印初始位置
#         if self.use_hyperedge_pos_emb: 
#             graph_node_emb = graph_node_emb + pos_emb
            
#         data['node'].x = self.emb_drop_op(graph_node_emb) 
        
#         history_x = [data['node'].x]
        
#         for i, (gnn, dhgat, relu, norm) in enumerate(zip(self.gnn_ops, self.dhgat_ops, self.grelu_ops, self.gnorm_ops)):
#             if self.use_dynamic_edges and i == (self.gnn_layers // 2): 
#                 self._add_dynamic_edges(data)
            
#             # NAG-AttnRes：从历史中按需检索
#             x_input = self.attn_res_ops[i](history_x) if self.use_attn_res else history_x[-1]
#             data['node'].x = x_input
            
#             x_dict = gnn(x_dict=data.x_dict, edge_index_dict=data.edge_index_dict)
#             out_x = x_dict['node']
            
#             if self.use_directed_hyperedges:
#                 a_s, a_d = data.edge_index_dict.get(('node','ast_dir_s','node')), data.edge_index_dict.get(('node','ast_dir_d','node'))
#                 if a_s is not None and a_s.numel()>0: 
#                     out_x = out_x + self.hetero_alpha[i, 4] * dhgat['ast_dir'](x_input, a_s, a_d)
#                 d_s, d_d = data.edge_index_dict.get(('node','dfg_dir_s','node')), data.edge_index_dict.get(('node','dfg_dir_d','node'))
#                 if d_s is not None and d_s.numel()>0: 
#                     out_x = out_x + self.hetero_alpha[i, 5] * dhgat['dfg_dir'](x_input, d_s, d_d)

#             # 残差与归一化
#             data['node'].x = norm(x_input + relu(out_x))
#             history_x.append(data['node'].x)

#         # 赋予最终坐标用于Copy机制
#         data['node'].x = history_x[-1] + pos_emb
        
#         code_x_batch = data.x_batch_dict['node'][data['node'].code_mask==True]   
#         graph_enc,_ = to_dense_batch(data.x_dict['node'], batch=data.x_batch_dict['node'], fill_value=self.pad_idx, max_num_nodes=self.graph_max_size)  
#         cm = data['node'].code_mask; cb = data.x_batch_dict['node'][cm]
#         code_src_map,_ = to_dense_batch(data.src_map_dict['node'][cm], batch=cb, fill_value=self.pad_idx, max_num_nodes=self.code_max_len)    
#         graph_code_enc,_ = to_dense_batch(data.x_dict['node'][cm], batch=cb, fill_value=self.pad_idx, max_num_nodes=self.code_max_len)    
#         return graph_enc, graph_code_enc, code_src_map

# class Dec(nn.Module):
#     def __init__(self, emb_dims, text_voc_size, text_emb_op, text_max_len, enc_out_dims, att_layers, att_heads, att_head_dims=None, ff_hid_dims=2048, drop_rate=0., **kwargs):
#         super().__init__()
#         self.emb_dims = emb_dims
#         self._copy = kwargs.get('copy', True)
#         self.text_emb_op = text_emb_op
#         self.pos_encoding = PosEnc(max_len=text_max_len+1, emb_dims=emb_dims, train=True, pad=True, pad_idx=kwargs.get('pad_idx', 0)) 
#         self.emb_layer_norm = nn.LayerNorm(emb_dims)
#         self.text_dec_op = TranDec(query_dims=emb_dims, key_dims=enc_out_dims, head_nums=att_heads, head_dims=att_head_dims, layer_num=att_layers, ff_hid_dims=ff_hid_dims, drop_rate=drop_rate, pad_idx=kwargs.get('pad_idx', 0), self_causality=True)
#         self.dropout = nn.Dropout(p=drop_rate)
#         self.out_fc = nn.Linear(emb_dims, text_voc_size)
#         self.copy_generator = MultiCopyGenerator(tgt_dims=emb_dims, tgt_voc_size=text_voc_size, src_dims=enc_out_dims, att_heads=att_heads, att_head_dims=att_head_dims, drop_rate=drop_rate, pad_idx=kwargs.get('pad_idx', 0))

#     def forward(self, graph_enc, graph_code_enc, code_src_map, text_input):
#         text_dec = self.emb_layer_norm(self.dropout(self.text_emb_op(text_input)*np.sqrt(self.emb_dims) + self.pos_encoding(text_input)))
#         text_dec = self.text_dec_op(query=text_dec, key=graph_enc, query_mask=text_input.abs().sign(), key_mask=graph_enc.abs().sum(-1).sign())  
#         if not self._copy:
#             return self.out_fc(text_dec).transpose(1, 2)
#         return self.copy_generator(text_dec, graph_code_enc, code_src_map).transpose(1, 2)

# class TNet(BaseNet):
#     def __init__(self, emb_dims, graph_max_size, code_max_len, text_max_len, io_voc_size, text_voc_size, graph_gnn_layers=6, drop_rate=0., **kwargs):
#         super().__init__()
#         self.use_cl, self.cl_temp, self.edge_drop_rate = kwargs.get('use_cl',True), kwargs.get('cl_temp',0.1), kwargs.get('edge_drop_rate',0.15)
#         io_emb = nn.Embedding(io_voc_size, emb_dims, padding_idx=kwargs.get('pad_idx',0)); nn.init.xavier_uniform_(io_emb.weight[1:, ])
#         self.enc_op = CodeGraphEnc(emb_dims, graph_max_size, code_max_len, io_emb, graph_gnn_layers, drop_rate, **kwargs)
#         self.dec_op = Dec(emb_dims, text_voc_size, io_emb, text_max_len, emb_dims, kwargs.get('text_att_layers',8), kwargs.get('text_att_heads',8), ff_hid_dims=kwargs.get('text_ff_hid_dims',2048), drop_rate=drop_rate, **kwargs)

#     def augment(self, data):
#         aug = deepcopy(data)
#         for et in aug.edge_index_dict.keys():
#             idx = aug.edge_index_dict[et]
#             if idx.numel() > 0: 
#                 aug.edge_index_dict[et] = idx[:, torch.rand(idx.size(1), device=idx.device) > self.edge_drop_rate]
#         return aug

#     def forward(self, data):
#         text_in = data['text'].text_token_input.clone()
#         del data['text']
#         if self.training and self.use_cl:
#             cg_orig, cg_aug = deepcopy(data), self.augment(data)
#             g_enc, g_code, src_m = self.enc_op(cg_orig)
#             g_enc_a, _, _ = self.enc_op(cg_aug)
#             out = self.dec_op(g_enc, g_code, src_m, text_in)
#             z1, z2 = F.normalize(g_enc.mean(1), p=2, dim=-1), F.normalize(g_enc_a.mean(1), p=2, dim=-1)
#             sim = torch.matmul(z1, z2.T) / self.cl_temp
#             loss_cl = (F.cross_entropy(sim, torch.arange(z1.size(0), device=z1.device)) + F.cross_entropy(sim.T, torch.arange(z1.size(0), device=z1.device))) / 2
#             return out, loss_cl
        
#         g_enc, g_code, src_m = self.enc_op(data)
#         return self.dec_op(g_enc, g_code, src_m, text_in)

# class TModel(TransSeq2Seq):
#     def __init__(self, model_dir, model_name='Transformer_based_model', model_id=None, emb_dims=512, graph_gnn_layers=3, graph_GNN=SAGEConv, graph_gnn_aggr='add', text_att_layers=3, text_att_heads=8, text_att_head_dims=None, text_ff_hid_dims=2048, drop_rate=0., copy=True, pad_idx=0, train_batch_size=32, pred_batch_size=32, max_train_size=-1, max_valid_size=32 * 10, max_big_epochs=20, regular_rate=1e-5, lr_base=0.001, lr_decay=0.9, min_lr_rate=0.01, warm_big_epochs=2, start_valid_epoch=20, early_stop=20, Net=TNet, Dataset=Datasetx, beam_width=1, train_metrics=[get_sent_bleu], valid_metric=get_sent_bleu, test_metrics=[get_sent_bleu], train_mode=True, **kwargs):
#         logging.info('Construct %s' % model_name)
#         super().__init__(model_name=model_name, model_dir=model_dir, model_id=model_id)
#         self.init_params = locals()
#         self.emb_dims = emb_dims
#         self.graph_gnn_layers = graph_gnn_layers
#         self.graph_GNN = graph_GNN
#         self.graph_gnn_aggr = graph_gnn_aggr
#         self.text_att_layers = text_att_layers
#         self.text_att_heads = text_att_heads
#         self.text_att_head_dims = text_att_head_dims
#         self.text_ff_hid_dims = text_ff_hid_dims
#         self.drop_rate = drop_rate
#         self.pad_idx = pad_idx
#         self.copy = copy
#         self.train_batch_size = train_batch_size
#         self.pred_batch_size = pred_batch_size
#         self.max_train_size = max_train_size
#         self.max_valid_size = max_valid_size
#         self.max_big_epochs = max_big_epochs
#         self.regular_rate = regular_rate
#         self.lr_base = lr_base
#         self.lr_decay = lr_decay
#         self.min_lr_rate = min_lr_rate
#         self.warm_big_epochs = warm_big_epochs
#         self.start_valid_epoch = start_valid_epoch
#         self.early_stop = early_stop
#         self.Net = Net
#         self.Dataset = Dataset
#         self.beam_width = beam_width
#         self.train_metrics = train_metrics
#         self.valid_metric = valid_metric
#         self.test_metrics = test_metrics
#         self.train_mode = train_mode
        
#         # 接收 kwargs 参数
#         self.use_attn_res = kwargs.get('use_attn_res', True)
#         self.use_directed_hyperedges = kwargs.get('use_directed_hyperedges', True)
#         self.use_hyperedge_pos_emb = kwargs.get('use_hyperedge_pos_emb', True)
#         self.use_dynamic_edges = kwargs.get('use_dynamic_edges', True)
#         self.use_cl = kwargs.get('use_cl', True)
#         self.cl_weight = kwargs.get('cl_weight', 0.05)
#         self.cl_temp = kwargs.get('cl_temp', 0.1)
#         self.edge_drop_rate = kwargs.get('edge_drop_rate', 0.15)
#         self.dynamic_threshold = kwargs.get('dynamic_threshold', 0.85)

#     def _logging_paramerter_num(self):
#         logging.info("{} have {} paramerters in total".format(self.model_name, sum( x.numel() for x in self.net.parameters() if x.requires_grad)))
#         code_graph_enc_param_num = sum(x.numel() for x in self.net.module.enc_op.gnn_ops.parameters() if x.requires_grad) + sum(x.numel() for x in self.net.module.enc_op.gnorm_ops.parameters() if x.requires_grad) + sum(x.numel() for x in self.net.module.enc_op.grelu_ops.parameters() if x.requires_grad)
#         text_dec_param_num = sum(x.numel() for x in self.net.module.dec_op.text_dec_op.parameters() if x.requires_grad)
#         enc_dec_param_num = code_graph_enc_param_num + text_dec_param_num
#         logging.info("{} have {} paramerters in encoder and decoder".format(self.model_name, enc_dec_param_num))

#     def fit(self, train_data, valid_data, **kwargs):
#         self.graph_max_size, self.code_max_len, self.io_voc_size, self.text_max_len=0, 0, 0, 0
#         for code_graph,text in zip(train_data['code_graphs'],train_data['texts']):
#             self.graph_max_size = max(self.graph_max_size,len(code_graph['nodes']))
#             self.code_max_len = max(self.code_max_len,code_graph['code_node_mask'].sum())
#             self.io_voc_size = max(self.io_voc_size,max(code_graph['nodes']))
#             self.text_max_len=max(self.text_max_len,len(text))
#         self.io_voc_size+=1

#         self.text_voc_size = len(train_data['text_dic']['text_i2w']) 
#         self.io_voc_size=max(self.io_voc_size,self.text_voc_size+2*self.code_max_len)
        
#         net = self.Net(
#             emb_dims=self.emb_dims, graph_max_size=self.graph_max_size, code_max_len=self.code_max_len, text_max_len=self.text_max_len, io_voc_size=self.io_voc_size, text_voc_size=self.text_voc_size, graph_gnn_layers=self.graph_gnn_layers, graph_GNN=self.graph_GNN, graph_gnn_aggr=self.graph_gnn_aggr,
#             text_att_layers=self.text_att_layers, text_att_heads=self.text_att_heads, text_att_head_dims=self.text_att_head_dims, text_ff_hid_dims=self.text_ff_hid_dims, 
#             drop_rate=self.drop_rate, pad_idx=self.pad_idx, copy=self.copy, 
#             use_attn_res=self.use_attn_res,
#             use_hyperedge_pos_emb=self.use_hyperedge_pos_emb, 
#             use_directed_hyperedges=self.use_directed_hyperedges, 
#             use_dynamic_edges=self.use_dynamic_edges, 
#             use_cl=self.use_cl,
#             cl_weight=self.cl_weight,
#             cl_temp=self.cl_temp,
#             edge_drop_rate=self.edge_drop_rate,
#             dynamic_threshold=self.dynamic_threshold
#         )
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
#         self.net =DataParallel(net.to(device),follow_batch=['x'])  
#         self._logging_paramerter_num()  
#         self.net.train()  

#         self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr_base, weight_decay=self.regular_rate)
#         self.criterion = LabelSmoothSoftmaxCEV2(reduction='mean', ignore_index=self.pad_idx, label_smooth=0.0)

#         self.text_begin_idx = self.text_voc_size - 1
#         self.text_end_idx = self.text_voc_size - 2
#         self.tgt_begin_idx,self.tgt_end_idx=self.text_begin_idx,self.text_end_idx

#         self.max_train_size = len(train_data['code_graphs']) if self.max_train_size == -1 else self.max_train_size
#         train_code_graphs, train_texts,train_ids = zip(*random.sample(list(zip(train_data['code_graphs'], train_data['texts'],train_data['ids'])), min(self.max_train_size, len(train_data['code_graphs']))))
#         train_set = self.Dataset(code_graphs=train_code_graphs, texts=train_texts, ids=train_ids, text_max_len=self.text_max_len, text_begin_idx=self.text_begin_idx, text_end_idx=self.text_end_idx, pad_idx=self.pad_idx)
#         train_loader=DataListLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True, drop_last=True) 

#         if self.warm_big_epochs is None: self.warm_big_epochs = max(self.max_big_epochs // 10, 2)
#         self.scheduler = LrWarmUp(self.optimizer, min_rate=self.min_lr_rate, lr_decay=self.lr_decay, warm_steps=self.warm_big_epochs * len(train_loader), reduce_steps=len(train_loader))  
        
#         if self.train_mode:  
#             for i in range(0,self.max_big_epochs):
#                 pbar = tqdm(train_loader)
#                 for j, batch_data in enumerate(pbar):
#                     batch_text_output = []
#                     ids=[]
#                     for data in batch_data:
#                         batch_text_output.append(data['text'].text_token_output.unsqueeze(0))
#                         del data['text'].text_token_output
#                         ids.append(data['idx'].idx.item())
#                         del data['idx']

#                     batch_text_output = torch.cat(batch_text_output, dim=0).to(device)
                    
#                     if self.use_cl:
#                         pred_text_output, loss_cl = self.net(batch_data)
#                         loss_cl = loss_cl.mean()
#                         loss_ce = self.criterion(pred_text_output, batch_text_output)
#                         loss = loss_ce + self.cl_weight * loss_cl
#                     else:
#                         pred_text_output = self.net(batch_data)
#                         loss_ce = self.criterion(pred_text_output, batch_text_output)
#                         loss = loss_ce

#                     self.optimizer.zero_grad()  
#                     loss.backward()  
#                     clip_grad_norm_(self.net.parameters(), 2.0)
#                     self.optimizer.step()  
#                     self.scheduler.step()  

#                     text_dic = {'text_i2w': train_data['text_dic']['text_i2w'], 'ex_text_i2ws': [train_data['text_dic']['ex_text_i2ws'][k] for k in ids]}
#                     log_info=self._get_log_fit_eval(loss=loss_ce, pred_tgt=pred_text_output, gold_tgt=batch_text_output, tgt_i2w=text_dic)
                    
#                     if self.use_cl:
#                         log_info = '[Big epoch:{}/{}, CE:{:.3f}, CL:{:.3f}, {}]'.format(i + 1, self.max_big_epochs, loss_ce.item(), loss_cl.item(), log_info)
#                     else:
#                         log_info = '[Big epoch:{}/{},{}]'.format(i + 1, self.max_big_epochs, log_info)
#                     pbar.set_description(log_info)
#                     del pred_text_output,batch_text_output,batch_data

#                 del pbar
#                 if i+1 >= self.start_valid_epoch:
#                     self.max_valid_size = len(valid_data['code_graphs']) if self.max_valid_size == -1 else self.max_valid_size
#                     valid_srcs, valid_tgts, ex_text_i2ws = zip(*random.sample(list(zip(valid_data['code_graphs'], valid_data['texts'], valid_data['text_dic']['ex_text_i2ws'])), min(self.max_valid_size, len(valid_data['code_graphs']))))
#                     text_dic = {'text_i2w': train_data['text_dic']['text_i2w'], 'ex_text_i2ws': ex_text_i2ws}
#                     worse_epochs = self._do_validation(valid_srcs=valid_srcs, valid_tgts=valid_tgts, tgt_i2w=text_dic, increase_better=True, last=False)  
#                     if worse_epochs>=self.early_stop: break
#         self._do_validation(valid_srcs=valid_data['code_graphs'], valid_tgts=valid_data['texts'], tgt_i2w=valid_data['text_dic'], increase_better=True, last=True)  
#         self._logging_paramerter_num()  

#     def predict(self, code_graphs, text_dic):
#         logging.info('Predict outputs of %s' % self.model_name)
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
#         self.net.eval()  
#         enc_op=DataParallel(self.net.module.enc_op,follow_batch=['x'])
#         dec_op=torch.nn.DataParallel(self.net.module.dec_op)
#         data_set = self.Dataset(code_graphs=code_graphs, texts=None, ids=None, text_max_len=self.text_max_len, text_begin_idx=self.text_begin_idx, text_end_idx=self.text_end_idx, pad_idx=self.pad_idx)  

#         data_loader = DataListLoader(dataset=data_set, batch_size=self.pred_batch_size, shuffle=False)
#         pred_text_id_np_batches = []  
#         with torch.no_grad():  
#             pbar = tqdm(data_loader)
#             for batch_data in pbar:
#                 batch_text_input = []
#                 for data in batch_data:
#                     batch_text_input.append(data['text'].text_token_input.unsqueeze(0))
#                     del data['text']
#                 batch_text_input = torch.cat(batch_text_input, dim=0).to(device)

#                 batch_graph_enc,batch_graph_code_enc,batch_code_src_map=enc_op(batch_data)
#                 batch_text_output: list = []  
#                 if self.beam_width == 1:
#                     for i in range(self.text_max_len + 1):  
#                         pred_out = dec_op(graph_enc=batch_graph_enc,graph_code_enc=batch_graph_code_enc,code_src_map=batch_code_src_map,text_input=batch_text_input)  
#                         batch_text_output.append(pred_out[:, :, i].unsqueeze(-1).to('cpu').data.numpy())  
#                         if i < self.text_max_len:  
#                             batch_text_input[:, i + 1] = torch.argmax(pred_out[:, :, i], dim=1)
#                     batch_pred_text = np.concatenate(batch_text_output, axis=-1)[:, :, :-1]  
#                     batch_pred_text[:, self.tgt_begin_idx, :] = -np.inf  
#                     batch_pred_text[:, self.pad_idx, :] = -np.inf  
#                     batch_pred_text_np = np.argmax(batch_pred_text, axis=1)  
#                     pred_text_id_np_batches.append(batch_pred_text_np)  
#                 else:
#                     batch_pred_text=trans_beam_search(net=dec_op, beam_width=self.beam_width, dec_input_arg_name='text_input', length_penalty=1, begin_idx=self.tgt_begin_idx, pad_idx=self.pad_idx, end_idx=self.tgt_end_idx, graph_enc=batch_graph_enc, graph_code_enc=batch_graph_code_enc, code_src_map=batch_code_src_map, text_input=batch_text_input)     
#                     pred_text_id_np_batches.append(batch_pred_text.to('cpu').data.numpy()[:,:-1])  

#         pred_text_id_np = np.concatenate(pred_text_id_np_batches,axis=0)  
#         self.net.train()  
#         return self._tgt_ids2tokens(pred_text_id_np, text_dic, self.text_end_idx)
    
#     def generate_texts(self,code_graphs,text_dic,res_path,gold_texts,raw_data,token_data,**kwargs):
#         logging.info('>>>>>>>Generate the targets according to sources and save the result to {}'.format(res_path))
#         kwargs.setdefault('beam_width',1)
#         res_dir=os.path.dirname(res_path)
#         if not os.path.exists(res_dir): os.makedirs(res_dir)
#         pred_texts=self.predict(code_graphs=code_graphs, text_dic=text_dic)
#         gold_texts=self._tgt_ids2tokens(gold_texts,text_dic,self.pad_idx)
#         res_data = []
#         for i,(pred_text,gold_text,raw_item,token_item) in enumerate(zip(pred_texts,gold_texts,raw_data,token_data)):
#             sent_bleu=self.valid_metric([pred_text],[gold_text])
#             res_data.append(dict(pred_text=' '.join(pred_text), gold_text=' '.join(gold_text), sent_bleu=sent_bleu, raw_code=raw_item['code'], raw_text=raw_item['text'], id=raw_item['id'], token_text=token_item['text'],))
#         with codecs.open(res_path,'w',encoding='utf-8') as f:
#             json.dump(res_data,f,indent=4, ensure_ascii=False)
#         self._logging_paramerter_num()  
#         logging.info('>>>>>>>The result has been saved to {}'.format(res_path))

#     def _code_ids2tokens(self,code_idss, code_i2w, end_idx):
#         return [[code_i2w[idx] for idx in (code_ids[:code_ids.tolist().index(end_idx)] if end_idx in code_ids else code_ids)] for code_ids in code_idss]
    
#     def _tgt_ids2tokens(self, text_id_np, text_dic, end_idx=0, **kwargs):
#         if self.copy:
#             text_tokens: list = []
#             for j, text_ids in enumerate(text_id_np):
#                 text_i2w = {**text_dic['text_i2w'], **text_dic['ex_text_i2ws'][j]}
#                 end_i = text_ids.tolist().index(end_idx) if end_idx in text_ids else len(text_ids)
#                 text_tokens.append([text_i2w[text_idx] for text_idx in text_ids[:end_i]])
#         else:
#             text_i2w=text_dic['text_i2w']
#             text_tokens = [[text_i2w[idx] for idx in (text_ids[:text_ids.tolist().index(end_idx)] if end_idx in text_ids else text_ids)] for text_ids in text_id_np]
#         return text_tokens

# if __name__ == '__main__':
#     # 从 params 中提取开关并默认填充
#     params.setdefault('use_attn_res', True)
#     params.setdefault('use_directed_hyperedges', True) 
#     params.setdefault('use_hyperedge_pos_emb', True)
#     params.setdefault('use_dynamic_edges', True)
#     params.setdefault('use_cl', True)
#     params.setdefault('cl_weight', 0.05)
#     params.setdefault('cl_temp', 0.1)
#     params.setdefault('edge_drop_rate', 0.15)
#     params.setdefault('dynamic_threshold', 0.85)

#     logging.info('Parameters are listed below: \n'+'\n'.join(['{}: {}'.format(key,value) for key,value in params.items()]))

#     model = TModel(
#                    model_dir=params['model_dir'],
#                    model_name=params['model_name'],
#                    model_id=params['model_id'],
#                    emb_dims=params['emb_dims'],
#                    graph_gnn_layers=params['graph_gnn_layers'],
#                    graph_GNN=params['graph_GNN'],
#                    graph_gnn_aggr=params['graph_gnn_aggr'],
#                    text_att_layers=params['text_att_layers'],
#                    text_att_heads=params['text_att_heads'],
#                    text_att_head_dims=params['text_att_head_dims'],
#                    text_ff_hid_dims=params['text_ff_hid_dims'],
#                    drop_rate=params['drop_rate'],
#                    copy=params['copy'],
#                    pad_idx=params['pad_idx'],
#                    train_batch_size=params['train_batch_size'],
#                    pred_batch_size=params['pred_batch_size'],
#                    max_train_size=params['max_train_size'],  
#                    max_valid_size=params['max_valid_size'],  
#                    max_big_epochs=params['max_big_epochs'],
#                    regular_rate=params['regular_rate'],
#                    lr_base=params['lr_base'],
#                    lr_decay=params['lr_decay'],
#                    min_lr_rate=params['min_lr_rate'],
#                    warm_big_epochs=params['warm_big_epochs'],
#                    early_stop=params['early_stop'],
#                    start_valid_epoch=params['start_valid_epoch'],
#                    # ===== 注入大招参数 =====
#                    use_attn_res=params['use_attn_res'],
#                    use_directed_hyperedges=params['use_directed_hyperedges'],
#                    use_hyperedge_pos_emb=params['use_hyperedge_pos_emb'],
#                    use_dynamic_edges=params['use_dynamic_edges'],
#                    use_cl=params['use_cl'],
#                    cl_weight=params['cl_weight'],
#                    cl_temp=params['cl_temp'],
#                    edge_drop_rate=params['edge_drop_rate'],
#                    dynamic_threshold=params['dynamic_threshold'],
#                    Net=TNet,
#                    Dataset=Datasetx,
#                    beam_width=params['beam_width'],
#                    train_metrics=train_metrics,
#                    valid_metric=valid_metric,
#                    test_metrics=test_metrics,
#                    train_mode=params['train_mode'])

#     logging.info('Load data ...')
#     with codecs.open(train_avail_data_path, 'rb') as f: train_data = pickle.load(f)
#     with codecs.open(valid_avail_data_path, 'rb') as f: valid_data = pickle.load(f)
#     with codecs.open(test_avail_data_path, 'rb') as f: test_data = pickle.load(f)
#     with codecs.open(test_token_data_path,'r') as f: test_token_data=json.load(f)
#     with codecs.open(test_raw_data_path,'r') as f: test_raw_data=json.load(f)

#     model.fit(train_data=train_data, valid_data=valid_data)

#     test_eval_df=model.eval(test_srcs=test_data['code_graphs'], test_tgts=test_data['texts'], tgt_i2w=test_data['text_dic'])
#     logging.info('Model performance on test dataset:\n')
#     for i in range(0,len(test_eval_df.columns),4): print(test_eval_df.iloc[:, i:i+4])

#     model.generate_texts(code_graphs=test_data['code_graphs'], text_dic=test_data['text_dic'], res_path=res_path, gold_texts=test_data['texts'], raw_data=test_raw_data, token_data=test_token_data)



#双端注意力残差

# # coding=utf-8
# import os
# import re
# import sys
# sys.path.append('../../../')
# from lib.neural_module.learn_strategy import LrWarmUp
# from lib.neural_module.transformer import TranEnc, TranDec, DualTranDec,ResFF,ResMHA
# from lib.neural_module.embedding import PosEnc,SinusoidalPositionalEncoding
# from lib.neural_module.loss import LabelSmoothSoftmaxCEV2, CriterionNet
# from lib.neural_module.balanced_data_parallel import BalancedDataParallel
# from lib.neural_module.copy_attention import DualMultiCopyGenerator,MultiCopyGenerator,DualCopyGenerator
# from lib.neural_module.beam_search import trans_beam_search
# from lib.neural_model.seq_to_seq_model import TransSeq2Seq
# from lib.neural_model.base_model import BaseNet
# from lib.neural_module.transformer import ResFF
    
# from typing import Any,Optional,Union
# from config import *

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler
# from torch.nn.utils import clip_grad_norm_
# from torch.utils.data import Dataset
# from torch_geometric.data import HeteroData
# from torch_geometric.loader.data_list_loader import DataListLoader
# from torch_geometric.utils import to_dense_batch
# from torch_geometric.data.storage import (BaseStorage, NodeStorage,EdgeStorage)
# from torch_geometric.nn.data_parallel import DataParallel
# from torch_geometric.nn import HeteroConv,GraphNorm,HypergraphConv,GATConv

# from torch_scatter import scatter, scatter_add
# from torch_geometric.utils import softmax as scatter_softmax
# from torch_geometric.utils import degree
# import random
# import numpy as np
# import logging
# import pickle
# import json
# import codecs
# from tqdm import tqdm
# import math
# from copy import deepcopy 

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# NodeOrEdgeStorage = Union[NodeStorage, EdgeStorage]

# # ================= 1. 深度架构组件：RMSNorm & 双端 AttnRes =================
# class RMSNorm(nn.Module):
#     def __init__(self, d, eps=1e-8):
#         super(RMSNorm, self).__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(d))

#     def forward(self, x):
#         normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
#         return normed * self.weight

# class NodeAwareAttentionResidual(nn.Module):
#     """Encoder专属：节点级图注意力残差。"""
#     def __init__(self, emb_dims):
#         super().__init__()
#         self.query = nn.Parameter(torch.zeros(emb_dims))
#         self.norm = RMSNorm(emb_dims)

#     def forward(self, history_x):
#         # history_x: [Layer_num, Total_Nodes, Emb_dims]
#         if len(history_x) == 1: return history_x[0]
#         V = torch.stack(history_x, dim=0) 
#         K = self.norm(V)
#         scores = torch.einsum('d, lnd -> ln', self.query, K)
#         alpha = F.softmax(scores, dim=0)
#         return torch.einsum('ln, lnd -> nd', alpha, V)

# class DecoderAttentionResidual(nn.Module):
#     """Decoder专属：时序级注意力残差。保护长距离序列生成的Copy特征"""
#     def __init__(self, emb_dims):
#         super().__init__()
#         self.query = nn.Parameter(torch.zeros(emb_dims))
#         self.norm = RMSNorm(emb_dims)

#     def forward(self, history_x):
#         # history_x: [Depth_L, Batch, Seq_Len, Emb_dims]
#         if len(history_x) == 1: return history_x[0]
#         V = torch.stack(history_x, dim=0) 
#         K = self.norm(V)
#         scores = torch.einsum('d, lbtd -> lbt', self.query, K)
#         alpha = F.softmax(scores, dim=0)
#         return torch.einsum('lbt, lbtd -> btd', alpha, V)

# # ================= 2. 图算子：防弹版 K阶扩散 & DHGAT =================
# class HyperedgeDiffusionConv(nn.Module):
#     def __init__(self, in_channels, out_channels, K=1, alpha=0.5, bias=True):
#         super(HyperedgeDiffusionConv, self).__init__()
#         self.K = K
#         self._alpha_init = alpha
#         self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
#         self.lin = nn.Linear(in_channels, out_channels, bias=bias)
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.lin.reset_parameters()
#         if hasattr(self, 'alpha'): self.alpha.data.fill_(self._alpha_init)

#     def forward(self, x, hyperedge_index, num_nodes=None, num_edges=None):
#         if num_nodes is None: num_nodes = x.size(0)
#         if num_edges is None: num_edges = int(hyperedge_index[1].max()) + 1
#         node_idx, edge_idx = hyperedge_index

#         d_v = degree(node_idx, num_nodes, dtype=x.dtype)
#         d_v[d_v == 0] = 1.0
#         d_v_inv_sqrt = d_v.pow(-0.5)
#         d_v_inv = d_v.pow(-1.0) 
#         d_v_norm_term = d_v_inv[node_idx] 

#         d_e = degree(edge_idx, num_edges, dtype=x.dtype)
#         d_e[d_e == 0] = 1.0
#         d_e_inv = d_e.pow(-1.0)

#         x = self.lin(x)
#         x_temp = x * d_v_inv_sqrt.unsqueeze(-1)
#         H_e = scatter_add(x_temp[node_idx], edge_idx, dim=0, dim_size=num_edges) * d_e_inv.unsqueeze(-1)
#         H_e_0 = H_e 

#         cur_alpha = torch.clamp(self.alpha, min=0.0, max=1.0)
#         for k in range(self.K):
#             H_node_temp = H_e[edge_idx] * d_v_norm_term.unsqueeze(-1)
#             H_e_diffused = scatter_add(H_node_temp, edge_idx, dim=0, dim_size=num_edges) * d_e_inv.unsqueeze(-1)
#             H_e = (1 - cur_alpha) * H_e_diffused + cur_alpha * H_e_0

#         out = scatter_add(H_e[edge_idx], node_idx, dim=0, dim_size=num_nodes) * d_v_inv_sqrt.unsqueeze(-1)
#         return out

# def build_directed_hyperedges_from_simple(edges, group_by='src'):
#     if edges is None or np.size(edges) == 0:
#         return np.empty((2, 0), dtype=np.int64), np.empty((2, 0), dtype=np.int64)
#     src_nodes, dst_nodes = np.array(edges[0]), np.array(edges[1])
#     src_hyper_edges, dst_hyper_edges = [], []
#     if group_by == 'src':
#         unique_srcs = np.unique(src_nodes)
#         for h_id, src in enumerate(unique_srcs):
#             src_hyper_edges.append([src, h_id])
#             for child in dst_nodes[src_nodes == src]: dst_hyper_edges.append([child, h_id])  
#     else:
#         unique_dsts = np.unique(dst_nodes)
#         for h_id, dst in enumerate(unique_dsts):
#             dst_hyper_edges.append([dst, h_id])
#             for parent in src_nodes[dst_nodes == dst]: src_hyper_edges.append([parent, h_id])
#     return np.array(src_hyper_edges, dtype=np.int64).T, np.array(dst_hyper_edges, dtype=np.int64).T

# class DirectedHypergraphAttention(nn.Module):
#     def __init__(self, emb_dims):
#         super().__init__()
#         self.att_src = nn.Linear(emb_dims, 1, bias=False)
#         self.att_dst = nn.Linear(emb_dims, 1, bias=False)
#         self.leaky_relu = nn.LeakyReLU(0.2)
#         self.out_proj = nn.Linear(emb_dims, emb_dims, bias=False)

#     def forward(self, x, edge_index_src, edge_index_dst):
#         if edge_index_src.numel() == 0: return torch.zeros_like(x)
#         src_nodes, src_edges = edge_index_src[0], edge_index_src[1]
#         dst_nodes, dst_edges = edge_index_dst[0], edge_index_dst[1]
#         num_he = max(src_edges.max().item(), dst_edges.max().item()) + 1
        
#         alpha_src = scatter_softmax(self.leaky_relu(self.att_src(x[src_nodes])), src_edges, dim=0)
#         he_x = scatter(x[src_nodes] * alpha_src, src_edges, dim=0, dim_size=num_he, reduce='sum')
        
#         alpha_dst = scatter_softmax(self.leaky_relu(self.att_dst(he_x[dst_edges] + x[dst_nodes])), dst_nodes, dim=0)
#         out = scatter(he_x[dst_edges] * alpha_dst, dst_nodes, dim=0, dim_size=x.size(0), reduce='sum')
#         return self.out_proj(out)

# # ================= 3. 数据预处理 =================
# class Datax(HeteroData):
#     def __cat_dim__(self, key: str, value: Any, store: Optional[NodeOrEdgeStorage] = None, *args, **kwargs) -> Any:
#         if bool(re.search('(token)', key)): return None  
#         if bool(re.search('(pos)', key)): return -1
#         return super().__cat_dim__(key, value,store)    

# class Datasetx(Dataset):
#     def __init__(self, code_graphs, texts=None, ids=None, text_max_len=None, text_begin_idx=1, text_end_idx=2, pad_idx=0):
#         self.len = len(code_graphs)  
#         self.text_max_len = max([len(t) for t in texts]) if text_max_len is None and texts is not None else text_max_len
#         self.text_begin_idx, self.text_end_idx, self.pad_idx = text_begin_idx, text_end_idx, pad_idx
#         self.code_graphs, self.texts, self.ids = code_graphs, texts, ids

#     def __getitem__(self, index):
#         if self.texts is None:
#             pad_text_in = np.zeros((self.text_max_len + 1,), dtype=np.int64)
#             pad_text_in[0] = self.text_begin_idx
#             pad_text_out = None
#         else:
#             tru_text = self.texts[index][:self.text_max_len]
#             pad_text_in = np.lib.pad(tru_text, (1, self.text_max_len - len(tru_text)), 'constant', constant_values=(self.text_begin_idx, self.pad_idx))
#             tru_text_out = np.lib.pad(np.lib.pad(tru_text, (0, 1), 'constant', constant_values=(0, self.text_end_idx)), (0, self.text_max_len + 1 - (len(tru_text)+1)), 'constant', constant_values=(self.pad_idx, self.pad_idx))
#             pad_text_out = tru_text_out
            
#         data=Datax()
#         cg = self.code_graphs[index]
#         data['node'].x = torch.tensor(cg['nodes'])
#         data['node'].src_map = torch.tensor(cg['node2text_map_ids']).long()
#         data['node'].code_mask = torch.tensor(cg['code_node_mask']).bool()
        
#         for key in ['parent_child_hyperedges', 'line_hyperedges', 'block_hyperedges', 'dfg_hyperedges', 'layout_sibling_hyperedges']:
#             data['node', key, 'node'].edge_index = torch.tensor(cg.get(key, [])).long()

#         ast_s, ast_d = build_directed_hyperedges_from_simple(cg.get('base_father2child_edges', []), 'src')
#         data['node', 'ast_dir_s', 'node'].edge_index, data['node', 'ast_dir_d', 'node'].edge_index = torch.tensor(ast_s).long(), torch.tensor(ast_d).long()
#         dfg_s, dfg_d = build_directed_hyperedges_from_simple(cg.get('dfg_prev2next_edges', []), 'dst')
#         data['node', 'dfg_dir_s', 'node'].edge_index, data['node', 'dfg_dir_d', 'node'].edge_index = torch.tensor(dfg_s).long(), torch.tensor(dfg_d).long()

#         data['text'].text_token_input = torch.tensor(pad_text_in).long()
#         if self.texts is not None: data['text'].text_token_output = torch.tensor(pad_text_out).long()
#         data['text'].num_nodes = pad_text_in.shape[0]
#         if self.ids is not None:
#             data['idx'].idx = torch.tensor(self.ids[index]); data['idx'].num_nodes = 1
#         return data

#     def __len__(self): return self.len

# # ================= 4. 核心网络：CodeGraphEnc & TNet =================
# class CodeGraphEnc(nn.Module):
#     def __init__(self, emb_dims, graph_max_size, code_max_len, graph_node_emb_op, graph_gnn_layers=6, drop_rate=0., **kwargs):
#         super().__init__()
#         self.graph_max_size, self.code_max_len, self.emb_dims = graph_max_size, code_max_len, emb_dims
#         self.pad_idx = kwargs.get('pad_idx', 0)
#         self.use_hyperedge_pos_emb = kwargs.get('use_hyperedge_pos_emb', True)
#         self.use_directed_hyperedges = kwargs.get('use_directed_hyperedges', True)
#         self.use_dynamic_edges = kwargs.get('use_dynamic_edges', True)
#         self.dynamic_threshold = kwargs.get('dynamic_threshold', 0.85)
#         self.gnn_layers = graph_gnn_layers

#         self.graph_node_emb_op = graph_node_emb_op
#         self.graph_pos_encoding = nn.Embedding(graph_max_size * 2 + 1, emb_dims, padding_idx=self.pad_idx)
#         nn.init.xavier_uniform_(self.graph_pos_encoding.weight[1:, ])
#         self.emb_drop_op = nn.Dropout(p=drop_rate)
        
#         # 接收 Encoder 残差开关
#         self.use_enc_attn_res = kwargs.get('use_enc_attn_res', True)
#         if self.use_enc_attn_res:
#             self.attn_res_ops = nn.ModuleList([NodeAwareAttentionResidual(emb_dims) for _ in range(graph_gnn_layers)])

#         self.gnn_ops, self.dhgat_ops, self.gnorm_ops, self.grelu_ops = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
#         self.hetero_alpha = nn.Parameter(torch.ones(graph_gnn_layers, 8))
        
#         for _ in range(graph_gnn_layers):
#             self.gnn_ops.append(HeteroConv({
#                 ('node', 'block_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, K=1),
#                 ('node', 'line_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, K=1),
#                 ('node', 'layout_sibling_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, K=1),
#                 ('node', 'parent_child_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, K=4),
#                 ('node', 'dfg_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, K=1),
#                 ('node', 'dynamic_semantic_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, K=1)
#             }, aggr='sum'))
#             self.dhgat_ops.append(nn.ModuleDict({
#                 'ast_dir': DirectedHypergraphAttention(emb_dims),
#                 'dfg_dir': DirectedHypergraphAttention(emb_dims)
#             }))
#             self.grelu_ops.append(nn.Sequential(nn.ReLU(), nn.Dropout(p=drop_rate)))
#             self.gnorm_ops.append(GraphNorm(emb_dims))

#     def _add_dynamic_edges(self, data):
#         dense_x, mask = to_dense_batch(data['node'].x, data.x_batch_dict['node'], fill_value=0.0) 
#         sim_matrix = torch.bmm(F.normalize(dense_x, p=2, dim=-1), F.normalize(dense_x, p=2, dim=-1).transpose(1, 2)) 
#         sim_matrix.diagonal(dim1=1, dim2=2).fill_(-1.0)
#         adj = (sim_matrix > self.dynamic_threshold) & (mask.unsqueeze(1) & mask.unsqueeze(2))
#         b_idx, r, c = adj.nonzero(as_tuple=True)
#         flat = torch.zeros_like(mask, dtype=torch.long); flat[mask] = torch.arange(mask.sum(), device=mask.device)
#         data.edge_index_dict[('node', 'dynamic_semantic_hyperedges', 'node')] = torch.stack([flat[b_idx, r], flat[b_idx, c]], dim=0)

#     def forward(self, data):
#         graph_node_emb = self.graph_node_emb_op(data.x_dict['node']) * np.sqrt(self.emb_dims)
#         batch_size = data.x_batch_dict['node'].max().item() + 1
#         pos_indices_list = []
#         for b in range(batch_size):
#             num = (data.x_batch_dict['node'] == b).sum().item()
#             max_p = self.graph_pos_encoding.num_embeddings - 1
#             pos_indices_list.append((torch.arange(1, num + 1, device=graph_node_emb.device) % max_p) + 1)
#         pos_emb = self.graph_pos_encoding(torch.cat(pos_indices_list)) 
        
#         if self.use_hyperedge_pos_emb: 
#             graph_node_emb = graph_node_emb + pos_emb
            
#         data['node'].x = self.emb_drop_op(graph_node_emb) 
        
#         history_x = [data['node'].x]
        
#         for i, (gnn, dhgat, relu, norm) in enumerate(zip(self.gnn_ops, self.dhgat_ops, self.grelu_ops, self.gnorm_ops)):
#             if self.use_dynamic_edges and i == (self.gnn_layers // 2): 
#                 self._add_dynamic_edges(data)
            
#             # 独立开关：只开启Encoder的注意力残差
#             x_input = self.attn_res_ops[i](history_x) if self.use_enc_attn_res else history_x[-1]
#             data['node'].x = x_input
            
#             x_dict = gnn(x_dict=data.x_dict, edge_index_dict=data.edge_index_dict)
#             out_x = x_dict['node']
            
#             if self.use_directed_hyperedges:
#                 a_s, a_d = data.edge_index_dict.get(('node','ast_dir_s','node')), data.edge_index_dict.get(('node','ast_dir_d','node'))
#                 if a_s is not None and a_s.numel()>0: 
#                     out_x = out_x + self.hetero_alpha[i, 4] * dhgat['ast_dir'](x_input, a_s, a_d)
#                 d_s, d_d = data.edge_index_dict.get(('node','dfg_dir_s','node')), data.edge_index_dict.get(('node','dfg_dir_d','node'))
#                 if d_s is not None and d_s.numel()>0: 
#                     out_x = out_x + self.hetero_alpha[i, 5] * dhgat['dfg_dir'](x_input, d_s, d_d)

#             data['node'].x = norm(x_input + relu(out_x))
#             history_x.append(data['node'].x)

#         data['node'].x = history_x[-1] + pos_emb
        
#         # 【防弹级修复】：锁死 batch_size，杜绝因为极端样本引发的维度坍塌
#         graph_enc,_ = to_dense_batch(data.x_dict['node'], batch=data.x_batch_dict['node'], fill_value=self.pad_idx, max_num_nodes=self.graph_max_size, batch_size=batch_size)  
#         cm = data['node'].code_mask; cb = data.x_batch_dict['node'][cm]
#         code_src_map,_ = to_dense_batch(data.src_map_dict['node'][cm], batch=cb, fill_value=self.pad_idx, max_num_nodes=self.code_max_len, batch_size=batch_size)    
#         graph_code_enc,_ = to_dense_batch(data.x_dict['node'][cm], batch=cb, fill_value=self.pad_idx, max_num_nodes=self.code_max_len, batch_size=batch_size)    
#         return graph_enc, graph_code_enc, code_src_map

# class Dec(nn.Module):
#     def __init__(self, emb_dims, text_voc_size, text_emb_op, text_max_len, enc_out_dims, att_layers, att_heads, att_head_dims=None, ff_hid_dims=2048, drop_rate=0., **kwargs):
#         super().__init__()
#         self.emb_dims = emb_dims
#         self._copy = kwargs.get('copy', True)
        
#         # 接收 Decoder 残差开关
#         self.use_dec_attn_res = kwargs.get('use_dec_attn_res', True)
        
#         self.text_emb_op = text_emb_op
#         self.pos_encoding = PosEnc(max_len=text_max_len+1, emb_dims=emb_dims, train=True, pad=True, pad_idx=kwargs.get('pad_idx', 0)) 
#         self.emb_layer_norm = nn.LayerNorm(emb_dims)
#         self.text_dec_op = TranDec(query_dims=emb_dims, key_dims=enc_out_dims, head_nums=att_heads, head_dims=att_head_dims, layer_num=att_layers, ff_hid_dims=ff_hid_dims, drop_rate=drop_rate, pad_idx=kwargs.get('pad_idx', 0), self_causality=True)
        
#         # 为 Decoder 注册专属时序注意力残差
#         if self.use_dec_attn_res:
#             self.dec_attn_res_ops = nn.ModuleList([DecoderAttentionResidual(emb_dims) for _ in range(att_layers)])
            
#         self.dropout = nn.Dropout(p=drop_rate)
#         self.out_fc = nn.Linear(emb_dims, text_voc_size)
#         self.copy_generator = MultiCopyGenerator(tgt_dims=emb_dims, tgt_voc_size=text_voc_size, src_dims=enc_out_dims, att_heads=att_heads, att_head_dims=att_head_dims, drop_rate=drop_rate, pad_idx=kwargs.get('pad_idx', 0))

#     def forward(self, graph_enc, graph_code_enc, code_src_map, text_input):
#         text_emb = self.text_emb_op(text_input)   
#         text_emb = text_emb * np.sqrt(self.emb_dims)
#         pos_emb = self.pos_encoding(text_input)  
#         text_dec = self.emb_layer_norm(self.dropout(text_emb.add(pos_emb)))  
        
#         graph_mask = graph_enc.abs().sum(-1).sign()  
#         text_mask = text_input.abs().sign()  
        
#         history_q = [text_dec] 
        
#         for i, block in enumerate(self.text_dec_op.dec_blocks):
#             # 独立开关：只开启Decoder的注意力残差
#             q_input = self.dec_attn_res_ops[i](history_q) if self.use_dec_attn_res else history_q[-1]
#             q_out = block(query=q_input, key=graph_enc, query_mask=text_mask, key_mask=graph_mask)
#             history_q.append(q_out)
            
#         text_dec = history_q[-1] 
        
#         if hasattr(self.text_dec_op, 'layer_norm') and getattr(self.text_dec_op, 'layer_norm') is not None:
#             text_dec = self.text_dec_op.layer_norm(text_dec)

#         if not self._copy: return self.out_fc(text_dec).transpose(1, 2)
#         return self.copy_generator(text_dec, graph_code_enc, code_src_map).transpose(1, 2)

# class TNet(BaseNet):
#     def __init__(self, emb_dims, graph_max_size, code_max_len, text_max_len, io_voc_size, text_voc_size, graph_gnn_layers=6, drop_rate=0., **kwargs):
#         super().__init__()
#         self.use_cl, self.cl_temp, self.edge_drop_rate = kwargs.get('use_cl',True), kwargs.get('cl_temp',0.1), kwargs.get('edge_drop_rate',0.15)
#         io_emb = nn.Embedding(io_voc_size, emb_dims, padding_idx=kwargs.get('pad_idx',0)); nn.init.xavier_uniform_(io_emb.weight[1:, ])
#         self.enc_op = CodeGraphEnc(emb_dims, graph_max_size, code_max_len, io_emb, graph_gnn_layers, drop_rate, **kwargs)
#         self.dec_op = Dec(emb_dims, text_voc_size, io_emb, text_max_len, emb_dims, kwargs.get('text_att_layers',8), kwargs.get('text_att_heads',8), ff_hid_dims=kwargs.get('text_ff_hid_dims',2048), drop_rate=drop_rate, **kwargs)

#     def augment(self, data):
#         aug = deepcopy(data)
#         for et in aug.edge_index_dict.keys():
#             idx = aug.edge_index_dict[et]
#             if idx.numel() > 0: 
#                 aug.edge_index_dict[et] = idx[:, torch.rand(idx.size(1), device=idx.device) > self.edge_drop_rate]
#         return aug

#     def forward(self, data):
#         text_in = data['text'].text_token_input.clone()
#         del data['text']
#         if self.training and self.use_cl:
#             cg_orig, cg_aug = deepcopy(data), self.augment(data)
#             g_enc, g_code, src_m = self.enc_op(cg_orig)
#             g_enc_a, _, _ = self.enc_op(cg_aug)
#             out = self.dec_op(g_enc, g_code, src_m, text_in)
#             z1, z2 = F.normalize(g_enc.mean(1), p=2, dim=-1), F.normalize(g_enc_a.mean(1), p=2, dim=-1)
#             sim = torch.matmul(z1, z2.T) / self.cl_temp
#             loss_cl = (F.cross_entropy(sim, torch.arange(z1.size(0), device=z1.device)) + F.cross_entropy(sim.T, torch.arange(z1.size(0), device=z1.device))) / 2
#             return out, loss_cl
        
#         g_enc, g_code, src_m = self.enc_op(data)
#         return self.dec_op(g_enc, g_code, src_m, text_in)

# class TModel(TransSeq2Seq):
#     def __init__(self, model_dir, model_name='Transformer_based_model', model_id=None, emb_dims=512, graph_gnn_layers=3, graph_GNN=SAGEConv, graph_gnn_aggr='add', text_att_layers=3, text_att_heads=8, text_att_head_dims=None, text_ff_hid_dims=2048, drop_rate=0., copy=True, pad_idx=0, train_batch_size=32, pred_batch_size=32, max_train_size=-1, max_valid_size=32 * 10, max_big_epochs=20, regular_rate=1e-5, lr_base=0.001, lr_decay=0.9, min_lr_rate=0.01, warm_big_epochs=2, start_valid_epoch=20, early_stop=20, Net=TNet, Dataset=Datasetx, beam_width=1, train_metrics=[get_sent_bleu], valid_metric=get_sent_bleu, test_metrics=[get_sent_bleu], train_mode=True, **kwargs):
#         logging.info('Construct %s' % model_name)
#         super().__init__(model_name=model_name, model_dir=model_dir, model_id=model_id)
#         self.init_params = locals()
#         self.emb_dims = emb_dims
#         self.graph_gnn_layers = graph_gnn_layers
#         self.graph_GNN = graph_GNN
#         self.graph_gnn_aggr = graph_gnn_aggr
#         self.text_att_layers = text_att_layers
#         self.text_att_heads = text_att_heads
#         self.text_att_head_dims = text_att_head_dims
#         self.text_ff_hid_dims = text_ff_hid_dims
#         self.drop_rate = drop_rate
#         self.pad_idx = pad_idx
#         self.copy = copy
#         self.train_batch_size = train_batch_size
#         self.pred_batch_size = pred_batch_size
#         self.max_train_size = max_train_size
#         self.max_valid_size = max_valid_size
#         self.max_big_epochs = max_big_epochs
#         self.regular_rate = regular_rate
#         self.lr_base = lr_base
#         self.lr_decay = lr_decay
#         self.min_lr_rate = min_lr_rate
#         self.warm_big_epochs = warm_big_epochs
#         self.start_valid_epoch = start_valid_epoch
#         self.early_stop = early_stop
#         self.Net = Net
#         self.Dataset = Dataset
#         self.beam_width = beam_width
#         self.train_metrics = train_metrics
#         self.valid_metric = valid_metric
#         self.test_metrics = test_metrics
#         self.train_mode = train_mode
        
#         # 接收最新分离的大招参数
#         self.use_enc_attn_res = kwargs.get('use_enc_attn_res', True)
#         self.use_dec_attn_res = kwargs.get('use_dec_attn_res', True)
#         self.use_directed_hyperedges = kwargs.get('use_directed_hyperedges', True)
#         self.use_hyperedge_pos_emb = kwargs.get('use_hyperedge_pos_emb', True)
#         self.use_dynamic_edges = kwargs.get('use_dynamic_edges', True)
#         self.use_cl = kwargs.get('use_cl', True)
#         self.cl_weight = kwargs.get('cl_weight', 0.05)
#         self.cl_temp = kwargs.get('cl_temp', 0.1)
#         self.edge_drop_rate = kwargs.get('edge_drop_rate', 0.15)
#         self.dynamic_threshold = kwargs.get('dynamic_threshold', 0.85)

#     def _logging_paramerter_num(self):
#         logging.info("{} have {} paramerters in total".format(self.model_name, sum( x.numel() for x in self.net.parameters() if x.requires_grad)))
#         code_graph_enc_param_num = sum(x.numel() for x in self.net.module.enc_op.gnn_ops.parameters() if x.requires_grad) + sum(x.numel() for x in self.net.module.enc_op.gnorm_ops.parameters() if x.requires_grad) + sum(x.numel() for x in self.net.module.enc_op.grelu_ops.parameters() if x.requires_grad)
#         text_dec_param_num = sum(x.numel() for x in self.net.module.dec_op.text_dec_op.parameters() if x.requires_grad)
#         enc_dec_param_num = code_graph_enc_param_num + text_dec_param_num
#         logging.info("{} have {} paramerters in encoder and decoder".format(self.model_name, enc_dec_param_num))

#     def fit(self, train_data, valid_data, **kwargs):
#         self.graph_max_size, self.code_max_len, self.io_voc_size, self.text_max_len=0, 0, 0, 0
#         for code_graph,text in zip(train_data['code_graphs'],train_data['texts']):
#             self.graph_max_size = max(self.graph_max_size,len(code_graph['nodes']))
#             self.code_max_len = max(self.code_max_len,code_graph['code_node_mask'].sum())
#             self.io_voc_size = max(self.io_voc_size,max(code_graph['nodes']))
#             self.text_max_len=max(self.text_max_len,len(text))
#         self.io_voc_size+=1

#         self.text_voc_size = len(train_data['text_dic']['text_i2w']) 
#         self.io_voc_size=max(self.io_voc_size,self.text_voc_size+2*self.code_max_len)
        
#         net = self.Net(
#             emb_dims=self.emb_dims, graph_max_size=self.graph_max_size, code_max_len=self.code_max_len, text_max_len=self.text_max_len, io_voc_size=self.io_voc_size, text_voc_size=self.text_voc_size, graph_gnn_layers=self.graph_gnn_layers, graph_GNN=self.graph_GNN, graph_gnn_aggr=self.graph_gnn_aggr,
#             text_att_layers=self.text_att_layers, text_att_heads=self.text_att_heads, text_att_head_dims=self.text_att_head_dims, text_ff_hid_dims=self.text_ff_hid_dims, 
#             drop_rate=self.drop_rate, pad_idx=self.pad_idx, copy=self.copy, 
#             use_enc_attn_res=self.use_enc_attn_res,
#             use_dec_attn_res=self.use_dec_attn_res,
#             use_hyperedge_pos_emb=self.use_hyperedge_pos_emb, 
#             use_directed_hyperedges=self.use_directed_hyperedges, 
#             use_dynamic_edges=self.use_dynamic_edges, 
#             use_cl=self.use_cl,
#             cl_weight=self.cl_weight,
#             cl_temp=self.cl_temp,
#             edge_drop_rate=self.edge_drop_rate,
#             dynamic_threshold=self.dynamic_threshold
#         )
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
#         self.net =DataParallel(net.to(device),follow_batch=['x'])  
#         self._logging_paramerter_num()  
#         self.net.train()  

#         self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr_base, weight_decay=self.regular_rate)
#         self.criterion = LabelSmoothSoftmaxCEV2(reduction='mean', ignore_index=self.pad_idx, label_smooth=0.0)

#         self.text_begin_idx = self.text_voc_size - 1
#         self.text_end_idx = self.text_voc_size - 2
#         self.tgt_begin_idx,self.tgt_end_idx=self.text_begin_idx,self.text_end_idx

#         self.max_train_size = len(train_data['code_graphs']) if self.max_train_size == -1 else self.max_train_size
#         train_code_graphs, train_texts,train_ids = zip(*random.sample(list(zip(train_data['code_graphs'], train_data['texts'],train_data['ids'])), min(self.max_train_size, len(train_data['code_graphs']))))
#         train_set = self.Dataset(code_graphs=train_code_graphs, texts=train_texts, ids=train_ids, text_max_len=self.text_max_len, text_begin_idx=self.text_begin_idx, text_end_idx=self.text_end_idx, pad_idx=self.pad_idx)
#         train_loader=DataListLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True, drop_last=True) 

#         if self.warm_big_epochs is None: self.warm_big_epochs = max(self.max_big_epochs // 10, 2)
#         self.scheduler = LrWarmUp(self.optimizer, min_rate=self.min_lr_rate, lr_decay=self.lr_decay, warm_steps=self.warm_big_epochs * len(train_loader), reduce_steps=len(train_loader))  
        
#         if self.train_mode:  
#             # 【核心修复】：物理Batch=32，通过两步累加实现等效Batch=64，规避显存OOM
#             accumulation_steps = 2
            
#             for i in range(0, self.max_big_epochs):
#                 pbar = tqdm(train_loader)
#                 self.optimizer.zero_grad() 
                
#                 for j, batch_data in enumerate(pbar):
#                     batch_text_output = []
#                     ids=[]
#                     for data in batch_data:
#                         batch_text_output.append(data['text'].text_token_output.unsqueeze(0))
#                         del data['text'].text_token_output
#                         ids.append(data['idx'].idx.item())
#                         del data['idx']

#                     batch_text_output = torch.cat(batch_text_output, dim=0).to(device)
                    
#                     if self.use_cl:
#                         pred_text_output, loss_cl = self.net(batch_data)
#                         loss_cl = loss_cl.mean()
#                         loss_ce = self.criterion(pred_text_output, batch_text_output)
#                         loss = loss_ce + self.cl_weight * loss_cl
#                     else:
#                         pred_text_output = self.net(batch_data)
#                         loss_ce = self.criterion(pred_text_output, batch_text_output)
#                         loss = loss_ce

#                     # 梯度累加魔法：Loss除以步数，保证更新幅度不变
#                     loss = loss / accumulation_steps
#                     loss.backward()  
                    
#                     # 当积攒满指定的步数时，统一更新参数
#                     if (j + 1) % accumulation_steps == 0 or (j + 1) == len(train_loader):
#                         clip_grad_norm_(self.net.parameters(), 2.0)
#                         self.optimizer.step()  
#                         self.scheduler.step()  
#                         self.optimizer.zero_grad() # 更新完清空梯度池

#                     text_dic = {'text_i2w': train_data['text_dic']['text_i2w'], 'ex_text_i2ws': [train_data['text_dic']['ex_text_i2ws'][k] for k in ids]}
#                     log_info=self._get_log_fit_eval(loss=loss_ce, pred_tgt=pred_text_output, gold_tgt=batch_text_output, tgt_i2w=text_dic)
                    
#                     if self.use_cl:
#                         log_info = '[Big epoch:{}/{}, CE:{:.3f}, CL:{:.3f}, {}]'.format(i + 1, self.max_big_epochs, loss_ce.item(), loss_cl.item(), log_info)
#                     else:
#                         log_info = '[Big epoch:{}/{},{}]'.format(i + 1, self.max_big_epochs, log_info)
#                     pbar.set_description(log_info)
                    
#                     # 及时释放局部变量，保障显存纯净
#                     del pred_text_output, batch_text_output, batch_data
                
#                 # 每一轮结束释放底层CUDA缓存
#                 torch.cuda.empty_cache()
#                 del pbar
                
#                 if i+1 >= self.start_valid_epoch:
#                     self.max_valid_size = len(valid_data['code_graphs']) if self.max_valid_size == -1 else self.max_valid_size
#                     valid_srcs, valid_tgts, ex_text_i2ws = zip(*random.sample(list(zip(valid_data['code_graphs'], valid_data['texts'], valid_data['text_dic']['ex_text_i2ws'])), min(self.max_valid_size, len(valid_data['code_graphs']))))
#                     text_dic = {'text_i2w': train_data['text_dic']['text_i2w'], 'ex_text_i2ws': ex_text_i2ws}
#                     worse_epochs = self._do_validation(valid_srcs=valid_srcs, valid_tgts=valid_tgts, tgt_i2w=text_dic, increase_better=True, last=False)  
#                     if worse_epochs>=self.early_stop: break
                    
#         self._do_validation(valid_srcs=valid_data['code_graphs'], valid_tgts=valid_data['texts'], tgt_i2w=valid_data['text_dic'], increase_better=True, last=True)  
#         self._logging_paramerter_num()  

#     def predict(self, code_graphs, text_dic):
#         logging.info('Predict outputs of %s' % self.model_name)
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
#         self.net.eval()  
#         enc_op=DataParallel(self.net.module.enc_op,follow_batch=['x'])
#         dec_op=torch.nn.DataParallel(self.net.module.dec_op)
#         data_set = self.Dataset(code_graphs=code_graphs, texts=None, ids=None, text_max_len=self.text_max_len, text_begin_idx=self.text_begin_idx, text_end_idx=self.text_end_idx, pad_idx=self.pad_idx)  

#         data_loader = DataListLoader(dataset=data_set, batch_size=self.pred_batch_size, shuffle=False)
#         pred_text_id_np_batches = []  
#         with torch.no_grad():  
#             pbar = tqdm(data_loader)
#             for batch_data in pbar:
#                 batch_text_input = []
#                 for data in batch_data:
#                     batch_text_input.append(data['text'].text_token_input.unsqueeze(0))
#                     del data['text']
#                 batch_text_input = torch.cat(batch_text_input, dim=0).to(device)

#                 batch_graph_enc,batch_graph_code_enc,batch_code_src_map=enc_op(batch_data)
#                 batch_text_output: list = []  
#                 if self.beam_width == 1:
#                     for i in range(self.text_max_len + 1):  
#                         pred_out = dec_op(graph_enc=batch_graph_enc,graph_code_enc=batch_graph_code_enc,code_src_map=batch_code_src_map,text_input=batch_text_input)  
#                         batch_text_output.append(pred_out[:, :, i].unsqueeze(-1).to('cpu').data.numpy())  
#                         if i < self.text_max_len:  
#                             batch_text_input[:, i + 1] = torch.argmax(pred_out[:, :, i], dim=1)
#                     batch_pred_text = np.concatenate(batch_text_output, axis=-1)[:, :, :-1]  
#                     batch_pred_text[:, self.tgt_begin_idx, :] = -np.inf  
#                     batch_pred_text[:, self.pad_idx, :] = -np.inf  
#                     batch_pred_text_np = np.argmax(batch_pred_text, axis=1)  
#                     pred_text_id_np_batches.append(batch_pred_text_np)  
#                 else:
#                     batch_pred_text=trans_beam_search(net=dec_op, beam_width=self.beam_width, dec_input_arg_name='text_input', length_penalty=1, begin_idx=self.tgt_begin_idx, pad_idx=self.pad_idx, end_idx=self.tgt_end_idx, graph_enc=batch_graph_enc, graph_code_enc=batch_graph_code_enc, code_src_map=batch_code_src_map, text_input=batch_text_input)     
#                     pred_text_id_np_batches.append(batch_pred_text.to('cpu').data.numpy()[:,:-1])  

#         pred_text_id_np = np.concatenate(pred_text_id_np_batches,axis=0)  
#         self.net.train()  
#         return self._tgt_ids2tokens(pred_text_id_np, text_dic, self.text_end_idx)
    
#     def generate_texts(self,code_graphs,text_dic,res_path,gold_texts,raw_data,token_data,**kwargs):
#         logging.info('>>>>>>>Generate the targets according to sources and save the result to {}'.format(res_path))
#         kwargs.setdefault('beam_width',1)
#         res_dir=os.path.dirname(res_path)
#         if not os.path.exists(res_dir): os.makedirs(res_dir)
#         pred_texts=self.predict(code_graphs=code_graphs, text_dic=text_dic)
#         gold_texts=self._tgt_ids2tokens(gold_texts,text_dic,self.pad_idx)
#         res_data = []
#         for i,(pred_text,gold_text,raw_item,token_item) in enumerate(zip(pred_texts,gold_texts,raw_data,token_data)):
#             sent_bleu=self.valid_metric([pred_text],[gold_text])
#             res_data.append(dict(pred_text=' '.join(pred_text), gold_text=' '.join(gold_text), sent_bleu=sent_bleu, raw_code=raw_item['code'], raw_text=raw_item['text'], id=raw_item['id'], token_text=token_item['text'],))
#         with codecs.open(res_path,'w',encoding='utf-8') as f:
#             json.dump(res_data,f,indent=4, ensure_ascii=False)
#         self._logging_paramerter_num()  
#         logging.info('>>>>>>>The result has been saved to {}'.format(res_path))

#     def _code_ids2tokens(self,code_idss, code_i2w, end_idx):
#         return [[code_i2w[idx] for idx in (code_ids[:code_ids.tolist().index(end_idx)] if end_idx in code_ids else code_ids)] for code_ids in code_idss]
    
#     def _tgt_ids2tokens(self, text_id_np, text_dic, end_idx=0, **kwargs):
#         if self.copy:
#             text_tokens: list = []
#             for j, text_ids in enumerate(text_id_np):
#                 text_i2w = {**text_dic['text_i2w'], **text_dic['ex_text_i2ws'][j]}
#                 end_i = text_ids.tolist().index(end_idx) if end_idx in text_ids else len(text_ids)
#                 text_tokens.append([text_i2w[text_idx] for text_idx in text_ids[:end_i]])
#         else:
#             text_i2w=text_dic['text_i2w']
#             text_tokens = [[text_i2w[idx] for idx in (text_ids[:text_ids.tolist().index(end_idx)] if end_idx in text_ids else text_ids)] for text_ids in text_id_np]
#         return text_tokens

# if __name__ == '__main__':
#     # 从 params 中提取开关并默认填充
#     params.setdefault('use_enc_attn_res', True)
#     params.setdefault('use_dec_attn_res', True)
#     params.setdefault('use_directed_hyperedges', True) 
#     params.setdefault('use_hyperedge_pos_emb', True)
#     params.setdefault('use_dynamic_edges', True)
#     params.setdefault('use_cl', True)
#     params.setdefault('cl_weight', 0.05)
#     params.setdefault('cl_temp', 0.1)
#     params.setdefault('edge_drop_rate', 0.15)
#     params.setdefault('dynamic_threshold', 0.85)

#     logging.info('Parameters are listed below: \n'+'\n'.join(['{}: {}'.format(key,value) for key,value in params.items()]))

#     model = TModel(
#                    model_dir=params['model_dir'],
#                    model_name=params['model_name'],
#                    model_id=params['model_id'],
#                    emb_dims=params['emb_dims'],
#                    graph_gnn_layers=params['graph_gnn_layers'],
#                    graph_GNN=params['graph_GNN'],
#                    graph_gnn_aggr=params['graph_gnn_aggr'],
#                    text_att_layers=params['text_att_layers'],
#                    text_att_heads=params['text_att_heads'],
#                    text_att_head_dims=params['text_att_head_dims'],
#                    text_ff_hid_dims=params['text_ff_hid_dims'],
#                    drop_rate=params['drop_rate'],
#                    copy=params['copy'],
#                    pad_idx=params['pad_idx'],
#                    train_batch_size=params['train_batch_size'],
#                    pred_batch_size=params['pred_batch_size'],
#                    max_train_size=params['max_train_size'],  
#                    max_valid_size=params['max_valid_size'],  
#                    max_big_epochs=params['max_big_epochs'],
#                    regular_rate=params['regular_rate'],
#                    lr_base=params['lr_base'],
#                    lr_decay=params['lr_decay'],
#                    min_lr_rate=params['min_lr_rate'],
#                    warm_big_epochs=params['warm_big_epochs'],
#                    early_stop=params['early_stop'],
#                    start_valid_epoch=params['start_valid_epoch'],
                   
#                    # ===== 注入全新分离的开关参数 =====
#                    use_enc_attn_res=params['use_enc_attn_res'],
#                    use_dec_attn_res=params['use_dec_attn_res'],
                   
#                    use_directed_hyperedges=params['use_directed_hyperedges'],
#                    use_hyperedge_pos_emb=params['use_hyperedge_pos_emb'],
#                    use_dynamic_edges=params['use_dynamic_edges'],
#                    use_cl=params['use_cl'],
#                    cl_weight=params['cl_weight'],
#                    cl_temp=params['cl_temp'],
#                    edge_drop_rate=params['edge_drop_rate'],
#                    dynamic_threshold=params['dynamic_threshold'],
#                    Net=TNet,
#                    Dataset=Datasetx,
#                    beam_width=params['beam_width'],
#                    train_metrics=train_metrics,
#                    valid_metric=valid_metric,
#                    test_metrics=test_metrics,
#                    train_mode=params['train_mode'])

#     logging.info('Load data ...')
#     with codecs.open(train_avail_data_path, 'rb') as f: train_data = pickle.load(f)
#     with codecs.open(valid_avail_data_path, 'rb') as f: valid_data = pickle.load(f)
#     with codecs.open(test_avail_data_path, 'rb') as f: test_data = pickle.load(f)
#     with codecs.open(test_token_data_path,'r') as f: test_token_data=json.load(f)
#     with codecs.open(test_raw_data_path,'r') as f: test_raw_data=json.load(f)

#     model.fit(train_data=train_data, valid_data=valid_data)

#     test_eval_df=model.eval(test_srcs=test_data['code_graphs'], test_tgts=test_data['texts'], tgt_i2w=test_data['text_dic'])
#     logging.info('Model performance on test dataset:\n')
#     for i in range(0,len(test_eval_df.columns),4): print(test_eval_df.iloc[:, i:i+4])

#     model.generate_texts(code_graphs=test_data['code_graphs'], text_dic=test_data['text_dic'], res_path=res_path, gold_texts=test_data['texts'], raw_data=test_raw_data, token_data=test_token_data)


#copy高速通道
# coding=utf-8
# import os
# import re
# import sys
# sys.path.append('../../../')
# from lib.neural_module.learn_strategy import LrWarmUp
# from lib.neural_module.transformer import TranEnc, TranDec, DualTranDec,ResFF,ResMHA
# from lib.neural_module.embedding import PosEnc,SinusoidalPositionalEncoding
# from lib.neural_module.loss import LabelSmoothSoftmaxCEV2, CriterionNet
# from lib.neural_module.balanced_data_parallel import BalancedDataParallel
# from lib.neural_module.copy_attention import DualMultiCopyGenerator,MultiCopyGenerator,DualCopyGenerator
# from lib.neural_module.beam_search import trans_beam_search
# from lib.neural_model.seq_to_seq_model import TransSeq2Seq
# from lib.neural_model.base_model import BaseNet
# from lib.neural_module.transformer import ResFF
    
# from typing import Any,Optional,Union
# from config import *

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler
# from torch.nn.utils import clip_grad_norm_
# from torch.utils.data import Dataset
# from torch_geometric.data import HeteroData
# from torch_geometric.loader.data_list_loader import DataListLoader
# from torch_geometric.utils import to_dense_batch
# from torch_geometric.data.storage import (BaseStorage, NodeStorage,EdgeStorage)
# from torch_geometric.nn.data_parallel import DataParallel
# from torch_geometric.nn import HeteroConv,GraphNorm,HypergraphConv,GATConv

# from torch_scatter import scatter, scatter_add
# from torch_geometric.utils import softmax as scatter_softmax
# from torch_geometric.utils import degree
# import random
# import numpy as np
# import logging
# import pickle
# import json
# import codecs
# from tqdm import tqdm
# import math
# from copy import deepcopy 

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# NodeOrEdgeStorage = Union[NodeStorage, EdgeStorage]

# # ================= 1. 深度架构组件：RMSNorm & 双端 AttnRes =================
# class RMSNorm(nn.Module):
#     def __init__(self, d, eps=1e-8):
#         super(RMSNorm, self).__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(d))

#     def forward(self, x):
#         normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
#         return normed * self.weight

# class NodeAwareAttentionResidual(nn.Module):
#     """Encoder专属：节点级图注意力残差。"""
#     def __init__(self, emb_dims):
#         super().__init__()
#         self.query = nn.Parameter(torch.zeros(emb_dims))
#         self.norm = RMSNorm(emb_dims)

#     def forward(self, history_x):
#         if len(history_x) == 1: return history_x[0]
#         V = torch.stack(history_x, dim=0) 
#         K = self.norm(V)
#         scores = torch.einsum('d, lnd -> ln', self.query, K)
#         alpha = F.softmax(scores, dim=0)
#         return torch.einsum('ln, lnd -> nd', alpha, V)

# class DecoderAttentionResidual(nn.Module):
#     """Decoder专属：时序级注意力残差。"""
#     def __init__(self, emb_dims):
#         super().__init__()
#         self.query = nn.Parameter(torch.zeros(emb_dims))
#         self.norm = RMSNorm(emb_dims)

#     def forward(self, history_x):
#         if len(history_x) == 1: return history_x[0]
#         V = torch.stack(history_x, dim=0) 
#         K = self.norm(V)
#         scores = torch.einsum('d, lbtd -> lbt', self.query, K)
#         alpha = F.softmax(scores, dim=0)
#         return torch.einsum('lbt, lbtd -> btd', alpha, V)

# # ================= 2. 图算子：防弹版 K阶扩散 & DHGAT =================
# class HyperedgeDiffusionConv(nn.Module):
#     def __init__(self, in_channels, out_channels, K=1, alpha=0.5, bias=True):
#         super(HyperedgeDiffusionConv, self).__init__()
#         self.K = K
#         self._alpha_init = alpha
#         self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
#         self.lin = nn.Linear(in_channels, out_channels, bias=bias)
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.lin.reset_parameters()
#         if hasattr(self, 'alpha'): self.alpha.data.fill_(self._alpha_init)

#     def forward(self, x, hyperedge_index, num_nodes=None, num_edges=None):
#         if num_nodes is None: num_nodes = x.size(0)
#         if num_edges is None: num_edges = int(hyperedge_index[1].max()) + 1
#         node_idx, edge_idx = hyperedge_index

#         d_v = degree(node_idx, num_nodes, dtype=x.dtype)
#         d_v[d_v == 0] = 1.0
#         d_v_inv_sqrt = d_v.pow(-0.5)
#         d_v_inv = d_v.pow(-1.0) 
#         d_v_norm_term = d_v_inv[node_idx] 

#         d_e = degree(edge_idx, num_edges, dtype=x.dtype)
#         d_e[d_e == 0] = 1.0
#         d_e_inv = d_e.pow(-1.0)

#         x = self.lin(x)
#         x_temp = x * d_v_inv_sqrt.unsqueeze(-1)
#         H_e = scatter_add(x_temp[node_idx], edge_idx, dim=0, dim_size=num_edges) * d_e_inv.unsqueeze(-1)
#         H_e_0 = H_e 

#         cur_alpha = torch.clamp(self.alpha, min=0.0, max=1.0)
#         for k in range(self.K):
#             H_node_temp = H_e[edge_idx] * d_v_norm_term.unsqueeze(-1)
#             H_e_diffused = scatter_add(H_node_temp, edge_idx, dim=0, dim_size=num_edges) * d_e_inv.unsqueeze(-1)
#             H_e = (1 - cur_alpha) * H_e_diffused + cur_alpha * H_e_0

#         out = scatter_add(H_e[edge_idx], node_idx, dim=0, dim_size=num_nodes) * d_v_inv_sqrt.unsqueeze(-1)
#         return out

# def build_directed_hyperedges_from_simple(edges, group_by='src'):
#     if edges is None or np.size(edges) == 0:
#         return np.empty((2, 0), dtype=np.int64), np.empty((2, 0), dtype=np.int64)
#     src_nodes, dst_nodes = np.array(edges[0]), np.array(edges[1])
#     src_hyper_edges, dst_hyper_edges = [], []
#     if group_by == 'src':
#         unique_srcs = np.unique(src_nodes)
#         for h_id, src in enumerate(unique_srcs):
#             src_hyper_edges.append([src, h_id])
#             for child in dst_nodes[src_nodes == src]: dst_hyper_edges.append([child, h_id])  
#     else:
#         unique_dsts = np.unique(dst_nodes)
#         for h_id, dst in enumerate(unique_dsts):
#             dst_hyper_edges.append([dst, h_id])
#             for parent in src_nodes[dst_nodes == dst]: src_hyper_edges.append([parent, h_id])
#     return np.array(src_hyper_edges, dtype=np.int64).T, np.array(dst_hyper_edges, dtype=np.int64).T

# class DirectedHypergraphAttention(nn.Module):
#     def __init__(self, emb_dims):
#         super().__init__()
#         self.att_src = nn.Linear(emb_dims, 1, bias=False)
#         self.att_dst = nn.Linear(emb_dims, 1, bias=False)
#         self.leaky_relu = nn.LeakyReLU(0.2)
#         self.out_proj = nn.Linear(emb_dims, emb_dims, bias=False)

#     def forward(self, x, edge_index_src, edge_index_dst):
#         if edge_index_src.numel() == 0: return torch.zeros_like(x)
#         src_nodes, src_edges = edge_index_src[0], edge_index_src[1]
#         dst_nodes, dst_edges = edge_index_dst[0], edge_index_dst[1]
#         num_he = max(src_edges.max().item(), dst_edges.max().item()) + 1
        
#         alpha_src = scatter_softmax(self.leaky_relu(self.att_src(x[src_nodes])), src_edges, dim=0)
#         he_x = scatter(x[src_nodes] * alpha_src, src_edges, dim=0, dim_size=num_he, reduce='sum')
        
#         alpha_dst = scatter_softmax(self.leaky_relu(self.att_dst(he_x[dst_edges] + x[dst_nodes])), dst_nodes, dim=0)
#         out = scatter(he_x[dst_edges] * alpha_dst, dst_nodes, dim=0, dim_size=x.size(0), reduce='sum')
#         return self.out_proj(out)

# # ================= 3. 数据预处理 =================
# class Datax(HeteroData):
#     def __cat_dim__(self, key: str, value: Any, store: Optional[NodeOrEdgeStorage] = None, *args, **kwargs) -> Any:
#         if bool(re.search('(token)', key)): return None  
#         if bool(re.search('(pos)', key)): return -1
#         return super().__cat_dim__(key, value,store)    

# class Datasetx(Dataset):
#     def __init__(self, code_graphs, texts=None, ids=None, text_max_len=None, text_begin_idx=1, text_end_idx=2, pad_idx=0):
#         self.len = len(code_graphs)  
#         self.text_max_len = max([len(t) for t in texts]) if text_max_len is None and texts is not None else text_max_len
#         self.text_begin_idx, self.text_end_idx, self.pad_idx = text_begin_idx, text_end_idx, pad_idx
#         self.code_graphs, self.texts, self.ids = code_graphs, texts, ids

#     def __getitem__(self, index):
#         if self.texts is None:
#             pad_text_in = np.zeros((self.text_max_len + 1,), dtype=np.int64)
#             pad_text_in[0] = self.text_begin_idx
#             pad_text_out = None
#         else:
#             tru_text = self.texts[index][:self.text_max_len]
#             pad_text_in = np.lib.pad(tru_text, (1, self.text_max_len - len(tru_text)), 'constant', constant_values=(self.text_begin_idx, self.pad_idx))
#             tru_text_out = np.lib.pad(np.lib.pad(tru_text, (0, 1), 'constant', constant_values=(0, self.text_end_idx)), (0, self.text_max_len + 1 - (len(tru_text)+1)), 'constant', constant_values=(self.pad_idx, self.pad_idx))
#             pad_text_out = tru_text_out
            
#         data=Datax()
#         cg = self.code_graphs[index]
#         data['node'].x = torch.tensor(cg['nodes'])
#         data['node'].src_map = torch.tensor(cg['node2text_map_ids']).long()
#         data['node'].code_mask = torch.tensor(cg['code_node_mask']).bool()
        
#         for key in ['parent_child_hyperedges', 'line_hyperedges', 'block_hyperedges', 'dfg_hyperedges', 'layout_sibling_hyperedges']:
#             data['node', key, 'node'].edge_index = torch.tensor(cg.get(key, [])).long()

#         ast_s, ast_d = build_directed_hyperedges_from_simple(cg.get('base_father2child_edges', []), 'src')
#         data['node', 'ast_dir_s', 'node'].edge_index, data['node', 'ast_dir_d', 'node'].edge_index = torch.tensor(ast_s).long(), torch.tensor(ast_d).long()
#         dfg_s, dfg_d = build_directed_hyperedges_from_simple(cg.get('dfg_prev2next_edges', []), 'dst')
#         data['node', 'dfg_dir_s', 'node'].edge_index, data['node', 'dfg_dir_d', 'node'].edge_index = torch.tensor(dfg_s).long(), torch.tensor(dfg_d).long()

#         data['text'].text_token_input = torch.tensor(pad_text_in).long()
#         if self.texts is not None: data['text'].text_token_output = torch.tensor(pad_text_out).long()
#         data['text'].num_nodes = pad_text_in.shape[0]
#         if self.ids is not None:
#             data['idx'].idx = torch.tensor(self.ids[index]); data['idx'].num_nodes = 1
#         return data

#     def __len__(self): return self.len

# # ================= 4. 核心网络：解耦架构 CodeGraphEnc & TNet =================
# class CodeGraphEnc(nn.Module):
#     def __init__(self, emb_dims, graph_max_size, code_max_len, graph_node_emb_op, graph_gnn_layers=6, drop_rate=0., **kwargs):
#         super().__init__()
#         self.graph_max_size, self.code_max_len, self.emb_dims = graph_max_size, code_max_len, emb_dims
#         self.pad_idx = kwargs.get('pad_idx', 0)
#         self.use_hyperedge_pos_emb = kwargs.get('use_hyperedge_pos_emb', True)
#         self.use_directed_hyperedges = kwargs.get('use_directed_hyperedges', True)
#         self.use_dynamic_edges = kwargs.get('use_dynamic_edges', True)
        
#         # 解耦开关接收
#         self.use_enc_attn_res = kwargs.get('use_enc_attn_res', True)
#         self.use_copy_highway = kwargs.get('use_copy_highway', True)
#         self.dynamic_threshold = kwargs.get('dynamic_threshold', 0.85)
#         self.gnn_layers = graph_gnn_layers

#         self.graph_node_emb_op = graph_node_emb_op
#         self.graph_pos_encoding = nn.Embedding(graph_max_size * 2 + 1, emb_dims, padding_idx=self.pad_idx)
#         nn.init.xavier_uniform_(self.graph_pos_encoding.weight[1:, ])
#         self.emb_drop_op = nn.Dropout(p=drop_rate)
        
#         if self.use_enc_attn_res:
#             self.attn_res_ops = nn.ModuleList([NodeAwareAttentionResidual(emb_dims) for _ in range(graph_gnn_layers)])

#         self.gnn_ops, self.dhgat_ops, self.gnorm_ops, self.grelu_ops = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
#         self.hetero_alpha = nn.Parameter(torch.ones(graph_gnn_layers, 8))
        
#         for _ in range(graph_gnn_layers):
#             self.gnn_ops.append(HeteroConv({
#                 ('node', 'block_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, K=1),
#                 ('node', 'line_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, K=1),
#                 ('node', 'layout_sibling_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, K=1),
#                 ('node', 'parent_child_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, K=4),
#                 ('node', 'dfg_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, K=1),
#                 ('node', 'dynamic_semantic_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, K=1)
#             }, aggr='sum'))
#             self.dhgat_ops.append(nn.ModuleDict({
#                 'ast_dir': DirectedHypergraphAttention(emb_dims),
#                 'dfg_dir': DirectedHypergraphAttention(emb_dims)
#             }))
#             self.grelu_ops.append(nn.Sequential(nn.ReLU(), nn.Dropout(p=drop_rate)))
#             self.gnorm_ops.append(GraphNorm(emb_dims))

#     def _add_dynamic_edges(self, data):
#         dense_x, mask = to_dense_batch(data['node'].x, data.x_batch_dict['node'], fill_value=0.0) 
#         sim_matrix = torch.bmm(F.normalize(dense_x, p=2, dim=-1), F.normalize(dense_x, p=2, dim=-1).transpose(1, 2)) 
#         sim_matrix.diagonal(dim1=1, dim2=2).fill_(-1.0)
#         adj = (sim_matrix > self.dynamic_threshold) & (mask.unsqueeze(1) & mask.unsqueeze(2))
#         b_idx, r, c = adj.nonzero(as_tuple=True)
#         flat = torch.zeros_like(mask, dtype=torch.long); flat[mask] = torch.arange(mask.sum(), device=mask.device)
#         data.edge_index_dict[('node', 'dynamic_semantic_hyperedges', 'node')] = torch.stack([flat[b_idx, r], flat[b_idx, c]], dim=0)

#     def forward(self, data):
#         graph_node_emb = self.graph_node_emb_op(data.x_dict['node']) * np.sqrt(self.emb_dims)
#         batch_size = data.x_batch_dict['node'].max().item() + 1
#         pos_indices_list = []
#         for b in range(batch_size):
#             num = (data.x_batch_dict['node'] == b).sum().item()
#             max_p = self.graph_pos_encoding.num_embeddings - 1
#             pos_indices_list.append((torch.arange(1, num + 1, device=graph_node_emb.device) % max_p) + 1)
#         pos_emb = self.graph_pos_encoding(torch.cat(pos_indices_list)) 
        
#         if self.use_hyperedge_pos_emb: 
#             graph_node_emb = graph_node_emb + pos_emb
            
#         data['node'].x = self.emb_drop_op(graph_node_emb) 
        
#         # 【旁路提纯】：克隆最底层的纯净特征，送给 Copy Highway
#         initial_node_x = data['node'].x.clone()
        
#         # 【主干深潜】：维护历史特征供 AttnRes 提取深层语义
#         history_x = [data['node'].x]
        
#         for i, (gnn, dhgat, relu, norm) in enumerate(zip(self.gnn_ops, self.dhgat_ops, self.grelu_ops, self.gnorm_ops)):
#             if self.use_dynamic_edges and i == (self.gnn_layers // 2): 
#                 self._add_dynamic_edges(data)
            
#             # 主干网络利用 AttnRes 按需检索历史层
#             x_input = self.attn_res_ops[i](history_x) if self.use_enc_attn_res else history_x[-1]
#             data['node'].x = x_input
            
#             x_dict = gnn(x_dict=data.x_dict, edge_index_dict=data.edge_index_dict)
#             out_x = x_dict['node']
            
#             if self.use_directed_hyperedges:
#                 a_s, a_d = data.edge_index_dict.get(('node','ast_dir_s','node')), data.edge_index_dict.get(('node','ast_dir_d','node'))
#                 if a_s is not None and a_s.numel()>0: 
#                     out_x = out_x + self.hetero_alpha[i, 4] * dhgat['ast_dir'](x_input, a_s, a_d)
#                 d_s, d_d = data.edge_index_dict.get(('node','dfg_dir_s','node')), data.edge_index_dict.get(('node','dfg_dir_d','node'))
#                 if d_s is not None and d_s.numel()>0: 
#                     out_x = out_x + self.hetero_alpha[i, 5] * dhgat['dfg_dir'](x_input, d_s, d_d)

#             data['node'].x = norm(x_input + relu(out_x))
#             history_x.append(data['node'].x)

#         data['node'].x = history_x[-1]
        
#         # 【终极解耦融合】：深层抽象特征 + 底层高速公路纯净特征
#         if self.use_copy_highway:
#             data['node'].x = data['node'].x + initial_node_x
        
#         # 严格锁死 batch_size 提取致密张量
#         graph_enc,_ = to_dense_batch(data.x_dict['node'], batch=data.x_batch_dict['node'], fill_value=self.pad_idx, max_num_nodes=self.graph_max_size, batch_size=batch_size)  
#         cm = data['node'].code_mask; cb = data.x_batch_dict['node'][cm]
#         code_src_map,_ = to_dense_batch(data.src_map_dict['node'][cm], batch=cb, fill_value=self.pad_idx, max_num_nodes=self.code_max_len, batch_size=batch_size)    
#         graph_code_enc,_ = to_dense_batch(data.x_dict['node'][cm], batch=cb, fill_value=self.pad_idx, max_num_nodes=self.code_max_len, batch_size=batch_size)    
#         return graph_enc, graph_code_enc, code_src_map

# class Dec(nn.Module):
#     def __init__(self, emb_dims, text_voc_size, text_emb_op, text_max_len, enc_out_dims, att_layers, att_heads, att_head_dims=None, ff_hid_dims=2048, drop_rate=0., **kwargs):
#         super().__init__()
#         self.emb_dims = emb_dims
#         self._copy = kwargs.get('copy', True)
        
#         # 解耦开关接收
#         self.use_dec_attn_res = kwargs.get('use_dec_attn_res', True)
#         self.use_copy_highway = kwargs.get('use_copy_highway', True)
        
#         self.text_emb_op = text_emb_op
#         self.pos_encoding = PosEnc(max_len=text_max_len+1, emb_dims=emb_dims, train=True, pad=True, pad_idx=kwargs.get('pad_idx', 0)) 
#         self.emb_layer_norm = nn.LayerNorm(emb_dims)
#         self.text_dec_op = TranDec(query_dims=emb_dims, key_dims=enc_out_dims, head_nums=att_heads, head_dims=att_head_dims, layer_num=att_layers, ff_hid_dims=ff_hid_dims, drop_rate=drop_rate, pad_idx=kwargs.get('pad_idx', 0), self_causality=True)
        
#         if self.use_dec_attn_res:
#             self.dec_attn_res_ops = nn.ModuleList([DecoderAttentionResidual(emb_dims) for _ in range(att_layers)])
            
#         self.dropout = nn.Dropout(p=drop_rate)
#         self.out_fc = nn.Linear(emb_dims, text_voc_size)
#         self.copy_generator = MultiCopyGenerator(tgt_dims=emb_dims, tgt_voc_size=text_voc_size, src_dims=enc_out_dims, att_heads=att_heads, att_head_dims=att_head_dims, drop_rate=drop_rate, pad_idx=kwargs.get('pad_idx', 0))

#     def forward(self, graph_enc, graph_code_enc, code_src_map, text_input):
#         text_emb = self.text_emb_op(text_input)   
#         text_emb = text_emb * np.sqrt(self.emb_dims)
#         pos_emb = self.pos_encoding(text_input)  
#         text_dec = self.emb_layer_norm(self.dropout(text_emb.add(pos_emb)))  
        
#         # 【旁路提纯】：克隆解码端初始特征
#         initial_text_dec = text_dec.clone()
        
#         graph_mask = graph_enc.abs().sum(-1).sign()  
#         text_mask = text_input.abs().sign()  
        
#         # 【主干深潜】：维护历史供时序级 AttnRes 使用
#         history_q = [text_dec] 
        
#         for i, block in enumerate(self.text_dec_op.dec_blocks):
#             q_input = self.dec_attn_res_ops[i](history_q) if self.use_dec_attn_res else history_q[-1]
#             q_out = block(query=q_input, key=graph_enc, query_mask=text_mask, key_mask=graph_mask)
#             history_q.append(q_out)
            
#         text_dec = history_q[-1] 
        
#         if hasattr(self.text_dec_op, 'layer_norm') and getattr(self.text_dec_op, 'layer_norm') is not None:
#             text_dec = self.text_dec_op.layer_norm(text_dec)

#         # 【终极解耦融合】：将未经注意力稀释的原始 Query 叠加进网络，彻底激活指针网络！
#         if self.use_copy_highway:
#             text_dec = text_dec + initial_text_dec

#         if not self._copy: return self.out_fc(text_dec).transpose(1, 2)
#         return self.copy_generator(text_dec, graph_code_enc, code_src_map).transpose(1, 2)

# class TNet(BaseNet):
#     def __init__(self, emb_dims, graph_max_size, code_max_len, text_max_len, io_voc_size, text_voc_size, graph_gnn_layers=6, drop_rate=0., **kwargs):
#         super().__init__()
#         self.use_cl, self.cl_temp, self.edge_drop_rate = kwargs.get('use_cl',True), kwargs.get('cl_temp',0.1), kwargs.get('edge_drop_rate',0.15)
#         io_emb = nn.Embedding(io_voc_size, emb_dims, padding_idx=kwargs.get('pad_idx',0)); nn.init.xavier_uniform_(io_emb.weight[1:, ])
#         self.enc_op = CodeGraphEnc(emb_dims, graph_max_size, code_max_len, io_emb, graph_gnn_layers, drop_rate, **kwargs)
#         self.dec_op = Dec(emb_dims, text_voc_size, io_emb, text_max_len, emb_dims, kwargs.get('text_att_layers',8), kwargs.get('text_att_heads',8), ff_hid_dims=kwargs.get('text_ff_hid_dims',2048), drop_rate=drop_rate, **kwargs)

#     def augment(self, data):
#         aug = deepcopy(data)
#         for et in aug.edge_index_dict.keys():
#             idx = aug.edge_index_dict[et]
#             if idx.numel() > 0: 
#                 aug.edge_index_dict[et] = idx[:, torch.rand(idx.size(1), device=idx.device) > self.edge_drop_rate]
#         return aug

#     def forward(self, data):
#         text_in = data['text'].text_token_input.clone()
#         del data['text']
#         if self.training and self.use_cl:
#             cg_orig, cg_aug = deepcopy(data), self.augment(data)
#             g_enc, g_code, src_m = self.enc_op(cg_orig)
#             g_enc_a, _, _ = self.enc_op(cg_aug)
#             out = self.dec_op(g_enc, g_code, src_m, text_in)
#             z1, z2 = F.normalize(g_enc.mean(1), p=2, dim=-1), F.normalize(g_enc_a.mean(1), p=2, dim=-1)
#             sim = torch.matmul(z1, z2.T) / self.cl_temp
#             loss_cl = (F.cross_entropy(sim, torch.arange(z1.size(0), device=z1.device)) + F.cross_entropy(sim.T, torch.arange(z1.size(0), device=z1.device))) / 2
#             return out, loss_cl
        
#         g_enc, g_code, src_m = self.enc_op(data)
#         return self.dec_op(g_enc, g_code, src_m, text_in)

# class TModel(TransSeq2Seq):
#     def __init__(self, model_dir, model_name='Transformer_based_model', model_id=None, emb_dims=512, graph_gnn_layers=3, graph_GNN=SAGEConv, graph_gnn_aggr='add', text_att_layers=3, text_att_heads=8, text_att_head_dims=None, text_ff_hid_dims=2048, drop_rate=0., copy=True, pad_idx=0, train_batch_size=32, pred_batch_size=32, max_train_size=-1, max_valid_size=32 * 10, max_big_epochs=20, regular_rate=1e-5, lr_base=0.001, lr_decay=0.9, min_lr_rate=0.01, warm_big_epochs=2, start_valid_epoch=20, early_stop=20, Net=TNet, Dataset=Datasetx, beam_width=1, train_metrics=[get_sent_bleu], valid_metric=get_sent_bleu, test_metrics=[get_sent_bleu], train_mode=True, **kwargs):
#         logging.info('Construct %s' % model_name)
#         super().__init__(model_name=model_name, model_dir=model_dir, model_id=model_id)
#         self.init_params = locals()
#         self.emb_dims = emb_dims
#         self.graph_gnn_layers = graph_gnn_layers
#         self.graph_GNN = graph_GNN
#         self.graph_gnn_aggr = graph_gnn_aggr
#         self.text_att_layers = text_att_layers
#         self.text_att_heads = text_att_heads
#         self.text_att_head_dims = text_att_head_dims
#         self.text_ff_hid_dims = text_ff_hid_dims
#         self.drop_rate = drop_rate
#         self.pad_idx = pad_idx
#         self.copy = copy
#         self.train_batch_size = train_batch_size
#         self.pred_batch_size = pred_batch_size
#         self.max_train_size = max_train_size
#         self.max_valid_size = max_valid_size
#         self.max_big_epochs = max_big_epochs
#         self.regular_rate = regular_rate
#         self.lr_base = lr_base
#         self.lr_decay = lr_decay
#         self.min_lr_rate = min_lr_rate
#         self.warm_big_epochs = warm_big_epochs
#         self.start_valid_epoch = start_valid_epoch
#         self.early_stop = early_stop
#         self.Net = Net
#         self.Dataset = Dataset
#         self.beam_width = beam_width
#         self.train_metrics = train_metrics
#         self.valid_metric = valid_metric
#         self.test_metrics = test_metrics
#         self.train_mode = train_mode
        
#         # 接收全套创新开关
#         self.use_enc_attn_res = kwargs.get('use_enc_attn_res', True)
#         self.use_dec_attn_res = kwargs.get('use_dec_attn_res', True)
#         self.use_copy_highway = kwargs.get('use_copy_highway', True)
        
#         self.use_directed_hyperedges = kwargs.get('use_directed_hyperedges', True)
#         self.use_hyperedge_pos_emb = kwargs.get('use_hyperedge_pos_emb', True)
#         self.use_dynamic_edges = kwargs.get('use_dynamic_edges', True)
#         self.use_cl = kwargs.get('use_cl', True)
#         self.cl_weight = kwargs.get('cl_weight', 0.05)
#         self.cl_temp = kwargs.get('cl_temp', 0.1)
#         self.edge_drop_rate = kwargs.get('edge_drop_rate', 0.15)
#         self.dynamic_threshold = kwargs.get('dynamic_threshold', 0.85)

#     def _logging_paramerter_num(self):
#         logging.info("{} have {} paramerters in total".format(self.model_name, sum( x.numel() for x in self.net.parameters() if x.requires_grad)))

#     def fit(self, train_data, valid_data, **kwargs):
#         self.graph_max_size, self.code_max_len, self.io_voc_size, self.text_max_len=0, 0, 0, 0
#         for code_graph,text in zip(train_data['code_graphs'],train_data['texts']):
#             self.graph_max_size = max(self.graph_max_size,len(code_graph['nodes']))
#             self.code_max_len = max(self.code_max_len,code_graph['code_node_mask'].sum())
#             self.io_voc_size = max(self.io_voc_size,max(code_graph['nodes']))
#             self.text_max_len=max(self.text_max_len,len(text))
#         self.io_voc_size+=1

#         self.text_voc_size = len(train_data['text_dic']['text_i2w']) 
#         self.io_voc_size=max(self.io_voc_size,self.text_voc_size+2*self.code_max_len)
        
#         net = self.Net(
#             emb_dims=self.emb_dims, graph_max_size=self.graph_max_size, code_max_len=self.code_max_len, text_max_len=self.text_max_len, io_voc_size=self.io_voc_size, text_voc_size=self.text_voc_size, graph_gnn_layers=self.graph_gnn_layers, graph_GNN=self.graph_GNN, graph_gnn_aggr=self.graph_gnn_aggr,
#             text_att_layers=self.text_att_layers, text_att_heads=self.text_att_heads, text_att_head_dims=self.text_att_head_dims, text_ff_hid_dims=self.text_ff_hid_dims, 
#             drop_rate=self.drop_rate, pad_idx=self.pad_idx, copy=self.copy, 
#             use_enc_attn_res=self.use_enc_attn_res,
#             use_dec_attn_res=self.use_dec_attn_res,
#             use_copy_highway=self.use_copy_highway,
#             use_hyperedge_pos_emb=self.use_hyperedge_pos_emb, 
#             use_directed_hyperedges=self.use_directed_hyperedges, 
#             use_dynamic_edges=self.use_dynamic_edges, 
#             use_cl=self.use_cl,
#             cl_weight=self.cl_weight,
#             cl_temp=self.cl_temp,
#             edge_drop_rate=self.edge_drop_rate,
#             dynamic_threshold=self.dynamic_threshold
#         )
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
#         self.net =DataParallel(net.to(device),follow_batch=['x'])  
#         self._logging_paramerter_num()  
#         self.net.train()  

#         self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr_base, weight_decay=self.regular_rate)
#         self.criterion = LabelSmoothSoftmaxCEV2(reduction='mean', ignore_index=self.pad_idx, label_smooth=0.0)

#         self.text_begin_idx = self.text_voc_size - 1
#         self.text_end_idx = self.text_voc_size - 2
#         self.tgt_begin_idx,self.tgt_end_idx=self.text_begin_idx,self.text_end_idx

#         self.max_train_size = len(train_data['code_graphs']) if self.max_train_size == -1 else self.max_train_size
#         train_code_graphs, train_texts,train_ids = zip(*random.sample(list(zip(train_data['code_graphs'], train_data['texts'],train_data['ids'])), min(self.max_train_size, len(train_data['code_graphs']))))
#         train_set = self.Dataset(code_graphs=train_code_graphs, texts=train_texts, ids=train_ids, text_max_len=self.text_max_len, text_begin_idx=self.text_begin_idx, text_end_idx=self.text_end_idx, pad_idx=self.pad_idx)
#         train_loader=DataListLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True, drop_last=True) 

#         if self.warm_big_epochs is None: self.warm_big_epochs = max(self.max_big_epochs // 10, 2)
#         self.scheduler = LrWarmUp(self.optimizer, min_rate=self.min_lr_rate, lr_decay=self.lr_decay, warm_steps=self.warm_big_epochs * len(train_loader), reduce_steps=len(train_loader))  
        
#         if self.train_mode:  
#             accumulation_steps = 1
            
#             for i in range(0, self.max_big_epochs):
#                 pbar = tqdm(train_loader)
#                 self.optimizer.zero_grad() 
                
#                 for j, batch_data in enumerate(pbar):
#                     batch_text_output = []
#                     ids=[]
#                     for data in batch_data:
#                         batch_text_output.append(data['text'].text_token_output.unsqueeze(0))
#                         del data['text'].text_token_output
#                         ids.append(data['idx'].idx.item())
#                         del data['idx']

#                     batch_text_output = torch.cat(batch_text_output, dim=0).to(device)
                    
#                     if self.use_cl:
#                         pred_text_output, loss_cl = self.net(batch_data)
#                         loss_cl = loss_cl.mean()
#                         loss_ce = self.criterion(pred_text_output, batch_text_output)
#                         loss = loss_ce + self.cl_weight * loss_cl
#                     else:
#                         pred_text_output = self.net(batch_data)
#                         loss_ce = self.criterion(pred_text_output, batch_text_output)
#                         loss = loss_ce

#                     loss = loss / accumulation_steps
#                     loss.backward()  
                    
#                     if (j + 1) % accumulation_steps == 0 or (j + 1) == len(train_loader):
#                         clip_grad_norm_(self.net.parameters(), 2.0)
#                         self.optimizer.step()  
#                         self.scheduler.step()  
#                         self.optimizer.zero_grad()

#                     text_dic = {'text_i2w': train_data['text_dic']['text_i2w'], 'ex_text_i2ws': [train_data['text_dic']['ex_text_i2ws'][k] for k in ids]}
#                     log_info=self._get_log_fit_eval(loss=loss_ce, pred_tgt=pred_text_output, gold_tgt=batch_text_output, tgt_i2w=text_dic)
                    
#                     if self.use_cl:
#                         log_info = '[Big epoch:{}/{}, CE:{:.3f}, CL:{:.3f}, {}]'.format(i + 1, self.max_big_epochs, loss_ce.item(), loss_cl.item(), log_info)
#                     else:
#                         log_info = '[Big epoch:{}/{},{}]'.format(i + 1, self.max_big_epochs, log_info)
#                     pbar.set_description(log_info)
#                     del pred_text_output, batch_text_output, batch_data
                
#                 torch.cuda.empty_cache()
#                 del pbar
                
#                 if i+1 >= self.start_valid_epoch:
#                     self.max_valid_size = len(valid_data['code_graphs']) if self.max_valid_size == -1 else self.max_valid_size
#                     valid_srcs, valid_tgts, ex_text_i2ws = zip(*random.sample(list(zip(valid_data['code_graphs'], valid_data['texts'], valid_data['text_dic']['ex_text_i2ws'])), min(self.max_valid_size, len(valid_data['code_graphs']))))
#                     text_dic = {'text_i2w': train_data['text_dic']['text_i2w'], 'ex_text_i2ws': ex_text_i2ws}
#                     worse_epochs = self._do_validation(valid_srcs=valid_srcs, valid_tgts=valid_tgts, tgt_i2w=text_dic, increase_better=True, last=False)  
#                     if worse_epochs>=self.early_stop: break
                    
#         self._do_validation(valid_srcs=valid_data['code_graphs'], valid_tgts=valid_data['texts'], tgt_i2w=valid_data['text_dic'], increase_better=True, last=True)  
#         self._logging_paramerter_num()  

#     def predict(self, code_graphs, text_dic):
#         logging.info('Predict outputs of %s' % self.model_name)
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
#         self.net.eval()  
#         enc_op=DataParallel(self.net.module.enc_op,follow_batch=['x'])
#         dec_op=torch.nn.DataParallel(self.net.module.dec_op)
#         data_set = self.Dataset(code_graphs=code_graphs, texts=None, ids=None, text_max_len=self.text_max_len, text_begin_idx=self.text_begin_idx, text_end_idx=self.text_end_idx, pad_idx=self.pad_idx)  

#         data_loader = DataListLoader(dataset=data_set, batch_size=self.pred_batch_size, shuffle=False)
#         pred_text_id_np_batches = []  
#         with torch.no_grad():  
#             pbar = tqdm(data_loader)
#             for batch_data in pbar:
#                 batch_text_input = []
#                 for data in batch_data:
#                     batch_text_input.append(data['text'].text_token_input.unsqueeze(0))
#                     del data['text']
#                 batch_text_input = torch.cat(batch_text_input, dim=0).to(device)

#                 batch_graph_enc,batch_graph_code_enc,batch_code_src_map=enc_op(batch_data)
#                 batch_text_output: list = []  
#                 if self.beam_width == 1:
#                     for i in range(self.text_max_len + 1):  
#                         pred_out = dec_op(graph_enc=batch_graph_enc,graph_code_enc=batch_graph_code_enc,code_src_map=batch_code_src_map,text_input=batch_text_input)  
#                         batch_text_output.append(pred_out[:, :, i].unsqueeze(-1).to('cpu').data.numpy())  
#                         if i < self.text_max_len:  
#                             batch_text_input[:, i + 1] = torch.argmax(pred_out[:, :, i], dim=1)
#                     batch_pred_text = np.concatenate(batch_text_output, axis=-1)[:, :, :-1]  
#                     batch_pred_text[:, self.tgt_begin_idx, :] = -np.inf  
#                     batch_pred_text[:, self.pad_idx, :] = -np.inf  
#                     batch_pred_text_np = np.argmax(batch_pred_text, axis=1)  
#                     pred_text_id_np_batches.append(batch_pred_text_np)  
#                 else:
#                     batch_pred_text=trans_beam_search(net=dec_op, beam_width=self.beam_width, dec_input_arg_name='text_input', length_penalty=1, begin_idx=self.tgt_begin_idx, pad_idx=self.pad_idx, end_idx=self.tgt_end_idx, graph_enc=batch_graph_enc, graph_code_enc=batch_graph_code_enc, code_src_map=batch_code_src_map, text_input=batch_text_input)     
#                     pred_text_id_np_batches.append(batch_pred_text.to('cpu').data.numpy()[:,:-1])  

#         pred_text_id_np = np.concatenate(pred_text_id_np_batches,axis=0)  
#         self.net.train()  
#         return self._tgt_ids2tokens(pred_text_id_np, text_dic, self.text_end_idx)
    
#     def generate_texts(self,code_graphs,text_dic,res_path,gold_texts,raw_data,token_data,**kwargs):
#         logging.info('>>>>>>>Generate the targets according to sources and save the result to {}'.format(res_path))
#         kwargs.setdefault('beam_width',1)
#         res_dir=os.path.dirname(res_path)
#         if not os.path.exists(res_dir): os.makedirs(res_dir)
#         pred_texts=self.predict(code_graphs=code_graphs, text_dic=text_dic)
#         gold_texts=self._tgt_ids2tokens(gold_texts,text_dic,self.pad_idx)
#         res_data = []
#         for i,(pred_text,gold_text,raw_item,token_item) in enumerate(zip(pred_texts,gold_texts,raw_data,token_data)):
#             sent_bleu=self.valid_metric([pred_text],[gold_text])
#             res_data.append(dict(pred_text=' '.join(pred_text), gold_text=' '.join(gold_text), sent_bleu=sent_bleu, raw_code=raw_item['code'], raw_text=raw_item['text'], id=raw_item['id'], token_text=token_item['text'],))
#         with codecs.open(res_path,'w',encoding='utf-8') as f:
#             json.dump(res_data,f,indent=4, ensure_ascii=False)
#         logging.info('>>>>>>>The result has been saved to {}'.format(res_path))

#     def _code_ids2tokens(self,code_idss, code_i2w, end_idx):
#         return [[code_i2w[idx] for idx in (code_ids[:code_ids.tolist().index(end_idx)] if end_idx in code_ids else code_ids)] for code_ids in code_idss]
    
#     def _tgt_ids2tokens(self, text_id_np, text_dic, end_idx=0, **kwargs):
#         if self.copy:
#             text_tokens: list = []
#             for j, text_ids in enumerate(text_id_np):
#                 text_i2w = {**text_dic['text_i2w'], **text_dic['ex_text_i2ws'][j]}
#                 end_i = text_ids.tolist().index(end_idx) if end_idx in text_ids else len(text_ids)
#                 text_tokens.append([text_i2w[text_idx] for text_idx in text_ids[:end_i]])
#         else:
#             text_i2w=text_dic['text_i2w']
#             text_tokens = [[text_i2w[idx] for idx in (text_ids[:text_ids.tolist().index(end_idx)] if end_idx in text_ids else text_ids)] for text_ids in text_id_np]
#         return text_tokens

# if __name__ == '__main__':
#     params.setdefault('use_enc_attn_res', True)
#     params.setdefault('use_dec_attn_res', True)
#     params.setdefault('use_copy_highway', True)
#     params.setdefault('use_directed_hyperedges', True) 
#     params.setdefault('use_hyperedge_pos_emb', True)
#     params.setdefault('use_dynamic_edges', True)
#     params.setdefault('use_cl', True)
#     params.setdefault('cl_weight', 0.05)
#     params.setdefault('cl_temp', 0.1)
#     params.setdefault('edge_drop_rate', 0.15)
#     params.setdefault('dynamic_threshold', 0.85)

#     logging.info('Parameters are listed below: \n'+'\n'.join(['{}: {}'.format(key,value) for key,value in params.items()]))

#     model = TModel(
#                    model_dir=params['model_dir'],
#                    model_name=params['model_name'],
#                    model_id=params['model_id'],
#                    emb_dims=params['emb_dims'],
#                    graph_gnn_layers=params['graph_gnn_layers'],
#                    graph_GNN=params['graph_GNN'],
#                    graph_gnn_aggr=params['graph_gnn_aggr'],
#                    text_att_layers=params['text_att_layers'],
#                    text_att_heads=params['text_att_heads'],
#                    text_att_head_dims=params['text_att_head_dims'],
#                    text_ff_hid_dims=params['text_ff_hid_dims'],
#                    drop_rate=params['drop_rate'],
#                    copy=params['copy'],
#                    pad_idx=params['pad_idx'],
#                    train_batch_size=params['train_batch_size'],
#                    pred_batch_size=params['pred_batch_size'],
#                    max_train_size=params['max_train_size'],  
#                    max_valid_size=params['max_valid_size'],  
#                    max_big_epochs=params['max_big_epochs'],
#                    regular_rate=params['regular_rate'],
#                    lr_base=params['lr_base'],
#                    lr_decay=params['lr_decay'],
#                    min_lr_rate=params['min_lr_rate'],
#                    warm_big_epochs=params['warm_big_epochs'],
#                    early_stop=params['early_stop'],
#                    start_valid_epoch=params['start_valid_epoch'],
                   
#                    use_enc_attn_res=params['use_enc_attn_res'],
#                    use_dec_attn_res=params['use_dec_attn_res'],
#                    use_copy_highway=params['use_copy_highway'],
                   
#                    use_directed_hyperedges=params['use_directed_hyperedges'],
#                    use_hyperedge_pos_emb=params['use_hyperedge_pos_emb'],
#                    use_dynamic_edges=params['use_dynamic_edges'],
#                    use_cl=params['use_cl'],
#                    cl_weight=params['cl_weight'],
#                    cl_temp=params['cl_temp'],
#                    edge_drop_rate=params['edge_drop_rate'],
#                    dynamic_threshold=params['dynamic_threshold'],
#                    Net=TNet,
#                    Dataset=Datasetx,
#                    beam_width=params['beam_width'],
#                    train_metrics=train_metrics,
#                    valid_metric=valid_metric,
#                    test_metrics=test_metrics,
#                    train_mode=params['train_mode'])

#     logging.info('Load data ...')
#     with codecs.open(train_avail_data_path, 'rb') as f: train_data = pickle.load(f)
#     with codecs.open(valid_avail_data_path, 'rb') as f: valid_data = pickle.load(f)
#     with codecs.open(test_avail_data_path, 'rb') as f: test_data = pickle.load(f)
#     with codecs.open(test_token_data_path,'r') as f: test_token_data=json.load(f)
#     with codecs.open(test_raw_data_path,'r') as f: test_raw_data=json.load(f)

#     model.fit(train_data=train_data, valid_data=valid_data)

#     test_eval_df=model.eval(test_srcs=test_data['code_graphs'], test_tgts=test_data['texts'], tgt_i2w=test_data['text_dic'])
#     logging.info('Model performance on test dataset:\n')
#     for i in range(0,len(test_eval_df.columns),4): print(test_eval_df.iloc[:, i:i+4])

#     model.generate_texts(code_graphs=test_data['code_graphs'], text_dic=test_data['text_dic'], res_path=res_path, gold_texts=test_data['texts'], raw_data=test_raw_data, token_data=test_token_data)

#跳跃知识网络
# # coding=utf-8
# import os
# import re
# import sys
# sys.path.append('../../../')
# from lib.neural_module.learn_strategy import LrWarmUp
# from lib.neural_module.transformer import TranEnc, TranDec, DualTranDec,ResFF,ResMHA
# from lib.neural_module.embedding import PosEnc,SinusoidalPositionalEncoding
# from lib.neural_module.loss import LabelSmoothSoftmaxCEV2, CriterionNet
# from lib.neural_module.balanced_data_parallel import BalancedDataParallel
# from lib.neural_module.copy_attention import DualMultiCopyGenerator,MultiCopyGenerator,DualCopyGenerator
# from lib.neural_module.beam_search import trans_beam_search
# from lib.neural_model.seq_to_seq_model import TransSeq2Seq
# from lib.neural_model.base_model import BaseNet
# from lib.neural_module.transformer import ResFF
    
# from typing import Any,Optional,Union
# from config import *

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler
# from torch.nn.utils import clip_grad_norm_
# from torch.utils.data import Dataset
# from torch_geometric.data import HeteroData
# from torch_geometric.loader.data_list_loader import DataListLoader
# from torch_geometric.utils import to_dense_batch
# from torch_geometric.data.storage import (BaseStorage, NodeStorage,EdgeStorage)
# from torch_geometric.nn.data_parallel import DataParallel
# from torch_geometric.nn import HeteroConv,GraphNorm,HypergraphConv,GATConv

# from torch_scatter import scatter, scatter_add
# from torch_geometric.utils import softmax as scatter_softmax
# from torch_geometric.utils import degree
# import random
# import numpy as np
# import logging
# import pickle
# import json
# import codecs
# from tqdm import tqdm
# import math
# from copy import deepcopy 

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# NodeOrEdgeStorage = Union[NodeStorage, EdgeStorage]

# # ================= 1. 图算子：防弹版 K阶扩散 & DHGAT =================
# class HyperedgeDiffusionConv(nn.Module):
#     def __init__(self, in_channels, out_channels, K=1, alpha=0.5, bias=True):
#         super(HyperedgeDiffusionConv, self).__init__()
#         self.K = K
#         self._alpha_init = alpha
#         self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
#         self.lin = nn.Linear(in_channels, out_channels, bias=bias)
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.lin.reset_parameters()
#         if hasattr(self, 'alpha'): self.alpha.data.fill_(self._alpha_init)

#     def forward(self, x, hyperedge_index, num_nodes=None, num_edges=None):
#         if num_nodes is None: num_nodes = x.size(0)
#         if num_edges is None: num_edges = int(hyperedge_index[1].max()) + 1
#         node_idx, edge_idx = hyperedge_index

#         d_v = degree(node_idx, num_nodes, dtype=x.dtype)
#         d_v[d_v == 0] = 1.0
#         d_v_inv_sqrt = d_v.pow(-0.5)
#         d_v_inv = d_v.pow(-1.0) 
#         d_v_norm_term = d_v_inv[node_idx] 

#         d_e = degree(edge_idx, num_edges, dtype=x.dtype)
#         d_e[d_e == 0] = 1.0
#         d_e_inv = d_e.pow(-1.0)

#         x = self.lin(x)
#         x_temp = x * d_v_inv_sqrt.unsqueeze(-1)
#         H_e = scatter_add(x_temp[node_idx], edge_idx, dim=0, dim_size=num_edges) * d_e_inv.unsqueeze(-1)
#         H_e_0 = H_e 

#         cur_alpha = torch.clamp(self.alpha, min=0.0, max=1.0)
#         for k in range(self.K):
#             H_node_temp = H_e[edge_idx] * d_v_norm_term.unsqueeze(-1)
#             H_e_diffused = scatter_add(H_node_temp, edge_idx, dim=0, dim_size=num_edges) * d_e_inv.unsqueeze(-1)
#             H_e = (1 - cur_alpha) * H_e_diffused + cur_alpha * H_e_0

#         out = scatter_add(H_e[edge_idx], node_idx, dim=0, dim_size=num_nodes) * d_v_inv_sqrt.unsqueeze(-1)
#         return out

# def build_directed_hyperedges_from_simple(edges, group_by='src'):
#     if edges is None or np.size(edges) == 0:
#         return np.empty((2, 0), dtype=np.int64), np.empty((2, 0), dtype=np.int64)
#     src_nodes, dst_nodes = np.array(edges[0]), np.array(edges[1])
#     src_hyper_edges, dst_hyper_edges = [], []
#     if group_by == 'src':
#         unique_srcs = np.unique(src_nodes)
#         for h_id, src in enumerate(unique_srcs):
#             src_hyper_edges.append([src, h_id])
#             for child in dst_nodes[src_nodes == src]: dst_hyper_edges.append([child, h_id])  
#     else:
#         unique_dsts = np.unique(dst_nodes)
#         for h_id, dst in enumerate(unique_dsts):
#             dst_hyper_edges.append([dst, h_id])
#             for parent in src_nodes[dst_nodes == dst]: src_hyper_edges.append([parent, h_id])
#     return np.array(src_hyper_edges, dtype=np.int64).T, np.array(dst_hyper_edges, dtype=np.int64).T

# class DirectedHypergraphAttention(nn.Module):
#     def __init__(self, emb_dims):
#         super().__init__()
#         self.att_src = nn.Linear(emb_dims, 1, bias=False)
#         self.att_dst = nn.Linear(emb_dims, 1, bias=False)
#         self.leaky_relu = nn.LeakyReLU(0.2)
#         self.out_proj = nn.Linear(emb_dims, emb_dims, bias=False)

#     def forward(self, x, edge_index_src, edge_index_dst):
#         if edge_index_src.numel() == 0: return torch.zeros_like(x)
#         src_nodes, src_edges = edge_index_src[0], edge_index_src[1]
#         dst_nodes, dst_edges = edge_index_dst[0], edge_index_dst[1]
#         num_he = max(src_edges.max().item(), dst_edges.max().item()) + 1
        
#         alpha_src = scatter_softmax(self.leaky_relu(self.att_src(x[src_nodes])), src_edges, dim=0)
#         he_x = scatter(x[src_nodes] * alpha_src, src_edges, dim=0, dim_size=num_he, reduce='sum')
        
#         alpha_dst = scatter_softmax(self.leaky_relu(self.att_dst(he_x[dst_edges] + x[dst_nodes])), dst_nodes, dim=0)
#         out = scatter(he_x[dst_edges] * alpha_dst, dst_nodes, dim=0, dim_size=x.size(0), reduce='sum')
#         return self.out_proj(out)

# # ================= 2. 数据预处理 =================
# class Datax(HeteroData):
#     def __cat_dim__(self, key: str, value: Any, store: Optional[NodeOrEdgeStorage] = None, *args, **kwargs) -> Any:
#         if bool(re.search('(token)', key)): return None  
#         if bool(re.search('(pos)', key)): return -1
#         return super().__cat_dim__(key, value,store)    

# class Datasetx(Dataset):
#     def __init__(self, code_graphs, texts=None, ids=None, text_max_len=None, text_begin_idx=1, text_end_idx=2, pad_idx=0):
#         self.len = len(code_graphs)  
#         self.text_max_len = max([len(t) for t in texts]) if text_max_len is None and texts is not None else text_max_len
#         self.text_begin_idx, self.text_end_idx, self.pad_idx = text_begin_idx, text_end_idx, pad_idx
#         self.code_graphs, self.texts, self.ids = code_graphs, texts, ids

#     def __getitem__(self, index):
#         if self.texts is None:
#             pad_text_in = np.zeros((self.text_max_len + 1,), dtype=np.int64)
#             pad_text_in[0] = self.text_begin_idx
#             pad_text_out = None
#         else:
#             tru_text = self.texts[index][:self.text_max_len]
#             pad_text_in = np.lib.pad(tru_text, (1, self.text_max_len - len(tru_text)), 'constant', constant_values=(self.text_begin_idx, self.pad_idx))
#             tru_text_out = np.lib.pad(np.lib.pad(tru_text, (0, 1), 'constant', constant_values=(0, self.text_end_idx)), (0, self.text_max_len + 1 - (len(tru_text)+1)), 'constant', constant_values=(self.pad_idx, self.pad_idx))
#             pad_text_out = tru_text_out
            
#         data=Datax()
#         cg = self.code_graphs[index]
#         data['node'].x = torch.tensor(cg['nodes'])
#         data['node'].src_map = torch.tensor(cg['node2text_map_ids']).long()
#         data['node'].code_mask = torch.tensor(cg['code_node_mask']).bool()
        
#         for key in ['parent_child_hyperedges', 'line_hyperedges', 'block_hyperedges', 'dfg_hyperedges', 'layout_sibling_hyperedges']:
#             data['node', key, 'node'].edge_index = torch.tensor(cg.get(key, [])).long()

#         ast_s, ast_d = build_directed_hyperedges_from_simple(cg.get('base_father2child_edges', []), 'src')
#         data['node', 'ast_dir_s', 'node'].edge_index, data['node', 'ast_dir_d', 'node'].edge_index = torch.tensor(ast_s).long(), torch.tensor(ast_d).long()
#         dfg_s, dfg_d = build_directed_hyperedges_from_simple(cg.get('dfg_prev2next_edges', []), 'dst')
#         data['node', 'dfg_dir_s', 'node'].edge_index, data['node', 'dfg_dir_d', 'node'].edge_index = torch.tensor(dfg_s).long(), torch.tensor(dfg_d).long()

#         data['text'].text_token_input = torch.tensor(pad_text_in).long()
#         if self.texts is not None: data['text'].text_token_output = torch.tensor(pad_text_out).long()
#         data['text'].num_nodes = pad_text_in.shape[0]
#         if self.ids is not None:
#             data['idx'].idx = torch.tensor(self.ids[index]); data['idx'].num_nodes = 1
#         return data

#     def __len__(self): return self.len

# # ================= 3. 核心网络：CodeGraphEnc (引入 JK-Net) =================
# class CodeGraphEnc(nn.Module):
#     def __init__(self, emb_dims, graph_max_size, code_max_len, graph_node_emb_op, graph_gnn_layers=6, drop_rate=0., **kwargs):
#         super().__init__()
#         self.graph_max_size, self.code_max_len, self.emb_dims = graph_max_size, code_max_len, emb_dims
#         self.pad_idx = kwargs.get('pad_idx', 0)
#         self.use_hyperedge_pos_emb = kwargs.get('use_hyperedge_pos_emb', True)
#         self.use_directed_hyperedges = kwargs.get('use_directed_hyperedges', True)
#         self.use_dynamic_edges = kwargs.get('use_dynamic_edges', True)
        
#         # 终极解耦大招：跳跃知识网络
#         self.use_jk_readout = kwargs.get('use_jk_readout', True)
#         self.dynamic_threshold = kwargs.get('dynamic_threshold', 0.85)
#         self.gnn_layers = graph_gnn_layers

#         self.graph_node_emb_op = graph_node_emb_op
#         self.graph_pos_encoding = nn.Embedding(graph_max_size * 2 + 1, emb_dims, padding_idx=self.pad_idx)
#         nn.init.xavier_uniform_(self.graph_pos_encoding.weight[1:, ])
#         self.emb_drop_op = nn.Dropout(p=drop_rate)
        
#         self.gnn_ops, self.dhgat_ops, self.gnorm_ops, self.grelu_ops = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
#         self.hetero_alpha = nn.Parameter(torch.ones(graph_gnn_layers, 8))
        
#         for _ in range(graph_gnn_layers):
#             self.gnn_ops.append(HeteroConv({
#                 ('node', 'block_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, K=1),
#                 ('node', 'line_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, K=1),
#                 ('node', 'layout_sibling_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, K=1),
#                 ('node', 'parent_child_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, K=4),
#                 ('node', 'dfg_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, K=1),
#                 ('node', 'dynamic_semantic_hyperedges', 'node'): HyperedgeDiffusionConv(emb_dims, emb_dims, K=1)
#             }, aggr='sum'))
#             self.dhgat_ops.append(nn.ModuleDict({
#                 'ast_dir': DirectedHypergraphAttention(emb_dims),
#                 'dfg_dir': DirectedHypergraphAttention(emb_dims)
#             }))
#             self.grelu_ops.append(nn.Sequential(nn.ReLU(), nn.Dropout(p=drop_rate)))
#             self.gnorm_ops.append(GraphNorm(emb_dims))
            
#         # JK-Net 读取层配置：拼接 (0层初始 + GNN层输出)，并通过线性层拉回 512 维
#         if self.use_jk_readout:
#             self.jk_linear = nn.Linear(emb_dims * (graph_gnn_layers + 1), emb_dims)
#             self.jk_norm = nn.LayerNorm(emb_dims)

#     def _add_dynamic_edges(self, data):
#         dense_x, mask = to_dense_batch(data['node'].x, data.x_batch_dict['node'], fill_value=0.0) 
#         sim_matrix = torch.bmm(F.normalize(dense_x, p=2, dim=-1), F.normalize(dense_x, p=2, dim=-1).transpose(1, 2)) 
#         sim_matrix.diagonal(dim1=1, dim2=2).fill_(-1.0)
#         adj = (sim_matrix > self.dynamic_threshold) & (mask.unsqueeze(1) & mask.unsqueeze(2))
#         b_idx, r, c = adj.nonzero(as_tuple=True)
#         flat = torch.zeros_like(mask, dtype=torch.long); flat[mask] = torch.arange(mask.sum(), device=mask.device)
#         data.edge_index_dict[('node', 'dynamic_semantic_hyperedges', 'node')] = torch.stack([flat[b_idx, r], flat[b_idx, c]], dim=0)

#     def forward(self, data):
#         graph_node_emb = self.graph_node_emb_op(data.x_dict['node']) * np.sqrt(self.emb_dims)
#         batch_size = data.x_batch_dict['node'].max().item() + 1
#         pos_indices_list = []
#         for b in range(batch_size):
#             num = (data.x_batch_dict['node'] == b).sum().item()
#             max_p = self.graph_pos_encoding.num_embeddings - 1
#             pos_indices_list.append((torch.arange(1, num + 1, device=graph_node_emb.device) % max_p) + 1)
#         pos_emb = self.graph_pos_encoding(torch.cat(pos_indices_list)) 
        
#         if self.use_hyperedge_pos_emb: 
#             graph_node_emb = graph_node_emb + pos_emb
            
#         data['node'].x = self.emb_drop_op(graph_node_emb) 
        
#         # 收集每一层输出，第 0 个是最纯净的未污染特征！
#         all_layer_outputs = [data['node'].x]
        
#         for i, (gnn, dhgat, relu, norm) in enumerate(zip(self.gnn_ops, self.dhgat_ops, self.grelu_ops, self.gnorm_ops)):
#             if self.use_dynamic_edges and i == (self.gnn_layers // 2): 
#                 self._add_dynamic_edges(data)
            
#             x_input = data['node'].x
#             x_dict = gnn(x_dict=data.x_dict, edge_index_dict=data.edge_index_dict)
#             out_x = x_dict['node']
            
#             if self.use_directed_hyperedges:
#                 a_s, a_d = data.edge_index_dict.get(('node','ast_dir_s','node')), data.edge_index_dict.get(('node','ast_dir_d','node'))
#                 if a_s is not None and a_s.numel()>0: 
#                     out_x = out_x + self.hetero_alpha[i, 4] * dhgat['ast_dir'](x_input, a_s, a_d)
#                 d_s, d_d = data.edge_index_dict.get(('node','dfg_dir_s','node')), data.edge_index_dict.get(('node','dfg_dir_d','node'))
#                 if d_s is not None and d_s.numel()>0: 
#                     out_x = out_x + self.hetero_alpha[i, 5] * dhgat['dfg_dir'](x_input, d_s, d_d)

#             # 标准图网络更新
#             data['node'].x = norm(x_input + relu(out_x))
            
#             # 将本层输出收集入列表
#             all_layer_outputs.append(data['node'].x)

#         # =========================================================
#         # 【致胜一击】：Jumping Knowledge 多分辨率特征融合
#         # 将 0层(词法) 到 6层(深层语义) 的所有特征 Concatenate 在一起
#         # 并通过 Linear 层完美拉回 512 维的对齐空间！
#         # =========================================================
#         if self.use_jk_readout:
#             concat_x = torch.cat(all_layer_outputs, dim=-1)  # shape: [Nodes, 7 * 512]
#             data['node'].x = self.jk_norm(self.jk_linear(concat_x)) # shape: [Nodes, 512]
#         else:
#             data['node'].x = all_layer_outputs[-1] # Fallback
            
#         # 严格锁死 batch_size 提取特征
#         graph_enc,_ = to_dense_batch(data.x_dict['node'], batch=data.x_batch_dict['node'], fill_value=self.pad_idx, max_num_nodes=self.graph_max_size, batch_size=batch_size)  
#         cm = data['node'].code_mask; cb = data.x_batch_dict['node'][cm]
#         code_src_map,_ = to_dense_batch(data.src_map_dict['node'][cm], batch=cb, fill_value=self.pad_idx, max_num_nodes=self.code_max_len, batch_size=batch_size)    
#         graph_code_enc,_ = to_dense_batch(data.x_dict['node'][cm], batch=cb, fill_value=self.pad_idx, max_num_nodes=self.code_max_len, batch_size=batch_size)    
        
#         return graph_enc, graph_code_enc, code_src_map

# class Dec(nn.Module):
#     def __init__(self, emb_dims, text_voc_size, text_emb_op, text_max_len, enc_out_dims, att_layers, att_heads, att_head_dims=None, ff_hid_dims=2048, drop_rate=0., **kwargs):
#         super().__init__()
#         self.emb_dims = emb_dims
#         self._copy = kwargs.get('copy', True)
        
#         self.text_emb_op = text_emb_op
#         self.pos_encoding = PosEnc(max_len=text_max_len+1, emb_dims=emb_dims, train=True, pad=True, pad_idx=kwargs.get('pad_idx', 0)) 
#         self.emb_layer_norm = nn.LayerNorm(emb_dims)
        
#         # 最纯净无污染的 TranDec
#         self.text_dec_op = TranDec(query_dims=emb_dims, key_dims=enc_out_dims, head_nums=att_heads, head_dims=att_head_dims, layer_num=att_layers, ff_hid_dims=ff_hid_dims, drop_rate=drop_rate, pad_idx=kwargs.get('pad_idx', 0), self_causality=True)
        
#         self.dropout = nn.Dropout(p=drop_rate)
#         self.out_fc = nn.Linear(emb_dims, text_voc_size)
#         self.copy_generator = MultiCopyGenerator(tgt_dims=emb_dims, tgt_voc_size=text_voc_size, src_dims=enc_out_dims, att_heads=att_heads, att_head_dims=att_head_dims, drop_rate=drop_rate, pad_idx=kwargs.get('pad_idx', 0))

#     def forward(self, graph_enc, graph_code_enc, code_src_map, text_input):
#         text_emb = self.text_emb_op(text_input)   
#         text_emb = text_emb * np.sqrt(self.emb_dims)
#         pos_emb = self.pos_encoding(text_input)  
#         text_dec = self.emb_layer_norm(self.dropout(text_emb.add(pos_emb)))  
        
#         graph_mask = graph_enc.abs().sum(-1).sign()  
#         text_mask = text_input.abs().sign()  
        
#         text_dec = self.text_dec_op(query=text_dec, key=graph_enc, query_mask=text_mask, key_mask=graph_mask)

#         if not self._copy: return self.out_fc(text_dec).transpose(1, 2)
#         return self.copy_generator(text_dec, graph_code_enc, code_src_map).transpose(1, 2)

# class TNet(BaseNet):
#     def __init__(self, emb_dims, graph_max_size, code_max_len, text_max_len, io_voc_size, text_voc_size, graph_gnn_layers=6, drop_rate=0., **kwargs):
#         super().__init__()
#         self.use_cl, self.cl_temp, self.edge_drop_rate = kwargs.get('use_cl',True), kwargs.get('cl_temp',0.1), kwargs.get('edge_drop_rate',0.15)
#         io_emb = nn.Embedding(io_voc_size, emb_dims, padding_idx=kwargs.get('pad_idx',0)); nn.init.xavier_uniform_(io_emb.weight[1:, ])
#         self.enc_op = CodeGraphEnc(emb_dims, graph_max_size, code_max_len, io_emb, graph_gnn_layers, drop_rate, **kwargs)
#         self.dec_op = Dec(emb_dims, text_voc_size, io_emb, text_max_len, emb_dims, kwargs.get('text_att_layers',8), kwargs.get('text_att_heads',8), ff_hid_dims=kwargs.get('text_ff_hid_dims',2048), drop_rate=drop_rate, **kwargs)

#     def augment(self, data):
#         aug = deepcopy(data)
#         for et in aug.edge_index_dict.keys():
#             idx = aug.edge_index_dict[et]
#             if idx.numel() > 0: 
#                 aug.edge_index_dict[et] = idx[:, torch.rand(idx.size(1), device=idx.device) > self.edge_drop_rate]
#         return aug

#     def forward(self, data):
#         text_in = data['text'].text_token_input.clone()
#         del data['text']
#         if self.training and self.use_cl:
#             cg_orig, cg_aug = deepcopy(data), self.augment(data)
#             g_enc, g_code, src_m = self.enc_op(cg_orig)
#             g_enc_a, _, _ = self.enc_op(cg_aug)
#             out = self.dec_op(g_enc, g_code, src_m, text_in)
#             z1, z2 = F.normalize(g_enc.mean(1), p=2, dim=-1), F.normalize(g_enc_a.mean(1), p=2, dim=-1)
#             sim = torch.matmul(z1, z2.T) / self.cl_temp
#             loss_cl = (F.cross_entropy(sim, torch.arange(z1.size(0), device=z1.device)) + F.cross_entropy(sim.T, torch.arange(z1.size(0), device=z1.device))) / 2
#             return out, loss_cl
        
#         g_enc, g_code, src_m = self.enc_op(data)
#         return self.dec_op(g_enc, g_code, src_m, text_in)

# class TModel(TransSeq2Seq):
#     def __init__(self, model_dir, model_name='Transformer_based_model', model_id=None, emb_dims=512, graph_gnn_layers=3, graph_GNN=SAGEConv, graph_gnn_aggr='add', text_att_layers=3, text_att_heads=8, text_att_head_dims=None, text_ff_hid_dims=2048, drop_rate=0., copy=True, pad_idx=0, train_batch_size=32, pred_batch_size=32, max_train_size=-1, max_valid_size=32 * 10, max_big_epochs=20, regular_rate=1e-5, lr_base=0.001, lr_decay=0.9, min_lr_rate=0.01, warm_big_epochs=2, start_valid_epoch=20, early_stop=20, Net=TNet, Dataset=Datasetx, beam_width=1, train_metrics=[get_sent_bleu], valid_metric=get_sent_bleu, test_metrics=[get_sent_bleu], train_mode=True, **kwargs):
#         logging.info('Construct %s' % model_name)
#         super().__init__(model_name=model_name, model_dir=model_dir, model_id=model_id)
#         self.init_params = locals()
#         self.emb_dims = emb_dims
#         self.graph_gnn_layers = graph_gnn_layers
#         self.graph_GNN = graph_GNN
#         self.graph_gnn_aggr = graph_gnn_aggr
#         self.text_att_layers = text_att_layers
#         self.text_att_heads = text_att_heads
#         self.text_att_head_dims = text_att_head_dims
#         self.text_ff_hid_dims = text_ff_hid_dims
#         self.drop_rate = drop_rate
#         self.pad_idx = pad_idx
#         self.copy = copy
#         self.train_batch_size = train_batch_size
#         self.pred_batch_size = pred_batch_size
#         self.max_train_size = max_train_size
#         self.max_valid_size = max_valid_size
#         self.max_big_epochs = max_big_epochs
#         self.regular_rate = regular_rate
#         self.lr_base = lr_base
#         self.lr_decay = lr_decay
#         self.min_lr_rate = min_lr_rate
#         self.warm_big_epochs = warm_big_epochs
#         self.start_valid_epoch = start_valid_epoch
#         self.early_stop = early_stop
#         self.Net = Net
#         self.Dataset = Dataset
#         self.beam_width = beam_width
#         self.train_metrics = train_metrics
#         self.valid_metric = valid_metric
#         self.test_metrics = test_metrics
#         self.train_mode = train_mode
        
#         self.use_jk_readout = kwargs.get('use_jk_readout', True)
#         self.use_directed_hyperedges = kwargs.get('use_directed_hyperedges', True)
#         self.use_hyperedge_pos_emb = kwargs.get('use_hyperedge_pos_emb', True)
#         self.use_dynamic_edges = kwargs.get('use_dynamic_edges', True)
#         self.use_cl = kwargs.get('use_cl', True)
#         self.cl_weight = kwargs.get('cl_weight', 0.05)
#         self.cl_temp = kwargs.get('cl_temp', 0.1)
#         self.edge_drop_rate = kwargs.get('edge_drop_rate', 0.15)
#         self.dynamic_threshold = kwargs.get('dynamic_threshold', 0.85)

#     def _logging_paramerter_num(self):
#         logging.info("{} have {} paramerters in total".format(self.model_name, sum( x.numel() for x in self.net.parameters() if x.requires_grad)))

#     def fit(self, train_data, valid_data, **kwargs):
#         self.graph_max_size, self.code_max_len, self.io_voc_size, self.text_max_len=0, 0, 0, 0
#         for code_graph,text in zip(train_data['code_graphs'],train_data['texts']):
#             self.graph_max_size = max(self.graph_max_size,len(code_graph['nodes']))
#             self.code_max_len = max(self.code_max_len,code_graph['code_node_mask'].sum())
#             self.io_voc_size = max(self.io_voc_size,max(code_graph['nodes']))
#             self.text_max_len=max(self.text_max_len,len(text))
#         self.io_voc_size+=1

#         self.text_voc_size = len(train_data['text_dic']['text_i2w']) 
#         self.io_voc_size=max(self.io_voc_size,self.text_voc_size+2*self.code_max_len)
        
#         net = self.Net(
#             emb_dims=self.emb_dims, graph_max_size=self.graph_max_size, code_max_len=self.code_max_len, text_max_len=self.text_max_len, io_voc_size=self.io_voc_size, text_voc_size=self.text_voc_size, graph_gnn_layers=self.graph_gnn_layers, graph_GNN=self.graph_GNN, graph_gnn_aggr=self.graph_gnn_aggr,
#             text_att_layers=self.text_att_layers, text_att_heads=self.text_att_heads, text_att_head_dims=self.text_att_head_dims, text_ff_hid_dims=self.text_ff_hid_dims, 
#             drop_rate=self.drop_rate, pad_idx=self.pad_idx, copy=self.copy, 
#             use_jk_readout=self.use_jk_readout,
#             use_hyperedge_pos_emb=self.use_hyperedge_pos_emb, 
#             use_directed_hyperedges=self.use_directed_hyperedges, 
#             use_dynamic_edges=self.use_dynamic_edges, 
#             use_cl=self.use_cl,
#             cl_weight=self.cl_weight,
#             cl_temp=self.cl_temp,
#             edge_drop_rate=self.edge_drop_rate,
#             dynamic_threshold=self.dynamic_threshold
#         )
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
#         self.net =DataParallel(net.to(device),follow_batch=['x'])  
#         self._logging_paramerter_num()  
#         self.net.train()  

#         self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr_base, weight_decay=self.regular_rate)
#         self.criterion = LabelSmoothSoftmaxCEV2(reduction='mean', ignore_index=self.pad_idx, label_smooth=0.0)

#         self.text_begin_idx = self.text_voc_size - 1
#         self.text_end_idx = self.text_voc_size - 2
#         self.tgt_begin_idx,self.tgt_end_idx=self.text_begin_idx,self.text_end_idx

#         self.max_train_size = len(train_data['code_graphs']) if self.max_train_size == -1 else self.max_train_size
#         train_code_graphs, train_texts,train_ids = zip(*random.sample(list(zip(train_data['code_graphs'], train_data['texts'],train_data['ids'])), min(self.max_train_size, len(train_data['code_graphs']))))
#         train_set = self.Dataset(code_graphs=train_code_graphs, texts=train_texts, ids=train_ids, text_max_len=self.text_max_len, text_begin_idx=self.text_begin_idx, text_end_idx=self.text_end_idx, pad_idx=self.pad_idx)
#         train_loader=DataListLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True, drop_last=True) 

#         if self.warm_big_epochs is None: self.warm_big_epochs = max(self.max_big_epochs // 10, 2)
#         self.scheduler = LrWarmUp(self.optimizer, min_rate=self.min_lr_rate, lr_decay=self.lr_decay, warm_steps=self.warm_big_epochs * len(train_loader), reduce_steps=len(train_loader))  
        
#         if self.train_mode:  
#             accumulation_steps = 1
            
#             for i in range(0, self.max_big_epochs):
#                 pbar = tqdm(train_loader)
#                 self.optimizer.zero_grad() 
                
#                 for j, batch_data in enumerate(pbar):
#                     batch_text_output = []
#                     ids=[]
#                     for data in batch_data:
#                         batch_text_output.append(data['text'].text_token_output.unsqueeze(0))
#                         del data['text'].text_token_output
#                         ids.append(data['idx'].idx.item())
#                         del data['idx']

#                     batch_text_output = torch.cat(batch_text_output, dim=0).to(device)
                    
#                     if self.use_cl:
#                         pred_text_output, loss_cl = self.net(batch_data)
#                         loss_cl = loss_cl.mean()
#                         loss_ce = self.criterion(pred_text_output, batch_text_output)
#                         loss = loss_ce + self.cl_weight * loss_cl
#                     else:
#                         pred_text_output = self.net(batch_data)
#                         loss_ce = self.criterion(pred_text_output, batch_text_output)
#                         loss = loss_ce

#                     loss = loss / accumulation_steps
#                     loss.backward()  
                    
#                     if (j + 1) % accumulation_steps == 0 or (j + 1) == len(train_loader):
#                         clip_grad_norm_(self.net.parameters(), 2.0)
#                         self.optimizer.step()  
#                         self.scheduler.step()  
#                         self.optimizer.zero_grad()

#                     text_dic = {'text_i2w': train_data['text_dic']['text_i2w'], 'ex_text_i2ws': [train_data['text_dic']['ex_text_i2ws'][k] for k in ids]}
#                     log_info=self._get_log_fit_eval(loss=loss_ce, pred_tgt=pred_text_output, gold_tgt=batch_text_output, tgt_i2w=text_dic)
                    
#                     if self.use_cl:
#                         log_info = '[Big epoch:{}/{}, CE:{:.3f}, CL:{:.3f}, {}]'.format(i + 1, self.max_big_epochs, loss_ce.item(), loss_cl.item(), log_info)
#                     else:
#                         log_info = '[Big epoch:{}/{},{}]'.format(i + 1, self.max_big_epochs, log_info)
#                     pbar.set_description(log_info)
#                     del pred_text_output, batch_text_output, batch_data
                
#                 torch.cuda.empty_cache()
#                 del pbar
                
#                 if i+1 >= self.start_valid_epoch:
#                     self.max_valid_size = len(valid_data['code_graphs']) if self.max_valid_size == -1 else self.max_valid_size
#                     valid_srcs, valid_tgts, ex_text_i2ws = zip(*random.sample(list(zip(valid_data['code_graphs'], valid_data['texts'], valid_data['text_dic']['ex_text_i2ws'])), min(self.max_valid_size, len(valid_data['code_graphs']))))
#                     text_dic = {'text_i2w': train_data['text_dic']['text_i2w'], 'ex_text_i2ws': ex_text_i2ws}
#                     worse_epochs = self._do_validation(valid_srcs=valid_srcs, valid_tgts=valid_tgts, tgt_i2w=text_dic, increase_better=True, last=False)  
#                     if worse_epochs>=self.early_stop: break
                    
#         self._do_validation(valid_srcs=valid_data['code_graphs'], valid_tgts=valid_data['texts'], tgt_i2w=valid_data['text_dic'], increase_better=True, last=True)  
#         self._logging_paramerter_num()  

#     def predict(self, code_graphs, text_dic):
#         logging.info('Predict outputs of %s' % self.model_name)
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
#         self.net.eval()  
#         enc_op=DataParallel(self.net.module.enc_op,follow_batch=['x'])
#         dec_op=torch.nn.DataParallel(self.net.module.dec_op)
#         data_set = self.Dataset(code_graphs=code_graphs, texts=None, ids=None, text_max_len=self.text_max_len, text_begin_idx=self.text_begin_idx, text_end_idx=self.text_end_idx, pad_idx=self.pad_idx)  

#         data_loader = DataListLoader(dataset=data_set, batch_size=self.pred_batch_size, shuffle=False)
#         pred_text_id_np_batches = []  
#         with torch.no_grad():  
#             pbar = tqdm(data_loader)
#             for batch_data in pbar:
#                 batch_text_input = []
#                 for data in batch_data:
#                     batch_text_input.append(data['text'].text_token_input.unsqueeze(0))
#                     del data['text']
#                 batch_text_input = torch.cat(batch_text_input, dim=0).to(device)

#                 batch_graph_enc,batch_graph_code_enc,batch_code_src_map=enc_op(batch_data)
#                 batch_text_output: list = []  
#                 if self.beam_width == 1:
#                     for i in range(self.text_max_len + 1):  
#                         pred_out = dec_op(graph_enc=batch_graph_enc,graph_code_enc=batch_graph_code_enc,code_src_map=batch_code_src_map,text_input=batch_text_input)  
#                         batch_text_output.append(pred_out[:, :, i].unsqueeze(-1).to('cpu').data.numpy())  
#                         if i < self.text_max_len:  
#                             batch_text_input[:, i + 1] = torch.argmax(pred_out[:, :, i], dim=1)
#                     batch_pred_text = np.concatenate(batch_text_output, axis=-1)[:, :, :-1]  
#                     batch_pred_text[:, self.tgt_begin_idx, :] = -np.inf  
#                     batch_pred_text[:, self.pad_idx, :] = -np.inf  
#                     batch_pred_text_np = np.argmax(batch_pred_text, axis=1)  
#                     pred_text_id_np_batches.append(batch_pred_text_np)  
#                 else:
#                     batch_pred_text=trans_beam_search(net=dec_op, beam_width=self.beam_width, dec_input_arg_name='text_input', length_penalty=1, begin_idx=self.tgt_begin_idx, pad_idx=self.pad_idx, end_idx=self.tgt_end_idx, graph_enc=batch_graph_enc, graph_code_enc=batch_graph_code_enc, code_src_map=batch_code_src_map, text_input=batch_text_input)     
#                     pred_text_id_np_batches.append(batch_pred_text.to('cpu').data.numpy()[:,:-1])  

#         pred_text_id_np = np.concatenate(pred_text_id_np_batches,axis=0)  
#         self.net.train()  
#         return self._tgt_ids2tokens(pred_text_id_np, text_dic, self.text_end_idx)
    
#     def generate_texts(self,code_graphs,text_dic,res_path,gold_texts,raw_data,token_data,**kwargs):
#         logging.info('>>>>>>>Generate the targets according to sources and save the result to {}'.format(res_path))
#         kwargs.setdefault('beam_width',1)
#         res_dir=os.path.dirname(res_path)
#         if not os.path.exists(res_dir): os.makedirs(res_dir)
#         pred_texts=self.predict(code_graphs=code_graphs, text_dic=text_dic)
#         gold_texts=self._tgt_ids2tokens(gold_texts,text_dic,self.pad_idx)
#         res_data = []
#         for i,(pred_text,gold_text,raw_item,token_item) in enumerate(zip(pred_texts,gold_texts,raw_data,token_data)):
#             sent_bleu=self.valid_metric([pred_text],[gold_text])
#             res_data.append(dict(pred_text=' '.join(pred_text), gold_text=' '.join(gold_text), sent_bleu=sent_bleu, raw_code=raw_item['code'], raw_text=raw_item['text'], id=raw_item['id'], token_text=token_item['text'],))
#         with codecs.open(res_path,'w',encoding='utf-8') as f:
#             json.dump(res_data,f,indent=4, ensure_ascii=False)
#         logging.info('>>>>>>>The result has been saved to {}'.format(res_path))

#     def _code_ids2tokens(self,code_idss, code_i2w, end_idx):
#         return [[code_i2w[idx] for idx in (code_ids[:code_ids.tolist().index(end_idx)] if end_idx in code_ids else code_ids)] for code_ids in code_idss]
    
#     def _tgt_ids2tokens(self, text_id_np, text_dic, end_idx=0, **kwargs):
#         if self.copy:
#             text_tokens: list = []
#             for j, text_ids in enumerate(text_id_np):
#                 text_i2w = {**text_dic['text_i2w'], **text_dic['ex_text_i2ws'][j]}
#                 end_i = text_ids.tolist().index(end_idx) if end_idx in text_ids else len(text_ids)
#                 text_tokens.append([text_i2w[text_idx] for text_idx in text_ids[:end_i]])
#         else:
#             text_i2w=text_dic['text_i2w']
#             text_tokens = [[text_i2w[idx] for idx in (text_ids[:text_ids.tolist().index(end_idx)] if end_idx in text_ids else text_ids)] for text_ids in text_id_np]
#         return text_tokens

# if __name__ == '__main__':
#     params.setdefault('use_jk_readout', True)
#     params.setdefault('use_directed_hyperedges', True) 
#     params.setdefault('use_hyperedge_pos_emb', True)
#     params.setdefault('use_dynamic_edges', True)
#     params.setdefault('use_cl', True)
#     params.setdefault('cl_weight', 0.05)
#     params.setdefault('cl_temp', 0.1)
#     params.setdefault('edge_drop_rate', 0.15)
#     params.setdefault('dynamic_threshold', 0.85)

#     logging.info('Parameters are listed below: \n'+'\n'.join(['{}: {}'.format(key,value) for key,value in params.items()]))

#     model = TModel(
#                    model_dir=params['model_dir'],
#                    model_name=params['model_name'],
#                    model_id=params['model_id'],
#                    emb_dims=params['emb_dims'],
#                    graph_gnn_layers=params['graph_gnn_layers'],
#                    graph_GNN=params['graph_GNN'],
#                    graph_gnn_aggr=params['graph_gnn_aggr'],
#                    text_att_layers=params['text_att_layers'],
#                    text_att_heads=params['text_att_heads'],
#                    text_att_head_dims=params['text_att_head_dims'],
#                    text_ff_hid_dims=params['text_ff_hid_dims'],
#                    drop_rate=params['drop_rate'],
#                    copy=params['copy'],
#                    pad_idx=params['pad_idx'],
#                    train_batch_size=params['train_batch_size'],
#                    pred_batch_size=params['pred_batch_size'],
#                    max_train_size=params['max_train_size'],  
#                    max_valid_size=params['max_valid_size'],  
#                    max_big_epochs=params['max_big_epochs'],
#                    regular_rate=params['regular_rate'],
#                    lr_base=params['lr_base'],
#                    lr_decay=params['lr_decay'],
#                    min_lr_rate=params['min_lr_rate'],
#                    warm_big_epochs=params['warm_big_epochs'],
#                    early_stop=params['early_stop'],
#                    start_valid_epoch=params['start_valid_epoch'],
                   
#                    use_jk_readout=params['use_jk_readout'],
                   
#                    use_directed_hyperedges=params['use_directed_hyperedges'],
#                    use_hyperedge_pos_emb=params['use_hyperedge_pos_emb'],
#                    use_dynamic_edges=params['use_dynamic_edges'],
#                    use_cl=params['use_cl'],
#                    cl_weight=params['cl_weight'],
#                    cl_temp=params['cl_temp'],
#                    edge_drop_rate=params['edge_drop_rate'],
#                    dynamic_threshold=params['dynamic_threshold'],
#                    Net=TNet,
#                    Dataset=Datasetx,
#                    beam_width=params['beam_width'],
#                    train_metrics=train_metrics,
#                    valid_metric=valid_metric,
#                    test_metrics=test_metrics,
#                    train_mode=params['train_mode'])

#     logging.info('Load data ...')
#     with codecs.open(train_avail_data_path, 'rb') as f: train_data = pickle.load(f)
#     with codecs.open(valid_avail_data_path, 'rb') as f: valid_data = pickle.load(f)
#     with codecs.open(test_avail_data_path, 'rb') as f: test_data = pickle.load(f)
#     with codecs.open(test_token_data_path,'r') as f: test_token_data=json.load(f)
#     with codecs.open(test_raw_data_path,'r') as f: test_raw_data=json.load(f)

#     model.fit(train_data=train_data, valid_data=valid_data)

#     test_eval_df=model.eval(test_srcs=test_data['code_graphs'], test_tgts=test_data['texts'], tgt_i2w=test_data['text_dic'])
#     logging.info('Model performance on test dataset:\n')
#     for i in range(0,len(test_eval_df.columns),4): print(test_eval_df.iloc[:, i:i+4])

#     model.generate_texts(code_graphs=test_data['code_graphs'], text_dic=test_data['text_dic'], res_path=res_path, gold_texts=test_data['texts'], raw_data=test_raw_data, token_data=test_token_data)


#双轨
# coding=utf-8
import os
import re
import sys
import math
import random
import numpy as np
import logging
import pickle
import json
import codecs
from copy import deepcopy 
from tqdm import tqdm
from typing import Any,Optional,Union

sys.path.append('../../../')
from lib.neural_module.learn_strategy import LrWarmUp
from lib.neural_module.transformer import TranEnc, TranDec, DualTranDec,ResFF,ResMHA
from lib.neural_module.embedding import PosEnc,SinusoidalPositionalEncoding
from lib.neural_module.loss import LabelSmoothSoftmaxCEV2, CriterionNet
from lib.neural_module.copy_attention import DualMultiCopyGenerator,MultiCopyGenerator,DualCopyGenerator
from lib.neural_module.beam_search import trans_beam_search
from lib.neural_model.seq_to_seq_model import TransSeq2Seq
from lib.neural_model.base_model import BaseNet

from config import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.loader.data_list_loader import DataListLoader
from torch_geometric.utils import to_dense_batch
from torch_geometric.data.storage import (BaseStorage, NodeStorage,EdgeStorage)
from torch_geometric.nn.data_parallel import DataParallel

# 引入 PyG 官方极其稳定且强大的超图与异构卷积算子
from torch_geometric.nn import HeteroConv, GraphNorm, HypergraphConv, GATConv,SAGEConv

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
NodeOrEdgeStorage = Union[NodeStorage, EdgeStorage]

# ================= 1. 数据预处理 (极细粒度解析所有边) =================
class Datax(HeteroData):
    def __cat_dim__(self, key: str, value: Any, store: Optional[NodeOrEdgeStorage] = None, *args, **kwargs) -> Any:
        if bool(re.search('(token)', key)): return None  
        if bool(re.search('(pos)', key)): return -1
        return super().__cat_dim__(key, value,store)    

class Datasetx(Dataset):
    def __init__(self, code_graphs, texts=None, ids=None, text_max_len=None, text_begin_idx=1, text_end_idx=2, pad_idx=0):
        self.len = len(code_graphs)  
        self.text_max_len = max([len(t) for t in texts]) if text_max_len is None and texts is not None else text_max_len
        self.text_begin_idx, self.text_end_idx, self.pad_idx = text_begin_idx, text_end_idx, pad_idx
        self.code_graphs, self.texts, self.ids = code_graphs, texts, ids

    def __getitem__(self, index):
        if self.texts is None:
            pad_text_in = np.zeros((self.text_max_len + 1,), dtype=np.int64)
            pad_text_in[0] = self.text_begin_idx
            pad_text_out = None
        else:
            tru_text = self.texts[index][:self.text_max_len]
            pad_text_in = np.lib.pad(tru_text, (1, self.text_max_len - len(tru_text)), 'constant', constant_values=(self.text_begin_idx, self.pad_idx))
            tru_text_out = np.lib.pad(np.lib.pad(tru_text, (0, 1), 'constant', constant_values=(0, self.text_end_idx)), (0, self.text_max_len + 1 - (len(tru_text)+1)), 'constant', constant_values=(self.pad_idx, self.pad_idx))
            pad_text_out = tru_text_out
            
        data=Datax()
        cg = self.code_graphs[index]
        data['node'].x = torch.tensor(cg['nodes'])
        data['node'].src_map = torch.tensor(cg['node2text_map_ids']).long()
        data['node'].code_mask = torch.tensor(cg['code_node_mask']).bool()
        
        # 【致胜细节】：将超边、AST边和代码/DFG流严格拆分命名，供后续双轨独立调用！
        def add_edge(raw_key, rename_key):
            if raw_key in cg and len(cg[raw_key]) > 0:
                data['node', rename_key, 'node'].edge_index = torch.tensor(cg[raw_key], dtype=torch.long)
                
        # 1. 宏观语义超边 (Macro Hyperedges)
        add_edge('parent_child_hyperedges', 'hc_parent_child')
        add_edge('block_hyperedges', 'hc_block')
        add_edge('line_hyperedges', 'hc_line')
        add_edge('layout_sibling_hyperedges', 'hc_layout')
        
        # 2. 宏观 AST 结构边 (Macro AST Edges)
        add_edge('base_child2father_edges', 'ast_c2f')
        add_edge('base_father2child_edges', 'ast_f2c')
        
        # 3. 微观词法与数据流边 (Micro Lexical Edges - 用于拯救 Copy 机制)
        add_edge('code_prev2next_edges', 'code_p2n')
        add_edge('code_next2prev_edges', 'code_n2p')
        add_edge('dfg_prev2next_edges', 'dfg_p2n')
        add_edge('dfg_next2prev_edges', 'dfg_n2p')

        data['text'].text_token_input = torch.tensor(pad_text_in).long()
        if self.texts is not None: data['text'].text_token_output = torch.tensor(pad_text_out).long()
        data['text'].num_nodes = pad_text_in.shape[0]
        if self.ids is not None:
            data['idx'].idx = torch.tensor(self.ids[index]); data['idx'].num_nodes = 1
        return data
    def __len__(self): return self.len

# ================= 2. 核心网络：HDT-Net 异构双轨网络 =================
class CodeGraphEnc(nn.Module):
    def __init__(self, emb_dims, graph_max_size, code_max_len, graph_node_emb_op, graph_gnn_layers=4, drop_rate=0.2, **kwargs):
        super().__init__()
        self.graph_max_size, self.code_max_len, self.emb_dims = graph_max_size, code_max_len, emb_dims
        self.pad_idx = kwargs.get('pad_idx', 0)
        self.gnn_layers = graph_gnn_layers
        self.micro_layers = kwargs.get('micro_gnn_layers', 2)

        self.graph_node_emb_op = graph_node_emb_op
        self.graph_pos_encoding = nn.Embedding(graph_max_size * 2 + 1, emb_dims, padding_idx=self.pad_idx)
        nn.init.xavier_uniform_(self.graph_pos_encoding.weight[1:, ])
        self.emb_drop_op = nn.Dropout(p=drop_rate)
        
        # ---------------------------------------------------------
        # 轨 1：宏观语义轨 (Macro-Semantic Track) - 喂给 Transformer Decoder
        # 【修复】：关闭 use_attention，让超图回归纯粹的全局上下文平滑！
        # ---------------------------------------------------------
        self.macro_convs, self.macro_norms, self.macro_relus = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(self.gnn_layers):
            self.macro_convs.append(HeteroConv({
                ('node', 'hc_parent_child', 'node'): HypergraphConv(emb_dims, emb_dims, use_attention=False, dropout=drop_rate),
                ('node', 'hc_block', 'node'): HypergraphConv(emb_dims, emb_dims, use_attention=False, dropout=drop_rate),
                ('node', 'hc_line', 'node'): HypergraphConv(emb_dims, emb_dims, use_attention=False, dropout=drop_rate),
                ('node', 'hc_layout', 'node'): HypergraphConv(emb_dims, emb_dims, use_attention=False, dropout=drop_rate),
                ('node', 'ast_c2f', 'node'): GATConv(emb_dims, emb_dims//4, heads=4, concat=True, dropout=drop_rate, add_self_loops=False),
                ('node', 'ast_f2c', 'node'): GATConv(emb_dims, emb_dims//4, heads=4, concat=True, dropout=drop_rate, add_self_loops=False)
            }, aggr='mean'))
            self.macro_relus.append(nn.Sequential(nn.ReLU(), nn.Dropout(p=drop_rate)))
            self.macro_norms.append(GraphNorm(emb_dims))
            
        # ---------------------------------------------------------
        # 轨 2：微观词法轨 (Micro-Lexical Track) - 喂给 Copy 机制
        # ---------------------------------------------------------
        self.micro_convs, self.micro_norms, self.micro_relus = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(self.micro_layers):
            self.micro_convs.append(HeteroConv({
                ('node', 'code_p2n', 'node'): GATConv(emb_dims, emb_dims//4, heads=4, concat=True, dropout=drop_rate, add_self_loops=False),
                ('node', 'code_n2p', 'node'): GATConv(emb_dims, emb_dims//4, heads=4, concat=True, dropout=drop_rate, add_self_loops=False),
                ('node', 'dfg_p2n', 'node'): GATConv(emb_dims, emb_dims//4, heads=4, concat=True, dropout=drop_rate, add_self_loops=False),
                ('node', 'dfg_n2p', 'node'): GATConv(emb_dims, emb_dims//4, heads=4, concat=True, dropout=drop_rate, add_self_loops=False)
            }, aggr='mean'))
            self.micro_relus.append(nn.Sequential(nn.ReLU(), nn.Dropout(p=drop_rate)))
            self.micro_norms.append(GraphNorm(emb_dims))

    def forward(self, data):
        graph_node_emb = self.graph_node_emb_op(data.x_dict['node']) * np.sqrt(self.emb_dims)
        batch_size = data.x_batch_dict['node'].max().item() + 1
        
        # 注入位置编码
        pos_indices_list = []
        for b in range(batch_size):
            num = (data.x_batch_dict['node'] == b).sum().item()
            max_p = self.graph_pos_encoding.num_embeddings - 1
            pos_indices_list.append((torch.arange(1, num + 1, device=graph_node_emb.device) % max_p) + 1)
        graph_node_emb = graph_node_emb + self.graph_pos_encoding(torch.cat(pos_indices_list)) 
        x_base = self.emb_drop_op(graph_node_emb) 
        
        # --- 轨 1：宏观超图传播 ---
        x_macro = x_base.clone()
        for conv, norm, relu in zip(self.macro_convs, self.macro_norms, self.macro_relus):
            macro_edges = {k: v for k, v in data.edge_index_dict.items() if k[1] in ['hc_parent_child', 'hc_block', 'hc_line', 'hc_layout', 'ast_c2f', 'ast_f2c']}
            if len(macro_edges) > 0:
                out = conv(x_dict={'node': x_macro}, edge_index_dict=macro_edges)
                if 'node' in out: x_macro = norm(x_macro + relu(out['node']))

        # --- 轨 2：微观词法传播 ---
        x_micro = x_base.clone()
        for conv, norm, relu in zip(self.micro_convs, self.micro_norms, self.micro_relus):
            micro_edges = {k: v for k, v in data.edge_index_dict.items() if k[1] in ['code_p2n', 'code_n2p', 'dfg_p2n', 'dfg_n2p']}
            if len(micro_edges) > 0:
                out = conv(x_dict={'node': x_micro}, edge_index_dict=micro_edges)
                if 'node' in out: x_micro = norm(x_micro + relu(out['node']))

        # 锁定批次，分别打包输出给对应的 Decoder 模块
        graph_enc,_ = to_dense_batch(x_macro, batch=data.x_batch_dict['node'], fill_value=self.pad_idx, max_num_nodes=self.graph_max_size, batch_size=batch_size)  
        
        cm = data['node'].code_mask; cb = data.x_batch_dict['node'][cm]
        code_src_map,_ = to_dense_batch(data.src_map_dict['node'][cm], batch=cb, fill_value=self.pad_idx, max_num_nodes=self.code_max_len, batch_size=batch_size)    
        
        graph_code_enc,_ = to_dense_batch(x_micro[cm], batch=cb, fill_value=self.pad_idx, max_num_nodes=self.code_max_len, batch_size=batch_size)    
        
        return graph_enc, graph_code_enc, code_src_map

class Dec(nn.Module):
    def __init__(self, emb_dims, text_voc_size, text_emb_op, text_max_len, enc_out_dims, att_layers, att_heads, att_head_dims=None, ff_hid_dims=2048, drop_rate=0., **kwargs):
        super().__init__()
        self.emb_dims = emb_dims
        self._copy = kwargs.get('copy', True)
        
        self.text_emb_op = text_emb_op
        self.pos_encoding = PosEnc(max_len=text_max_len+1, emb_dims=emb_dims, train=True, pad=True, pad_idx=kwargs.get('pad_idx', 0)) 
        self.emb_layer_norm = nn.LayerNorm(emb_dims)
        
        self.text_dec_op = TranDec(query_dims=emb_dims, key_dims=enc_out_dims, head_nums=att_heads, head_dims=att_head_dims, layer_num=att_layers, ff_hid_dims=ff_hid_dims, drop_rate=drop_rate, pad_idx=kwargs.get('pad_idx', 0), self_causality=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.out_fc = nn.Linear(emb_dims, text_voc_size)
        self.copy_generator = MultiCopyGenerator(tgt_dims=emb_dims, tgt_voc_size=text_voc_size, src_dims=enc_out_dims, att_heads=att_heads, att_head_dims=att_head_dims, drop_rate=drop_rate, pad_idx=kwargs.get('pad_idx', 0))

    def forward(self, graph_enc, graph_code_enc, code_src_map, text_input):
        text_emb = self.text_emb_op(text_input) * np.sqrt(self.emb_dims)
        text_dec = self.emb_layer_norm(self.dropout(text_emb.add(self.pos_encoding(text_input))))  
        
        # Transformer 处理宏观语义 graph_enc
        text_dec = self.text_dec_op(query=text_dec, key=graph_enc, query_mask=text_input.abs().sign(), key_mask=graph_enc.abs().sum(-1).sign())
        
        if not self._copy: return self.out_fc(text_dec).transpose(1, 2)
        # Copy 机制处理微观词法 graph_code_enc
        return self.copy_generator(text_dec, graph_code_enc, code_src_map).transpose(1, 2)

class TNet(BaseNet):
    def __init__(self, emb_dims, graph_max_size, code_max_len, text_max_len, io_voc_size, text_voc_size, graph_gnn_layers=4, drop_rate=0.2, **kwargs):
        super().__init__()
        self.use_cl, self.cl_temp, self.edge_drop_rate = kwargs.get('use_cl',True), kwargs.get('cl_temp',0.1), kwargs.get('edge_drop_rate',0.15)
        io_emb = nn.Embedding(io_voc_size, emb_dims, padding_idx=kwargs.get('pad_idx',0)); nn.init.xavier_uniform_(io_emb.weight[1:, ])
        self.enc_op = CodeGraphEnc(emb_dims, graph_max_size, code_max_len, io_emb, graph_gnn_layers, drop_rate, **kwargs)
        self.dec_op = Dec(emb_dims, text_voc_size, io_emb, text_max_len, emb_dims, kwargs.get('text_att_layers',8), kwargs.get('text_att_heads',8), ff_hid_dims=kwargs.get('text_ff_hid_dims',2048), drop_rate=drop_rate, **kwargs)

    def augment(self, data):
        aug = deepcopy(data)
        for et in aug.edge_index_dict.keys():
            idx = aug.edge_index_dict[et]
            if idx.numel() > 0: aug.edge_index_dict[et] = idx[:, torch.rand(idx.size(1), device=idx.device) > self.edge_drop_rate]
        return aug

    def forward(self, data):
        text_in = data['text'].text_token_input.clone()
        del data['text']
        if self.training and self.use_cl:
            cg_orig, cg_aug = deepcopy(data), self.augment(data)
            g_enc, g_code, src_m = self.enc_op(cg_orig)
            g_enc_a, _, _ = self.enc_op(cg_aug)
            out = self.dec_op(g_enc, g_code, src_m, text_in)
            
            # 【完美兼容】：对比学习的 Pull/Push 仅作用于宏观超图语义 (g_enc)！
            # 微观轨彻底免于对比学习强制对齐造成的破坏。
            z1, z2 = F.normalize(g_enc.mean(1), p=2, dim=-1), F.normalize(g_enc_a.mean(1), p=2, dim=-1)
            sim = torch.matmul(z1, z2.T) / self.cl_temp
            loss_cl = (F.cross_entropy(sim, torch.arange(z1.size(0), device=z1.device)) + F.cross_entropy(sim.T, torch.arange(z1.size(0), device=z1.device))) / 2
            return out, loss_cl
        
        g_enc, g_code, src_m = self.enc_op(data)
        return self.dec_op(g_enc, g_code, src_m, text_in)

class TModel(TransSeq2Seq):
    def __init__(self, model_dir, model_name='Transformer_based_model', model_id=None, emb_dims=512, graph_gnn_layers=4, graph_GNN=SAGEConv, graph_gnn_aggr='mean', text_att_layers=8, text_att_heads=8, text_att_head_dims=None, text_ff_hid_dims=2048, drop_rate=0.2, copy=True, pad_idx=0, train_batch_size=64, pred_batch_size=64, max_train_size=-1, max_valid_size=32 * 10, max_big_epochs=100, regular_rate=1e-5, lr_base=0.0005, lr_decay=0.95, min_lr_rate=0.01, warm_big_epochs=4, start_valid_epoch=60, early_stop=12, Net=TNet, Dataset=Datasetx, beam_width=5, train_metrics=[get_sent_bleu], valid_metric=get_sent_bleu, test_metrics=[get_sent_bleu], train_mode=True, **kwargs):
        logging.info('Construct %s' % model_name)
        super().__init__(model_name=model_name, model_dir=model_dir, model_id=model_id)
        self.init_params = locals()
        self.emb_dims, self.graph_gnn_layers, self.graph_GNN, self.graph_gnn_aggr = emb_dims, graph_gnn_layers, graph_GNN, graph_gnn_aggr
        self.text_att_layers, self.text_att_heads, self.text_att_head_dims, self.text_ff_hid_dims = text_att_layers, text_att_heads, text_att_head_dims, text_ff_hid_dims
        self.drop_rate, self.pad_idx, self.copy, self.train_batch_size, self.pred_batch_size = drop_rate, pad_idx, copy, train_batch_size, pred_batch_size
        self.max_train_size, self.max_valid_size, self.max_big_epochs, self.regular_rate = max_train_size, max_valid_size, max_big_epochs, regular_rate
        self.lr_base, self.lr_decay, self.min_lr_rate, self.warm_big_epochs, self.start_valid_epoch, self.early_stop = lr_base, lr_decay, min_lr_rate, warm_big_epochs, start_valid_epoch, early_stop
        self.Net, self.Dataset, self.beam_width, self.train_metrics, self.valid_metric, self.test_metrics, self.train_mode = Net, Dataset, beam_width, train_metrics, valid_metric, test_metrics, train_mode
        
        self.use_hdt_dual_track = kwargs.get('use_hdt_dual_track', True)
        self.use_cl = kwargs.get('use_cl', True)
        self.cl_weight, self.cl_temp, self.edge_drop_rate = kwargs.get('cl_weight', 0.05), kwargs.get('cl_temp', 0.1), kwargs.get('edge_drop_rate', 0.15)
        self.micro_gnn_layers = kwargs.get('micro_gnn_layers', 2)

    def _logging_paramerter_num(self):
        logging.info("{} have {} paramerters in total".format(self.model_name, sum( x.numel() for x in self.net.parameters() if x.requires_grad)))

    def fit(self, train_data, valid_data, **kwargs):
        self.graph_max_size, self.code_max_len, self.io_voc_size, self.text_max_len=0, 0, 0, 0
        for code_graph,text in zip(train_data['code_graphs'],train_data['texts']):
            self.graph_max_size = max(self.graph_max_size,len(code_graph['nodes']))
            self.code_max_len = max(self.code_max_len,code_graph['code_node_mask'].sum())
            self.io_voc_size = max(self.io_voc_size,max(code_graph['nodes']))
            self.text_max_len=max(self.text_max_len,len(text))
        self.io_voc_size+=1
        self.text_voc_size = len(train_data['text_dic']['text_i2w']) 
        self.io_voc_size=max(self.io_voc_size,self.text_voc_size+2*self.code_max_len)
        
        net = self.Net(
            emb_dims=self.emb_dims, graph_max_size=self.graph_max_size, code_max_len=self.code_max_len, text_max_len=self.text_max_len, io_voc_size=self.io_voc_size, text_voc_size=self.text_voc_size, graph_gnn_layers=self.graph_gnn_layers, graph_GNN=self.graph_GNN, graph_gnn_aggr=self.graph_gnn_aggr,
            text_att_layers=self.text_att_layers, text_att_heads=self.text_att_heads, text_att_head_dims=self.text_att_head_dims, text_ff_hid_dims=self.text_ff_hid_dims, 
            drop_rate=self.drop_rate, pad_idx=self.pad_idx, copy=self.copy, micro_gnn_layers=self.micro_gnn_layers,
            use_hdt_dual_track=self.use_hdt_dual_track, use_cl=self.use_cl, cl_weight=self.cl_weight, cl_temp=self.cl_temp, edge_drop_rate=self.edge_drop_rate
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        self.net =DataParallel(net.to(device),follow_batch=['x'])  
        self._logging_paramerter_num()  
        self.net.train()  

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr_base, weight_decay=self.regular_rate)
        self.criterion = LabelSmoothSoftmaxCEV2(reduction='mean', ignore_index=self.pad_idx, label_smooth=0.0)
        self.text_begin_idx, self.text_end_idx = self.text_voc_size - 1, self.text_voc_size - 2
        self.tgt_begin_idx, self.tgt_end_idx = self.text_begin_idx, self.text_end_idx

        self.max_train_size = len(train_data['code_graphs']) if self.max_train_size == -1 else self.max_train_size
        train_code_graphs, train_texts,train_ids = zip(*random.sample(list(zip(train_data['code_graphs'], train_data['texts'],train_data['ids'])), min(self.max_train_size, len(train_data['code_graphs']))))
        train_set = self.Dataset(code_graphs=train_code_graphs, texts=train_texts, ids=train_ids, text_max_len=self.text_max_len, text_begin_idx=self.text_begin_idx, text_end_idx=self.text_end_idx, pad_idx=self.pad_idx)
        train_loader=DataListLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True, drop_last=True) 

        if self.warm_big_epochs is None: self.warm_big_epochs = max(self.max_big_epochs // 10, 2)
        self.scheduler = LrWarmUp(self.optimizer, min_rate=self.min_lr_rate, lr_decay=self.lr_decay, warm_steps=self.warm_big_epochs * len(train_loader), reduce_steps=len(train_loader))  
        
        if self.train_mode:  
            accumulation_steps = 1
            for i in range(0, self.max_big_epochs):
                pbar = tqdm(train_loader)
                self.optimizer.zero_grad() 
                for j, batch_data in enumerate(pbar):
                    batch_text_output, ids = [], []
                    for data in batch_data:
                        batch_text_output.append(data['text'].text_token_output.unsqueeze(0)); ids.append(data['idx'].idx.item())
                        del data['text'].text_token_output; del data['idx']
                    batch_text_output = torch.cat(batch_text_output, dim=0).to(device)
                    
                    if self.use_cl:
                        pred_text_output, loss_cl = self.net(batch_data)
                        loss_ce = self.criterion(pred_text_output, batch_text_output)
                        loss = loss_ce + self.cl_weight * loss_cl.mean()
                    else:
                        pred_text_output = self.net(batch_data)
                        loss_ce, loss = self.criterion(pred_text_output, batch_text_output), loss_ce
                    
                    (loss / accumulation_steps).backward()  
                    if (j + 1) % accumulation_steps == 0 or (j + 1) == len(train_loader):
                        clip_grad_norm_(self.net.parameters(), 2.0); self.optimizer.step(); self.scheduler.step(); self.optimizer.zero_grad()

                    text_dic = {'text_i2w': train_data['text_dic']['text_i2w'], 'ex_text_i2ws': [train_data['text_dic']['ex_text_i2ws'][k] for k in ids]}
                    log_info=self._get_log_fit_eval(loss=loss_ce, pred_tgt=pred_text_output, gold_tgt=batch_text_output, tgt_i2w=text_dic)
                    log_info = '[Big epoch:{}/{}, CE:{:.3f}, CL:{:.3f}, {}]'.format(i + 1, self.max_big_epochs, loss_ce.item(), loss_cl.item(), log_info) if self.use_cl else '[Big epoch:{}/{},{}]'.format(i + 1, self.max_big_epochs, log_info)
                    pbar.set_description(log_info)
                    del pred_text_output, batch_text_output, batch_data
                
                torch.cuda.empty_cache()
                del pbar
                
                if i+1 >= self.start_valid_epoch:
                    self.max_valid_size = len(valid_data['code_graphs']) if self.max_valid_size == -1 else self.max_valid_size
                    valid_srcs, valid_tgts, ex_text_i2ws = zip(*random.sample(list(zip(valid_data['code_graphs'], valid_data['texts'], valid_data['text_dic']['ex_text_i2ws'])), min(self.max_valid_size, len(valid_data['code_graphs']))))
                    text_dic = {'text_i2w': train_data['text_dic']['text_i2w'], 'ex_text_i2ws': ex_text_i2ws}
                    worse_epochs = self._do_validation(valid_srcs=valid_srcs, valid_tgts=valid_tgts, tgt_i2w=text_dic, increase_better=True, last=False)  
                    if worse_epochs>=self.early_stop: break
                    
        self._do_validation(valid_srcs=valid_data['code_graphs'], valid_tgts=valid_data['texts'], tgt_i2w=valid_data['text_dic'], increase_better=True, last=True)  
        self._logging_paramerter_num()  

    def predict(self, code_graphs, text_dic):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        self.net.eval()  
        enc_op=DataParallel(self.net.module.enc_op,follow_batch=['x']); dec_op=torch.nn.DataParallel(self.net.module.dec_op)
        data_set = self.Dataset(code_graphs=code_graphs, texts=None, ids=None, text_max_len=self.text_max_len, text_begin_idx=self.text_begin_idx, text_end_idx=self.text_end_idx, pad_idx=self.pad_idx)  
        data_loader = DataListLoader(dataset=data_set, batch_size=self.pred_batch_size, shuffle=False)
        pred_text_id_np_batches = []  
        with torch.no_grad():  
            for batch_data in tqdm(data_loader):
                batch_text_input = []
                for data in batch_data:
                    batch_text_input.append(data['text'].text_token_input.unsqueeze(0)); del data['text']
                batch_text_input = torch.cat(batch_text_input, dim=0).to(device)
                batch_graph_enc,batch_graph_code_enc,batch_code_src_map=enc_op(batch_data)
                batch_text_output = []  
                if self.beam_width == 1:
                    for i in range(self.text_max_len + 1):  
                        pred_out = dec_op(graph_enc=batch_graph_enc,graph_code_enc=batch_graph_code_enc,code_src_map=batch_code_src_map,text_input=batch_text_input)  
                        batch_text_output.append(pred_out[:, :, i].unsqueeze(-1).to('cpu').data.numpy())  
                        if i < self.text_max_len: batch_text_input[:, i + 1] = torch.argmax(pred_out[:, :, i], dim=1)
                    batch_pred_text = np.concatenate(batch_text_output, axis=-1)[:, :, :-1]  
                    batch_pred_text[:, self.tgt_begin_idx, :] = -np.inf; batch_pred_text[:, self.pad_idx, :] = -np.inf  
                    pred_text_id_np_batches.append(np.argmax(batch_pred_text, axis=1))  
                else:
                    batch_pred_text=trans_beam_search(net=dec_op, beam_width=self.beam_width, dec_input_arg_name='text_input', length_penalty=1, begin_idx=self.tgt_begin_idx, pad_idx=self.pad_idx, end_idx=self.tgt_end_idx, graph_enc=batch_graph_enc, graph_code_enc=batch_graph_code_enc, code_src_map=batch_code_src_map, text_input=batch_text_input)     
                    pred_text_id_np_batches.append(batch_pred_text.to('cpu').data.numpy()[:,:-1])  
        self.net.train()  
        return self._tgt_ids2tokens(np.concatenate(pred_text_id_np_batches,axis=0), text_dic, self.text_end_idx)
    
    def generate_texts(self,code_graphs,text_dic,res_path,gold_texts,raw_data,token_data,**kwargs):
        kwargs.setdefault('beam_width',1)
        if not os.path.exists(os.path.dirname(res_path)): os.makedirs(os.path.dirname(res_path))
        pred_texts=self.predict(code_graphs=code_graphs, text_dic=text_dic)
        gold_texts=self._tgt_ids2tokens(gold_texts,text_dic,self.pad_idx)
        res_data = []
        for i,(pred_text,gold_text,raw_item,token_item) in enumerate(zip(pred_texts,gold_texts,raw_data,token_data)):
            res_data.append(dict(pred_text=' '.join(pred_text), gold_text=' '.join(gold_text), sent_bleu=self.valid_metric([pred_text],[gold_text]), raw_code=raw_item['code'], raw_text=raw_item['text'], id=raw_item['id'], token_text=token_item['text'],))
        with codecs.open(res_path,'w',encoding='utf-8') as f: json.dump(res_data,f,indent=4, ensure_ascii=False)

    def _code_ids2tokens(self,code_idss, code_i2w, end_idx):
        return [[code_i2w[idx] for idx in (code_ids[:code_ids.tolist().index(end_idx)] if end_idx in code_ids else code_ids)] for code_ids in code_idss]
    
    def _tgt_ids2tokens(self, text_id_np, text_dic, end_idx=0, **kwargs):
        if self.copy:
            text_tokens = []
            for j, text_ids in enumerate(text_id_np):
                text_i2w = {**text_dic['text_i2w'], **text_dic['ex_text_i2ws'][j]}
                end_i = text_ids.tolist().index(end_idx) if end_idx in text_ids else len(text_ids)
                text_tokens.append([text_i2w[text_idx] for text_idx in text_ids[:end_i]])
        else:
            text_i2w=text_dic['text_i2w']
            text_tokens = [[text_i2w[idx] for idx in (text_ids[:text_ids.tolist().index(end_idx)] if end_idx in text_ids else text_ids)] for text_ids in text_id_np]
        return text_tokens

if __name__ == '__main__':
    params.setdefault('use_hdt_dual_track', True)
    params.setdefault('use_cl', True)
    params.setdefault('cl_weight', 0.05)
    params.setdefault('cl_temp', 0.1)
    params.setdefault('edge_drop_rate', 0.15)

    logging.info('Parameters are listed below: \n'+'\n'.join(['{}: {}'.format(key,value) for key,value in params.items()]))

    model = TModel(
                   model_dir=params['model_dir'],
                   model_name=params['model_name'],
                   model_id=params['model_id'],
                   emb_dims=params['emb_dims'],
                   graph_gnn_layers=params['graph_gnn_layers'],
                   micro_gnn_layers=params.get('micro_gnn_layers', 2),
                   text_att_layers=params['text_att_layers'],
                   text_att_heads=params['text_att_heads'],
                   text_att_head_dims=params['text_att_head_dims'],
                   text_ff_hid_dims=params['text_ff_hid_dims'],
                   drop_rate=params['drop_rate'],
                   copy=params['copy'],
                   pad_idx=params['pad_idx'],
                   train_batch_size=params['train_batch_size'],
                   pred_batch_size=params['pred_batch_size'],
                   max_train_size=params['max_train_size'],  
                   max_valid_size=params['max_valid_size'],  
                   max_big_epochs=params['max_big_epochs'],
                   regular_rate=params['regular_rate'],
                   lr_base=params['lr_base'],
                   lr_decay=params['lr_decay'],
                   min_lr_rate=params['min_lr_rate'],
                   warm_big_epochs=params['warm_big_epochs'],
                   early_stop=params['early_stop'],
                   start_valid_epoch=params['start_valid_epoch'],
                   
                   use_hdt_dual_track=params['use_hdt_dual_track'],
                   use_cl=params['use_cl'],
                   cl_weight=params['cl_weight'],
                   cl_temp=params['cl_temp'],
                   edge_drop_rate=params['edge_drop_rate'],
                   Net=TNet,
                   Dataset=Datasetx,
                   beam_width=params['beam_width'],
                   train_metrics=train_metrics,
                   valid_metric=valid_metric,
                   test_metrics=test_metrics,
                   train_mode=params['train_mode'])

    logging.info('Load data ...')
    with codecs.open(train_avail_data_path, 'rb') as f: train_data = pickle.load(f)
    with codecs.open(valid_avail_data_path, 'rb') as f: valid_data = pickle.load(f)
    with codecs.open(test_avail_data_path, 'rb') as f: test_data = pickle.load(f)
    with codecs.open(test_token_data_path,'r') as f: test_token_data=json.load(f)
    with codecs.open(test_raw_data_path,'r') as f: test_raw_data=json.load(f)

    model.fit(train_data=train_data, valid_data=valid_data)

    test_eval_df=model.eval(test_srcs=test_data['code_graphs'], test_tgts=test_data['texts'], tgt_i2w=test_data['text_dic'])
    logging.info('Model performance on test dataset:\n')
    for i in range(0,len(test_eval_df.columns),4): print(test_eval_df.iloc[:, i:i+4])

    model.generate_texts(code_graphs=test_data['code_graphs'], text_dic=test_data['text_dic'], res_path=res_path, gold_texts=test_data['texts'], raw_data=test_raw_data, token_data=test_token_data)