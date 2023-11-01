import math

import torch
from Param import *
from torch import nn
from EnvGraph import *
import json


def _init_weights(m):
    if type(m) == nn.Linear:
        fanin = m.weight.data.size(0)
        fanout = m.weight.data.size(1)
        nn.init.xavier_uniform_(m.weight, gain=math.sqrt(2.0/(fanin + fanout)))


class RoomEmbedding(nn.Module):
    def __init__(self):
        super(RoomEmbedding, self).__init__()
        self.fcnn_obs = nn.Sequential(
            nn.Linear(Param.obs_feat_in_num, Param.obs_feat_out_num)
        )
        self.fcnn_gate = nn.Sequential(
            nn.Linear(Param.gate_feat_in_num, Param.gate_feat_out_num)
        )
        self.fcnn_room = nn.Sequential(
            nn.ReLU(),
            nn.Linear(Param.room_emb_size_in, Param.room_emb_size)
        )
        self.fcnn_gate.apply(_init_weights)
        self.fcnn_obs.apply(_init_weights)
        # env_graph -> used for cal node emb with graph structure
        self.env_graph = EnvGraph()

    def forward(self, obs_info, gate_info, method="cat", env_ids=None, route_len=None):
        obs_emb = self.fcnn_obs(obs_info)
        gate_emb = self.fcnn_gate(gate_info)
        if method == "avg":
            avg_obs_emb = self._avg_emb(obs_emb)
            avg_gate_emb = self._avg_emb(gate_emb)
            assert avg_obs_emb.shape == (Param.batch_size, Param.max_room_num, Param.obs_feat_out_num)
            assert avg_gate_emb.shape == (Param.batch_size, Param.max_room_num, Param.gate_feat_out_num)
            #沿着第三维拼接
            room_emb = torch.cat((avg_obs_emb, avg_gate_emb), dim=2)
            room_emb = self.fcnn_room(room_emb)
        elif method == "cat":
            cat_obs_emb = obs_emb.reshape((Param.batch_size, Param.max_room_num, Param.max_obs_num * Param.obs_feat_out_num))
            cat_gate_emb = gate_emb.reshape((Param.batch_size, Param.max_room_num, Param.max_gate_num * Param.gate_feat_out_num))
            room_emb = torch.cat((cat_obs_emb, cat_gate_emb), dim=2)
            # NOTE add another fcnn layer
            room_emb = self.fcnn_room(room_emb)
        else:
            print("there is no method called {}".format(method))
            raise NameError
        # NOTE mask for now
        # if env_ids is not None:
        #     ori_room_emb_shape = room_emb.shape
        #     room_emb = self.env_graph.cal_node_emb(env_ids, room_emb, route_len)
        #     assert ori_room_emb_shape == room_emb.shape
        return room_emb
    def _avg_emb(self, emb):
        res = torch.mean(emb, dim=2)
        return res

    def _LSTM_emb(self, room_info):
        # TODO
        raise NotImplementedError
'''
class RoomEmbedding(nn.Module):
    def __init__(self):
        super(RoomEmbedding, self).__init__()
        self.fcnn_room = nn.Sequential(
            nn.Linear(Param.room_emb_size_in, Param.room_emb_size)
        )
        self.fcnn_room.apply(_init_weights)
        # env_graph -> used for cal node emb with graph structure
        self.env_graph = EnvGraph()

    def forward(self, obs_info, gate_info, method="cat", env_ids=None, route_len=None):
        obs_emb = obs_info
        gate_emb = gate_info
        if method == "avg":
            avg_obs_emb = self._avg_emb(obs_emb)
            avg_gate_emb = self._avg_emb(gate_emb)
            assert avg_obs_emb.shape == (Param.batch_size, Param.max_room_num, Param.obs_feat_in_num)
            assert avg_gate_emb.shape == (Param.batch_size, Param.max_room_num, Param.gate_feat_in_num)
            #沿着第三维拼接
            room_emb = torch.cat((avg_obs_emb, avg_gate_emb), dim=2)
            room_emb = self.fcnn_room(room_emb)
        elif method == "cat":
            cat_obs_emb = obs_emb.reshape((Param.batch_size, Param.max_room_num, Param.max_obs_num * Param.obs_feat_in_num))
            cat_gate_emb = gate_emb.reshape((Param.batch_size, Param.max_room_num, Param.max_gate_num * Param.gate_feat_in_num))
            room_emb = torch.cat((cat_obs_emb, cat_gate_emb), dim=2)
            # NOTE add another fcnn layer
            room_emb = self.fcnn_room(room_emb)
        else:
            print("there is no method called {}".format(method))
            raise NameError
        # NOTE mask for now
        # if env_ids is not None:
        #     ori_room_emb_shape = room_emb.shape
        #     room_emb = self.env_graph.cal_node_emb(env_ids, room_emb, route_len)
        #     assert ori_room_emb_shape == room_emb.shape
        return room_emb
'''



class GateEmbedding(nn.Module):
    """
    TODO maybe it is better to share the fcnn_gate in RoomEmbedding
    """
    def __init__(self):
        super(GateEmbedding, self).__init__()
        self.fcnn_gate = nn.Sequential(
            nn.Linear(Param.gate_feat_in_num, Param.gate_feat_out_num)
        )
        self.fcnn_transform = nn.Sequential(
            nn.ReLU(),
            nn.Linear(Param.gate_feat_out_num, Param.room_emb_size)  # NOTE same shape as RoomEmbedding
        )

    def forward(self, gates_info):
        gates_emb = self.fcnn_gate(gates_info)
        gates_emb = self.fcnn_transform(gates_emb)
        return gates_emb


class Utils(object):
    @staticmethod
    def construct_room_graph(env_ids, is_train=True):
        """
        :param env_ids:
        :return: room_graph[room][gate] = room [room_graph1, ....]
        """
        room_graph_dict = {}
        for cur_env_id in env_ids:
            if is_train and Param.is_dynamic_data is False: cur_path = "{}/env{}.txt".format(Param.env_dir, cur_env_id)
            elif is_train and Param.is_dynamic_data is True: cur_path = "{}/env{}.txt".format(Param.dynamic_env_dir, cur_env_id)
            else: cur_path = "{}/env{}.txt".format(Param.eval_env_dir, cur_env_id)
            with open(cur_path, 'r') as f:
                cur_env_info = json.load(f)
                cur_room_graph = Utils._room_graph(cur_env_info)
                room_graph_dict[int(cur_env_id)] = cur_room_graph
        return room_graph_dict

    @staticmethod
    def _room_graph(env_info):
        room_graph = {}
        for room_id, room_info in env_info.items():
            room_id_int = int(room_id)
            if room_id not in room_graph: room_graph[room_id_int] = {}
            gates_info = room_info["gates"]
            for gate_id, cur_gate_info in gates_info.items():
                neighbor_id_int = int(cur_gate_info["neighbor"])
                gate_id_int = int(gate_id)
                room_graph[room_id_int][gate_id_int] = neighbor_id_int
        return room_graph


