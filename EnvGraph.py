import torch
from torch import nn
from Param import *
from random import sample
import numpy as np
import json


# TODO have not been checked yet
class EnvGraph(nn.Module):
    def __init__(self):
        super(EnvGraph, self).__init__()
        # NOTE mask for now
        # self.route_encoder = nn.GRU(input_size=Param.room_emb_size, hidden_size=Param.route_encoder_hidden_size, batch_first=True)  # TODO gru for now

    def _construct_graph_batch(self, env_info):
        env_graph = {}
        for room_id, room_info in env_info.items():
            room_id_str = str(room_id)
            if room_id_str not in env_graph: env_graph[room_id_str] = {}
            gates_info = room_info["gates"]
            for gate_id, cur_gate_info in gates_info.items():
                neighbor_id_str = str(cur_gate_info["neighbor"])
                env_graph[room_id_str][neighbor_id_str] = str(gate_id)
        return env_graph

    # def _sample_route(self, env_graph, start_node, route_len, route_num=Param.route_num, is_l2r=True):
    #     """
    #     :param env_graph:
    #     :param start_node:
    #     :param route_len:
    #     :param route_num:
    #     :param is_l2r: is the sampled route is from left to right.
    #     if it is left to right: fix the first guess.
    #     if it is right to left: get a more accurate guess about current position.
    #     :return:
    #     """
    #     routes = []
    #     for i in range(route_num):
    #         cur_node = start_node; cur_route = [int(cur_node)]
    #         for j in range(route_len):
    #             cur_node_neighbors = env_graph[cur_node].keys()
    #             if len(cur_node_neighbors) == 0:
    #                 print("There should not be any room with no neighbors")
    #                 raise RuntimeError
    #             cur_node = sample(cur_node_neighbors, 1)[0]
    #             cur_route.append(int(cur_node))
    #         if is_l2r:
    #             routes.append(cur_route)
    #         else:
    #             routes.append(list(reversed(cur_route)))
    #     return routes

    # def _node_route_embs(self, env_info, init_node_emb, route_len):
    #     """
    #     node route embs
    #     :param env_info: info which directly read from env files
    #     :param init_node_emb: (max_room_num, room_emb_size)  Not in batch
    #     :param route_len: route len
    #     :return:
    #     """
    #     node_route_embs = []
    #     assert init_node_emb.shape == (Param.max_room_num, Param.room_emb_size)
    #     env_graph = self._construct_graph_batch(env_info)
    #     for cur_node_id in range(len(env_info)):
    #         cur_routes = self._sample_route(env_graph, str(cur_node_id), route_len-1, route_num=Param.route_num, is_l2r=False)
    #         cur_routes = np.array(cur_routes)
    #         assert cur_routes.shape == (Param.route_num, route_len)
    #         # NOTE encode from left to right -> fix the first guess
    #         # NOTE encode from right to left -> a more accurate guess about current position
    #         # cur_routes_embs = torch.cat([cur_routes[route_id, :] for route_id in range(Param.route_num)], dim=0)  # TODO check dim
    #         cur_routes_embs = torch.cat([init_node_emb[cur_routes[route_id, :], :].unsqueeze(0) for route_id in range(Param.route_num)], dim=0)
    #         assert cur_routes_embs.shape == (Param.route_num, route_len, Param.room_emb_size)
    #         node_route_embs.append(cur_routes_embs.unsqueeze(0))
    #     node_route_embs = torch.cat(node_route_embs, dim=0)
    #     node_route_embs = nn.functional.pad(node_route_embs, (0, Param.max_room_num - node_route_embs.shape[0], 0, 0, 0, 0, 0, 0))
    #     assert node_route_embs.shape == (Param.max_room_num, Param.route_num, route_len, Param.room_emb_size)
    #     return node_route_embs

    # def _node_emb(self, env_info_batch, init_node_emb_batch, route_len):
    #     """
    #     node emb with structure info
    #     :param env_info_batch: [env_info1, env_info2, ...]
    #     :param init_node_emb_batch: (batch_size, max_room_num, room_emb_size)
    #     :param route_len:
    #     :return:
    #     """
    #     route_embs = []
    #     for i, cur_env_info in enumerate(env_info_batch):
    #         cur_route_embs = self._node_route_embs(cur_env_info, init_node_emb_batch[i, :, :], route_len)
    #         route_embs.append(cur_route_embs.unsqueeze(0))
    #     route_embs = torch.cat(route_embs, dim=0)  # TODO check dim
    #     cur_batch = route_embs.shape[0]
    #     assert route_embs.shape[1:] == (Param.max_room_num, Param.route_num, route_len, Param.room_emb_size)
    #     route_embs = torch.reshape(route_embs, (cur_batch * Param.max_room_num * Param.route_num, route_len, Param.room_emb_size))
    #     output, hx = self.route_encoder(route_embs)
    #     hx = torch.reshape(hx, (cur_batch, Param.max_room_num, Param.route_num, Param.route_encoder_hidden_size))   # TODO check
    #     hx = torch.sum(hx, dim=2)
    #     assert hx.shape == (cur_batch, Param.max_room_num, Param.route_encoder_hidden_size)
    #     return hx
    #
    # def cal_node_emb(self, env_ids, init_node_emb_batch, route_len):
    #     env_info_batch = []
    #     for cur_env_id in env_ids:
    #         with open("{}/env{}.txt".format(Param.env_dir, cur_env_id), 'r') as f:
    #             cur_env_info = json.load(f)
    #             env_info_batch.append(cur_env_info)
    #     return self._node_emb(env_info_batch, init_node_emb_batch, route_len)








