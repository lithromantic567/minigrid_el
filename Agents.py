import torch
from torch import nn
from util import *
from Param import *
import numpy as np
import os
from RoutePlan import *
from torch.nn.utils.rnn import pack_padded_sequence


class ELG(nn.Module):
    def __init__(self):
        super(ELG, self).__init__()
        #词汇表大小，每个词汇维度
        self.voc_embedding = nn.Embedding(Param.voc_size, Param.voc_emb_size)  # start token, and end token
        self.gru = nn.GRU(input_size=Param.voc_emb_size, hidden_size=Param.room_emb_size)
        #序贯模型
        self.emb2idx = nn.Sequential(
            nn.Linear(Param.room_emb_size, Param.voc_size),
            nn.ReLU(),
            nn.Linear(Param.voc_size,Param.voc_size),
            nn.Softmax()
        )

    def forward(self, cur_room_emb, max_length, choose_token_method="sample"):
        spoken_token_prob = []; spoken_token = []; next_token_prob = []
        #语句为真，则继续运行
        assert cur_room_emb.shape == (Param.batch_size, Param.room_emb_size)
        #0表示在张量最外层加一个中括号变成第一维，也就是变成(1,batch_size,room_emb_size)
        hx = cur_room_emb.unsqueeze(0)
        #查询词汇表矩阵中的start token？
        token_before = self.voc_embedding(torch.LongTensor([[Param.sos_idx for _ in range(Param.batch_size)]]))
        for i in range(max_length):
            #token_before是当前的输入，hx是上一时刻传递下来的隐状态
            output, hx = self.gru(token_before, hx)
            next_token_pred = self.emb2idx(output.squeeze(0))
            if choose_token_method == "greedy":
                token_idx = torch.argmax(next_token_pred, dim=1)  # TODO check dim
            elif choose_token_method == "sample":
                next_token_prob.append(next_token_pred)
                token_sampler = torch.distributions.Categorical(next_token_pred)
                token_idx = token_sampler.sample()
                # record actions
                spoken_token_prob.append(token_sampler.log_prob(token_idx))
            spoken_token.append(token_idx)
            token_before = self.voc_embedding(token_idx).unsqueeze(0)
        spoken_token = torch.reshape(torch.cat(spoken_token, axis=0), (Param.max_sent_len, Param.batch_size))
        spoken_token = torch.transpose(spoken_token, 0, 1)
        assert spoken_token.shape[0] == Param.batch_size
        return spoken_token, spoken_token_prob

    def cal_loss(self, spoken_token_prob, reward):
        #沿着新维度对张量序列连接
        spoken_token_prob_arr = torch.stack(spoken_token_prob)
        #矩阵相乘
        loss = torch.mm(spoken_token_prob_arr, torch.Tensor(reward).unsqueeze(0).transpose(0, 1))
        loss = -torch.mean(loss)
        return loss


class ELU(nn.Module):
    def __init__(self):
        super(ELU, self).__init__()
        self.voc_embedding = nn.Embedding(Param.voc_size + 2, Param.voc_emb_size)
        #batch_first把batch放在第一维
        self.sent_encoder = nn.GRU(Param.voc_emb_size, Param.room_emb_size, batch_first=True)
        self.history_encoder = nn.GRU(Param.room_emb_size, Param.room_emb_size, batch_first=True)
        self.softmax = nn.Softmax()

    def _encode_message(self, message):
        # message -> (batch, history, message len)
        msg_shape = message.shape
        reshaped_msg = message.reshape((msg_shape[0] * msg_shape[1], msg_shape[2])).long()
        # msg_embs -> (batch, history, message len, voc_emb_size)
        msg_embs = self.voc_embedding(reshaped_msg)
        _, hx = self.sent_encoder(msg_embs)
        hx = torch.reshape(hx, (msg_shape[0], msg_shape[1], Param.room_emb_size))
        _, hx = self.history_encoder(hx)
        return hx

    def forward(self, env_emb, message, choose_room_method="sample", obj_nums=None):
        #shape从(a,b)到(a,1,b)
        if len(message.shape) == 2: message = message.unsqueeze(1)
        room_prob = []
        hx = self._encode_message(message)
        res = torch.bmm(env_emb, hx.permute(1, 2, 0)).squeeze()
        # make sure that the scores for each padding gate is 0
        score_mask = torch.zeros_like(res)
        if obj_nums is not None:
            obj_nums_list = obj_nums.tolist()
            for i, cur_obj_num in enumerate(obj_nums_list):
                if cur_obj_num < res.shape[1]:
                    res[i, int(cur_obj_num):] = -torch.inf
                    score_mask[i, int(cur_obj_num):] = 1
        scores = self.softmax(res)
        assert obj_nums is None or torch.sum(scores[score_mask == 1]) == 0
        if choose_room_method == "greedy":
            room_idx = torch.argmax(scores, dim=1)
        elif choose_room_method == "sample":
            room_sampler = torch.distributions.Categorical(scores)
            room_idx = room_sampler.sample()
            # record actions
            room_prob.append(room_sampler.log_prob(room_idx))
        return room_idx, room_prob


    def backward(self, room_prob, reward):
        loss = -sum([room_prob[i] * reward[i] for i in range(len(room_prob))])
        loss.backward()

    def cal_loss(self, room_prob, reward):
        loss = -sum([room_prob[i] * reward[i] for i in range(len(room_prob))])
        return loss


class AgentA(nn.Module):
    def __init__(self, lang_understand=None, lang_generate=None):
        super(AgentA, self).__init__()
        self.lang_understand = ELU() if lang_understand is None else lang_understand
        self.lang_generate = ELG() if lang_generate is None else lang_generate

    def describe_room(self, cur_room_emb, max_length, choose_token_method="sample"):
        return self.lang_generate(cur_room_emb, max_length, choose_token_method)

    def guess_gate(self, gates_emb, message, choose_gate_method="sample", gates_num=None):
        return self.lang_understand(gates_emb, message, choose_gate_method, obj_nums=gates_num)

    def cal_guess_room_loss(self, spoken_token_prob, reward):
        return self.lang_generate.cal_loss(spoken_token_prob, reward)

    def cal_guess_gate_loss(self, gate_prob, reward):
        return self.lang_understand.cal_loss(gate_prob, reward)


class AgentB(nn.Module):
    def __init__(self, lang_understand=None, lang_generate=None, is_cal_all_route_init=False):
        super(AgentB, self).__init__()
        self.lang_understand = ELU() if lang_understand is None else lang_understand
        self.lang_generate = ELG() if lang_generate is None else lang_generate
        self.is_cal_all_route_init = is_cal_all_route_init
        if self.is_cal_all_route_init is True:
            self.route_plan_gate_table = None; self.route_plan_room_table = None
            self._fill_route_plan_table()

    def _fill_route_plan_table(self):
        # TODO this structure is not good
        self.route_plan_gate_table =[]; self.route_plan_room_table = []
        env_files = os.listdir(Param.env_dir)
        env_files = [f for f in env_files if f.startswith(".")]  # remove .DS_Store
        for file_id in range(len(env_files)):
            cur_next_room, cur_next_door = RoutePlan.find_shortest_path("{}/env{}.txt".format(Param.env_dir, file_id))
            self.route_plan_room_table.append(cur_next_room); self.route_plan_gate_table.append(cur_next_door)

    def _cal_route_plan(self, env_ids):
        route_plan_rooms = {}; route_plan_gates = {}
        for cur_env_id in env_ids:
            cur_next_room, cur_next_door = RoutePlan.find_shortest_path("{}/env{}.txt".format(Param.env_dir, cur_env_id))
            route_plan_rooms[int(cur_env_id)] = cur_next_room
            route_plan_gates[int(cur_env_id)] = cur_next_door
        return route_plan_rooms, route_plan_gates

    def next_movement(self, env_ids, now_rooms, tgt_rooms):
        """
        :param env_ids: the id of cur envs
        :param now_rooms: NOTE now_rooms means the rooms where Agent B thinks the Agent A is
        :param tgt_rooms: the rooms where the final goal is
        :return: cur_next_door, expected_next_room
        """
        if self.is_cal_all_route_init is True:
            route_plan_rooms = {int(i): self.route_plan_room_table[i] for i in env_ids}
            route_plan_gates = {int(i): self.route_plan_gate_table[i] for i in env_ids}
        else:
            route_plan_rooms, route_plan_gates = self._cal_route_plan(env_ids)
        # print(type(now_rooms), type(tgt_rooms))
        assert type(now_rooms) == torch.Tensor and type(tgt_rooms) == np.ndarray and type(env_ids) == torch.Tensor
        next_doors = [route_plan_gates[cur_env_id][cur_now_room][cur_tgt_room] for cur_env_id, cur_now_room, cur_tgt_room in zip(env_ids.tolist(), now_rooms.tolist(), tgt_rooms.tolist())]
        expected_next_rooms = [route_plan_rooms[cur_env_id][cur_now_room][cur_tgt_room] for cur_env_id, cur_now_room, cur_tgt_room in zip(env_ids.tolist(), now_rooms.tolist(), tgt_rooms.tolist())]
        return next_doors, expected_next_rooms

    def guess_room(self, env_emb, message, choose_room_method="sample"):
        return self.lang_understand(env_emb, message, choose_room_method)

    def describe_gate(self, ordered_gate_emb, max_length, choose_token_method="sample"):
        return self.lang_generate(ordered_gate_emb, max_length, choose_token_method)

    def cal_guess_room_loss(self, room_prob, reward):
        return self.lang_understand.cal_loss(room_prob, reward)

    def cal_guess_gate_loss(self, spoken_token_prob, reward):
        return self.lang_generate.cal_loss(spoken_token_prob, reward)

    # NOTE maybe the observation is just used for location
    # def cal_movement_reward(self, env_ids, room_prob):
    #     # TODO
    #     raise NotImplementedError
    #     assert room_prob.shape == (room_prob.shape[0], Param.max_room_num)
    #     cur_reward = - room_prob[np.arange(room_prob.shape[0]), expected_rooms] * Param.reward
    #     return cur_reward


