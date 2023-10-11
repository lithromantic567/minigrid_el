import torch
from torch import nn
from Agents import *
from GuessGate import *
from GuessRoom import *


class Navigation(nn.Module):
    def __init__(self):
        super(Navigation, self).__init__()
        self.elu_A = ELU(); self.elg_A = ELG()
        # self.elu_B = ELU(); self.elg_B = ELG()  # dont share
        self.elu_B = self.elu_A; self.elg_B = self.elg_A
        self.agentA = AgentA(self.elu_A, self.elg_A)
        self.agentB = AgentB(self.elu_B, self.elg_B)
        self.guess_gate_task = GuessGate(self.agentA, self.agentB)
        self.guess_room_task = GuessRoom(self.agentA, self.agentB)

    def _move(self, room_graph, room, gate, env_ids):
        assert type(room) == list and type(gate) == list and type(env_ids) == list
        new_rooms = []
        for cur_room, cur_gate, cur_env_id in zip(room, gate, env_ids):
            cur_new_room = room_graph[cur_env_id][cur_room][cur_gate]
            new_rooms.append(cur_new_room)
        return np.array(new_rooms)

    def forward(self, env_ids, room_graph, cat_info, start_room, goal_room, max_move_len, gates_num, is_train=True, choose_method="sample"):
        """
        TODO have not dealt with early stop problem, A should know when it should stop, because it could see the goal at the right room
        TODO it is also possible to add an room emb which means end of routes, add a gate emb which means do not choose any gate
        NOTE just ignore rooms and gates after the tgt rooms, and loss will not count them in
        guess room, guess gate, guess room, ....
        :param env_ids: used for reading env files, then construct graph
        :param room_graph: room_graph[room_id][gate_id] = neighbor_id, then env can track the real path of A
        :param cat_info: (obs_info, gate_info)
        :param start_room: B does not know where the start room is
        :param goal_room: B knows where the goal room is
        :param max_move_len: max num of moving actions of A
        :param is_train:
        if True, the input of guess gate is the real pos of A.
        if False, the input of guess gate is the output of last guess room
        :return:
        action probs -> (token_prob, guess_prob),
        total route (used for cal loss, a strong signal)
        sents (analysis)
        """
        token_probs_room = []; token_probs_gate = []
        guess_probs_room = []; guess_probs_gate = []
        actual_route_room = []; actual_route_gate = []
        sents_room = []; sents_gate = []
        total_guess_room_idx = []; total_guess_gate_idx = []
        is_right_room = []; is_right_gate = []
        obs_info, gate_info = cat_info
        assert obs_info.shape == (obs_info.shape[0], Param.max_room_num, Param.max_obs_num, Param.obs_feat_in_num)
        assert gate_info.shape == (gate_info.shape[0], Param.max_room_num, Param.max_gate_num, Param.gate_feat_in_num)
        real_room_A = start_room; actual_route_room.append(real_room_A)
        cur_is_stop = (torch.Tensor(real_room_A) == torch.Tensor(goal_room))
        history_sents_room = None; history_sents_gate = None
        for cur_step in range(max_move_len):
            cur_is_stop[torch.Tensor(real_room_A) == torch.Tensor(goal_room)] = True
            # ----- Guess Room ----
            # TODO do not add history sents
            # guess_room_idx, cur_token_probs_room, cur_room_prob, cur_sent_room = self.guess_room_task(obs_info, gate_info, real_room_A, history_sents=None, env_ids=env_ids, route_len=cur_step + 1)
            if is_train:
                guess_room_idx, cur_token_probs_room, cur_room_prob, cur_sent_room = self.guess_room_task(obs_info, gate_info, real_room_A, history_sents=history_sents_room, env_ids=env_ids, route_len=cur_step + 1)
            else:
                guess_room_idx, cur_token_probs_room, cur_room_prob, cur_sent_room = self.guess_room_task(obs_info, gate_info, real_room_A, history_sents=history_sents_room, env_ids=env_ids, route_len=cur_step + 1, choose_method=choose_method)
            history_sents_room = cur_sent_room  # include all history sents
            if len(history_sents_room.shape) == 2: history_sents_room = history_sents_room.unsqueeze(1)
            if is_train:
                token_probs_room.append(cur_token_probs_room); guess_probs_room.append(cur_room_prob[0])
            sents_room.append(cur_sent_room)
            total_guess_room_idx.append(guess_room_idx)
            # ------ is right room ----
            cur_is_right_room = torch.zeros_like(guess_room_idx)
            cur_is_right_room[torch.Tensor(real_room_A) == guess_room_idx] = 1
            cur_is_right_room[cur_is_stop] = -1
            is_right_room.append(cur_is_right_room)
            # ----- Route Plan ----
            # TODO in training process, input of guess gate should be the real pos
            # TODO in test process, input should be the res of guess room task
            if is_train is True:
                cur_next_doors, cur_expected_next_rooms = self.agentB.next_movement(env_ids, torch.Tensor(real_room_A), goal_room)
            else:
                cur_next_doors, cur_expected_next_rooms = self.agentB.next_movement(env_ids, guess_room_idx, goal_room)
            guess_room_gates_info = gate_info[np.arange(gate_info.shape[0]), guess_room_idx, :, :]
            assert guess_room_gates_info.shape == (guess_room_gates_info.shape[0], Param.max_gate_num, Param.gate_feat_in_num)
            # ----- Guess Gate ----
            # TODO in training process, input of guess gate should be the real pos
            # TODO in test process, input should be the res of guess room task
            real_room_gates_info = gate_info[np.arange(gate_info.shape[0]), real_room_A, :, :]
            real_room_gates_num = gates_num[np.arange(gate_info.shape[0]), real_room_A]
            assert real_room_gates_info.shape == (real_room_gates_info.shape[0], Param.max_gate_num, Param.gate_feat_in_num)
            assert real_room_gates_num.shape == (real_room_gates_info.shape[0],)
            if is_train is True:
                guess_gate_idx, cur_token_probs_gate, cur_gate_prob, cur_sent_gate = self.guess_gate_task(real_room_gates_info, cur_next_doors, gates_num=real_room_gates_num, history_sents=None)
            else:
                guess_gate_idx, cur_token_probs_gate, cur_gate_prob, cur_sent_gate = self.guess_gate_task(real_room_gates_info, cur_next_doors, guess_gates_info=guess_room_gates_info, gates_num=real_room_gates_num, choose_method=choose_method, history_sents=None)
            history_sents_gate = cur_sent_gate
            if len(history_sents_gate.shape) == 2: history_sents_gate = history_sents_gate.unsqueeze(1)
            if is_train is True:
                token_probs_gate.append(cur_token_probs_gate); guess_probs_gate.append(cur_gate_prob[0])
            sents_gate.append(cur_sent_gate)
            total_guess_gate_idx.append(guess_gate_idx)
            # ----- is right gate ----
            cur_is_right_gate = torch.zeros_like(guess_gate_idx)
            cur_is_right_gate[torch.Tensor(cur_next_doors) == guess_gate_idx] = 1
            cur_is_right_gate[cur_is_stop] = -1
            is_right_gate.append(cur_is_right_gate)
            # ----- Actual Movement ---- -> update real_room_A
            real_room_A = self._move(room_graph, real_room_A.tolist(), guess_gate_idx.tolist(), env_ids.tolist())
            actual_route_room.append(real_room_A)
            actual_route_gate.append(cur_next_doors)
        # return (total_guess_room_idx, total_guess_gate_idx), (token_probs_room, token_probs_gate), (guess_probs_room, guess_probs_gate), (actual_route_room, actual_route_gate), (sents_room, sents_gate)
        is_right_room = torch.stack(is_right_room); is_right_gate = torch.stack(is_right_gate)
        return (sents_room, sents_gate), (actual_route_room, actual_route_gate), (token_probs_room, token_probs_gate), (guess_probs_room, guess_probs_gate), (is_right_room, is_right_gate)

    def backward(self, token_probs, guess_probs, rewards):
        """
        :param token_probs: (token_probs_room, token_probs_gate)
        :param guess_probs: (room_prob, gate_prob)
        :param rewards: (reward_room, reward_gate)
        :return:
        """
        token_probs_room, token_probs_gate = token_probs
        room_prob, gate_prob = guess_probs
        reward_room, reward_gate = rewards
        lossA_room = []; lossB_room = []; lossA_gate = []; lossB_gate = []
        for cur_step in range(Param.max_move_len):
            cur_token_prob_room = token_probs_room[cur_step]; cur_token_prob_gate = token_probs_gate[cur_step]
            cur_room_prob = room_prob[cur_step]; cur_gate_prob = gate_prob[cur_step]
            cur_room_reward = reward_room[cur_step]; cur_gate_reward = reward_gate[cur_step]

            cur_lossA_room, cur_lossB_room = self.guess_room_task.backward(cur_token_prob_room, cur_room_prob, cur_room_reward)
            cur_lossA_gate, cur_lossB_gate = self.guess_gate_task.backward(cur_token_prob_gate, cur_gate_prob, cur_gate_reward)
            lossA_room.append(cur_lossA_room); lossB_room.append(cur_lossB_room)
            lossA_gate.append(cur_lossA_gate); lossB_gate.append(cur_lossB_gate)
        return (lossA_room, lossB_room), (lossA_gate, lossB_gate)
