from Agents import *


class GuessRoom(nn.Module):
    def __init__(self, agentA=None, agentB=None):
        super(GuessRoom, self).__init__()
        self.agentA = AgentA() if agentA is None else agentA
        self.agentB = AgentB() if agentB is None else agentB
        self.room_embedding_A = RoomEmbedding()  # initial state
        self.room_embedding_B = RoomEmbedding()
        # self.room_embedding_B = self.room_embedding_A  # share

    def forward(self, obs_info, gate_info, tgt_rooms, choose_method="sample", history_sents=None, env_ids=None, route_len=None):
        tgt_rooms_arr = np.array(tgt_rooms).astype(int)
        room_embs_A = self.room_embedding_A(obs_info, gate_info)[np.arange(tgt_rooms_arr.shape[0]), tgt_rooms_arr, :]
        sent, token_probs = self.agentA.describe_room(room_embs_A, Param.max_sent_len, choose_method)
        if history_sents is not None:
            sent = torch.cat([history_sents, sent.unsqueeze(1)], dim=1)
            room_embs_B = self.room_embedding_B(obs_info, gate_info, env_ids=env_ids, route_len=route_len)
        else:
            room_embs_B = self.room_embedding_B(obs_info, gate_info)
        room_idx, room_prob = self.agentB.guess_room(room_embs_B, sent, choose_method)
        return room_idx, token_probs, room_prob, sent

    def backward(self, token_probs, room_prob, reward, step=0):
        lossA = self.agentA.cal_guess_room_loss(token_probs, reward)
        lossB = self.agentB.cal_guess_room_loss(room_prob, reward)
        lossA.backward()
        lossB.backward()
        return lossA, lossB