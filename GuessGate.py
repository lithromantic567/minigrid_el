from Agents import *


class GuessGate(nn.Module):
    def __init__(self, agentA=None, agentB=None):
        super(GuessGate, self).__init__()
        self.agentA = AgentA() if agentA is None else agentA
        self.agentB = AgentB() if agentB is None else agentB
        self.gate_embedding_A = GateEmbedding()
        self.gate_embedding_B = GateEmbedding()

    def forward(self, gates_info, tgt_gates_idx, choose_method="sample", guess_gates_info=None, gates_num=None, history_sents=None):
        """
        :param gates_info: (batch, max_gate_num, gate_feat_num)
        :param tgt_gates_idx: (batch)
        :param choose_method:
        :param guess_gates_info: if rooms of A is not consistent with room which B thought,
        then guess_gates_info describes rooms that B thought
        :param gates_num: num of gates in each room
        :param history_sents: (history_sents_room, history_sents_gate)
        :return:
        """
        # TODO need to check whether this mask way is right
        # --- split tgt & distractor ---
        cur_batch_size = len(tgt_gates_idx)
        assert gates_info.shape == (cur_batch_size, Param.max_gate_num, Param.gate_feat_in_num)
        # --- forward ---
        if guess_gates_info is not None:
            ordered_gates_emb_B = self.gate_embedding_B(guess_gates_info)[np.arange(cur_batch_size), tgt_gates_idx, :]
        else:
            ordered_gates_emb_B = self.gate_embedding_B(gates_info)[np.arange(cur_batch_size), tgt_gates_idx, :]
        # unitize emb size of gate(stage1) and room(stage3)
        assert ordered_gates_emb_B.shape == (cur_batch_size, Param.room_emb_size)
        sent, token_probs = self.agentB.describe_gate(ordered_gates_emb_B, Param.max_sent_len, choose_method)
        ordered_gates_emb_A = self.gate_embedding_A(gates_info)
        if history_sents is not None:
            history_sents_room, history_sents_gate = history_sents
            if history_sents_gate is None:
                history_sents_gate = sent.unsqueeze(1)
            else:
                history_sents_gate = torch.cat([history_sents_gate, sent.unsqueeze(1)], dim=1)
            sent = torch.cat([history_sents_room, history_sents_gate], dim=2)
            assert sent.shape[-1] == Param.max_sent_len + Param.max_sent_len  # NOTE sent len = room sent len + gate sent len
        gate_idx, gate_prob = self.agentA.guess_gate(ordered_gates_emb_A, sent, choose_method, gates_num=gates_num)
        return gate_idx, token_probs, gate_prob, sent if history_sents is None else history_sents_gate

    def backward(self, token_probs, gate_prob, reward):
        lossB = self.agentB.cal_guess_gate_loss(token_probs, reward)
        lossA = self.agentA.cal_guess_gate_loss(gate_prob, reward)
        lossB.backward()
        lossA.backward()
        return lossB, lossA
