from Agents import *
from GuessGate import *
from GuessRoom import *


class GuessJoint(nn.Module):
    def __init__(self):
        super(GuessJoint, self).__init__()
        self.elu = ELU()
        self.elg = ELG()
        self.agentA = AgentA(self.elu, self.elg)
        self.agentB = AgentB(self.elu, self.elg)
        self.task_gate = GuessGate(self.agentA, self.agentB)
        self.task_room = GuessRoom(self.agentA, self.agentB)

    def forward(self, obs_info, gate_info, tgt_rooms, gates_info, tgt_gates_idx, choose_method="sample"):
        room_idx, token_probs_room, room_prob, sent_room = self.task_room(obs_info, gate_info, tgt_rooms, choose_method)
        gate_idx, token_probs_gate, gate_prob, sent_gate = self.task_gate(gates_info, tgt_gates_idx, choose_method)
        return (room_idx, token_probs_room, room_prob, sent_room), (gate_idx, token_probs_gate, gate_prob, sent_gate)

    def backward(self, token_probs_room, room_prob, reward_room, token_probs_gate, gate_prob, reward_gate):
        lossA_room, lossB_room = self.task_room.backward(token_probs_room, room_prob, reward_room)
        lossB_gate, lossA_gate = self.task_gate.backward(token_probs_gate, gate_prob, reward_gate)
        # lossA_room.backward(); lossB_room.backward()
        # lossA_gate.backward(); lossB_gate.backward()
        return (lossA_room, lossB_room), (lossA_gate, lossB_gate)