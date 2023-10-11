from torch import nn
from Param import *

class ELG(nn.Module):
    def __init__(self):
        super(ELG, self).__init__()
        self.voc_embedding = nn.Embedding(Param.voc_size, Param.voc_emb_size)  # start token, and end token
        self.gru = nn.GRU(input_size=Param.voc_emb_size, hidden_size=Param.room_emb_size)
        self.emb2idx = nn.Sequential(
            nn.Linear(Param.room_emb_size, Param.voc_size),
            nn.Softmax()
        )
    def forward(self, obs, max_length, choose_token_method="sample"):
        spoken_token_prob = []; spoken_token = []; next_token_prob = []
        assert obs.shape == (Param.batch_size, Param.room_emb_size)
        
        hx = obs.unsqueeze(0)
        
        token_before = self.voc_embedding(torch.LongTensor([[Param.sos_idx for _ in range(Param.batch_size)]]))
        for i in range(max_length):
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
        spoken_token_prob_arr = torch.stack(spoken_token_prob)
        loss = torch.mm(spoken_token_prob_arr, torch.Tensor(reward).unsqueeze(0).transpose(0, 1))
        loss = -torch.mean(loss)
        return loss
    
class ELU(nn.Module):
    def __init__(self):
        super(ELU, self).__init__()
        self.voc_embedding = nn.Embedding(Param.voc_size + 2, Param.voc_emb_size)
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

class Tourist(nn.Module):
    def __init__(self, lang_understand=None, lang_generate=None):
        super(Tourist, self).__init__()
        self.lang_understand=ELU() if lang_understand is None else lang_understand
        self.lang_generate=ELG() if lang_generate is None else lang_generate
    
    def decribe_room(self,obs, max_length,choose_token_method="sample"):
        return self.lang_generate(obs, max_length, choose_token_method)
    
    def cal_guess_room_loss(self, spoken_token_prob, reward):
        return self.lang_generate.cal_loss(spoken_token_prob, reward)
    
class Guide(nn.Module):
    def __init__(self, lang_understand=None, lang_generate=None):
        super(Guide, self).__init__()
        self.lang_understand=ELU() if lang_understand is None else lang_understand
        self.lang_generate=ELG() if lang_generate is None else lang_generate
    
    def guess_room(self, env_obs, message, choose_room_method="sample"):
        return self.lang_understand(env_obs, message, choose_room_method)
    
    def cal_guess_room_loss(self, room_prob,reward):
        return self.lang_understand.cal_loss(room_prob,reward)