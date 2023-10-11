import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from el.try.Agents import *
from Dataset import EnvDataset
from el.try.Param import *
import random
from el.try.GuessRoom import *
from Env import *


def guess_room_train():
    """
    train
    :return:
    """
    env=GoTo(render_mode="human")
    env._gen_grid(env.width,env.height)
    task = GuessRoom()
    # if Param.is_gpu: task = task.to(Param.gpu_device)
    opt = Adam(task.parameters(), lr=Param.lr_task, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    # opt = SGD(task.parameters(), lr=Param.lr, momentum=0.9)
    accum_tgt = []; accum_pred = []
    for i in range(Param.epoch):
        tgt = []
        pred = []
        cur_sent = None
        total_loss_A = 0; total_loss_B = 0
        opt.zero_grad()
        for i in range(10):  
            env_obs = DictObservationSpaceWrapper(env)
            obs,_ = env_obs.reset()         
            task.train()
            task.agentA.train()
            task.agentB.train()
            # --- FORWARD ----
            tgt_rooms = env.agent_room
            tgt.append(tgt_rooms)
            # num_room = num_room.to(torcht.float32); num_obs = num_obs.to(torch.float32)
            # cur_env_info = cur_env_info.to(torch.float32)
            obs_A= obs['image']
            room_idxes, token_probs, room_probs, sent = task(cur_obs_info, cur_gate_info, tgt_rooms)
            cur_sent = sent
            pred.append(room_idxes)
            # --- BACKWARD ---
            reward = np.ones_like(room_idxes)
            reward[room_idxes.numpy() != tgt_rooms] = -1

            reward *= Param.reward
            cur_loss_A, cur_loss_B = task.backward(token_probs, room_probs[0], reward.tolist(), step)
            total_loss_A += cur_loss_A; total_loss_B += cur_loss_B
        opt.step()
        tgt = np.concatenate(tgt, axis=0)
        pred = np.concatenate(pred, axis=0)
        accum_tgt.append(tgt)
        accum_pred.append(pred)
        if i % 50 == 0:
            task.eval()
            task.agentA.eval()
            task.agentB.eval()
            accum_pred = np.concatenate(accum_pred, axis=0)
            accum_tgt = np.concatenate(accum_tgt, axis=0)
            print("epoch{}: \nacc = {}, loss A = {}, loss B = {}".format(i, np.mean(accum_tgt == accum_pred), total_loss_A, total_loss_B))
            accum_pred = []; accum_tgt = []
            guess_room_evaluate(task)


def guess_room_evaluate(model):
    """
    evaluation
    :param model:
    :return:
    """
    model.eval()
    with torch.no_grad():
        tgt = []; pred = []
        dataset = EnvDataset(Param.eval_env_dir, 'GuessRoom')
        loader = DataLoader(dataset, batch_size=Param.batch_size)
        for step, (num_room, cur_obs_info, cur_gate_info) in enumerate(loader):
            # TODO maybe it is better to use this random way
            # tgt_rooms = np.random.randint(np.zeros(cur_obs_info.shape[0]), num_room)
            tgt_rooms = np.ones(cur_obs_info.shape[0])
            tgt.append(tgt_rooms)
            cur_obs_info = cur_obs_info.to(torch.float32); cur_gate_info = cur_gate_info.to(torch.float32)
            room_idxes, token_probs, room_probs, sent = model(cur_obs_info, cur_gate_info, tgt_rooms, choose_method="greedy")
            pred.append(room_idxes)
        tgt = np.concatenate(tgt, axis=0)
        pred = np.concatenate(pred, axis=0)
        print("eval acc = {}".format(np.mean(tgt == pred)))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    setup_seed(1)
    guess_room_train()
