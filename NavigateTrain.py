# TODO: the guess room is just a special case of stage3, so expand it
# NOTE: the whole process is really like the self-regression process, so maybe, the train process can copy it
# NOTE: evaluation metric is success rate
import numpy as np
import random

import torch
import pickle

from Dataset import EnvDataset
from Param import *
from Navigation import *
from torch.utils.data import DataLoader
from torch.optim import Adam
from util import *
from env_generation.generate_gridworld import generate_envs


def _cal_gain():
    """
    gain for each step
    :return:
    """
    pass


def navigation_train(task_continue=None):
    dataset = EnvDataset(Param.env_dir, "Navigation") if Param.is_dynamic_data is False else EnvDataset(Param.dynamic_env_dir, "Navigation")
    loader = DataLoader(dataset, batch_size=Param.batch_size)
    task = Navigation() if task_continue is None else task_continue
    opt = Adam(task.parameters(), lr=Param.lr_task, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    # accum_tgt_room = []; accum_pred_room = []
    # accum_tgt_gate = []; accum_pred_gate = []
    accum_is_right_room = []; accum_is_right_gate = []
    for i in range(Param.epoch):
        opt.zero_grad()
        total_loss_room_A = 0.0; total_loss_room_B = 0.0
        total_loss_gate_A = 0.0; total_loss_gate_B = 0.0
        if Param.is_dynamic_data is True and i % 50 == 0: generate_envs(Param.dynamic_env_dir, Param.dynamic_datasize)
        for step, (num_room, num_gate, cur_obs_info, cur_gate_info, env_ids) in enumerate(loader):
            task.train()
            # --- FORWARD ---
            num_room_list = num_room.tolist()
            temp = np.array([random.sample(range(num_room_list[k]), 2) for k in range(len(num_room_list))])
            start_rooms = temp[:, 0]; goal_rooms = temp[:, 1]
            cur_obs_info = cur_obs_info.to(torch.float32); cur_gate_info = cur_gate_info.to(torch.float32)
            room_graph = Utils.construct_room_graph(env_ids)
            # cur_guess_idx, cur_token_probs, cur_guess_probs, cur_routes, cur_sents = task(env_ids, room_graph, (cur_obs_info, cur_gate_info), start_rooms, goal_rooms, Param.max_move_len, num_gate)
            cur_sents, cur_actual_route, cur_token_probs, cur_guess_probs, cur_is_right = task(env_ids, room_graph, (cur_obs_info, cur_gate_info), start_rooms, goal_rooms, Param.max_move_len, num_gate)
            cur_is_right_room, cur_is_right_gate = cur_is_right
            # --- BACKWARD ---
            rewards_room = np.zeros_like(cur_is_right_room); rewards_gate = np.zeros_like(cur_is_right_gate)
            # print(cur_is_right_room.shape)
            rewards_room[cur_is_right_room.numpy() == 1] = 1; rewards_gate[cur_is_right_gate.numpy() == 1] = 1
            rewards_room[cur_is_right_room.numpy() == 0] = -1; rewards_gate[cur_is_right_gate.numpy() == 0] = -1
            cur_loss_room, cur_loss_gate = task.backward(cur_token_probs, cur_guess_probs, (rewards_room.tolist(), rewards_gate.tolist()))
            cur_loss_room_A, cur_loss_room_B = cur_loss_room
            cur_loss_gate_A, cur_loss_gate_B = cur_loss_gate
            total_loss_room_A += sum(cur_loss_room_A); total_loss_room_B += sum(cur_loss_room_B)
            total_loss_gate_A += sum(cur_loss_gate_A); total_loss_gate_B += sum(cur_loss_gate_B)
            accum_is_right_gate.append(cur_is_right_gate); accum_is_right_room.append(cur_is_right_room)
            opt.step()
        print("|", end="")
        if i % 50 == 0:
            task.eval()
            accum_is_right_room = np.stack(accum_is_right_room); accum_is_right_gate = np.stack(accum_is_right_gate)
            print()
            print("epoch {}:".format(i))
            print("    room: acc = {}, loss A = {}, loss B = {}".format(np.mean(accum_is_right_room[accum_is_right_room != -1]), total_loss_room_A, total_loss_room_B))
            print("    gate: acc = {}, loss A = {}, loss B = {}".format(np.mean(accum_is_right_gate[accum_is_right_gate != -1]), total_loss_gate_A, total_loss_gate_B))
            # accum_tgt_room = []; accum_pred_room = []; accum_tgt_gate = []; accum_pred_gate = []
            accum_is_right_room = []; accum_is_right_gate = []
            sents_info = navigation_evaluate(task)
            # TODO
            if i % 100 == 0:
                with open("{}/{}.pkl".format(Param.sent_dir, i), 'wb') as f:
                    pickle.dump(sents_info, f)
                save_model(task, "{}/{}.pth".format(Param.model_dir, i))


def navigation_evaluate(model):
    model.eval()
    with torch.no_grad():
        success = []; sents_room = []; sents_gate = []; route_room = []; route_gate = []
        first_room_success = []; first_gate_success = []
        dataset = EnvDataset(Param.eval_env_dir, "Navigation")
        loader = DataLoader(dataset, batch_size=Param.batch_size)
        for step, (num_room, num_gate, cur_obs_info, cur_gate_info, env_ids) in enumerate(loader):
            num_room_list = num_room.tolist()
            # temp = np.array([random.sample(range(num_room_list[k]), 2) for k in range(len(num_room_list))])
            start_rooms = np.zeros_like(num_room); goal_rooms = np.full_like(num_room, 5)
            cur_obs_info = cur_obs_info.to(torch.float32); cur_gate_info = cur_gate_info.to(torch.float32)
            room_graph = Utils.construct_room_graph(env_ids, is_train=False)
            cur_sents, cur_actual_route, cur_token_probs, cur_guess_probs, cur_is_right = model(env_ids, room_graph, (cur_obs_info, cur_gate_info), start_rooms, goal_rooms, Param.max_move_len, num_gate, is_train=False, choose_method="greedy")
            cur_is_right_room, cur_is_right_gate = cur_is_right
            cur_sents_room, cur_sents_gate = cur_sents
            cur_route_room, cur_route_gate = cur_actual_route
            sents_room.append(cur_sents_room[-1]); sents_gate.append(cur_sents_gate)
            route_room.append(cur_route_room); route_gate.append(cur_route_gate)
            # print("       ", [cur_route_room[i][0] for i in range(len(cur_route_room))], end=" ")
            # print("       ", cur_is_right_room[:, 0])
            success.append(cur_is_right_room[-1, :])
            first_room_success.append(cur_is_right_room[3, :])
            first_gate_success.append(cur_is_right_gate[3, :])
        success = np.stack(success)
        first_room_success = np.stack(first_room_success)
        first_gate_success = np.stack(first_gate_success)
        print()
        print("    eval acc = {}".format(np.mean(success.astype(int) == -1)))
        print("    eval first guess room acc = {}, first guess gate acc = {}".format(
            np.mean(first_room_success.astype(int)[first_room_success.astype(int) != -1] == 1),
            np.mean(first_gate_success[first_room_success.astype(int) == 1].astype(int) == 1)), flush=True)
        print()
        return {"sents_room": sents_room, "sents_gate": sents_gate, "route_room": route_room, "route_gate": route_gate}


def save_model(model, path):
    torch.save(model, path)


def load_model(model, path):
    return torch.load(path)


def setup_seed(seed):
    gen = torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return gen


if __name__ == "__main__":
    gen = setup_seed(0)
    navigation_train()
    # task = Navigation()
    # task = load_model(task, "{}/{}.pth".format(Param.model_dir, 50))
    # navigation_evaluate(task)
    # navigation_train(task)