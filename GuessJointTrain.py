from Dataset import *
from Agents import *
from torch.optim import Adam
import random
from GuessJoint import *


def guess_joint_train():
    dataset_gate = EnvDataset(Param.env_dir, "GuessGate")
    dataset_room = EnvDataset(Param.env_dir, "GuessRoom")
    loader_gate = DataLoader(dataset_gate, batch_size=Param.batch_size)
    loader_room = DataLoader(dataset_room, batch_size=Param.batch_size)
    task = GuessJoint()
    opt = Adam(task.parameters(), lr=Param.lr_task, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    accum_tgt_room = []; accum_pred_room = []
    accum_tgt_gate = []; accum_pred_gate = []
    for i in range(Param.epoch):
        cur_tgt_room = []; cur_pred_room = []
        cur_tgt_gate = []; cur_pred_gate = []
        total_loss_room_A = 0; total_loss_room_B = 0
        total_loss_gate_A = 0; total_loss_gate_B = 0
        opt.zero_grad()
        for step, ((num_room_room, cur_obs_info, cur_gate_info_room), (num_room_gate, cur_num_gate, cur_gate_info_gate)) in enumerate(zip(loader_room, loader_gate)):
            task.train()
            # --- Prepare room info ---
            tgt_rooms = np.random.randint(np.zeros(cur_obs_info.shape[0]), num_room_room)
            cur_tgt_room.append(tgt_rooms)
            cur_obs_info = cur_obs_info.to(torch.float32); cur_gate_info_room = cur_gate_info_room.to(torch.float32)
            # --- Prepare gate info ---
            cur_rooms = np.random.randint(np.zeros(num_room_gate.shape[0]), num_room_gate)
            selected_rooms_gates_info = cur_gate_info_gate[np.arange(cur_rooms.shape[0]), cur_rooms, :, :]
            selected_rooms_gates_num = cur_num_gate[np.arange(cur_rooms.shape[0]), cur_rooms]
            tgt_gates = np.random.randint(np.zeros(selected_rooms_gates_num.shape[0]), selected_rooms_gates_num)
            cur_tgt_gate.append(tgt_gates)
            selected_rooms_gates_info = selected_rooms_gates_info.to(torch.float32)
            # --- FORWARD ---
            cur_room_res, cur_gate_res = task(cur_obs_info, cur_gate_info_room, tgt_rooms, selected_rooms_gates_info, tgt_gates, "sample")
            room_idx, token_probs_room, room_prob, sent_room = cur_room_res
            gate_idx, token_probs_gate, gate_prob, sent_gate = cur_gate_res
            cur_pred_room.append(room_idx); cur_pred_gate.append(gate_idx)
            # --- BACKWARD ---
            reward_room = np.ones_like(room_idx)
            reward_room[room_idx.numpy() != tgt_rooms] = -1
            reward_gate = np.ones_like(gate_idx)
            reward_gate[gate_idx.numpy() != tgt_gates] = -1

            reward_room *= Param.reward; reward_gate *= Param.reward
            cur_loss_room, cur_loss_gate = task.backward(token_probs_room, room_prob[0], reward_room, token_probs_gate, gate_prob[0], reward_gate)
            cur_loss_room_A, cur_loss_room_B = cur_loss_room
            cur_loss_gate_A, cur_loss_gate_B = cur_loss_gate
            total_loss_room_A += cur_loss_room_A; total_loss_room_B += cur_loss_room_B
            total_loss_gate_A += cur_loss_gate_A; total_loss_gate_B += cur_loss_gate_B
        opt.step()
        cur_tgt_room = np.concatenate(cur_tgt_room, axis=0); cur_pred_room = np.concatenate(cur_pred_room, axis=0)
        cur_tgt_gate = np.concatenate(cur_tgt_gate, axis=0); cur_pred_gate = np.concatenate(cur_pred_gate, axis=0)
        accum_tgt_room.append(cur_tgt_room); accum_pred_room.append(cur_pred_room)
        accum_tgt_gate.append(cur_tgt_gate); accum_pred_gate.append(cur_pred_gate)
        if i % 50 == 0:
            task.eval()
            accum_tgt_room = np.concatenate(accum_tgt_room, axis=0); accum_pred_room = np.concatenate(accum_pred_room, axis=0)
            accum_tgt_gate = np.concatenate(accum_tgt_gate, axis=0); accum_pred_gate = np.concatenate(accum_pred_gate, axis=0)
            print("epoch{}: \n".format(i))
            print("    room: acc = {}, loss A = {}, loss B = {}".format(np.mean(cur_tgt_room == cur_pred_room), total_loss_room_A, total_loss_room_B))
            print("    gate: acc = {}, loss A = {}, loss B = {}".format(np.mean(cur_tgt_gate == cur_pred_gate), total_loss_gate_A, total_loss_gate_B))
            accum_tgt_room = []; accum_pred_room = []
            accum_tgt_gate = []; accum_pred_gate = []
            guess_joint_evaluate(task)


def guess_joint_evaluate(model):
    model.eval()
    with torch.no_grad():
        cur_tgt_room = []; cur_pred_room = []
        cur_tgt_gate = []; cur_pred_gate = []
        dataset_gate = EnvDataset(Param.env_dir, "GuessGate"); dataset_room = EnvDataset(Param.env_dir, "GuessRoom")
        loader_gate = DataLoader(dataset_gate, batch_size=Param.batch_size); loader_room = DataLoader(dataset_room, batch_size=Param.batch_size)
        for step, ((num_room_room, cur_obs_info, cur_gate_info_room), (num_room_gate, cur_num_gate, cur_gate_info_gate)) in enumerate(zip(loader_room, loader_gate)):
            # --- Prepare room info ---
            tgt_rooms = np.ones(cur_obs_info.shape[0])
            cur_tgt_room.append(tgt_rooms)
            cur_obs_info = cur_obs_info.to(torch.float32); cur_gate_info_room = cur_gate_info_room.to(torch.float32)
            # --- Prepare gate info ---
            cur_rooms = np.ones(num_room_gate.shape[0])
            selected_rooms_gates_info = cur_gate_info_gate[np.arange(cur_rooms.shape[0]), cur_rooms, :, :]
            selected_rooms_gates_num = cur_num_gate[np.arange(cur_rooms.shape[0]), cur_rooms]
            tgt_gates = np.zeros(selected_rooms_gates_num.shape[0])
            cur_tgt_gate.append(tgt_gates)
            selected_rooms_gates_info = selected_rooms_gates_info.to(torch.float32)
            # --- FORWARD ---
            cur_room_res, cur_gate_res = model(cur_obs_info, cur_gate_info_room, tgt_rooms, selected_rooms_gates_info, tgt_gates, "sample")
            room_idx, token_probs_room, room_prob, sent_room = cur_room_res
            gate_idx, token_probs_gate, gate_prob, sent_gate = cur_gate_res
            cur_pred_room.append(room_idx); cur_pred_gate.append(gate_idx)
        cur_tgt_room = np.concatenate(cur_tgt_room, axis=0); cur_pred_room = np.concatenate(cur_pred_room, axis=0)
        cur_tgt_gate = np.concatenate(cur_tgt_gate, axis=0); cur_pred_gate = np.concatenate(cur_pred_gate, axis=0)
        print("    eval room: acc = {}".format(np.mean(cur_tgt_room == cur_pred_room)))
        print("    eval gate: acc = {}".format(np.mean(cur_tgt_gate == cur_pred_gate)))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    setup_seed(3)
    guess_joint_train()