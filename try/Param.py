import torch


class Param(object):
    obs_feat_in_num = 10
    # obs_feat_mid_num = 64
    obs_feat_out_num = 20
    gate_feat_in_num = 4
    # gate_feat_mid_num = 32
    gate_feat_out_num = 20

    # room_emb_size = obs_feat_out_num + gate_feat_out_num
    voc_emb_size = 20
    voc_size = 40
    
    sos_idx = 0
    # eos_idx = 1
    # max_sent_len = 5
    max_sent_len = 5
    # env_dir = "/home_data/yh/dataset/EnvVersion1/res_files"
    # eval_env_dir = "/home_data/yh/dataset/EnvVersion1/eval_res_files"
    env_dir = "./env_generation/env03/env_files"
    eval_env_dir = "./env_generation/env03/eval_env_files"
    eval_newenv_dir = "./env_generation/env03/eval_newenv_files"
    dynamic_env_dir = "./env_catch/env_files"
    # dynamic_eval_env_dir = "/Users/hanyu/PycharmProjects/pythonProject/workstation/env_catch/eval_env_files"

    model_dir = "./models"
    sent_dir = "./sents"

    batch_size = 10
    # batch_size = 10
    epoch = 20000
    # max_obj_num = 8
    max_room_num = 6
    max_obs_num = 3
    max_gate_num = max_room_num - 1
    reward = 5
    # NOTE reward for the final goal
    # final_reward = 20
    # each_step_penalty = 10
    # lr_A = 0.001
    # lr_B = 0.001
    lr_task = 0.001
    # TODO
    room_emb_size_in = obs_feat_out_num * max_obs_num + gate_feat_out_num * max_gate_num
    # room_emb_size = obs_feat_out_num * max_obs_num + gate_feat_out_num * max_gate_num
    room_emb_size = 50
    # room_emb_size = obs_feat_out_num + gate_feat_out_num
    # NOTE if move too much, then this task could fail
    max_move_len = 6
    route_num = 3   # the num of sampled route in calculating the node emb with structure info
    end_of_route = -1
    route_encoder_hidden_size = 50
    gpu_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    is_gpu = False
    debug_mode = False
    is_dynamic_data = True
    dynamic_datasize = 100