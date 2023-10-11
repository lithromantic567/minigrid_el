from __future__ import annotations
import json

import numpy as np

import pygame


from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid
from minigrid.core.world_object import Ball, Door, Goal, Key, Wall, Box
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

from minigrid.envs.babyai.core.levelgen import LevelGen
from minigrid.envs.babyai.core.roomgrid_level import RejectSampling, RoomGridLevel
from minigrid.envs.babyai.core.verifier import GoToInstr, ObjDesc

from minigrid.wrappers import DictObservationSpaceWrapper, FullyObsWrapper

from el.try.localization import MinigridFeaturesExtractor
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO

class GoTo(RoomGridLevel):
    """

    ## Description

    Go to an object, the object may be in another room. Many distractors.

    ## Mission Space

    "go to a/the {color} {type}"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    {type} is the type of the object. Can be "ball", "box" or "key".

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused            |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent goes to the object.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-GoTo-v0`
    - `BabyAI-GoToOpen-v0`
    - `BabyAI-GoToObjMaze-v0`
    - `BabyAI-GoToObjMazeOpen-v0`
    - `BabyAI-GoToObjMazeS4R2-v0`
    - `BabyAI-GoToObjMazeS4-v0`
    - `BabyAI-GoToObjMazeS5-v0`
    - `BabyAI-GoToObjMazeS6-v0`
    - `BabyAI-GoToObjMazeS7-v0`
    """

    def __init__(
        self,
        room_size=8,
        num_rows=3,
        num_cols=3,
        num_dists=18,
        doors_open=True,
        **kwargs,
    ):
        self.num_dists = num_dists
        self.doors_open = doors_open
        self.agent_room: int = None
        
        super().__init__(
            
            num_rows=num_rows, 
            num_cols=num_cols, 
            room_size=room_size, 
            **kwargs
        )
       

    def gen_mission(self):
        # We catch RecursionError to deal with rare cases where
        # rejection sampling gets stuck in an infinite loop
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), self.width - 2, self.height - 2)
        self.place_agent()
        '''
        #增加了room
        self.agent_row=self.agent_pos[1]//7
        self.agent_col=self.agent_pos[0]//7
        self.agent_room = self.agent_row*self.num_rows+self.agent_col
        print(self.agent_pos)
        print(self.agent_room)
        print(self.agent_row)
        print(self.agent_col)
        '''
        
        self.connect_all()
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))

        # If requested, open all the doors
        if self.doors_open:
            self.open_all_doors()

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
#房间id，用于定位       
def get_room_id(env,room,i,j):
    room.id=i*env.num_rows+j
    return room.id

def get_neighbor_id(env,room,i,j):
        #计算邻居房间的id
    if room.doors[0] is not None:
        room.neighbors[0].id=get_room_id(env,room.neighbors[0],i,j+1)
    if room.doors[1] is not None:
        room.neighbors[1].id=get_room_id(env,room.neighbors[1],i+1,j)
    if room.doors[2] is not None:
        room.neighbors[2].id=get_room_id(env,room.neighbors[2],i,j-1)
    if room.doors[3] is not None:
        room.neighbors[3].id=get_room_id(env,room.neighbors[3],i-1,j)
        
def output_envs(env, output_file_path):
    directions_str = [ "right",  "down","left","up"]
    res = {}
    for i in range(3):  #row
        for j in range(3):  #col
            room=env.get_room(j,i)
            id=str(get_room_id(env,room,i,j))
            get_neighbor_id(env,room,i,j)
            res[id] = {}
            res[id]["pos"] = [7*i+1, 7*i+6,7*j+1, 7*j+6]  # up. down, left, right
            # gates
            res_gates = {}
            for d in range(4):    
                gate_id = len(res_gates.keys())
                if  room.doors[d] is not None:
                    if d in [0, 2]:  # left or right
                        res_gates[gate_id] = {"direction": directions_str[d],
                                            "pos": room.doors[d].cur_pos[1] - res[id]["pos"][0],
                                            "neighbor": room.neighbors[d].id}
                    elif d in [2, 3]:  # up or down
                        res_gates[gate_id] = {"direction": directions_str[d],
                                            "pos": room.doors[d].cur_pos[0] - res[id]["pos"][2],
                                            "neighbor": room.neighbors[d].id}
            res[id]["gates"] = res_gates
            # obstacles
            res_obstacles = {}            
            for item in room.objs:
                obstacle_id =len(res_obstacles)
                res_obstacles[obstacle_id] = {"pos": [item.cur_pos[1] - res[id]["pos"][0],item.cur_pos[0] - res[id]["pos"][2]],
                                            "color": item.color, "shape": item.type}
            res[id]["obstacles"] = res_obstacles
    
    with open(output_file_path, 'w') as f:
        json.dump(res, f, indent=4,cls=NpEncoder)


def main():
    env = GoTo(render_mode="human")
    env._gen_grid(env.width,env.height)
    env_obs =FullyObsWrapper(env)
    for i in range(100):
        obs,_ = env_obs.reset()  
        output_file='./env_full_eval'+'/env{}.txt'.format(i)
        output_envs(env,output_file)
        env.render()
        pygame.image.save(env.window, "pic/env{}.png".format(i))
             
    '''
    #输出整个环境的三维向量表示
        output_file='./env_full'+'/env{}.txt'.format(i)
        np.set_printoptions(threshold=np.inf)
        with open(output_file,'w') as f:
            f.write(str(obs['image']))
         
    #Room类型
    agent_room=env.room_grid[env.agent_row][env.agent_col]
    #agent所在房间的障碍物的三维向量表示；障碍物的位置（列，行）
    for j in agent_room.objs:
        print(j.encode(),j.cur_pos)
    #door_pos每个方位都存在，但是door不一定真的存在
    print(agent_room.doors)
    #door的位置，right,down,left,top
            
    agent_room.id= get_room_id(env,agent_room,env.agent_row,env.agent_col)
    print(agent_room.id) 
           
    for r in range(4):
        if agent_room.doors[r] is not None:
            print(agent_room.doors[r].cur_pos)
            print(agent_room.neighbors[r].id)
    '''          
    #env.gen_mission()
    
    

    
if __name__ == "__main__":
    main()