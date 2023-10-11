import json
from EnvGraph import *


class RoutePlan(object):
    @staticmethod
    def _construct_graph(env_info):
        env_graph = {}
        for room_id, room_info in env_info.items():
            room_id_str = str(room_id)
            if room_id_str not in env_graph: env_graph[room_id_str] = {}
            gates_info = room_info["gates"]
            for gate_id, cur_gate_info in gates_info.items():
                neighbor_id_str = str(cur_gate_info["neighbor"])
                env_graph[room_id_str][neighbor_id_str] = str(gate_id)
        return env_graph

    @staticmethod
    def _floyd(env_graph):
        # --- init ---
        distances = {}; next_room = {}; next_door = {}
        for room, neighbors in env_graph.items():
            room_int = int(room)
            if room_int not in distances:
                distances[room_int] = {}; next_room[room_int] = {}
            for cur_neighbor in neighbors:
                cur_neighbor_int = int(cur_neighbor)
                distances[room_int][cur_neighbor_int] = 1; next_room[room_int][cur_neighbor_int] = cur_neighbor_int
                if cur_neighbor_int not in distances:
                    distances[cur_neighbor_int] = {}; next_room[cur_neighbor_int] = {}
                distances[cur_neighbor_int][room_int] = 1; next_room[cur_neighbor_int][room_int] = room_int
        # --- insert node ---
        for room, neighbors in distances.items():
            neighbors_list = list(neighbors.keys())
            for room_i_idx in range(len(neighbors_list)):
                for room_j_idx in range(room_i_idx + 1, len(neighbors_list)):
                    room_i = neighbors_list[room_i_idx]; room_j = neighbors_list[room_j_idx]
                    new_distance = distances[room][room_i] + distances[room][room_j]
                    ori_distance = -1 if room_j not in distances[room_i] else distances[room_i][room_j]
                    if ori_distance == -1 or new_distance < ori_distance:
                        distances[room_i][room_j] = new_distance; distances[room_j][room_i] = new_distance
                        next_room[room_i][room_j] = next_room[room_i][room]; next_room[room_j][room_i] = next_room[room_j][room]
        # --- next door ---
        for room, targets in next_room.items():
            if room not in next_door: next_door[room] = {}
            for target, cur_next_room in targets.items():
                next_door[room][target] = int(env_graph[str(room)][str(cur_next_room)])
        # NOTE special case: self to self
        for room in next_door.keys():
            next_door[room][room] = -1  # choose no gate
            next_room[room][room] = room
        return next_room, next_door

    @staticmethod
    def find_shortest_path(file_path, method="floyd"):
        with open(file_path, 'r') as f:
            env_info = json.load(f)
        env_graph = RoutePlan._construct_graph(env_info)
        if method == "floyd":
            next_room, next_door = RoutePlan._floyd(env_graph)
        else:
            print("there is no method called {}".format(method))
            raise NameError
        return next_room, next_door


# if __name__ == "__main__":
#     RoutePlan._floyd(env_graph)
