import torch
from NavigateTrain import navigation_evaluate
import matplotlib.pyplot as plt


def load_performance(path):
    with open(path, 'r') as f:
        success_rate = []; room_acc = []; gate_acc = []
        for line in f:
            if line.strip().startswith("eval acc"):
                temp = line.strip().split("=")
                cur_success_rate = float(temp[-1].strip())
                success_rate.append(cur_success_rate)
            if line.strip().startswith("eval first guess room acc"):
                temp = line.strip().split(",")
                room_acc.append(float(temp[0].strip().split("=")[-1].strip()))
                gate_acc.append(float(temp[1].strip().split("=")[-1].strip()))
    return success_rate, room_acc, gate_acc



if __name__ == "__main__":
    # model_path = "/Users/hanyu/PycharmProjects/pythonProject/workstation/models/4500.pth"
    # task = torch.load(model_path)
    # navigation_evaluate(task)

    model_dir = "/Users/hanyu/PycharmProjects/pythonProject/workstation/models"
    for i in range(0, 18200, 100):
        task = torch.load("{}/{}.pth".format(model_dir, i))
        navigation_evaluate(task)

    # success_rate, room_acc, gate_acc = load_performance("/Users/hanyu/PycharmProjects/pythonProject/workstation/logs/eval_performance_route01.log")
    # plt.subplot(3, 1, 1)
    # plt.plot(success_rate)
    # plt.subplot(3, 1, 2)
    # plt.plot(room_acc)
    # plt.subplot(3, 1, 3)
    # plt.plot(gate_acc)
    # plt.show()