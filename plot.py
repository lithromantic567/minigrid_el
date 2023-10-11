import numpy as np
import random

import torch

from Dataset import *
from Agents import *
from torch.utils.data import DataLoader
from torch.optim import Adam
from GuessGate import *
import matplotlib.pyplot as plt

def plotfig(fp):
    x=np.arange(0, Param.epoch,50 )
    y=[]
    with open(fp,'r') as f:
        data=f.read().strip().split()
        y.extend([float(i) for i in data])
    plt.plot(x,y)
    plt.show()
    plt.savefig("guessroom_40000.png")

if __name__ == "__main__":
    plotfig("guessRoom_acc_40000.txt")