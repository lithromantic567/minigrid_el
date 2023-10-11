import os
import json
import numpy
import re
import torch
import torch_ac
import gymnasium as gym

import utils

# 将输入（观测到）的图像和文字转化成可索引的列表对象——PyTorch张量

def get_obss_preprocessor(obs_space):
    # Check if obs_space is an image space
    if isinstance(obs_space, gym.spaces.Box): #Box数据类型，其实就是array
        obs_space = {"image": obs_space.shape}

        def preprocess_obss(obss, device=None):
            return torch_ac.DictList({
                "image": preprocess_images(obss, device=device)
            })

    # Check if it is a MiniGrid observation space
    elif isinstance(obs_space, gym.spaces.Dict) and "image" in obs_space.spaces.keys():
        obs_space = {"image": obs_space.spaces["image"].shape, "text": 100}

        vocab = Vocabulary(obs_space["text"])

        def preprocess_obss(obss, device=None):
            return torch_ac.DictList({
                "image": preprocess_images([obs["image"] for obs in obss], device=device),
                "text": preprocess_texts([obs["mission"] for obs in obss], vocab, device=device)
            })

        preprocess_obss.vocab = vocab

    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss

#转化成tensor数据类型，适用于深度学习
def preprocess_images(images, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = numpy.array(images)
    return torch.tensor(images, device=device, dtype=torch.float)


def preprocess_texts(texts, vocab, device=None):
    var_indexed_texts = []
    max_text_len = 0

    for text in texts:
        tokens = re.findall("([a-z]+)", text.lower())
        # vocab[token]表示vocab中token第几个加入, var_indexed_text长度就是token的个数，token可能重复
        var_indexed_text = numpy.array([vocab[token] for token in tokens])
        var_indexed_texts.append(var_indexed_text)
        # 句子的最长值
        max_text_len = max(len(var_indexed_text), max_text_len)
    #填充
    indexed_texts = numpy.zeros((len(texts), max_text_len))

    for i, indexed_text in enumerate(var_indexed_texts):
        indexed_texts[i, :len(indexed_text)] = indexed_text

    return torch.tensor(indexed_texts, device=device, dtype=torch.long)


class Vocabulary:
    """A mapping from tokens to ids with a capacity of `max_size` words.
    It can be saved in a `vocab.json` file."""

    def __init__(self, max_size):
        self.max_size = max_size
        self.vocab = {}

    def load_vocab(self, vocab):
        self.vocab = vocab
        
    # 如果在类中定义了__getitem__()方法，那么他的实例对象（假设为P）就可以这样P[key]取值。
    # 当实例对象做P[key]运算时，就会调用类中的__getitem__()方法。
    # 输出的是token是第几个加入vocab
    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]
