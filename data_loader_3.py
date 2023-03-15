import collections
import os
import numpy as np
import math
from tqdm import tqdm
from collections import defaultdict
from geopy.distance import geodesic
import datetime
import time
import random
import pickle as pkl
import pandas as pd


def load_data(args):
    '''
    TODO: 工程的主要接入函数
    :param args:
    :return:
    '''
    train_data, eval_data, test_data, pos_data = load_rating_2()
    statistic = load_kg_2()
    rippleset_fun, rippleset_geo = load_ripple_set(args.n_memory)

    return train_data, eval_data, test_data, statistic, rippleset_fun, rippleset_geo, pos_data


def load_rating_2():
    train_data = pd.read_csv('data3/dataset/train.csv', encoding="utf-8", sep="\t").values
    eval_data = pd.read_csv('data3/dataset/eval.csv', encoding="utf-8", sep="\t").values
    test_data = pd.read_csv('data3/dataset/test.csv', encoding="utf-8", sep="\t").values

    pos_data = pd.read_csv('data3/dataset/pos_data.csv', encoding="utf-8", sep="\t").values

    return train_data, eval_data, test_data, pos_data


def load_kg_2():
    df_fun = pd.read_csv('data3/kg/kg_fun_rehashed.csv', sep="\t", encoding="utf-8", dtype=np.int32)
    kg_np_fun = df_fun.values
    df_geo = pd.read_csv('data3/kg/kg_geo_rehashed.csv', sep="\t", encoding="utf-8", dtype=np.int32)
    kg_np_geo = df_geo.values

    n_entity_fun = len(set(kg_np_fun[:, 0]) | set(kg_np_fun[:, 2]))
    n_relation_fun = len(set(kg_np_fun[:, 1]))
    n_entity_geo = len(set(kg_np_geo[:, 0]) | set(kg_np_geo[:, 2]))
    n_relation_geo = len(set(kg_np_geo[:, 1]))

    print("num of entities:", n_entity_fun, n_entity_geo)
    print("num of relations:", n_relation_fun, n_relation_geo)

    residence_dict = pkl.load(open('data2.4/residence/resi_user_dict.pkl', 'rb'))
    n_residence = len(residence_dict)
    print("num_residence:", n_residence)

    return (n_entity_fun, n_relation_fun, n_entity_geo, n_relation_geo, n_residence)


def load_ripple_set(n_memory):
    rippleset_fun = pkl.load(open('data3/rippleset/rippleset_fun_align_{}.pkl'.format(n_memory), 'rb'))
    rippleset_geo = pkl.load(open("data3/rippleset/rippleset_geo_align_{}.pkl".format(n_memory), 'rb'))

    return rippleset_fun, rippleset_geo
