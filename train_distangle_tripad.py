import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from tensorflow.python import debug as tf_debug
import numpy as np
from model_distangle_3_resi import RippleNet_distangle
import datetime
import time
import scipy.io as sio
from collections import defaultdict
import pickle as pkl

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(args, data_info, show_loss):
    '''
    TODO: 整个文件的核心函数，串联训练过程的各个部分
    :param args:
    :param data_info: data_loader.py加载进来的所有数据（6部分），以一个tuple的形式存储
    :param show_loss: 决定是否每个batch完成后，输出一次当前loss
    :return:
    '''
    # data_info是data_loader.py加载进来的所有数据，以一个tuple的形式存储
    train_data = data_info[0]  # 训练数据集（60%），numpy.ndarray形式（二维数组），“[[user1,book1,rating], [user2,book2,rating],…]”
    eval_data = data_info[1]  # 验证数据集（20%），numpy.ndarray形式（二维数组），“[[user1,book1,rating], [user2,book2,rating],…]”
    test_data = data_info[2]  # 测试数据集（20%），numpy.ndarray形式（二维数组），“[[user1,book1,rating], [user2,book2,rating],…]”
    statistic = data_info[3]  # Satori实体个数：77,903个实体【包含图书和作者等各种相关的实体】
    rippleset_fun = data_info[4]  # 各个用户的各阶ripple set
    rippleset_geo = data_info[5]
    pos_data = data_info[6]  # 所有正例数据

    # 实例化一个RippleNet对象，整个网络结构和模型框架都在RippleNet的初始化函数中做了
    # 也就是说，网络结构使用是在train.py的train中进行的
    start_time = time.time()
    model = RippleNet_distangle(args, statistic)
    end_time = time.time()
    print("\nexecution time to build model:\t{} s\n".format(end_time - start_time))

    print("num_batch:\t{}".format(train_data.shape[0] // args.batch_size))

    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1

    # TensorFlow.Session提供了Operation执行和Tensor求值的环境。
    # Session叫做回话，掌控TensorFlow计算和存储的所有空间
    # tensorflow所有的真实操作，都要在tf.Session()中执行
    with tf.Session() as sess:
        # 尝试TensorFlow的debug工具，包装一下sess，就可以进入debug环境
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # 调用Session.run(fetches, feed_dict=None)方法来执行Operation或者求值Tensor：
        # 参数fetches便是一个或者多个Operation或者Tensor
        # tf.global_variables_initializer()初始化模型的所有参数
        # 这是tensorflow训练神经网络所必需的
        sess.run(tf.global_variables_initializer())

        # 训练n个epoch，这里设置的是10个
        for step in range(args.n_epoch):
            # training
            start_time = datetime.datetime.now()
            # numpy.random.shuffle(x)打乱序列的顺序，对于多维数组只按第一维打乱，
            # 比如二维数组，只能打乱行的排列，列是没有变化的，非常符合这里的使用要求。
            np.random.shuffle(train_data)
            start = 0
            end = args.batch_size
            while end <= train_data.shape[0]:
                # shape属性是数组的维度。返回的是一个整数的元组，元组中元素对应着每一维度的大小(size)。
                # shape[0]指的是二维矩阵的行数，即训练集的大小。
                # 训练的过程中，要把整个训练集都用上，都得遍历了
                # 一个batch学一次model
                # end = start + args.batch_size
                # start_time = time.time()
                _, loss = model.train(
                    sess, get_feed_dict(args, model, train_data, rippleset_fun, rippleset_geo,
                                        start, end))

                start += args.batch_size  # 一个batch里面有多少样本，默认设的是1024
                end += args.batch_size

                # 输出每个batch完成后的训练误差
                if show_loss:  # 是不是需要每个batch完都show一下loss，一般不用，因为这样太多了
                    # 输出训练过程进行到了百分之几，且损失还有剩余多少
                    print('%.1f%% %.4f' % (start / train_data.shape[0] * 100, loss))

            # evaluation
            # 评估算法在各个数据集上的表现，每个epoch都需要进行一次
            train_auc, train_acc = evaluation(sess, args, model, train_data, rippleset_fun, rippleset_geo,
                                              args.batch_size)
            eval_auc, eval_acc = evaluation(sess, args, model, eval_data, rippleset_fun, rippleset_geo,
                                            args.batch_size)
            test_auc, test_acc = evaluation(sess, args, model, test_data, rippleset_fun, rippleset_geo,
                                            args.batch_size)
            # 输出，每个epoch结束后，在三个数据集上的算法表现（AUC和Accuracy）
            print('epoch %d    train auc: %.4f  acc: %.4f    eval auc: %.4f  acc: %.4f    test auc: %.4f  acc: %.4f'
                  % (step, train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc))

            end_time = datetime.datetime.now()
            print("epoch %d\ttime %f s" % (step, (end_time - start_time).seconds))
            print(end_time)

        # 最后再输出attention
        print("-" * 100)
        output_attention(sess, args, model, pos_data, rippleset_fun, rippleset_geo)
        print('output attention time:', datetime.datetime.now())


def get_feed_dict(args, model, data, rippleset_fun, rippleset_geo, start, end):
    '''
    TODO: 给model.train()中的sess.run([self.optimizer, self.loss], feed_dict)中的feed_dict赋值
    TODO: 取出训练所用的数据，包括训练集（user-item interaction）和ripple set
    :param args:
    :param model:
    :param data:
    :param ripple_set:
    :param start:
    :param end:
    :return:
    '''
    # {entity_id: eid_fun}
    # {entity_id: eid_geo}
    eid_dict_fun = pkl.load(open("data2.3/kg_lookup/eid_dict_fun.pkl", 'rb'))
    eid_dict_geo = pkl.load(open("data2.3/kg_lookup/eid_dict_geo.pkl", 'rb'))

    feed_dict = dict()
    # 调用时，data是训练集，二维ndarry
    # [
    #   [user1, spot1, rating1, ]，
    #   [user2, spot2, rating2, ]，
    #    ...
    # ]
    # feed_dict[model.items]: [batch_size]
    feed_dict[model.users] = data[start:end, 0]
    feed_dict[model.residences_id] = data[start:end, 3]
    # feed_dict[model.items]: [batch_size]
    items = data[start:end, 1]
    feed_dict[model.items_eid_fun] = [eid_dict_fun[item] for item in items]  # 训练model用的items（图书）的id，shape是[batch_size]
    feed_dict[model.items_eid_geo] = [eid_dict_geo[item] for item in items]
    # feed_dict[model.items]: [batch_size]
    feed_dict[model.labels] = data[start:end, 2]  # label即这本书是否被读过，1或0，shape是[batch_size]

    for i in range(args.n_hop):
        # i是RippleSet的阶数
        # RippleSet的user标识去哪了？
        # 不需要user标识了，因为每条记录都对应一个用户，用户被他的rippleset所替代掉了
        # feed_dict[model.memories_h[i]]: [batch_size, n_memory]
        feed_dict[model.memories_h_fun[i]] = [rippleset_fun[user][i][0] for user in data[start:end, 0]]
        feed_dict[model.memories_r_fun[i]] = [rippleset_fun[user][i][1] for user in data[start:end, 0]]
        feed_dict[model.memories_t_fun[i]] = [rippleset_fun[user][i][2] for user in data[start:end, 0]]
        feed_dict[model.memories_h_geo[i]] = [rippleset_geo[user][i][0] for user in data[start:end, 0]]
        feed_dict[model.memories_r_geo[i]] = [rippleset_geo[user][i][1] for user in data[start:end, 0]]
        feed_dict[model.memories_t_geo[i]] = [rippleset_geo[user][i][2] for user in data[start:end, 0]]

    return feed_dict


def evaluation(sess, args, model, data, rippleset_fun, rippleset_geo, batch_size):
    '''
    TODO: 返回在数据集上的算法表现，AUC和Accuracy
    :param sess:
    :param args:
    :param model:
    :param data:
    :param ripple_set:
    :param batch_size:
    :return:
    '''
    # 每个batch的起止index
    start = 0
    end = batch_size
    auc_list = []
    acc_list = []
    while end <= data.shape[0]:
        # 一个batch评估一次结果，最后把一个epoch内的所有batch结果取个均值
        auc, acc = model.eval(sess, get_feed_dict(args, model, data, rippleset_fun, rippleset_geo,
                                                  start, end))
        auc_list.append(auc)
        acc_list.append(acc)
        start += batch_size
        end += batch_size
    return float(np.mean(auc_list)), float(np.mean(acc_list))


def output_attention(sess, args, model, data, rippleset_fun, rippleset_geo):
    '''
    TODO: 将网络中所有的注意力权重进行输出
    :param sess:
    :param args:
    :param model:
    :param data:
    :param rippleset_fun:
    :param rippleset_geo:
    :return:
    '''
    # micro pi
    # [data_size, n_hop, n_memory]
    p_fun_tensor = np.empty(shape=[0, args.n_hop, args.n_memory])
    p_geo_tensor = np.empty(shape=[0, args.n_hop, args.n_memory])
    # macro alpha
    # [data_size]
    alpha_fun_list = []
    alpha_geo_list = []
    # residence_determinism beta
    # [data_size]
    beta_list = []

    # 数据输入每个batch_size的起止位
    start = 0
    end = args.batch_size

    while end <= data.shape[0]:
        p_fun_batch, p_geo_batch, alpha_fun_batch, alpha_geo_batch, beta_batch = model.show_attention(
            sess, get_feed_dict(args, model, data, rippleset_fun, rippleset_geo, start, end)
        )
        p_fun_tensor = np.concatenate((p_fun_tensor, p_fun_batch), axis=0)
        p_geo_tensor = np.concatenate((p_geo_tensor, p_geo_batch), axis=0)

        alpha_fun_list += list(alpha_fun_batch)
        alpha_geo_list += list(alpha_geo_batch)
        beta_list += list(beta_batch)

        start += args.batch_size
        end += args.batch_size

    np.savez('result3/attentions.npz', p_fun_tensor, p_geo_tensor, alpha_fun_list, alpha_geo_list, beta_list)

    return p_fun_tensor, p_geo_tensor, alpha_fun_list, alpha_geo_list, beta_list
