'''
TODO: 此模型是在2_1的基础上所进行的改进，把最终的居住环境变量Resi加入了对u影响中
'''

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import numpy as np
from sklearn.metrics import roc_auc_score
import time
import json


class RippleNet_distangle(object):
    def __init__(self, args, statistics):
        '''
        TODO: 初始化，模型的接口
        :param args: 模型超参
        :param n_entity: Satori中的实体数，头尾实体数相加
        :param n_relation: Satori中的关系种数
        '''
        self._parse_args(args, statistics)
        self._build_inputs()
        self._build_embeddings()
        self._build_model()
        self._build_loss()
        self._build_train()

    def _parse_args(self, args, statistics):
        '''
        TODO: 设置超参
        TODO: 声明并赋值一堆的数据成员
        :param args:
        :param n_entity:
        :param n_relation:
        :return:
        '''
        self.n_entity_fun = statistics[0]  # Satori的实体个数：77,903
        self.n_relation_fun = statistics[1]  # Satori的关系种数：25
        self.n_entity_geo = statistics[2]  # Satori的实体个数：77,903
        self.n_relation_geo = statistics[3]  # Satori的关系种数：25
        self.n_city = statistics[4]  # 居住地的总数：346
        self.dim = args.dim  # 实体和关系embedding的维数：16
        self.n_hop = args.n_hop  # 几跳之后进行截断：2
        self.kge_weight = args.kge_weight  # KGE任务设的权重：0.01
        self.l2_weight = args.l2_weight  # 正则化项的权重：1e-7
        self.lr = args.lr  # 学习率：0.02
        self.batch_size = args.batch_size  # 一个batch的样本数
        self.n_memory = args.n_memory  # 每阶RippleSet最多有多少个三元组：32
        self.item_update_mode = args.item_update_mode  # how to update item at the end of each hop：plus_transform
        self.using_all_hops = args.using_all_hops  # whether using outputs of all hops or just the last hop when making prediction：True

    def _build_inputs(self):
        '''
        TODO: 输入层
        TODO: 读入user-spot interaction，各书的id号和label（是否读过）
        :return:
        '''
        # tf.placeholder(dtype, shape=None, name=None)
        # 此函数用于定义过程，在执行的时候再赋具体的值
        # 神经网络的输入都是用占位符声明的，训练测试的时候，placeholder就负责接数据。
        # 参数：
        #   dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
        #   shape：数据形状。默认是None，就是一维值，也可以多维，
        #           比如：[None，3]，表示列是3，行不一定
        #   name：名称。
        # 返回：
        #   Tensor类型
        # 使用的时候，赋值方法如下：
        # 赋值一般用sess.run(feed_dict = {x:xs, y_:ys})，其中x,y_是用placeholder创建出来的
        #
        # items是user-item矩阵中的图书id，可以直接用作图书的索引
        # labels是user-item矩阵中的每本图书是否被读过，即1或0
        # users是每条记录的用户信息，为了评估HIT@K上的表现
        # [batch_size]
        self.users = tf.placeholder(dtype=tf.int32, shape=[None], name="users")
        # [batch_size]
        self.residences_id = tf.placeholder(dtype=tf.int32, shape=[None], name="user_residence")

        # item在kg_fun和kg_geo中的entity_id
        # [batch_size]
        self.items_eid_fun = tf.placeholder(dtype=tf.int32, shape=[None], name="items_eid_fun")
        self.items_eid_geo = tf.placeholder(dtype=tf.int32, shape=[None], name="items_eid_geo")
        # [batch_size]
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name="labels")  # 之所以将类型设为浮点型，是为了方便计算RS的损失项

        # 用来接RippleSet的，list中的每一个元素是一阶RippleSet的h，r或t，即memories_h_1, memories_h_2, ...
        # 这几个list的shape: [n_nop, batch_size, n_memory]
        self.memories_h_fun = []
        self.memories_r_fun = []
        self.memories_t_fun = []

        self.memories_h_geo = []
        self.memories_r_geo = []
        self.memories_t_geo = []

        for hop in range(self.n_hop):
            # 这几个placeholder的shape是行不定的，列数为RippleSet的size：n_memory(32)。
            self.memories_h_fun.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_h_fun_" + str(hop)))
            self.memories_r_fun.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_r_fun_" + str(hop)))
            self.memories_t_fun.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_t_fun_" + str(hop)))
            self.memories_h_geo.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_h_geo_" + str(hop)))
            self.memories_r_geo.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_r_geo_" + str(hop)))
            self.memories_t_geo.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_t_geo_" + str(hop)))

    def _build_embeddings(self):
        '''
        Note: 说明entity和rel的向量表示也是待训练的参数。
        TODO: 定义和初始化entity和rel的表达：entity是向量，rel是矩阵，存这俩的就是矩阵和张量
        :return:
        '''
        # tf.get_variable(): 创建一个新变量
        # name是变量的名称，shape是变量的形状，initializer是初始化变量的方式
        # 方式tf.contrib.layers.xavier_initializer()：
        #   该函数返回一个用于初始化权重的初始化程序“Xavier”，
        #   这个初始化器是用来保持每一层的梯度大小都差不多相同。
        # 对于神经网络，所有需要训练的参数和权重都是tf.get_variable()创建和声明的。
        # [num_of_entities, dim]
        self.entity_fun_emb_matrix = tf.get_variable(name="entity_fun_emb_matrix", dtype=tf.float32,
                                                     shape=[self.n_entity_fun, self.dim],
                                                     initializer=tf.keras.initializers.glorot_normal(),
                                                     )
        self.entity_geo_emb_matrix = tf.get_variable(name="entity_geo_emb_matrix", dtype=tf.float32,
                                                     shape=[self.n_entity_geo, self.dim],
                                                     initializer=tf.keras.initializers.glorot_normal(),
                                                     )
        # [num_of_relations, dim, dim]
        # [44, 64, 64]
        self.rel_fun_emb_matrix = tf.get_variable(name="rel_fun_emb_matrix", dtype=tf.float32,
                                                  shape=[self.n_relation_fun, self.dim, self.dim],
                                                  initializer=tf.keras.initializers.glorot_normal(),
                                                  )
        # [14, 64, 64]
        self.rel_geo_emb_matrix = tf.get_variable(name="rel_geo_emb_matrix", dtype=tf.float32,
                                                  shape=[self.n_relation_geo, self.dim, self.dim],
                                                  initializer=tf.keras.initializers.glorot_normal(),
                                                  )

        # 以Rescal模型的KGE的结果作为各个embed初始化
        with open('data3/embed/embed_fun_{}.vec'.format(self.dim), 'r') as loader:
            json_dict = json.load(loader)
            ent_emb = np.array(json_dict['ent_embeddings.weight'])
            rel_emb = np.array(json_dict['rel_matrices.weight'])
            self.entity_fun_emb_matrix.assign(ent_emb)
            self.rel_fun_emb_matrix.assign(rel_emb.reshape([self.n_relation_fun, self.dim, self.dim]))

        with open('data3/embed/embed_geo_{}.vec'.format(self.dim), 'r') as loader:
            json_dict = json.load(loader)
            ent_emb = np.array(json_dict['ent_embeddings.weight'])
            rel_emb = np.array(json_dict['rel_matrices.weight'])
            self.entity_geo_emb_matrix.assign(ent_emb)
            self.rel_geo_emb_matrix.assign(rel_emb.reshape([self.n_relation_geo, self.dim, self.dim]))

        # 居住环境变量的encoding_matrix
        # [居住地数量, dim]
        self.resi_emb_matrix = tf.get_variable(name="residence_encoding_matrix", dtype=tf.float32,
                                               shape=[self.n_city, self.dim],
                                               initializer=tf.keras.initializers.glorot_normal(), )
        # Residence与u融合的参数矩阵
        # [batch_size, dim, dim*2]
        self.w_context = tf.get_variable(name="weight_context", dtype=tf.float32,
                                         shape=[self.batch_size, self.dim, self.dim * 2],
                                         initializer=tf.keras.initializers.glorot_normal(), )

    def _build_model(self):
        '''
        TODO: 网络的主函数，串联输入层到输出层
        :return:
        '''
        # transformation matrix for updating item embeddings at the end of each hop
        # transform_matrix是用于更新每个跳结束时的更新item向量的转换矩阵，
        # 在update_item_embedding()中会被用到，关系到是怎么迭代o的。
        # 详情见update_item_embedding()
        # [dim, dim]
        self.transform_matrix_fun = tf.get_variable(name="transform_matrix_fun", shape=[self.dim, self.dim],
                                                    dtype=tf.float32,
                                                    initializer=tf.keras.initializers.glorot_normal())
        self.transform_matrix_geo = tf.get_variable(name="transform_matrix_geo", shape=[self.dim, self.dim],
                                                    dtype=tf.float32,
                                                    initializer=tf.keras.initializers.glorot_normal())

        # tf.nn.embedding_lookup(params, ids)函数的用法主要是选取一个张量里面索引对应的元素:
        #   params可以是张量也可以是数组等，id就是对应的索引。
        # 可以看出self.items就是一堆用户看过的书的id号，可以直接被用来当做索引。
        # note：item和KG中的entity是共享embedding表示的。
        # [batch size, dim]
        self.item_fun_embeddings = tf.nn.embedding_lookup(self.entity_fun_emb_matrix, self.items_eid_fun)
        self.item_geo_embeddings = tf.nn.embedding_lookup(self.entity_geo_emb_matrix, self.items_eid_geo)

        # 居住地的embedding
        # [batch_size, dim]
        self.resi_embeddings = tf.nn.embedding_lookup(self.resi_emb_matrix, self.residences_id)

        # [n_hop, batch_size, n_memory, dim]
        self.h_fun_emb_list = []
        self.h_geo_emb_list = []
        # [n_hop, batch_size, n_memory, dim, dim]
        self.r_fun_emb_list = []
        self.r_geo_emb_list = []
        # [n_hop, batch_size, n_memory, dim]
        self.t_fun_emb_list = []
        self.t_geo_emb_list = []

        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            # [1024, 32, 16]
            self.h_fun_emb_list.append(tf.nn.embedding_lookup(self.entity_fun_emb_matrix, self.memories_h_fun[i]))
            self.h_geo_emb_list.append(tf.nn.embedding_lookup(self.entity_geo_emb_matrix, self.memories_h_geo[i]))

            # [batch size, n_memory, dim, dim]
            # [1024, 32, 16, 16]
            self.r_fun_emb_list.append(tf.nn.embedding_lookup(self.rel_fun_emb_matrix, self.memories_r_fun[i]))
            self.r_geo_emb_list.append(tf.nn.embedding_lookup(self.rel_geo_emb_matrix, self.memories_r_geo[i]))

            # [batch size, n_memory, dim]
            # [1024, 32, 16]
            self.t_fun_emb_list.append(tf.nn.embedding_lookup(self.entity_fun_emb_matrix, self.memories_t_fun[i]))
            self.t_geo_emb_list.append(tf.nn.embedding_lookup(self.entity_geo_emb_matrix, self.memories_t_geo[i]))

        # o是第i阶RippleSet的波动总和
        # 这里把RippleNet兴趣传播的核心过程加了进来
        # [n_hop, batch_size, dim]
        # start_time = time.time()
        self.o_fun_list, self.o_geo_list = self._key_addressing()
        # end_time = time.time()
        # print("\ntime to execute _key_addressing per batch:\t{}s\n".format(end_time - start_time))

        # tf.squeeze()函数返回一个张量，
        # 这个张量是将原始input中所有维度为1的那些维都删掉的结果
        # [batch_size]
        self.scores = tf.squeeze(
            self.predict(self.item_fun_embeddings, self.item_geo_embeddings, self.o_fun_list, self.o_geo_list),
            name="scores")

        # tf.sigmoid(x)计算 x 元素的sigmoid，具体来说,就是：y = 1/(1 + exp (-x)).
        # 输入一个Tensor，返回一个相同shape的Tensor
        # [batch_size]
        self.scores_normalized = tf.sigmoid(self.scores, name="scores_normalized")

    def _key_addressing(self):
        '''
        TODO: RippleNet中，user对候选item的兴趣传播的核心网络结构
        函数名的直译：关键解决
        TODO: 输出每一阶RippleSet的波动加权和o
        :return o_list:每个元素是一阶RippleSet的波动加权和o的list
        '''
        o_fun_list = []
        o_geo_list = []

        self.probs_fun_list = []
        self.probs_geo_list = []

        for hop in range(self.n_hop):  # 与原文一致，一阶一阶的向外扩散式地计算o
            # tf.expand_dims(input, axis)功能是给定一个input，在axis轴处给input增加一个size为1的维度。
            # [batch_size, n_memory, dim, 1]
            h_fun_expanded = tf.expand_dims(self.h_fun_emb_list[hop], axis=3)
            h_geo_expanded = tf.expand_dims(self.h_geo_emb_list[hop], axis=3)

            # tf.matmul()矩阵乘法，rel的矩阵表示 * entity的向量表示，乘完之后的Rh是一行
            # 该函数返回一个张量，这个张量是将原始input中所有维度为1的那些维都删掉的结果
            # axis可以用来指定要删掉的为1的维度
            # [batch_size, n_memory, dim]，其中n_memory是一阶RippleSet的大小
            # Note: 这种高维矩阵相乘，实际上还是二维矩阵的乘法，乘的是后两维度，
            # 前面的维度保持一致，只是循环而已
            # self.r_emb_list[hop]是[batch_size, n_memory, dim, dim]
            # Rh的shape: [batch_size, n_memory, dim]
            Rh_fun = tf.squeeze(tf.matmul(self.r_fun_emb_list[hop], h_fun_expanded), axis=3)
            Rh_geo = tf.squeeze(tf.matmul(self.r_geo_emb_list[hop], h_geo_expanded), axis=3)

            # v是item的embedding表示向量
            # [batch_size, dim, 1]
            v_fun = tf.expand_dims(self.item_fun_embeddings, axis=2)
            v_geo = tf.expand_dims(self.item_geo_embeddings, axis=2)

            # probs=vRh
            # probs_normalized就是论文中的pi，p=sigmoid(vRh)
            # [batch_size, n_memory]
            probs_fun = tf.squeeze(tf.matmul(Rh_fun, v_fun), axis=2)
            probs_geo = tf.squeeze(tf.matmul(Rh_geo, v_geo), axis=2)

            # softmax是sigmoid的n类版
            # [batch_size, n_memory]
            probs_fun_normalized = tf.nn.softmax(probs_fun)
            probs_geo_normalized = tf.nn.softmax(probs_geo)

            # 每一阶的probs追加到后面
            # [n_hop, batch_size, n_memory]
            self.probs_fun_list.append(probs_fun_normalized)
            self.probs_geo_list.append(probs_geo_normalized)

            # [batch_size, n_memory, 1]
            probs_fun_expanded = tf.expand_dims(probs_fun_normalized, axis=2)
            probs_geo_expanded = tf.expand_dims(probs_geo_normalized, axis=2)

            # tf.reduce_sum()是压缩求和，这里是取第二维度进行求和
            # 这里的shape变化，充分体现了一阶RippleSet的波动是求和了
            # [batch_size, dim]
            o_fun = tf.reduce_sum(self.t_fun_emb_list[hop] * probs_fun_expanded, axis=1)
            o_geo = tf.reduce_sum(self.t_geo_emb_list[hop] * probs_geo_expanded, axis=1)

            # 迭代，将上一阶的o变成下一阶的v
            # [batch_size, dim]
            self.item_fun_embeddings = self.update_item_embedding(self.item_fun_embeddings, o_fun, 'fun')
            self.item_geo_embeddings = self.update_item_embedding(self.item_geo_embeddings, o_geo, 'geo')

            # 把第i阶的RippleSet的o存上
            # o_list: [n_hop, batch_size, dim]
            o_fun_list.append(o_fun)
            o_geo_list.append(o_geo)

        return o_fun_list, o_geo_list

    def update_item_embedding(self, item_embeddings, o, kg_type):
        '''
        TODO：这部分是每个RippleSet的兴趣计算完毕后，要迭代地进行下一次。第一阶的输入是item，后面的就得是上一阶的o了。
        TODO：这里让你选后面RippleSet怎么迭代前一阶的RippleSet产生的o，可以直接替换，也可以叠加，或者再上一个转移矩阵，这个转移矩阵是可以被学出来的。
        Note: 论文中用的是第4种，最复杂的。
        :param item_embeddings:
        :param o:
        :return:
        '''
        if kg_type == 'fun':
            item_embeddings = tf.matmul(item_embeddings + o, self.transform_matrix_fun)
        if kg_type == 'geo':
            item_embeddings = tf.matmul(item_embeddings + o, self.transform_matrix_geo)

        return item_embeddings

    def predict(self, item_fun_embeddings, item_geo_embeddings, o_fun_list, o_geo_list):
        '''
        TODO: 模型的输出层，y^=sigmoid(u*v)
        Note: sigmoid这个操作在_build_model()中进行的
        TODO: 预测用户对item的评分，在_build_model()中被调用
        :param item_embeddings: [batch size, dim]
        :param o_list:
        :return scores: 一个batch内的所有item的得分情况，没有过sigmoid，过了sigmoid就是点击率了
        '''
        # 用户的embedding表达，各阶o之和
        # u:[batch size, dim]
        u_fun = o_fun_list[-1]
        u_geo = o_geo_list[-1]
        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                u_fun += o_fun_list[i]
                u_geo += o_geo_list[i]

        # 分开算的score作为特征融合的注意力权重
        # [batch_size]
        alpha_fun_logits = tf.reduce_sum(item_fun_embeddings * u_fun, axis=1)
        alpha_geo_logits = tf.reduce_sum(item_geo_embeddings * u_geo, axis=1)

        # [batch_size, 2]
        alpha_logits = tf.stack([alpha_fun_logits, alpha_geo_logits], axis=1)
        alpha_logits /= np.sqrt(self.dim)
        alpha = tf.nn.softmax(alpha_logits, axis=1)

        # [batch_size]
        alpha_fun, alpha_geo = tf.split(alpha, num_or_size_splits=2, axis=1)
        # for show attention
        self.alpha_fun = alpha_fun
        self.alpha_geo = alpha_geo

        # 特征进行拼接操作
        # [batch size, dim]
        u = alpha_fun * u_fun + alpha_geo * u_geo

        # 环境决定论，调节方式按元素乘
        # # [batch_size]
        # beta = tf.reduce_sum(self.resi_embeddings * u, axis=1)
        # beta = tf.sigmoid(beta, name='normalized_weight')
        # beta = tf.expand_dims(beta, axis=1)  # 按元素乘要求列数为1或一致，所以需要进行转置，即加上一列
        # [batch size, dim]
        # u = beta * self.resi_embeddings + (1 - beta) * u
        u = self._residence_determinism(u)

        # 用户对一个batch内的item的打分情况
        # note: 这个打分是未归一化的，在_build_model()里面
        # item_embeddings是[batch size, dim]，y也是[batch size, dim]
        v = item_fun_embeddings + item_geo_embeddings

        # 这个*是向量的点积运算
        # [batch_size]
        scores = tf.reduce_sum(v * u, axis=1)

        return scores

    def _residence_determinism(self, u):
        '''
        TODO: 刻画环境决定论
        :param u:
        :return:
        '''
        # [batch_size]
        beta = tf.reduce_sum(self.resi_embeddings * u, axis=1)  # 内积
        beta = tf.sigmoid(beta, name='normalized_weight')  # normalized

        # to show attention
        self.beta = beta

        # [batch_size, 1]
        beta = tf.expand_dims(beta, axis=1)
        # [batch_size, dim]
        context = beta * self.resi_embeddings

        # [batch_size, dim]
        context_u = tf.expand_dims(tf.concat([context, u], axis=1), axis=-1)
        u_logits = tf.matmul(self.w_context, context_u)
        u = tf.squeeze(tf.nn.tanh(u_logits), axis=-1)

        return u

    def _build_loss(self):
        '''
        TODO: 建立损失函数
        :return:
        '''
        # tf.reduce_mean(input_tensor, axis)根据给出的axis在input_tensor上求平均值。,
        # 如果axis没有条目, 则减少所有维度, 并返回具有单个元素的张量。
        # tf.nn.sigmoid_cross_entropy_with_logits()对于给定的logits计算sigmoid的交叉熵。
        # 推荐系统的损失项
        # sigmoid_cross_entropy_with_logits中的logits参数是未标准化的predictions
        base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.scores),
                                   name="base_loss")

        # KGE的损失项，loss=Rh-t
        kge_fun_loss = 0
        kge_geo_loss = 0
        for hop in range(self.n_hop):
            h_fun_expanded = tf.expand_dims(self.h_fun_emb_list[hop], axis=2)
            h_geo_expanded = tf.expand_dims(self.h_geo_emb_list[hop], axis=2)
            t_fun_expanded = tf.expand_dims(self.t_fun_emb_list[hop], axis=3)
            t_geo_expanded = tf.expand_dims(self.t_geo_emb_list[hop], axis=3)
            hRt_fun = tf.squeeze(tf.matmul(tf.matmul(h_fun_expanded, self.r_fun_emb_list[hop]), t_fun_expanded))
            hRt_geo = tf.squeeze(tf.matmul(tf.matmul(h_geo_expanded, self.r_geo_emb_list[hop]), t_geo_expanded))
            # note：这里也暴露了没有负采样，只有正例，所以损失只有正例的，正例的hRt越接近于1越好，
            # note：所以到loss中就取了一个负号。
            kge_fun_loss += tf.reduce_mean(tf.sigmoid(hRt_fun))
            kge_geo_loss += tf.reduce_mean(tf.sigmoid(hRt_geo))

        kge_loss = -self.kge_weight * (kge_fun_loss + kge_geo_loss)  # KGE任务有一个调整权重，避免反客为主

        # 正则化项，L2范数 sum(tensor ** 2) / 2
        # 计算的是每一个元素的平方之后相加最后除以2
        l2_fun_loss = 0
        l2_geo_loss = 0
        for hop in range(self.n_hop):
            l2_fun_loss += tf.reduce_mean(tf.reduce_sum(self.h_fun_emb_list[hop] * self.h_fun_emb_list[hop]))
            l2_fun_loss += tf.reduce_mean(tf.reduce_sum(self.t_fun_emb_list[hop] * self.t_fun_emb_list[hop]))
            l2_fun_loss += tf.reduce_mean(tf.reduce_sum(self.r_fun_emb_list[hop] * self.r_fun_emb_list[hop]))
            l2_geo_loss += tf.reduce_mean(tf.reduce_sum(self.h_geo_emb_list[hop] * self.h_geo_emb_list[hop]))
            l2_geo_loss += tf.reduce_mean(tf.reduce_sum(self.t_geo_emb_list[hop] * self.t_geo_emb_list[hop]))
            l2_geo_loss += tf.reduce_mean(tf.reduce_sum(self.r_geo_emb_list[hop] * self.r_geo_emb_list[hop]))

        l2_fun_loss += tf.nn.l2_loss(self.transform_matrix_fun)
        l2_geo_loss += tf.nn.l2_loss(self.transform_matrix_geo)

        # l2_resi_loss = tf.nn.l2_loss(self.resi_embeddings)
        # l2_attention_loss = tf.nn.l2_loss(self.w_context)

        l2_loss = self.l2_weight * (l2_fun_loss + l2_geo_loss)  # 调节正则化项的影响，避免影响训练目标

        # all_loss = RS_loss + KGE_loss + l2_regularization
        self.loss = base_loss + kge_loss + l2_loss

    def _build_train(self):
        '''
        TODO: 选择优化器
        :return:
        '''
        # 此函数是Adam优化算法：是一个寻找全局最优点的优化算法，引入了二次方梯度校正。
        # 相比于基础SGD算法，1.不容易陷于局部优点。2.速度更快
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        '''
        TODO: 训练，供train.py的train()调用
        :param sess: train.py里面建的Session
        :param feed_dict: user-item interaction和RippleSet
        :return sess:
        '''
        # sess = tf.Session()，是TensorFlow运算模型–会话，
        # 拥有并管理TensorFlow程序运行时的所有资源

        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        '''
        TODO: 供train.py的train()和evaluation()调用
        TODO: 评价模型的效果，两个指标AUC和Accuracy
        :param sess:
        :param feed_dict:
        :return auc, acc: 俩数，都是浮点型
        '''
        # scores的含义实际上是点击率预测（CTR），论文中说的
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)

        auc = roc_auc_score(y_true=labels, y_score=scores)  # 报错代码行
        predictions = [1 if i >= 0.5 else 0 for i in scores]  # 将点击率，以0.5作阈值，变为1、0预测标签
        acc = np.mean(np.equal(predictions, labels))

        return auc, acc

    def show_attention(self, sess, feed_dict):
        '''
        TODO: 输出5个attention权重给train.py
        :param sess:
        :param feed_dict:
        :return:
        '''
        # [n_hop, batch_size, n_memory]
        probs_fun_tensor, probs_geo_tensor = sess.run([self.probs_fun_list, self.probs_geo_list], feed_dict)
        # [batch_size, n_hop, n_memory]
        probs_fun_tensor = np.swapaxes(probs_fun_tensor, axis1=0, axis2=1)
        probs_geo_tensor = np.swapaxes(probs_geo_tensor, axis1=0, axis2=1)

        # [batch_size]
        alpha_fun, alpha_geo, beta = sess.run([self.alpha_fun, self.alpha_geo, self.beta], feed_dict)

        return probs_fun_tensor, probs_geo_tensor, alpha_fun, alpha_geo, beta