import argparse  # argparse是python标准库里面用来处理命令行参数的库
import numpy as np
from data_loader_3 import load_data
from train_distangle_3 import train
import time


np.random.seed(555)

parser = argparse.ArgumentParser(description="超参接口")  # 创建一个解析对象

# 向parser对象中添加命令行参数和选项
'''
name or flags...    - 必选，指定参数的形式，一般写两个，一个短参数，一个长参数
type    - 指定参数类型
help    - 可以写帮助信息
'''

parser.add_argument('--dim', type=int, default=16, help='dimension of entity and relation embeddings')
parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')
parser.add_argument('--kge_weight', type=float, default=0.01, help='weight of the KGE term')
parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--n_epoch', type=int, default=10, help='the number of epochs')
parser.add_argument('--n_memory', type=int, default=32, help='size of ripple set for each hop')
parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                    help='how to update item at the end of each hop')
parser.add_argument('--using_all_hops', type=bool, default=True,
                    help='whether using outputs of all hops or just the last hop when making prediction')
parser.add_argument('--reusing_ripple_set', type=bool, default=False,
                    help='whether reusing the ripple set in last execution or '
                         'generating a new one according to args when getting ripple set')
# parser.add_argument('--n_seed', type=int, default=16, help='size of seed for 1st and 3st hop ripple set')
# parser.add_argument('--min_hist', type=int, default=32, help='minimum threshold of historical set size '
#                                                              'that can support the prediction')

args = parser.parse_args()  # 把所设的参数解析出来

show_loss = False  # 训练过程不显式当前的损失

# grid search调参
args.n_epoch = 60
args.batch_size = 1024
args.n_hop = 3
args.n_memory = 16
args.dim = 64   # 如设为64，则GPU内存溢出
args.kge_weight = 0.001
args.lr = 1e-7
args.l2_weight = 1e-11

print("\n超参设置:\n{}\n".format(args))

start_time = time.time()

# 调用data_loader.py
# load_data()返回多个值，这里用一个data_info接住，这个data_info是个tuple
# tuple：(train_data, eval_data, test_data, n_entity, n_relation, ripple_set)
# 因为每次load_data时间非常长，不在每次测试都重新生成RippleSet
data_info = load_data(args)

end_time = time.time()
print("\ntime to load data:\t{} s\n".format(end_time - start_time))

# 调用train.py
# 所有的网络调用和数据输入输出都是在train.py的train()里搞的
# 包括模型测试验证结果的显示，全部在train()里
train(args, data_info, show_loss)

print("-" * 150)
