import pickle as pkl
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import math


def rating_city_revise():
    ratings = pd.read_csv('data2.3/rating.csv', sep='\t',
                          dtype={'uid': str, 'poi_id': str}).values

    spot_city_dict = pkl.load(open('data2.1/spot_city_dict_pro_final.pkl', 'rb'))

    wrong_count = 0
    data = []
    for record in tqdm(ratings):
        user = record[0]
        poi_id = record[1]
        rating = record[2]
        destination = record[3]
        residence = record[4]
        year = record[6]
        month = record[7]

        dest_revise = spot_city_dict[poi_id]['city']

        if destination != dest_revise:
            wrong_count += 1

        if pd.isna(residence):
            if user == '80392157':
                residence = '喀什地区'
            if user == '80597241':
                residence = '阿勒泰地区'

        data.append([user, poi_id, rating, dest_revise, residence, year, month])

    print('destination wrong counting', wrong_count)

    df_output = pd.DataFrame(data=data, columns=['user', 'poi_id', 'rating', 'destination', 'residence', 'year', 'month'])
    df_output.to_csv('data3/ratings_2.4.csv', index=False, sep='\t')


def convert_rating():
    '''
    TODO: 将评分转成1和0，0的样本是负采样出来的，正负例平衡
    :return:
    '''
    df_rating = pd.read_csv("data3/ratings_2.4.csv", sep="\t", encoding="utf-8",
                            dtype={"user": str, "destination": str, "poi_id": str, "residence": str})

    user_pos_dict = dict()  # key是uid，value是所有去过的item的set
    user_pos_ratings = dict()  # key是uid，value是{destination: set(pos_item)}

    # Note: ，用于给目的城市打编码，用这个字典根据目的城市名称查其编码
    destination_index = 0
    destination_index_dict = dict()

    # Note: 用这个字典根据item，查其所属的城市的编码
    spot_destination_dict = dict()

    # user从0重新开始编码
    uid_index = 0
    uid_index_dict = dict()

    # 将item的id与在KG中eid进行对齐
    entity_id_dict = pkl.load(open("data2.3/kg_lookup/entity_id_dict_dis.pkl", "rb"))

    # poi_index与匹配的entity同id
    poi_index_dict = dict()

    # 检查是否有item没有匹配进kg中
    count_poi_not_in_kg = 0

    # 将residence从0进行一次重新编码
    residence_index = 0
    residence_id_dict = dict()
    user_resi_dict = defaultdict()

    print("rehashing uid, poi_id, destination, rating...")

    spot_multi_d_dict = dict()
    for index, row in tqdm(df_rating.iterrows()):  # 跳过表头
        # row是用户景点访问记录中的各行数据

        # 用户的旧索引
        uid_old = row["user"]
        if uid_old not in uid_index_dict:
            uid_index_dict[uid_old] = uid_index
            uid_index += 1

        # 居住地
        residence = row["residence"]
        if residence not in residence_id_dict:
            residence_id_dict[residence] = residence_index
            residence_index += 1

        user_resi_dict[uid_old] = residence

        # Note: 景点还在第2个位置
        poi_id_old = row["poi_id"]  # item_index_old老索引就是景点名称

        # item_index是景点的新索引
        if poi_id_old in entity_id_dict:
            poi_index_dict[poi_id_old] = entity_id_dict[poi_id_old][0]
        else:
            count_poi_not_in_kg += 1

        # Note: 把景点所对应的目的城市获得到
        destination = row["destination"]
        # 给目的城市打上编码
        if destination not in destination_index_dict.keys():
            destination_index_dict[destination] = destination_index
            destination_index += 1

        # spot_destination_dict：景点对应的目的城市编码
        if poi_id_old not in spot_destination_dict:
            spot_destination_dict[poi_id_old] = destination  # 用来根据景点的新索引查找目的城市

        if spot_destination_dict[poi_id_old] != destination:
            if poi_id_old not in spot_multi_d_dict:
                spot_multi_d_dict[poi_id_old] = dict()
            if destination not in spot_multi_d_dict[poi_id_old]:
                spot_multi_d_dict[poi_id_old][destination] = 0
            spot_multi_d_dict[poi_id_old][destination] += 1

    # 修正景点目的地查询字典
    spot_dict = pkl.load(open("data2.3/spot_label_dict.pkl", "rb"))
    city_looking_dict = pkl.load(open("data2.1/city_looking_dict_pro_final.pkl", 'rb'))
    for poi_id_old in spot_multi_d_dict:
        city_raw = spot_dict[int(poi_id_old)]['city']
        if city_raw in city_looking_dict:
            city = city_looking_dict[city_raw]['city']
            if city in spot_multi_d_dict[poi_id_old].keys():
                spot_destination_dict[poi_id_old] = city
        else:
            max_city = max(zip(spot_multi_d_dict[poi_id_old].values(), spot_multi_d_dict[poi_id_old].keys()))
            spot_destination_dict[poi_id_old] = max_city[1]

    for index, row in tqdm(df_rating.iterrows()):  # 跳过表头
        # Note: 打分变到第4个位置上去了
        uid_old = row["user"]
        poi_id_old = row["poi_id"]
        rating = row["rating"]
        destination = spot_destination_dict[poi_id_old]
        if rating >= 0:  # 有分即正向
            if uid_old not in user_pos_ratings:
                user_pos_ratings[uid_old] = {}
            if destination not in user_pos_ratings[uid_old]:
                user_pos_ratings[uid_old][destination] = set()

            user_pos_ratings[uid_old][destination].add(poi_id_old)

            if uid_old not in user_pos_dict:
                user_pos_dict[uid_old] = set()
            user_pos_dict[uid_old].add(poi_id_old)

    print("check poi_not_in_kg:\t", count_poi_not_in_kg)
    print("spot_destination_dict size:\t", len(spot_destination_dict))

    item_total = set()
    item_destination = dict()
    for poi_id_old in spot_destination_dict:
        destination = spot_destination_dict[poi_id_old]
        if destination not in item_destination:
            item_destination[destination] = set()
        item_destination[destination].add(poi_id_old)
        item_total.add(poi_id_old)

    # with open("data2.1/spot_multi_d.pkl", 'wb') as dumper:
    #     pkl.dump(spot_multi_d_dict, dumper, pkl.HIGHEST_PROTOCOL)
    # with open("data2.1/spot_destination_dict.pkl", 'wb') as dumper:
    #     pkl.dump(spot_destination_dict, dumper, pkl.HIGHEST_PROTOCOL)
    # with open("data2.1/item_destination.pkl", 'wb') as dumper:
    #     pkl.dump(item_destination, dumper, pkl.HIGHEST_PROTOCOL)
    with open("data3/lookup/destination_index_dict.pkl", 'wb') as dumper:
        pkl.dump(destination_index_dict, dumper, pkl.HIGHEST_PROTOCOL)
    with open('data3/lookup/residence_index_dict.pkl', 'wb') as output:
        pkl.dump(residence_id_dict, output, pkl.HIGHEST_PROTOCOL)

    print('converting rating file...')

    data_output = []

    # 输出1,0后的interaction
    for user_index_old, pos_item_set in tqdm(user_pos_ratings.items()):
        # items()返回一个包含字典中所有键值项的PyListObject。
        # 这样操作的目的是遍历user_pos_ratings字典
        # TODO: 将user也重新编码，从0开始

        user_index = uid_index_dict[user_index_old]
        residence_id = residence_id_dict[user_resi_dict[user_index_old]]

        for destination in pos_item_set:
            for poi_id_old in pos_item_set[destination]:
                item_index = poi_index_dict[poi_id_old]
                destination = spot_destination_dict[poi_id_old]
                # 输出正向反馈的景点
                data_output.append([
                    user_index, destination_index_dict[destination], item_index, 1, residence_id])  # 输出user的新id，图书新id，和1

            # unwatched_set未到过的景点
            # destination_index = destination_index_dict[destination]
            unwatched_set = item_destination[destination] - pos_item_set[destination]  # 将某用户正例的图书部分去除

            negative_cases = None
            if len(unwatched_set) < len(pos_item_set[destination]):
                # 这里要注意，如果一个城市的景点比较少，这老哥要是都踩过了，负例就不够了
                # 不要强行增加，其他城市的景点作负例，不然会导致该有RippleSet的用户没有了
                # 正负例不均衡就不均衡吧
                negative_cases = unwatched_set
            else:
                negative_cases = np.random.choice(list(unwatched_set), size=len(pos_item_set[destination]),
                                                  replace=False)

            # TODO：负采样，在用户未到过的景点中采样，采样大小与正例一致
            # 对于每个用户正负例数都是平衡的。
            for poi_id_old in negative_cases:
                # Note: 由于是负采样出来的，去的时间是未知的，所以随机一个月份即可
                item_index = poi_index_dict[poi_id_old]
                destination = spot_destination_dict[poi_id_old]
                data_output.append([
                    user_index, destination_index_dict[destination], item_index, 0, residence_id])  # 输出user的新id，图书新id，和0

    print('number of users: %d' % len(uid_index_dict))
    print('number of items: %d' % len(item_total))
    print('number of cities: %d' % len(destination_index_dict.keys()))
    print('number of interactions after negative sampling: %d' % len(data_output))

    with open('data3/lookup/uid_index_dict.pkl', 'wb') as output:
        pkl.dump(uid_index_dict, output, pkl.HIGHEST_PROTOCOL)

    df_output = pd.DataFrame(data=data_output, columns=["user_index", "destination_index", "spot_index", "interaction", "residence_index"])
    df_output.to_csv("data3/rating_rehashed.csv", index=False, sep="\t", encoding="utf-8")


def dataset_split():
    '''
    TODO: 将数据集一分为3，train:evaluation:test = 6:2:2
    Note: 将同属一个目的地景点完整切出来
    :param rating_np: user-spot打分记录的二维 narray
    :return: train_data，训练集
    :return: eval_data，验证集（调超参）
    :return: test_data，测试集
    :return: user_history_dict，用户历史阅览过哪些书，key是用户，value是阅览过的书集合;
            这个字典是用于产生用户的RippleSet，不会作为最终的返回。
    '''
    print('reading rating file...')
    df = pd.read_csv('data3/rating_rehashed.csv', sep="\t", encoding="utf-8", dtype=np.int32)
    rating_np = df.values

    print('splitting dataset...')
    # 数据集的比例，train:evaluation:test = 0.6: 0.2: 0.2
    eval_ratio = 0.2
    test_ratio = 0.2

    # shape属性是第一维度中元素的个数（即二维列表的行数），这里指的是num_interation = 139,746
    # n_ratings = num_interation = 139,746
    n_ratings = rating_np.shape[0]
    print('total dataset size:', n_ratings)

    # 对rating_np进行打包处理，同一个目的地景点打成一个包，一会直接分这个包
    # 一个大包中的interaction都是同一个用户的，同用户的大包中的小包，景点都是同目的城市的
    # rating_np_bag_dict中的key是user，value是一个用户的所有footprint，
    # 每个footprint也是个字典，key是目的城市，value是interaction的标准型(user, spot, rate)
    # rating_np_bag_dict：
    # 数据结构{用户:{目的地: [interaction]}}
    # {
    #   0:{1: [[0, 4250,  1], [0, 4268, 1]],
    #      0: [[0, 2319,  0]]}
    #   1:{1: [[1, 490,   1]],
    #      2: [[2, 5443,  0], [2, 5789, 1]]}
    #   2:{2: [[2, 12960, 1]],
    #      1: [[2, 5    , 1]]}
    # }
    rating_bag_dict = dict()

    for rating_record in rating_np:
        user = int(rating_record[0])
        destination = int(rating_record[1])
        spot = int(rating_record[2])
        feedback = int(rating_record[3])
        residence = int(rating_record[4])

        if user not in rating_bag_dict:
            rating_bag_dict[user] = dict()
        if destination not in rating_bag_dict[user]:
            rating_bag_dict[user][destination] = list()
        rating_bag_dict[user][destination].append([user, spot, feedback, residence])

    print('num_users_total', len(rating_bag_dict))
    users_can_be_test = set()
    for user in rating_bag_dict.keys():
        if len(rating_bag_dict[user].values()) > 3:  # 将比较稠密的用户选出来做测试和验证
            users_can_be_test.add(user)
    users_size_can_test = len(users_can_be_test)
    print("num users that can be tested:", users_size_can_test)

    # Note: 训练集中有所有的用户，从训练集的用户中选了小部分去验证和测试，
    # Note: 用于验证和测试的用户，也会出现在训练集中，只是他们的有部分记录用于了测试和验证。
    # Note: 这里*2是因为两层抽取之后，验证集和测试集太小了
    users_for_eval = np.random.choice(list(users_can_be_test), math.ceil(users_size_can_test * eval_ratio * 2),
                                      replace=False)
    left = users_can_be_test - set(users_for_eval)
    users_for_test = np.random.choice(list(left), math.ceil(users_size_can_test * test_ratio * 2), replace=False)

    print("n_user_eval", len(users_for_eval))
    print("n_user_test", len(users_for_test))

    # 构造训练集，验证集，测试集
    train_bags = list()
    eval_bags = list()
    test_bags = list()

    for user in rating_bag_dict.keys():
        # ratings_user = np.array(rating_user_dict[user])
        bags = list(rating_bag_dict[user].items())
        # fp_dict = rating_np_bag_dict[user]  # 某个用户的footprint集合

        if user in users_for_eval:
            np.random.shuffle(bags)
            break_point = math.floor(len(bags) * eval_ratio)
            eval_bags += list(bags[:break_point])
            train_bags += list(bags[break_point:])
        elif user in users_for_test:
            np.random.shuffle(bags)
            break_point = math.floor(len(bags) * test_ratio)
            test_bags += list(bags[:break_point])
            train_bags += list(bags[break_point:])
        else:
            train_bags += bags

    # 将interactions从bag里面拆出来
    test_data = [interaction for bag in test_bags for interaction in bag[1]]
    eval_data = [interaction for bag in eval_bags for interaction in bag[1]]
    train_data = [interaction for bag in train_bags for interaction in bag[1]]

    # 用户去过的景点字典，用于之后产生RippleSet用的
    # Note：这个字典也是用于训练的，所以只有训练集部分。
    user_history_dict = dict()
    for interaction in train_data:
        user = interaction[0]
        spot = interaction[1]
        rating = interaction[2]

        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = list()
            user_history_dict[user].append(spot)

    print('num_user_train', len(user_history_dict))

    print("total_before_cleaning:\t", len(test_data) + len(eval_data) + len(train_data))

    # Note! Note!: 原文中之所以有一次用user_history_dict过滤train_data, eval_data和test_data
    # Note！: 是因为必须保证这个用户在训练集中是有正例的，即保证他有访问历史记录
    train_data = [interaction for interaction in train_data if interaction[0] in user_history_dict]
    eval_data = [interaction for interaction in eval_data if interaction[0] in user_history_dict]
    test_data = [interaction for interaction in test_data if interaction[0] in user_history_dict]

    print("total_after_eliminate_non_pos_train:\t", len(test_data) + len(eval_data) + len(train_data))

    # 把切好的数据集转从二维list转为二维array
    train_data = np.array(train_data)
    eval_data = np.array(eval_data)
    test_data = np.array(test_data)

    print("size of train_data:\t{} records".format(np.shape(train_data)[0]))
    print("size of validation_data:\t{} records".format(np.shape(eval_data)[0]))
    print("size of test_data:\t{} records".format(np.shape(test_data)[0]))

    print("users for train", np.unique(train_data[:, 0]).shape)
    print("users for eval", np.unique(eval_data[:, 0]).shape)
    print("users for test", np.unique(test_data[:, 0]).shape)

    with open("data3/user_history_dict.pkl", "wb") as writer:
        pkl.dump(user_history_dict, writer, protocol=3)

    df_train = pd.DataFrame(data=train_data, columns=["user_index", "poi_index", "feedback", "residence_index"])
    df_test = pd.DataFrame(data=test_data, columns=["user_index", "poi_index", "feedback", "residence_index"])
    df_eval = pd.DataFrame(data=eval_data, columns=["user_index", "poi_index", "feedback", "residence_index"])

    df_train.to_csv("data3/dataset/train.csv", index=False, sep="\t", encoding="utf-8")
    df_test.to_csv("data3/dataset/test.csv", index=False, sep="\t", encoding="utf-8")
    df_eval.to_csv("data3/dataset/eval.csv", index=False, sep="\t", encoding="utf-8")

    return train_data, eval_data, test_data, user_history_dict


# def create_residence_dict():
#     uid_dict = pkl.load(open('data2.4/uid_index_dict.pkl', 'rb'))
#
#     rating_df = pd.read_csv('data2.4/ratings_2.4.csv', sep='\t',
#                             dtype={'user': str})
#
#     user_resi_dict = defaultdict()
#     resi_user_dict = defaultdict(set)
#
#     resi_index_dict = dict()
#
#     for record in tqdm(rating_df.values):
#         user = record[0]
#         residence = record[4]
#         uid = uid_dict[user]
#         if pd.isna(residence):
#             print(record)
#         if residence not in resi_index_dict:
#             resi_index_dict[residence] = len(resi_index_dict)
#
#         resi_id = resi_index_dict[residence]
#         user_resi_dict[uid] = resi_id
#         resi_user_dict[resi_id].add(uid)
#
#     print(resi_index_dict)
#     print(len(resi_user_dict))
#     print(len(user_resi_dict))
#
#     pkl.dump(resi_index_dict, open('data2.4/residence/residence_index_dict.pkl', 'wb'))
#     pkl.dump(user_resi_dict, open('data2.4/residence/user_resi_dict.pkl', 'wb'))
#     pkl.dump(resi_user_dict, open('data2.4/residence/resi_user_dict.pkl', 'wb'))


def get_positive_data():
    train_data = pd.read_csv("data3/dataset/train.csv", sep='\t').values
    eval_data = pd.read_csv("data3/dataset/test.csv", sep='\t').values
    test_data = pd.read_csv("data3/dataset/eval.csv", sep='\t').values

    all_data = np.concatenate((train_data, eval_data, test_data), axis=0)

    pos_data = all_data[all_data[:, 2] == 1, :]

    df_output = pd.DataFrame(data=pos_data, columns=["user_index", "poi_index", "feedback", "residence_index"])
    df_output.to_csv("data3/dataset/pos_data.csv", sep="\t", index=False, encoding='utf-8')


if __name__ == '__main__':
    # rating_city_revise()

    # convert_rating()
    # dataset_split()

    get_positive_data()


