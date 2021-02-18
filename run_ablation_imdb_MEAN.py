import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import metapath as mp
import sklearn.metrics as sm
import time
from multiprocessing.dummy import Pool as ThreadPool
from model import GraphMSE
from model import Discriminator
from data import HeteData


def node_search_wrapper(index):
    return mp.search_all_path(graph_list, index, metapath_name, metapath_list, data.get_metapath_name(),
                              single_path_limit)


def train(model, epochs, method="all_node", ablation="all"):
    # 提前加载训练集和验证集到内存，节约时间。

    def index_to_feature_wrapper(dict):
        return mp.index_to_features(dict, data.x, method)

    start_select = 50
    train_index = train_list[:, 0].tolist()
    print("Loading dataset with thread pool...")
    train_metapath = pool.map(node_search_wrapper, train_index)
    train_features = pool.map(index_to_feature_wrapper, train_metapath)
    val_index = val_list[:, 0].tolist()
    val_label = val_list[:, 1]
    val_metapath = pool.map(node_search_wrapper, val_index)
    val_features = pool.map(index_to_feature_wrapper, val_metapath)
    lr = learning_rate
    model.train()  # 训练模式
    best_micro_f1 = 0
    best_macro_f1 = 0

    type_set = set()
    metapath_set = {}
    for node in val_metapath:
        for key in node:
            type_set.add(key[0])
    for type in type_set:
        metapath_set[type] = set()
    for node in val_metapath:
        for key in node:
            if len(node) - 1 > len(metapath_set[key[0]]):
                if len(key) > 1:
                    metapath_set[key[0]].add(key)

    metapath_label = {}
    metapath_onehot = {}
    discriminator = {}
    d_optimizer = {}
    label = {}
    for type in type_set:
        metapath_label[type] = {}
        metapath_onehot[type] = {}
        label[type] = []
        for i, metapath in enumerate(metapath_set[type]):
            metapath_label[type][metapath] = torch.zeros(batch_size, data.type_num, device=DEVICE)
            metapath_onehot[type][metapath] = torch.zeros(batch_size, device=DEVICE).long()
            metapath_onehot[type][metapath][:] = i
            for all_type in data.node_dict:
                if all_type in metapath[1:]:
                    metapath_label[type][metapath][:, data.node_dict[all_type]] = 1
            label[type].append(metapath_label[type][metapath])
        label[type] = torch.cat(label[type], dim=0)
        discriminator[type] = Discriminator(info_section, data.type_num).to(DEVICE)
        d_optimizer[type] = optim.Adam(discriminator[type].parameters(), lr=0.01, weight_decay=5e-4)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    select_flag = False
    time1 = time.time()
    for e in range(epochs):
        # if single_path_limit is not None and (e + 1) % 20 == 0:
        #     print("Re-sampling...")
        #     train_metapath = pool.map(node_search_wrapper, train_index)
        #     train_features = pool.map(index_to_feature_wrapper, train_metapath)
        for batch in range(num_batch_per_epoch):
            batch_src_choice = np.random.choice(range(train_list.shape[0]), size=(batch_size,), replace=False)
            batch_src_index = train_list[batch_src_choice, 0]
            batch_src_label = train_list[batch_src_choice, 1]
            batch_feature_list = [train_features[i] for i in batch_src_choice]
            batch_train_feature_dict, batch_src_index_dict, batch_src_label_dict, batch_train_rows_dict = mp.combine_features_dict(
                batch_feature_list,
                batch_src_index,
                batch_src_label, DEVICE)

            optimizer.zero_grad()
            for type in d_optimizer:
                d_optimizer[type].zero_grad()
            if e >= start_select:
                batch_train_logits_dict, GAN_input = model(batch_train_rows_dict, batch_train_feature_dict, True)
            else:
                batch_train_logits_dict, GAN_input = model(batch_train_rows_dict, batch_train_feature_dict,
                                                           False)  # 获取模型的输出

            for type in batch_train_logits_dict:
                Loss_Classification = criterion(batch_train_logits_dict[type], batch_src_label_dict[type])
                assert ablation in ["all", "no_align"]
                if ablation == "all":
                    Pred_D = discriminator[type](GAN_input[type])
                    Pred_Shuffle = discriminator[type](GAN_input[type], True)
                    Sorted_Pred_D = []
                    Sorted_Pred_Shuffle = []

                    for metapath in metapath_set[type]:
                        Sorted_Pred_D.append(Pred_D[metapath])
                        Sorted_Pred_Shuffle.append(Pred_Shuffle[metapath])

                    Sorted_Pred_D = torch.cat(Sorted_Pred_D, dim=0)
                    Sorted_Pred_Shuffle = torch.cat(Sorted_Pred_Shuffle, dim=0)

                    Loss_D = nn.BCELoss()(Sorted_Pred_D, label[type])
                    Loss_D_Shuffle = nn.BCELoss()(Sorted_Pred_Shuffle,
                                                  torch.zeros_like(Sorted_Pred_Shuffle, device=DEVICE))

                    Loss = Loss_Classification + Loss_D + Loss_D_Shuffle

                else:
                    Loss = Loss_Classification

                Loss.backward()
                d_optimizer[type].step()
            optimizer.step()

        if e >= start_select and select_flag == False:
            select_flag = True
            pretrain_convergence = time2 - time1
            print("Start select! Best f1-score reset to 0.")
            print("Pretrain convergence time:", pretrain_convergence)
            time1 = time.time()
            best_micro_f1 = 0
            best_macro_f1 = 0

        if select_flag:
            micro_f1, macro_f1 = val(model, val_features, val_index, val_label, True)
            model.show_metapath_importance()
        else:
            micro_f1, macro_f1 = val(model, val_features, val_index, val_label)
        if micro_f1 >= best_micro_f1:
            if micro_f1 > best_micro_f1:
                time2 = time.time()
                best_micro_f1 = micro_f1
                best_macro_f1 = macro_f1
                if select_flag:
                    torch.save(model.state_dict(), "checkpoint/" + dataset + "_best_val")
            elif macro_f1 > best_macro_f1:
                best_micro_f1 = micro_f1
                best_macro_f1 = macro_f1
                if select_flag:
                    torch.save(model.state_dict(), "checkpoint/" + dataset + "_best_val")
        select_convergence = time2 - time1
        print("Epoch ", e, ",Val Micro_f1 is ", micro_f1, ", Macro_f1 is ", macro_f1, ", the best micro is ",
              best_micro_f1, ", the best macro is ",
              best_macro_f1)
    torch.save(model.state_dict(), "checkpoint/" + dataset + "_final")


def val(model, val_features, val_index, val_label, start_select=False):
    val_feature_dict, val_index_dict, val_label_dict, val_rows_dict = mp.combine_features_dict(val_features,
                                                                                               val_index, val_label,
                                                                                               DEVICE)
    model.eval()  # 测试模型
    with torch.no_grad():  # 关闭无用的梯度计算-防止显存爆炸
        val_logits_dict, _ = model(val_rows_dict, val_feature_dict, start_select)
        # 若要测试每类节点上的f1值，则不能把它们拼在一起
        # 我们的数据集只有一类点，拼接与否效果一样
        y_pred = []
        y_true = []
        for type in val_logits_dict:
            y_pred.extend(val_logits_dict[type].max(1)[1].cpu().numpy().tolist())  # 预测标签：对预测结果按行取argmax
            y_true.extend(val_label_dict[type].cpu().numpy().tolist())  # 计算在测试节点/数据上的准确率
        micro_f1 = sm.f1_score(y_true, y_pred, average='micro')
        macro_f1 = sm.f1_score(y_true, y_pred, average='macro')
        return micro_f1, macro_f1


def test(model, batch_size=200, test_method="best_val"):
    model.load_state_dict(torch.load("checkpoint/" + dataset + "_" + test_method))

    def index_to_feature_wrapper(dict):
        return mp.index_to_features(dict, data.x)

    test_index = test_list[:, 0].tolist()
    print("Loading dataset with thread pool...")
    time1 = time.time()
    test_metapath = pool.map(node_search_wrapper, test_index)
    test_features = pool.map(index_to_feature_wrapper, test_metapath)
    time2 = time.time()
    print("Dataset Loaded. Time consumption:", time2 - time1)

    # 若要测试每类节点上的f1值，则不能把它们拼在一起
    # 我们的数据集只有一类点，拼接与否效果一样
    y_pred = []
    y_true = []
    model.eval()  # 测试模型
    with torch.no_grad():
        batch = 0
        while batch < len(test_index):
            end = batch + batch_size if batch + batch_size <= len(test_index) else len(test_index)
            batch_test_index = test_list[batch:end, 0]
            batch_test_label = test_list[batch:end, 1]
            batch_feature_list = [test_features[i] for i in range(batch, end)]

            batch_test_feature_dict, batch_test_index_dict, batch_test_label_dict, batch_test_rows_dict = mp.combine_features_dict(
                batch_feature_list,
                batch_test_index,
                batch_test_label, DEVICE)
            batch += batch_size
            batch_test_logits_dict, _ = model(batch_test_rows_dict, batch_test_feature_dict, True)
            for type in batch_test_logits_dict:
                y_pred.extend(batch_test_logits_dict[type].max(1)[1].cpu().numpy().tolist())
                y_true.extend(batch_test_label_dict[type].cpu().numpy().tolist())
        micro_f1 = sm.f1_score(y_true, y_pred, average='micro')
        macro_f1 = sm.f1_score(y_true, y_pred, average='macro')
        print("Final F1 @ " + test_method + ":")
        print("Micro_F1:\t", micro_f1, "\tMacro_F1:", macro_f1)
        model.show_metapath_importance()
        if shuffle == True:
            feature_mode = "shuffle_"
        else:
            feature_mode = ""
        with open("result/" + dataset + "_" + str(
                train_percent) + "_" + test_method + "_" + feature_mode + ablation + "_MEAN.txt",
                  "a") as f:
            f.write("Micro_F1:" + str(micro_f1) + "\tMacro_F1:" + str(macro_f1) + "\n")
        return micro_f1


if __name__ == '__main__':
    # important hyperparameters
    dataset = "IMDB"
    train_percent = 20  # 20, 40, 60, 80
    ablation = "all"
    shuffle = False
    metapath_length = 4
    mlp_settings = {'layer_list': [256], 'dropout_list': [0.5], 'activation': 'sigmoid'}
    info_section = 40  # total embedding dim = info_section *3 = 120
    learning_rate = 0.01  #
    select_method = "all_node"  # Only used in end-node study
    single_path_limit = 5  # lambda = 5

    # Automatically calculated parameters
    num_batch_per_epoch = 5  # 每个epoch循环的批次数
    batch_size = train_percent // 20 * 96
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    data = HeteData(dataset=dataset, train_percent=train_percent, shuffle=shuffle)
    graph_list = data.get_dict_of_list()
    homo_graph = nx.to_dict_of_lists(data.homo_graph)
    input_dim = data.x.shape[1]

    pre_embed_dim = data.type_num * info_section
    output_dim = max(data.train_list[:, 1].tolist()) + 1  # 隐藏单元节点数  两层

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 获取预处理数据

    assert select_method in ["end_node", "all_node"]

    x = data.x
    train_list = data.train_list  # 训练节点/数据对应的标签
    test_list = data.test_list  # 测试节点/数据对应的索引
    val_list = data.val_list  # 验证节点/数据对应的索引
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    num_thread = 12
    pool = ThreadPool(num_thread)

    # 此处枚举metapath并初始化选择模型
    metapath_name = mp.enum_metapath_name(data.get_metapath_name(), data.get_metapath_dict(), metapath_length)
    metapath_list = mp.enum_longest_metapath_index(data.get_metapath_name(), data.get_metapath_dict(), metapath_length)

    select_model = GraphMSE(metapath_list=metapath_name, input_dim=input_dim, pre_embed_dim=pre_embed_dim,
                            select_dim=output_dim, mlp_settings=mlp_settings, mean_test=True).to(DEVICE)
    train(model=select_model, epochs=100, method="all_node", ablation=ablation)
    test(model=select_model, test_method="best_val")
