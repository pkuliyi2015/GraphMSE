import torch
import torch.nn as nn
import torch.nn.init as init
import random


class GraphMSE(nn.Module):
    def __init__(self, metapath_list, input_dim, pre_embed_dim, select_dim, mlp_settings, endnode_test=False, mean_test=False):
        super(GraphMSE, self).__init__()
        assert 'layer_list' in mlp_settings.keys()
        assert 'dropout_list' in mlp_settings.keys()
        assert 'activation' in mlp_settings.keys()
        assert mlp_settings['activation'] in ['sigmoid', 'relu', 'tanh']
        self.mean_test = mean_test
        self.metapath = metapath_list
        self.node_type = len(metapath_list)
        self.input_dim = input_dim
        self.pre_embed_dim = pre_embed_dim
        self.select_dim = select_dim
        self.metapath_mlp = nn.ModuleDict()
        self.mlp_settings = mlp_settings
        self.type_weight = nn.ParameterDict()
        self.type_trained = {}
        self.classify_layer = nn.Linear(pre_embed_dim, select_dim)
        self.weight = {}
        self.attn = nn.ModuleDict()
        self.metapath2index = {}

        for node_type in metapath_list:
            self.metapath_mlp[node_type] = nn.ModuleDict()
            self.type_weight[node_type] = nn.Parameter(torch.Tensor(input_dim, pre_embed_dim))
            init.kaiming_uniform_(self.type_weight[node_type])
            self.metapath2index[node_type] = {}
            metapath_index = 0
            for metapath in metapath_list[node_type]:
                self.metapath2index[node_type][metapath] = metapath_index
                metapath_index += 1

                if len(self.mlp_settings['layer_list']) == 0:
                    if endnode_test:
                        self.metapath_mlp[node_type][metapath] = nn.Sequential(
                            nn.Linear(input_dim, pre_embed_dim),
                        )
                    else:
                        self.metapath_mlp[node_type][metapath] = nn.Sequential(
                            nn.Linear((len(metapath) - 1) * input_dim, pre_embed_dim),
                        )
                else:
                    layer_dim = self.mlp_settings['layer_list']
                    if endnode_test:
                        self.metapath_mlp[node_type][metapath] = nn.Sequential(
                            nn.Linear(input_dim, layer_dim[0]), )
                    else:
                        self.metapath_mlp[node_type][metapath] = nn.Sequential(
                            nn.Linear((len(metapath) - 1) * input_dim, layer_dim[0]), )
                    self.metapath_mlp[node_type][metapath].add_module('1',
                                                                      self.mlp_activation(
                                                                          mlp_settings['activation']))
                    cur = 2
                    for i in range(len(layer_dim) - 1):
                        self.metapath_mlp[node_type][metapath].add_module(str(cur),
                                                                          nn.Linear(layer_dim[i], layer_dim[i + 1]))
                        cur += 1
                        if i < len(mlp_settings['dropout_list']):
                            self.metapath_mlp[node_type][metapath].add_module(str(cur), nn.Dropout(
                                mlp_settings['dropout_list'][i]))
                            cur += 1
                        self.metapath_mlp[node_type][metapath].add_module(str(cur), self.mlp_activation(
                            mlp_settings['activation']))
                        cur += 1
                    self.metapath_mlp[node_type][metapath].add_module(str(cur),
                                                                      nn.Linear(layer_dim[-1], pre_embed_dim))

            self.attn[node_type] = nn.Sequential(
                nn.Tanh(),
                nn.Linear(2 * pre_embed_dim, 2 * pre_embed_dim, bias=False),
                nn.Linear(2 * pre_embed_dim, 1, bias=False),
                nn.LeakyReLU(0.1),
            )

    def mlp_activation(self, type):
        if type == 'sigmoid':
            return nn.Sigmoid()
        if type == 'tanh':
            return nn.Tanh()
        if type == 'relu':
            return nn.ReLU()

    def metapath_aggregate(self, tensor):
        if self.mean_test:
            return tensor.mean(dim=0, keepdim=True)
        return tensor.sum(dim=0, keepdim=True)

    def forward(self, rows_dict, feature_dict, start_select=False):
        center_node = {}
        injective = {}
        pre_embed = {}
        weight = {}
        GAN_input = {}
        for type in feature_dict:
            self.type_trained[type] = True
            injective[type] = {}
            for metapath in feature_dict[type]:
                if len(metapath) == 1:
                    center_node[type] = feature_dict[type][metapath].mm(self.type_weight[type])
                    continue
                # 将相同种类metapth做映射，再相加
                x_dim = self.pre_embed_dim
                injective_mapping = self.metapath_mlp[type][metapath](feature_dict[type][metapath])
                features_per_node = list(injective_mapping.split(rows_dict[type][metapath]))
                for i in range(len(rows_dict[type][metapath])):
                    if rows_dict[type][metapath][i] == 0:
                        features_per_node[i] = torch.zeros([1, x_dim]).to(
                            "cuda" if next(self.parameters()).is_cuda else "cpu")
                injective[type][metapath] = torch.cat(list(map(self.metapath_aggregate, features_per_node)), dim=0)
            GAN_input[type] = injective[type]
            weight[type] = torch.empty(len(self.metapath2index[type])).to(
                "cuda" if next(self.parameters()).is_cuda else "cpu")
            for metapath in injective[type]:
                if start_select:
                    weight[type][self.metapath2index[type][metapath]] = self.attn[type](torch.cat(
                        [center_node[type], injective[type][metapath]], dim=1)).mean()

            if start_select:
                weight[type] = weight[type].softmax(dim=0)
                for metapath in injective[type]:
                    injective[type][metapath] *= weight[type][self.metapath2index[type][metapath]]
                self.weight[type] = weight[type]

            embed = torch.stack(list(injective[type].values()), dim=2).mean(dim=2)
            pre_embed[type] = self.classify_layer(center_node[type] + embed)
        return pre_embed, GAN_input

    def show_metapath_importance(self, type=None):
        if type != None:
            weight = self.weight[type]
            for metapath in self.metapath2index[type]:
                print("Meta-path: ", metapath, "\tWeight: ", weight[self.metapath2index[type][metapath]].item())
            return
        for type in self.type_trained:
            weight = self.weight[type]
            for metapath in self.metapath2index[type]:
                print("type: ", type, "\tMeta-path: ", metapath, "\tWeight: ",
                      weight[self.metapath2index[type][metapath]].item())

    def save_metapath_importance(self, file):
        for type in self.type_trained:
            weight = self.weight[type]
            for metapath in self.metapath2index[type]:
                file.write(metapath + "," + str(weight[self.metapath2index[type][metapath]].item()) + ",")
            file.write('end\n')



class Discriminator(nn.Module):
    def __init__(self, info_section, type_num):
        super(Discriminator, self).__init__()
        self.info_section = info_section
        self.type_num = type_num
        self.classifier = nn.ModuleList()
        for i in range(type_num):
            self.classifier.append(
                nn.Sequential(
                    nn.Linear(info_section, 1),
                    nn.Sigmoid(),
                )
            )

    def forward(self, GAN_input, Shuffle=False):
        result = {}
        for metapath in GAN_input:
            result[metapath] = []
            section = GAN_input[metapath].split([self.info_section] * self.type_num, dim=1)
            for i in range(self.type_num):
                if not Shuffle:
                    result[metapath].append(self.classifier[i](section[i]))
                else:
                    choice = list(range(self.type_num))
                    choice.remove(i)
                    result[metapath].append(self.classifier[i](section[random.choice(choice)]))
            result[metapath] = torch.cat(result[metapath], dim=1)
        return result
