import argparse
import scipy.sparse as sp
import numpy as np
import torch
import ipdb
from scipy.io import loadmat
import networkx as nx
import multiprocessing as mp
import torch.nn.functional as F
from functools import partial
import random
from sklearn.metrics import roc_auc_score, f1_score
from copy import deepcopy
from scipy.spatial.distance import pdist,squareform
from sklearn.neighbors import NearestNeighbors
import copy
import math
import sklearn

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
    # Todo seed选择
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--nhid', type=int, default=64)
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--size', type=int, default=100)
    # Todo GPU选择
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--epochs', type=int, default=1010,
                help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--batch_nums', type=int, default=6000, help='number of batches per epoch')
    parser.add_argument('--batch_size', type=int, default=40, help='number of batches per epoch')

    # Todo imbalance
     parser.add_argument('--imbalance', action='store_true', default=False)
    # Todo setting
    parser.add_argument('--setting', type=str, default='no',
        choices=['no','upsampling', 'smote','reweight','embed_up', 'recon','newG_cls','recon_newG'])
    #upsampling: oversample in the raw input; smote: ; reweight: reweight minority classes; 
    # embed_up: 
    # recon: pretrain; newG_cls: pretrained decoder; recon_newG: also finetune the decoder

    parser.add_argument('--opt_new_G', action='store_true', default=False) # whether optimize the decoded graph based on classification result.
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--up_scale', type=float, default=1)
    # Todo im——ratio
    parser.add_argument('--im_ratio', type=float, default=0.5)
    parser.add_argument('--rec_weight', type=float, default=0.002)

    parser.add_argument('--up_beta', type=float, default=0.6) # 0.3 0.4 0.5 0.6 0.7
    parser.add_argument('--low_beta', type=float, default=0.1) #0.05 0.1 0.15 0.2 0.25
    # todo --umsample
    parser.add_argument('--umsample', type=str, default='recon_upsample')
    parser.add_argument('--decoder', type=str, default='Decoder')
    parser.add_argument('--loss_rec', type=str, default='origin')
    parser.add_argument('--loss_type', type=str, default='regular')
    parser.add_argument('--model', type=str, default='sage',
        choices=['sage','gcn','GAT','MLP'])



    return parser

def split_arti(labels, c_train_num):
    #labels: n-dim Longtensor, each element in [0,...,m-1].
    #cora: m=7
    num_classes = len(set(labels.tolist()))
    c_idxs = [] # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)

    nums = []

    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        c_num = len(c_idx)
        # print('{:d}-th class sample number: {:d}'.format(i,len(c_idx)))
        random.shuffle(c_idx)
        c_idxs.append(c_idx)
        nums.append(len(c_idx))

        c_num_mat[i, 0] = c_train_num[i]
        c_num_mat[i, 1] = c_train_num[i]
        c_num_mat[i, 2] = c_num - 2 * c_train_num[i]
        train_idx = train_idx + c_idx[:c_num_mat[i, 0]]
        val_idx = val_idx + c_idx[c_num_mat[i, 0]:c_num_mat[i, 0] + c_num_mat[i, 1]]
        test_idx = test_idx + c_idx[c_num_mat[i, 0] + c_num_mat[i, 1]:c_num_mat[i, 0] + c_num_mat[i, 1] + c_num_mat[i, 2]]
        
    print('class sample number:'+str(nums))
    print('class total number:'+str(sum(nums)))

    random.shuffle(train_idx)

    #ipdb.set_trace()

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)
    #c_num_mat = torch.LongTensor(c_num_mat)


    return train_idx, val_idx, test_idx, c_num_mat

def split_genuine(labels):
    #labels: n-dim Longtensor, each element in [0,...,m-1].
    #cora: m=7
    num_classes = len(set(labels.tolist()))
    c_idxs = [] # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    nums = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)

    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        c_num = len(c_idx)
        # print('{:d}-th class sample number: {:d}'.format(i,len(c_idx)))
        nums.append(len(c_idx))
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        if c_num <4:
            if c_num < 3:
                print("too small class type")
                ipdb.set_trace()
            c_num_mat[i,0] = 1
            c_num_mat[i,1] = 1
            c_num_mat[i,2] = 1
        else:
            c_num_mat[i,0] = int(c_num/4)
            c_num_mat[i,1] = int(c_num/4)
            c_num_mat[i,2] = int(c_num/2)

        train_idx = train_idx + c_idx[:c_num_mat[i,0]]
        val_idx = val_idx + c_idx[c_num_mat[i,0]:c_num_mat[i,0]+c_num_mat[i,1]]
        test_idx = test_idx + c_idx[c_num_mat[i,0]+c_num_mat[i,1]:c_num_mat[i,0]+c_num_mat[i,1]+c_num_mat[i,2]]
    print('class sample number:'+str(nums))
    print('class total number:'+str(sum(nums)))

    random.shuffle(train_idx)

    #ipdb.set_trace()

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)
    #c_num_mat = torch.LongTensor(c_num_mat)


    return train_idx, val_idx, test_idx, c_num_mat


def print_edges_num(dense_adj, labels):
    c_num = labels.max().item()+1
    dense_adj = np.array(dense_adj)
    labels = np.array(labels)
    homophily = []
    edge_total = 0
    for i in range(c_num):
        total = 0
        homo = 0
        for j in range(c_num):
            #ipdb.set_trace()
            row_ind = labels == i
            col_ind = labels == j
            edge_num = dense_adj[row_ind].transpose()[col_ind].sum()
            # print("edges between class {:d} and class {:d}: {:f}".format(i,j,edge_num))
            total = total + edge_num
            if i == j:
                homo = homo + edge_num
        homophily.append(float(homo/total))
        edge_total = edge_total + total
    print("homophily class"+str(homophily))
    print('edge total number:'+str(edge_total))


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def cmAccuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    result = sklearn.metrics.balanced_accuracy_score(labels.cpu().detach(), preds.cpu().detach())
    return result


def print_class_acc(output, labels, class_num_list, pre='valid'):
    pre_num = 0
    #print class-wise performance
    class_true_positive_rate = []
    # for i in range(labels.max()+1):
    #
    #     cur_tpr = accuracy(output[pre_num:pre_num+class_num_list[i]], labels[pre_num:pre_num+class_num_list[i]])
    #     print(str(pre)+" class {:d} True Positive Rate: {:.3f}".format(i,cur_tpr.item()))
    #     class_true_positive_rate.append(cur_tpr.item())
    #     index_negative = labels != i
    #     labels_negative = labels.new(labels.shape).fill_(i)
    #
    #     cur_fpr = accuracy(output[index_negative,:], labels_negative[index_negative])
    #     print(str(pre)+" class {:d} False Positive Rate: {:.3f}".format(i,cur_fpr.item()))
    #
    #     pre_num = pre_num + class_num_list[i]
    # print("class_true_positive_rate:" + str(class_true_positive_rate))

    #ipdb.set_trace()
    if labels.max() > 1:
        auc_score = roc_auc_score(labels.cpu().detach(), F.softmax(output, dim=-1).cpu().detach(), average='macro', multi_class='ovr')
    else:
        auc_score = roc_auc_score(labels.cpu().detach(), F.softmax(output, dim=-1)[:,1].cpu().detach(), average='macro')

    macro_F = f1_score(labels.cpu().detach(), torch.argmax(output, dim=-1).cpu().detach(), average='macro')
    # cmA = float(total/count)
    cmA = cmAccuracy(output, labels)
    print(str(pre)+' current auc-roc score: {:f}, current macro_F score: {:f}, '
                   'current cmA score: {:f}'.format(auc_score,macro_F,cmA))

    return auc_score, macro_F, cmA

minority_class_list = {  #调参
    "citeseer" : [0],
    "amazon_cs": [4, 5],
    "ms_cs": [5, 8]
}

def src_upsample(adj,features,labels,idx_train, dataset, portion=1.0, im_class_num=3):
    c_largest = labels.max().item()
    adj_back = adj.to_dense()
    chosen = None

    #ipdb.set_trace()
    avg_number = int(idx_train.shape[0]/(c_largest+1))

    for i in range(im_class_num):
        if dataset in minority_class_list:
            new_chosen = idx_train[(labels==minority_class_list[dataset][i])[idx_train]]
            # print("new_chosen"+str(new_chosen))
        else:
            new_chosen = idx_train[(labels==(c_largest-i))[idx_train]]
        if portion == 0:#refers to even distribution
            c_portion = int(avg_number/new_chosen.shape[0])

            for j in range(c_portion):
                if chosen is None:
                    chosen = new_chosen
                else:
                    chosen = torch.cat((chosen, new_chosen), 0)

        else:
            c_portion = int(portion)
            portion_rest = portion-c_portion
            for j in range(c_portion):
                num = int(new_chosen.shape[0])
                new_chosen = new_chosen[:num]

                if chosen is None:
                    chosen = new_chosen
                else:
                    chosen = torch.cat((chosen, new_chosen), 0)
            
            num = int(new_chosen.shape[0]*portion_rest)
            new_chosen = new_chosen[:num]

            if chosen is None:
                chosen = new_chosen
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
            

    add_num = chosen.shape[0]
    new_adj = adj_back.new(torch.Size((adj_back.shape[0]+add_num, adj_back.shape[0]+add_num)))
    new_adj[:adj_back.shape[0], :adj_back.shape[0]] = adj_back[:,:]
    new_adj[adj_back.shape[0]:, :adj_back.shape[0]] = adj_back[chosen,:]
    new_adj[:adj_back.shape[0], adj_back.shape[0]:] = adj_back[:,chosen]
    new_adj[adj_back.shape[0]:, adj_back.shape[0]:] = adj_back[chosen,:][:,chosen]

    #ipdb.set_trace()
    features_append = deepcopy(features[chosen,:])
    labels_append = deepcopy(labels[chosen])
    idx_new = np.arange(adj_back.shape[0], adj_back.shape[0]+add_num)
    idx_train_append = idx_train.new(idx_new)

    features = torch.cat((features,features_append), 0)
    labels = torch.cat((labels,labels_append), 0)
    idx_train = torch.cat((idx_train,idx_train_append), 0)
    adj = new_adj.to_sparse()

    return adj, features, labels, idx_train

def src_smote(adj,features,labels,idx_train, dataset, portion=1.0, im_class_num=3):
    c_largest = labels.max().item()
    adj_back = adj.to_dense()
    chosen = None
    new_features = None

    #ipdb.set_trace()
    avg_number = int(idx_train.shape[0]/(c_largest+1))

    for i in range(im_class_num):
        if dataset in minority_class_list:
            new_chosen = idx_train[(labels==minority_class_list[dataset][i])[idx_train]]
            print("new_chosen"+str(new_chosen))
        else:
            new_chosen = idx_train[(labels==(c_largest-i))[idx_train]]
        if portion == 0:#refers to even distribution
            c_portion = int(avg_number/new_chosen.shape[0])

            portion_rest = (avg_number/new_chosen.shape[0]) - c_portion

        else:
            c_portion = int(portion)
            portion_rest = portion-c_portion
            
        for j in range(c_portion):
            num = int(new_chosen.shape[0])
            new_chosen = new_chosen[:num]

            chosen_embed = features[new_chosen,:]
            distance = squareform(pdist(chosen_embed.detach()))
            np.fill_diagonal(distance,distance.max()+100)

            idx_neighbor = distance.argmin(axis=-1)
            
            interp_place = random.random()
            embed = chosen_embed + (chosen_embed[idx_neighbor,:]-chosen_embed)*interp_place

            if chosen is None:
                chosen = new_chosen
                new_features = embed
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
                new_features = torch.cat((new_features, embed),0)
            
        num = int(new_chosen.shape[0]*portion_rest)
        new_chosen = new_chosen[:num]

        chosen_embed = features[new_chosen,:]
        distance = squareform(pdist(chosen_embed.detach()))
        np.fill_diagonal(distance,distance.max()+100)

        idx_neighbor = distance.argmin(axis=-1)
            
        interp_place = random.random()
        embed = chosen_embed + (chosen_embed[idx_neighbor,:]-chosen_embed)*interp_place

        if chosen is None:
            chosen = new_chosen
            new_features = embed
        else:
            chosen = torch.cat((chosen, new_chosen), 0)
            new_features = torch.cat((new_features, embed),0)
            

    add_num = chosen.shape[0]
    new_adj = adj_back.new(torch.Size((adj_back.shape[0]+add_num, adj_back.shape[0]+add_num)))
    new_adj[:adj_back.shape[0], :adj_back.shape[0]] = adj_back[:,:]
    new_adj[adj_back.shape[0]:, :adj_back.shape[0]] = adj_back[chosen,:]
    new_adj[:adj_back.shape[0], adj_back.shape[0]:] = adj_back[:,chosen]
    new_adj[adj_back.shape[0]:, adj_back.shape[0]:] = adj_back[chosen,:][:,chosen]

    #ipdb.set_trace()
    features_append = deepcopy(new_features)
    labels_append = deepcopy(labels[chosen])
    idx_new = np.arange(adj_back.shape[0], adj_back.shape[0]+add_num)
    idx_train_append = idx_train.new(idx_new)

    features = torch.cat((features,features_append), 0)
    labels = torch.cat((labels,labels_append), 0)
    idx_train = torch.cat((idx_train,idx_train_append), 0)
    adj = new_adj.to_sparse()

    return adj, features, labels, idx_train

def recon_upsample(embed, labels, idx_train, dataset ,adj=None, portion=1.0, im_class_num=3, epoch=10, max_epoch=1000):
    c_largest = labels.max().item()
    avg_number = int(idx_train.shape[0]/(c_largest+1))
    #ipdb.set_trace()
    adj_new = None
    # print("idx_train"+str(idx_train))
    for i in range(im_class_num):
        if dataset in minority_class_list:
            obj_label = minority_class_list[dataset][i]
        else:
            obj_label = c_largest-i
        chosen = idx_train[(labels == obj_label)[idx_train]]
        # print("chosen" + str(chosen))
        num = int(chosen.shape[0]*portion)
        if portion == 0:
            c_portion = int(avg_number/chosen.shape[0])
            num = chosen.shape[0]
        else:
            c_portion = 1

        for j in range(c_portion):
            chosen = chosen[:num]

            chosen_embed = embed[chosen,:]
            distance = squareform(pdist(chosen_embed.cpu().detach()))
            np.fill_diagonal(distance,distance.max()+100)

            idx_neighbor = distance.argmin(axis=-1)
            
            interp_place = random.random()
            # TODO 标准的生成过程 新节点
            new_embed = embed[chosen,:] + (embed[idx_neighbor,:]-embed[chosen,:])*interp_place

            new_labels = labels.new(torch.Size((chosen.shape[0],1))).reshape(-1).fill_(obj_label)
            idx_new = np.arange(embed.shape[0], embed.shape[0]+chosen.shape[0])
            idx_train_append = idx_train.new(idx_new)

            embed = torch.cat((embed,new_embed), 0)
            labels = torch.cat((labels,new_labels), 0)
            idx_train = torch.cat((idx_train,idx_train_append), 0)

            if adj is not None:
                if adj_new is None:
                    adj_new = adj.new(torch.clamp_(adj[chosen,:] + adj[idx_neighbor,:], min=0.0, max = 1.0))
                else:
                    temp = adj.new(torch.clamp_(adj[chosen,:] + adj[idx_neighbor,:], min=0.0, max = 1.0))
                    adj_new = torch.cat((adj_new, temp), 0)

    if adj is not None:
        add_num = adj_new.shape[0]
        new_adj = adj.new(torch.Size((adj.shape[0]+add_num, adj.shape[0]+add_num))).fill_(0.0)
        new_adj[:adj.shape[0], :adj.shape[0]] = adj[:,:]
        new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new[:,:]
        new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new, 0, 1)[:,:]

        return embed, labels, idx_train, new_adj.detach()

    else:
        return embed, labels, idx_train

'''
def danger_upsample(embed, labels, idx_train, dataset, adj=None, portion=1.0, im_class_num=3, epoch=10, max_epoch=1000):
    c_largest = labels.max().item()
    avg_number = int(idx_train.shape[0] / (c_largest + 1))
    # ipdb.set_trace()
    adj_new = None
    # print("idx_train"+str(idx_train))
    for i in range(im_class_num):
        if dataset in minority_class_list:
            obj_label = minority_class_list[dataset][i]
        else:
            obj_label = c_largest - i
        chosen = idx_train[(labels == obj_label)[idx_train]]
        # print("chosen" + str(chosen))

        portion = portion * (1.0 - 0.5 * cal_cir_score(epoch, max_epoch))

        num = int(chosen.shape[0] * portion)
        if portion == 0:
            c_portion = int(avg_number / chosen.shape[0])
            num = chosen.shape[0]
        else:
            c_portion = 1

        for j in range(c_portion):
            chosen = chosen[:num]

            chosen_embed = embed[chosen, :]
            distance = squareform(pdist(chosen_embed.cpu().detach()))
            np.fill_diagonal(distance, distance.max() + 100)

            idx_neighbor = distance.argmin(axis=-1)

            interp_place = random.random()
            new_embed = embed[chosen, :] + (embed[idx_neighbor, :] - embed[chosen, :]) * interp_place

            new_labels = labels.new(torch.Size((chosen.shape[0], 1))).reshape(-1).fill_(obj_label)
            idx_new = np.arange(embed.shape[0], embed.shape[0] + chosen.shape[0])
            idx_train_append = idx_train.new(idx_new)

            embed = torch.cat((embed, new_embed), 0)
            labels = torch.cat((labels, new_labels), 0)
            idx_train = torch.cat((idx_train, idx_train_append), 0)

            if adj is not None:
                if adj_new is None:
                    adj_new = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
                else:
                    temp = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
                    adj_new = torch.cat((adj_new, temp), 0)

    if adj is not None:
        add_num = adj_new.shape[0]
        new_adj = adj.new(torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num))).fill_(0.0)
        new_adj[:adj.shape[0], :adj.shape[0]] = adj[:, :]
        new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new[:, :]
        new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new, 0, 1)[:, :]

        return embed, labels, idx_train, new_adj.detach()

    else:
        return embed, labels, idx_train
'''


def danger_upsample(embed, labels, idx_train, dataset, adj=None, n_neighbors=10, im_class_num=3, epoch=10, max_epoch=1000, cuda=False, setting="no"):
    # print("old_embed" + str(embed))
    # print("old_embed shape" + str(embed.shape))
    # print("old_adj" + str(adj))
    # print("old_adj shape" + str(adj.shape))
    # print("old_adj sum" + str(torch.sum(adj)))


    c_largest = labels.max().item()
    adj_new = None
    im_class_ids = []
    im_class_ids_length = []
    for i in range(im_class_num):
        if dataset in minority_class_list:
            obj_label = minority_class_list[dataset][i]
        else:
            obj_label = c_largest-i
        chosen = idx_train[(labels == obj_label)[idx_train]]
        print("chosen:"+str(obj_label))
        length = chosen.shape[0]
        im_class_ids.append(chosen)
        im_class_ids_length.append(length)

    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(embed[idx_train, :].cpu().detach())
    idx_train_dict = {}
    for idx, i in enumerate(idx_train):
        idx_train_dict[idx] = i

    for i in range(im_class_num):
        if setting=="recon":   #调参
            portion = 1.0 #预训练时是否要加？
        else:
            portion = 1.0 * (1.0 - cal_cir_score(epoch, max_epoch))
        c_portion = 1
        for j in range(c_portion):
            chosen = copy.deepcopy(im_class_ids[i])
            chosen = get_random_list(chosen, portion)
            if chosen.shape[0] > 0:
                print("chosen"+str(chosen))
                chosen_embed = embed[chosen, :].cpu().detach()
                diff_adj = torch.zeros_like(adj[chosen, :])
                chosen_kneighbors = neigh.kneighbors(chosen_embed, return_distance=False)
                if cuda:
                    diff_embed = torch.zeros_like(chosen_embed).cuda()
                    chosen_embed = chosen_embed.cuda()
                else:
                    diff_embed = torch.zeros_like(chosen_embed)

                for idx, neighbors in enumerate(chosen_kneighbors):
                    neighbors_idx = [idx_train_dict[tmp] for tmp in neighbors][1:]
                    neighbors_idxs = torch.stack(neighbors_idx)
                    intersection_list = np.intersect1d(neighbors_idxs.cpu().detach().numpy(), im_class_ids[i].cpu().detach().numpy())
                    # print("intersection_list"+str(intersection_list))

                    if(len(intersection_list) > 0):
                        random_interp_place = np.random.rand(len(intersection_list))
                        ratio = float(1.0 / sum(random_interp_place))
                        interp_places = random_interp_place * ratio

                        for random_idx, inter_id in enumerate(intersection_list):
                            tmp_random = random.random()
                            diff_embed[idx] = diff_embed[idx] +\
                                              (embed[inter_id, :] - chosen_embed[idx, :]) * tmp_random * interp_places[random_idx]
                            # print("tmp_random * interp_places[random_idx]" + str(tmp_random * interp_places[random_idx]))
                            diff_adj[idx] = diff_adj[idx] + adj[inter_id]
                        # diff_adj[idx][intersection_list] = 1.0  # ?????邻居 #调参

                new_embed = embed[chosen, :] + diff_embed
                new_labels = labels.new(torch.Size((chosen.shape[0], 1))).reshape(-1).fill_(obj_label)
                idx_new = np.arange(embed.shape[0], embed.shape[0] + chosen.shape[0])
                idx_train_append = idx_train.new(idx_new)

                embed = torch.cat((embed, new_embed), 0)
                labels = torch.cat((labels, new_labels), 0)
                idx_train = torch.cat((idx_train, idx_train_append), 0)

                if adj is not None:
                    if adj_new is None:
                        adj_new = adj.new(torch.clamp_(adj[chosen, :] + diff_adj, min=0.0, max=1.0))
                    else:
                        temp = adj.new(torch.clamp_(adj[chosen, :] + diff_adj, min=0.0, max=1.0))
                        adj_new = torch.cat((adj_new, temp), 0)


    if adj is not None:
        if adj_new is None:
            new_adj = adj.new(torch.Size((adj.shape[0] + 0, adj.shape[0] + 0))).fill_(0.0)
            new_adj[:adj.shape[0], :adj.shape[0]] = adj[:, :]
        else:
            add_num = adj_new.shape[0]
            new_adj = adj.new(torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num))).fill_(0.0)
            new_adj[:adj.shape[0], :adj.shape[0]] = adj[:, :]
            new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new[:, :]
            new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new, 0, 1)[:, :]

        # print("new_embed" + str(embed))
        # print("new_embed shape" + str(embed.shape))
        # print("new_adj" + str(new_adj))
        # print("new_adj shape" + str(new_adj.shape))
        # print("new_adj sum" + str(torch.sum(new_adj)))

        return embed, labels, idx_train, new_adj.detach()

    else:
        return embed, labels, idx_train

def adj_mse_loss(adj_rec, adj_tgt, adj_mask = None):
    edge_num = adj_tgt.nonzero().shape[0]
    total_num = adj_tgt.shape[0]**2

    neg_weight = edge_num / (total_num-edge_num)

    weight_matrix = adj_rec.new(adj_tgt.shape).fill_(1.0)
    weight_matrix[adj_tgt==0] = neg_weight

    loss = torch.sum(weight_matrix * (adj_rec - adj_tgt) ** 2)

    return loss

def adj_homo_mse_loss(adj_rec, adj_tgt, adj_label, adj_homo, alpha, beta):

    # adj_none_label = adj_label.new(adj_label.shape).fill_(1.0) - adj_label

    ################
    edge_num = adj_tgt.nonzero().shape[0]
    total_num = adj_tgt.shape[0] ** 2

    neg_weight = edge_num / (total_num - edge_num)

    weight_matrix = adj_rec.new(adj_tgt.shape).fill_(1.0)
    weight_matrix[adj_tgt == 0] = neg_weight

    loss1 = torch.sum(weight_matrix * (adj_rec - adj_tgt) ** 2)

    loss = alpha * loss1
    # print("loss1" + str(loss1))
    #################

    label_adj_rec = adj_label * adj_rec

    label_edge_num = adj_homo.nonzero().shape[0]
    label_total_num = adj_label.nonzero().shape[0]

    if label_total_num != label_edge_num:
        label_neg_weight = label_edge_num / (label_total_num - label_edge_num)
        label_weight_matrix = label_adj_rec.new(adj_homo.shape).fill_(1.0)
        label_weight_matrix[adj_homo == 0] = label_neg_weight

        loss2 = torch.sum(label_weight_matrix * (label_adj_rec - adj_homo) ** 2)

        loss = loss + beta * loss2
        # print("loss2" + str(loss2))
    return loss

def cosine_distance(feature1, feature2):
    return 1 - torch.cosine_similarity(feature1, feature2)

def cal_cir_score(cur_epoch, max_epoch):
    # return math.pow(0.99, cur_epoch)
    # return math.cos(float(cur_epoch/max_epoch)*math.pi/2)
    return 1.0
    # return 1 - float(cur_epoch/max_epoch)

def get_random_list(candidates, prob):
    res = []
    for i in candidates:
        p = random.random()
        if p < prob:
            res.append(i)
    return torch.LongTensor(res)

def write_record(filename, content):
    f1 = open('new_runs/' + filename+'.txt', 'a')
    f1.write(content)