import argparse
import numpy as np

import torch
print(torch.cuda.is_available())
import torch.nn.functional as F
import torch.optim as optim

import models
import utils
import data_load
import random
import ipdb
import copy
import time


import os
# from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"
#from torch.utils.tensorboard import SummaryWriter

# Training setting
parser = utils.get_parser()

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.cuda.set_device('cuda:'+str(args.gpu)) 

minority_class_list = {  
    "citeseer" : [0],
    "amazon_cs": [4, 5],
    "ms_cs": [5, 8]
}
acc = []
auc = []
macrof1 = []
cmA = []

# default `log_dir` is "runs" - we'll be more specific here
name = args.model+'/'+args.dataset+'/'+str(time.strftime("%Y-%m-%d%H:%M:%S", time.localtime()))
record_name = args.loss_type+str(args.up_beta)+str(args.low_beta)+str(args.up_scale)+str(args.im_ratio)+args.umsample+'_'+args.model+'_'+args.dataset+'_'+args.setting+'_'+str(time.strftime("%Y-%m-%d%H:%M:%S", time.localtime()))

# writer = SummaryWriter('runs/origin_recon_cosine'+name)

'''
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
'''

# Load data
if args.dataset == 'cora':
    adj, features, labels = data_load.load_data()
    im_class_num = 3
    class_sample_num = 20
elif args.dataset == 'BlogCategory': 
    adj, features, labels = data_load.load_data_Blog()
    im_class_num = 14 #set it to be the number less than 100
    class_sample_num = 20 #not used
elif args.dataset == 'twitter':
    adj, features, labels = data_load.load_sub_data_twitter()
    im_class_num = 1
    class_sample_num = 20  # not used
elif args.dataset == 'citeseer':
    adj, features, labels = data_load.load_citeseer()
    im_class_num = 1
    class_sample_num = 20  # not used
elif args.dataset == 'pubmed':
    adj, features, labels = data_load.load_citeseer(dataset_str="pubmed")
    im_class_num = 1
    class_sample_num = 20  # not used

elif args.dataset == 'amazon_cs':
    adj, features, labels = data_load.load_new_data(dataset_str="amazon_cs")
    im_class_num = 2
    class_sample_num = 20  # not used
elif args.dataset == 'amazon_photo':
    adj, features, labels = data_load.load_new_data(dataset_str="amazon_photo")
    im_class_num = 2
    class_sample_num = 20  # not used
elif args.dataset == 'ms_cs':
    adj, features, labels = data_load.load_new_data(dataset_str="ms_cs")
    im_class_num = 2
    class_sample_num = 20  # not used
elif args.dataset == 'ms_phy':
    adj, features, labels = data_load.load_new_data(dataset_str="ms_phy")
    im_class_num = 2
    class_sample_num = 20  # not used
else:
    print("no this dataset: {args.dataset}")


#for artificial imbalanced setting: only the last im_class_num classes are imbalanced
c_train_num = []
for i in range(labels.max().item() + 1):
    if args.imbalance and i > labels.max().item()-im_class_num: #only imbalance the last classes
        c_train_num.append(int(class_sample_num*args.im_ratio))
    else:
        c_train_num.append(class_sample_num)

#get train, validatio, test data split
if args.dataset == 'BlogCategory':
    idx_train, idx_val, idx_test, class_num_mat = utils.split_genuine(labels)
elif args.dataset == 'cora':
    idx_train, idx_val, idx_test, class_num_mat = utils.split_arti(labels, c_train_num)
elif args.dataset == 'twitter':
    idx_train, idx_val, idx_test, class_num_mat = utils.split_genuine(labels)
elif args.dataset == 'citeseer':
    idx_train, idx_val, idx_test, class_num_mat = utils.split_genuine(labels)
elif args.dataset == 'pubmed':
    idx_train, idx_val, idx_test, class_num_mat = utils.split_genuine(labels)
elif args.dataset == 'amazon_cs':
    idx_train, idx_val, idx_test, class_num_mat = utils.split_genuine(labels)
elif args.dataset == 'amazon_photo':
    idx_train, idx_val, idx_test, class_num_mat = utils.split_genuine(labels)
elif args.dataset == 'ms_cs':
    idx_train, idx_val, idx_test, class_num_mat = utils.split_genuine(labels)
elif args.dataset == 'ms_phy':
    idx_train, idx_val, idx_test, class_num_mat = utils.split_genuine(labels)
#

if os.path.exists("tensor/"+args.dataset+"_adj_label.pt"):
    adj_label = torch.load("tensor/"+args.dataset+"_adj_label.pt")
    adj_homo = torch.load("tensor/"+args.dataset+"_adj_homo.pt")
else:
    adj_label = torch.zeros_like(adj.to_dense())
    adj_homo = torch.zeros_like(adj.to_dense())
    adj2numpy=adj.to_dense().numpy()
    indexes = np.nonzero(adj2numpy)
    xindex = indexes[0]
    yindex = indexes[1]

    for id, tmp_i in enumerate(xindex):
        if xindex[id] in idx_train and yindex[id] in idx_train:
            if adj[xindex[id]][yindex[id]] == 1.0 :
                if xindex[id] != yindex[id]:
                    adj_label[xindex[id]][yindex[id]] = 1.0
                    adj_label[yindex[id]][xindex[id]] = 1.0
                    if labels[xindex[id]] == labels[yindex[id]]:
                        adj_homo[xindex[id]][yindex[id]] = 1.0
                        adj_homo[yindex[id]][xindex[id]] = 1.0
    torch.save(adj_label, "tensor/" + args.dataset + "_adj_label.pt")
    torch.save(adj_label, "tensor/" + args.dataset + "_adj_homo.pt")
print("final____")
# print(adj_label.to_sparse())
# print(adj_homo.to_sparse())


#method_1: oversampling in input domain
if args.setting == 'upsampling':
    adj,features,labels,idx_train = utils.src_upsample(adj,features,labels,idx_train,args.dataset,portion=args.up_scale, im_class_num=im_class_num)
if args.setting == 'smote':
    adj,features,labels,idx_train = utils.src_smote(adj,features,labels,idx_train,args.dataset,portion=args.up_scale, im_class_num=im_class_num)


# Model and optimizer
#if oversampling in the embedding space is required, model need to be changed
if args.setting != 'embed_up':
    if args.model == 'sage':
        encoder = models.Sage_En(nfeat=features.shape[1],
                nhid=args.nhid,
                nembed=args.nhid,
                dropout=args.dropout)
        classifier = models.Sage_Classifier(nembed=args.nhid, 
                nhid=args.nhid, 
                nclass=labels.max().item() + 1, 
                dropout=args.dropout)
    elif args.model == 'gcn':
        encoder = models.GCN_En(nfeat=features.shape[1],
                nhid=args.nhid,
                nembed=args.nhid,
                dropout=args.dropout)
        classifier = models.GCN_Classifier(nembed=args.nhid, 
                nhid=args.nhid, 
                nclass=labels.max().item() + 1, 
                dropout=args.dropout)
    elif args.model == 'GAT':
        encoder = models.GAT_En(nfeat=features.shape[1],
                nhid=args.nhid,
                nembed=args.nhid,
                dropout=args.dropout)
        classifier = models.GAT_Classifier(nembed=args.nhid, 
                nhid=args.nhid, 
                nclass=labels.max().item() + 1, 
                dropout=args.dropout)
    elif args.model == 'MLP':
        encoder = models.MLP_En(nfeat=features.shape[1],
                nhid=args.nhid,
                nembed=args.nhid,
                dropout=args.dropout)
        classifier = models.Classifier(nembed=args.nhid,
                nhid=args.nhid,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
else:
    if args.model == 'sage':
        encoder = models.Sage_En2(nfeat=features.shape[1],
                nhid=args.nhid,
                nembed=args.nhid,
                dropout=args.dropout)
        classifier = models.Classifier(nembed=args.nhid, 
                nhid=args.nhid, 
                nclass=labels.max().item() + 1, 
                dropout=args.dropout)
    elif args.model == 'gcn':
        encoder = models.GCN_En2(nfeat=features.shape[1],
                nhid=args.nhid,
                nembed=args.nhid,
                dropout=args.dropout)
        classifier = models.Classifier(nembed=args.nhid, 
                nhid=args.nhid, 
                nclass=labels.max().item() + 1, 
                dropout=args.dropout)
    elif args.model == 'GAT':
        encoder = models.GAT_En2(nfeat=features.shape[1],
                nhid=args.nhid,
                nembed=args.nhid,
                dropout=args.dropout)
        classifier = models.Classifier(nembed=args.nhid, 
                nhid=args.nhid, 
                nclass=labels.max().item() + 1, 
                dropout=args.dropout)


if args.decoder == "Decoder":
    decoder = models.Decoder(nembed=args.nhid,
        dropout=args.dropout)
elif args.decoder == "SmoothDecoder":
    decoder = models.SmoothDecoder(nembed=args.nhid,
        dropout=args.dropout)


optimizer_en = optim.Adam(encoder.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
optimizer_cls = optim.Adam(classifier.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
optimizer_de = optim.Adam(decoder.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


gcn_adj = data_load.normalize_adj(adj.to_dense())


if args.cuda:
    encoder = encoder.cuda()
    classifier = classifier.cuda()
    decoder = decoder.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    gcn_adj = gcn_adj.cuda()
    adj_label = adj_label.cuda()
    adj_homo = adj_homo.cuda()

tmp_ratio = float(1/(labels.max().item() + 1))

def train(epoch):
    global gcn_adj
    t = time.time()
    encoder.train()
    classifier.train()
    decoder.train()
    optimizer_en.zero_grad()
    optimizer_cls.zero_grad()
    optimizer_de.zero_grad()


    if args.model == 'MLP':
        # embed = encoder(features, gcn_adj)
        embed = encoder(features)
    else:
        embed = encoder(features, adj)

    if args.setting == 'recon_newG' or args.setting == 'recon' or args.setting == 'newG_cls':
        ori_num = labels.shape[0]
        ori_num_homo = adj_label.nonzero().shape[0]

        if args.umsample == 'recon_upsample':
            embed, labels_new, idx_train_new, adj_up = utils.recon_upsample(embed, labels, idx_train, args.dataset, adj=adj.detach().to_dense(),portion=args.up_scale,
                                                                       im_class_num=im_class_num, epoch=epoch, max_epoch=args.epochs)
        # elif args.umsample == 'danger_upsample':
        #     embed, labels_new, idx_train_new, adj_up = utils.danger_upsample(embed, labels, idx_train, args.dataset, adj=adj.detach().to_dense(),
        #                                                                      n_neighbors=10, im_class_num=im_class_num, epoch=epoch, max_epoch=args.epochs, cuda=args.cuda, setting=args.setting)
        elif args.umsample == 'danger_upsample':
            # embed, labels_new, idx_train_new, adj_up = utils.danger_upsample(embed, labels, idx_train, args.dataset,
            #                                                                 adj=adj.detach().to_dense(),
            #                                                                 portion=args.up_scale,
            #                                                                 im_class_num=im_class_num, epoch=epoch,
            #                                                                 max_epoch=args.epochs)
            embed, labels_new, idx_train_new, adj_up = utils.danger_upsample(embed, labels, idx_train, args.dataset,
                                                                            adj=adj.detach().to_dense(),
                                                                            n_neighbors=4,
                                                                            im_class_num=im_class_num, epoch=epoch,
                                                                            max_epoch=args.epochs)

            # elif args.umsample == 'danger_upsample':
        generated_G = decoder(embed)

        if args.loss_rec == 'origin':
            loss_rec = utils.adj_mse_loss(generated_G[:ori_num, :][:, :ori_num], adj.detach().to_dense())
        elif args.loss_rec == 'homo_edge_loss':
            loss_rec = utils.adj_homo_mse_loss(generated_G[:ori_num, :][:, :ori_num], adj.detach().to_dense(),
                                               adj_label.detach(), adj_homo.detach(), 1.0, 1.0) #调参loss比例

        #ipdb.set_trace()

        if not args.opt_new_G:
            adj_new = copy.deepcopy(generated_G.detach())
            # print("adj_new" + str(adj_new))
            # print("adj_shape" + str(adj_new.shape))
            # print("adj_sum" + str(torch.sum(adj_new)))
            threshold = 0.5
            adj_new[adj_new<threshold] = 0.0
            adj_new[adj_new>=threshold] = 1.0

            #ipdb.set_trace()
            edge_ac = adj_new[:ori_num, :ori_num].eq(adj.to_dense()).double().sum()/(ori_num**2)
            edge_ac_homo = torch.tensor(0.0)
            # adj_tmp = adj_new[:ori_num, :ori_num].eq(adj_homo).double() * adj_label.double()
            # edge_ac_homo = adj_tmp.sum()/(ori_num_homo)

        else:
            adj_new = generated_G
            edge_ac = F.l1_loss(adj_new[:ori_num, :ori_num], adj.to_dense(), reduction='mean')
            # adj_tmp = adj_new[:ori_num, :ori_num].eq(adj_homo).double() * adj_label.double()
            # edge_ac_homo = F.l1_loss(adj_tmp, adj_homo, reduction='mean')


        #calculate generation information
        exist_edge_prob = adj_new[:ori_num, :ori_num].mean() #edge prob for existing nodes
        generated_edge_prob = adj_new[ori_num:, :ori_num].mean() #edge prob for generated nodes
        print("edge acc: {:.4f}, exist_edge_prob: {:.4f}, generated_edge_prob: {:.4f}, edge_ac_homo acc: {:.4f}".format(edge_ac.item(), exist_edge_prob.item(), generated_edge_prob.item(), edge_ac_homo.item()))


        adj_new = torch.mul(adj_up, adj_new)

        exist_edge_prob = adj_new[:ori_num, :ori_num].mean() #edge prob for existing nodes
        generated_edge_prob = adj_new[ori_num:, :ori_num].mean() #edge prob for generated nodes
        print("after filtering, edge acc: {:.4f}, exist_edge_prob: {:.4f}, generated_edge_prob: {:.4f}, edge_ac_homo acc: {:.4f}".format(edge_ac.item(), exist_edge_prob.item(), generated_edge_prob.item(), edge_ac_homo.item()))


        adj_new[:ori_num, :][:, :ori_num] = adj.detach().to_dense()
        #adj_new = adj_new.to_sparse()
        #ipdb.set_trace()

        if not args.opt_new_G:
            adj_new = adj_new.detach()

        if args.setting == 'newG_cls':
            idx_train_new = idx_train

    elif args.setting == 'embed_up':
        #perform SMOTE in embedding space
        embed, labels_new, idx_train_new = utils.recon_upsample(embed, labels, idx_train, args.dataset, portion=args.up_scale, im_class_num=im_class_num)
        adj_new = adj
    else:
        labels_new = labels
        idx_train_new = idx_train
        adj_new = adj


    if args.model == 'gcn':
        # gcn_adj_new = data_load.normalize_adj(adj_new)
        output = classifier(embed, adj_new)
    else:
        output = classifier(embed, adj_new)
    #ipdb.set_trace()

    if args.setting == 'reweight':
        weight = features.new((labels.max().item()+1)).fill_(1)
        if args.dataset in minority_class_list:
            weight[minority_class_list[args.dataset]:] = 1+args.up_scale
        else:
            weight[-im_class_num:] = 1+args.up_scale
        loss_train = F.cross_entropy(output[idx_train_new], labels_new[idx_train_new], weight=weight)
    else:
        loss_train = F.cross_entropy(output[idx_train_new], labels_new[idx_train_new])

    ##########
    ##########
    ##########
    ##########
    if args.loss_type == 'cosine':
        label_ratio = F.softmax(output, dim=1)
        pred_label_max = label_ratio.max(1)
        pred_label = pred_label_max[1]
        pred_score = pred_label_max[0]
        up_alpha = (1.0-args.up_beta*utils.cal_cir_score(epoch, args.epochs)) 
        # up_alpha = 0.4*(2.0-utils.cal_cir_score(epoch, args.epochs))
        down_alpha = args.low_beta*(utils.cal_cir_score(epoch, args.epochs))  
        adj_new_dense = adj_new
        all_class_loss = []
        c_largest = labels.max().item()
        for cur_num in range(im_class_num):
            if args.dataset in minority_class_list:
                obj_label = minority_class_list[args.dataset][cur_num]
            else:
                obj_label = c_largest - cur_num
            one_class_loss = []
            small_label_idx = (pred_label == obj_label).nonzero()[:, -1].tolist()
            for ss in small_label_idx:
                if ss < adj.shape[0] and pred_score[ss] > up_alpha:
                    # neighbor_idx = adj_new_dense.to_dense()[ss].nonzero()[:, -1].tolist()
                    neighbor_idx = adj_new_dense[ss].nonzero()[:, -1].tolist()
                    plus_ids = []
                    minus_ids = []
                    for nn in neighbor_idx:
                        if nn < adj.shape[0]:
                            neighbor_score = label_ratio[nn][obj_label]
                            if neighbor_score > up_alpha:
                                plus_ids.append(nn)
                            elif neighbor_score < down_alpha:
                                minus_ids.append(nn)

                    plus_distance = utils.cosine_distance(output[ss].reshape(1, -1), output[plus_ids])
                    minus_distance = utils.cosine_distance(output[ss].reshape(1, -1), output[minus_ids])

                    if(plus_distance.shape[0] > 0 and minus_distance.shape[0] > 0):
                        plus_dikar = torch.transpose(plus_distance.repeat(minus_distance.shape[0], 1), 0, 1).reshape(1, -1)[0]
                        mius_dikar = minus_distance.repeat(plus_distance.shape[0])
                        tmp = torch.sub(plus_dikar, mius_dikar)+0.5 
                        zeros = torch.zeros_like(tmp)
                        final_dikar = torch.where(torch.lt(tmp, 0.0), zeros, tmp)
                        one_class_loss.append(final_dikar)
            if(len(one_class_loss) > 0):
                final_dikar_mean = torch.mean(torch.cat(one_class_loss))
                all_class_loss.append(final_dikar_mean)

        print("regular_loss" + str(all_class_loss))
        regular_loss = float(sum(all_class_loss))
    elif args.loss_type == 'regular':
        regular_loss = float(0.0)

####################
    acc_train = utils.accuracy(output[idx_train], labels_new[idx_train])
    if args.setting == 'recon_newG':
        if args.loss_type == 'cosine':
            loss = loss_train+loss_rec*args.rec_weight + regular_loss * 2.0 #* utils.cal_cir_score(epoch, args.epochs)
        elif args.loss_type == 'regular':
            loss = loss_train+loss_rec*args.rec_weight
    elif args.setting == 'recon':
        loss = loss_rec + 0*loss_train
    else:
        if args.loss_type == 'cosine':
            loss = loss_train + regular_loss * 0.5 #* utils.cal_cir_score(epoch, args.epochs)  
            loss_rec = loss_train + regular_loss * 0.5 #* utils.cal_cir_score(epoch, args.epochs)  
        elif args.loss_type == 'regular':
            loss = loss_train
            loss_rec = loss_train

    loss.backward()
    if args.setting == 'newG_cls':
        optimizer_en.zero_grad()
        optimizer_de.zero_grad()
    else:
        optimizer_en.step()

    optimizer_cls.step()

    if args.setting == 'recon_newG' or args.setting == 'recon':
        optimizer_de.step()

    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
    acc_val = utils.accuracy(output[idx_val], labels[idx_val])

    #ipdb.set_trace()
    utils.print_class_acc(output[idx_val], labels[idx_val], class_num_mat[:,1])

    auc_score, macro_F, class_balance_mean = utils.print_class_acc(output[idx_test], labels[idx_test], class_num_mat[:,2], pre='valid')
    """
    writer.add_scalar('train metric_loss',
                      regular_loss,
                      epoch)

    writer.add_scalar('train classification_loss',
                      loss_train.item(),
                      epoch)

    writer.add_scalar('train recognization_loss',
                      loss_rec.item(),
                      epoch)

    writer.add_scalar('train_cmA',
                      class_balance_mean,
                      epoch)

    writer.add_scalar('train AUC-ROC',
                      auc_score,
                      epoch)
    """
    print('Epoch: {:05d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_rec: {:.4f}'.format(loss_rec.item()),
          'regular_loss: {:.4f}'.format(regular_loss),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test(epoch = 0):
    encoder.eval()
    classifier.eval()
    decoder.eval()
    if args.model == 'MLP':
        # embed = encoder(features, gcn_adj)
        embed = encoder(features)
    else:
        embed = encoder(features, adj)
    output = classifier(embed, adj)
    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test = utils.accuracy(output[idx_test], labels[idx_test])

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    acc.append(acc_test.item())

    auc_score, macro_F, class_balance_mean = utils.print_class_acc(output[idx_test], labels[idx_test], class_num_mat[:,2], pre='test')
    auc.append(auc_score)
    macrof1.append(macro_F)
    cmA.append(class_balance_mean)
    '''
    writer.add_scalar('test cmA',
                      class_balance_mean,
                      epoch)

    writer.add_scalar('test ROC-AUC',
                      auc_score,
                      epoch)
    
    if epoch==40:
        torch
    '''


def save_model(epoch):
    saved_content = {}

    saved_content['encoder'] = encoder.state_dict()
    saved_content['decoder'] = decoder.state_dict()
    saved_content['classifier'] = classifier.state_dict()

    torch.save(saved_content, 'checkpoint/{}/{}_{}_{}_{}_{}_{}_{}_{}.pth'.format(args.dataset,args.model,args.umsample,args.decoder,args.loss_rec,args.setting,epoch, args.opt_new_G, args.im_ratio))
    print("save checkpoint/{}/{}_{}_{}_{}_{}_{}_{}_{}.pth".format(args.dataset,args.model,args.umsample,args.decoder,args.loss_rec,args.setting,epoch, args.opt_new_G, args.im_ratio))
    return

def load_model(filename):
    loaded_content = torch.load('checkpoint/{}/{}.pth'.format(args.dataset,filename), map_location=lambda storage, loc: storage)

    encoder.load_state_dict(loaded_content['encoder'])
    decoder.load_state_dict(loaded_content['decoder'])
    classifier.load_state_dict(loaded_content['classifier'])

    print("successfully loaded: "+ filename)

    return

# Train model
if args.load is not None:
    load_model(args.load)

t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)

    if epoch % 10 == 0:
        test(epoch)

    if epoch % 100 == 0:
        if args.setting == 'recon':
            save_model(epoch)


print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

num_classes = len(set(labels.tolist()))
class_training_sample_number = []
class_validating_sample_number = []
class_testing_sample_number = []

for i in range(num_classes):
    class_training_sample_number.append(class_num_mat[i,0])
    class_validating_sample_number.append(class_num_mat[i,1])
    class_testing_sample_number.append(class_num_mat[i,2])

print("class_training_sample_number:" + str(class_training_sample_number) + str(sum(class_training_sample_number)))
print("class_validating_sample_number:" + str(class_validating_sample_number) + str(sum(class_validating_sample_number)))
print("class_testing_sample_number:" + str(class_testing_sample_number) + str(sum(class_testing_sample_number)))

# Testing
test(args.epochs)
utils.write_record(record_name, "acc:" + str(acc) + "\n")
utils.write_record(record_name, "macrof1:" + str(macrof1) + "\n")
utils.write_record(record_name, "cmA:" + str(cmA) + "\n")
utils.write_record(record_name, "auc:" + str(auc) + "\n")

# python ../main.py --gpu=2 --imbalance --dataset=cora --model=sage --umsample=danger_upsample --setting=recon --decoder=SmoothDecoder --loss_rec=homo_edge_loss

# python ../main.py --gpu=2 --imbalance --dataset=cora --model=sage --umsample=danger_upsample --load=sage_danger_upsample_SmoothDecoder_homo_edge_loss_recon_900_False_0.5 --setting=newG_cls --decoder=SmoothDecoder --loss_rec=homo_edge_loss
