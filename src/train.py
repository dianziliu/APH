import argparse
import os
import random
import sys
from time import time
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
import torch as t
import torch.optim as optim

from data_loader import GraphData
from model import APM
sys.path.append(".")
# os.chdir(os.path.dirname(__file__))

import numpy as np
from torchmetrics import MeanAbsoluteError 

# from torch_summary import model_summary 

MAELoss=MeanAbsoluteError().cuda()

def model_summary(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def gpu(batch, is_long=True, use_cuda=True):
    if is_long:
        batch = t.LongTensor(batch)
    else:
        batch = t.FloatTensor(batch)
    if use_cuda:
        batch = batch.cuda()
    return batch


def early_stopping(log_value, best_value, test_mse, stopping_step, flag_step=3):
    # early stopping strategy:
    global test_value_r
    global test_value_m
    if best_value is None or log_value <= best_value:
        stopping_step = 0
        best_value = log_value
        test_value_r = test_mse
    else:
        stopping_step += 1
    if stopping_step >= flag_step:
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop


def train_model(args):
    print('################Training model###################')
    stopping_step = 0
    best_cu = None
    for epoch in range(1, args.epochs + 1):
        t1 = time()
        train_mse = train()
        t2 = time()
        test_mse,test_mae = test(dtype='eval')
        print('epoch{: d}: train_time:{: .2f}s, train_mse:{: .4f}; test_mse:{: .4f},test_rmse:{: .4f}.test_mae:{:.4f}'.format(
            epoch, t2 - t1, train_mse, test_mse,np.math.sqrt(test_mse),test_mae))
        best_cu, stopping_step, should_stop = early_stopping(
            test_mse, best_cu, test_mse, stopping_step, flag_step=5)
        # if should_stop:
        #     break
    print('best mse: {:.4f}'.format(test_value_r))


def train():
    model.train()
    epoch_mse = 0.0
    
    for uid_batch, iid_batch, rate_batch, user_info, item_info in data_generator.generate_batch(data_type='train'):
        optimizer.zero_grad()
        uid_batch = gpu(uid_batch)
        iid_batch = gpu(iid_batch)
        rate_batch = gpu(rate_batch, is_long=False)
        u_nodes = [gpu(nodes) for nodes in user_info[0]]
        u_adj_ind = [gpu(adj) for adj in user_info[1]]
        u_adj_type = [gpu(val) for val in user_info[2]]
        i_nodes = [gpu(nodes) for nodes in item_info[0]]
        i_adj_ind = [gpu(adj) for adj in item_info[1]]
        i_adj_type = [gpu(val) for val in item_info[2]]
        preds = model(uid_batch, iid_batch, u_nodes, u_adj_ind,
                      u_adj_type, i_nodes, i_adj_ind, i_adj_type)
        loss = loss_function(preds, rate_batch)
        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_mse += loss.item()
    train_mse = epoch_mse / data_generator.train_length

    return train_mse


def test(dtype):
    model.eval()
    epoch_loss = 0.0
    epoch_mae = 0.0
    if dtype == 'eval':
        data_length = data_generator.eval_length
    elif dtype == 'test':
        data_length = data_generator.test_length
    else:
        sys.exit()
    n_batch=0
    for uid_batch, iid_batch, rate_batch, user_info, item_info in data_generator.generate_batch(data_type=dtype):
        uid_batch = gpu(uid_batch)
        iid_batch = gpu(iid_batch)
        rate_batch = gpu(rate_batch, is_long=False)
        u_nodes = [gpu(nodes) for nodes in user_info[0]]
        u_adj_ind = [gpu(adj) for adj in user_info[1]]
        u_adj_type = [gpu(val) for val in user_info[2]]
        i_nodes = [gpu(nodes) for nodes in item_info[0]]
        i_adj_ind = [gpu(adj) for adj in item_info[1]]
        i_adj_type = [gpu(val) for val in item_info[2]]
        preds = model(uid_batch, iid_batch, u_nodes, u_adj_ind,
                      u_adj_type, i_nodes, i_adj_ind, i_adj_type)
        loss = loss_function(preds, rate_batch)
        mae_loss=MAELoss(preds,rate_batch)
        epoch_loss += loss.item()
        epoch_mae += mae_loss.item()
        n_batch+=1
    final_mse = epoch_loss / data_length
    final_mae = epoch_mae/ n_batch
    model.train()
    return final_mse,final_mae


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str,
                        default='Musical_Instruments_5', help='type of dataset')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='size of batch of data')
    parser.add_argument('--num_layers', type=int,
                        default=2, help='number of GAT layer')
    parser.add_argument('--dim', type=int, default=16,
                        help='dim of user/item embedding')
    parser.add_argument('--word_dim', type=int, default=16,
                        help='dim of word embedding')
    parser.add_argument('--hidd_dim', type=int, default=8,
                        help='dim of graph node in TGAT')
    parser.add_argument('--factors', type=int, default=8,
                        help='dim of bilinear interaction')  # 这一项没有用到

    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--l2_re', type=float,
                        default=1e-4, help='weight of l2')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epoch')
    parser.add_argument('--dropout', type=float,
                        default=0, help='dropout rate')
    parser.add_argument('--alpha', type=float,
                        default=0.9, help='alpha of aspect pooling')
    args = parser.parse_known_args()[0]

    np.random.seed(2019)
    random.seed(2019)
    t.manual_seed(2019)
    t.cuda.manual_seed_all(2019)
    # 
    data_generator = GraphData(args)
    test_value_r = 0
    loss_function = t.nn.MSELoss(reduction='sum')
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['n_words'] = data_generator.word_num
    model = APM(config=config, args=args)
    print( model_summary(model))
    model = model.cuda()
   

    # optimizer = optim.Adam(model.parameters(), lr=args.lr,
    #                        weight_decay=args.l2_re)
    optimizer=optim.SGD(model.parameters(), lr=args.lr,
                        weight_decay=args.l2_re)                
    train_model(args)
