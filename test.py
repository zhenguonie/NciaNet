import os
import time

import argparse
import random
import numpy as np
import paddle
import paddle.nn.functional as F
from pgl.utils.data import Dataloader
from dataset import ComplexDataset, collate_fn
from model import NciaNet
from NciaNet.utils import rmse, mae, sd, pearson
from tqdm import tqdm


paddle.seed(123)

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

@paddle.no_grad()
def evaluate(model, loader):
    model.eval()
    y_hat_list = []
    y_list = []
    result1 = ''
    result3 = ''
    n=1
    i=0
    edges_num= []
    for batch_data in loader:
        a2a_g,  Hbond_graphs, hyb_graphs, van_graphs,PAI_graphs,y = batch_data
        y_hat = model( a2a_g, Hbond_graphs, hyb_graphs,van_graphs,PAI_graphs,y)
        if Hbond_graphs.edge_feat["coe"]==1:
            A= Hbond_graphs.num_edges
            A= paddle.tolist(A)
            edges_num.append(A[0])
        result = '%d predict:%f.\n'% (n,y_hat.tolist()[0][0])
        result2 = '%d label:%f.\n'% (n,y.tolist()[0][0])
        n =n +1
        y_hat_list += y_hat.tolist()
        y_list += y.tolist()
        result1 += result
        result3 += result2
    f = open(os.path.join(args.model_dir, 'predict.txt'), 'w')
    f.write(result1)
    f = open(os.path.join(args.model_dir, 'label.txt'), 'w')
    f.write(result3)
    f.close
    y_hat = np.array(y_hat_list).reshape(-1,)
    y = np.array(y_list).reshape(-1,)
    
    return rmse(y, y_hat), mae(y, y_hat), sd(y, y_hat), pearson(y, y_hat)


def train(args, model, trn_loader, tst_loader, val_loader):
    epoch_step = len(trn_loader)
    boundaries = [i for i in range(args.dec_step, args.epochs*epoch_step, args.dec_step)]
    values = [args.lr * args.lr_dec_rate ** i for i in range(0, len(boundaries) + 1)]
    scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=boundaries, values=values, verbose=False)
    optim = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())

    rmse_val_best, res_tst_best = 1e9, ''
    running_log = ''
    print('Start training model...')
    standard = 1000000
    for epoch in range(1, args.epochs + 1):
        rmse_tst, mae_tst, sd_tst, r_tst = evaluate(model, tst_loader)
#        end_val = time.time()
        log = '-----------------------------------------------------------------------\n'
        log += 'Test - RMSE: %.6f, MAE: %.6f, SD: %.6f, R: %.6f.\n' % (rmse_tst, mae_tst, sd_tst, r_tst)
        print(log)
        running_log += log
        f = open(os.path.join(args.model_dir, 'log.txt'), 'w')
        f.write(running_log)
        f.close()

    f = open(os.path.join(args.model_dir, 'log.txt'), 'w')
    f.write(running_log + res_tst_best)
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
#    parser.add_argument('--data_dir', type=str, default=r'C:/data/')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--dataset', type=str, default='pdbbind2016')
    parser.add_argument('--model_dir', type=str, default='./output/NciaNet')
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument("--save_model", action="store_true", default=True)

    parser.add_argument("--lambda_", type=float, default=1.75)
    parser.add_argument("--feat_drop", type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=1                                                                  )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--lr_dec_rate", type=float, default=0.5)
    parser.add_argument("--dec_step", type=int, default=8000)
    parser.add_argument('--stop_epoch', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)

    parser.add_argument("--num_convs", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--infeat_dim", type=int, default=36)
    parser.add_argument("--dense_dims", type=str, default='128*5,128*3,128*2,128')

    parser.add_argument('--num_heads', type=int, default=3)
    parser.add_argument('--cut_dist', type=float, default=5)
    parser.add_argument('--num_angle', type=int, default=6)
    parser.add_argument('--merge_b2b', type=str, default='cat')
    parser.add_argument('--merge_a2b', type=str, default='cat')#我的
    parser.add_argument('--merge_b2a', type=str, default='mean')

    args = parser.parse_args()
    args.activation = F.relu
    args.dense_dims = [eval(dim) for dim in args.dense_dims.split(',')]
    if args.seed:
        setup_seed(args.seed)

    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    
    if int(args.cuda) == -1:
        paddle.set_device('cpu')
    else:
        paddle.set_device('gpu:%s' % args.cuda)
    trn_complex = ComplexDataset(args.data_dir, "%s_train" % args.dataset, args.cut_dist)
    tst_complex = ComplexDataset(args.data_dir, "%s_test" % args.dataset, args.cut_dist)
    val_complex = ComplexDataset(args.data_dir, "%s_val" % args.dataset, args.cut_dist)
    trn_loader = Dataloader(trn_complex, args.batch_size, shuffle=True, num_workers=1, collate_fn=collate_fn)#别忘了改
    tst_loader = Dataloader(tst_complex, args.batch_size, shuffle=False,  num_workers=1, collate_fn=collate_fn)
    val_loader = Dataloader(val_complex, args.batch_size, shuffle=False, num_workers=1, collate_fn=collate_fn)


    model = NciaNet(args)
    train(args, model, trn_loader, tst_loader, val_loader)