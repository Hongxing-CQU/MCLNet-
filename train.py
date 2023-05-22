import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
from utils import npmat2euler
import datetime
from model import MCLNet


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def train_one_epoch(args, net, train_loader, opt):
    net.train()
    R_list = []
    t_list = []
    R_pred_list = []
    t_pred_list = []
    euler_list = []

    cnt = 0
    train_total_loss = 0.

    for src, tgt, euler, T in tqdm(train_loader):
        src = src.cuda()
        tgt = tgt.cuda()
        R = T[:,:3,:3].cuda()
        t = T[:,:3, 3].squeeze(-1).cuda()
        euler = euler

        # forward
        R_pred, t_pred, loss = net(src, tgt, R, t)

        # backward
        opt.zero_grad()
        loss.backward()
        opt.step()

        cnt += 1
        train_total_loss += loss.item()

        R_list.append(R.detach().cpu().numpy())
        t_list.append(t.detach().cpu().numpy())
        R_pred_list.append(R_pred.detach().cpu().numpy())
        t_pred_list.append(t_pred.detach().cpu().numpy())
        euler_list.append(euler.numpy())

    R = np.concatenate(R_list, axis=0)  # B,3,3
    t = np.concatenate(t_list, axis=0)
    R_pred = np.concatenate(R_pred_list, axis=0)
    t_pred = np.concatenate(t_pred_list, axis=0)
    euler = np.concatenate(euler_list, axis=0)

    euler_pred = npmat2euler(R_pred)
    r_mse = np.mean((euler_pred - np.degrees(euler)) ** 2)
    r_rmse = np.sqrt(r_mse)
    r_mae = np.mean(np.abs(euler_pred - np.degrees(euler)))
    t_mse = np.mean((t - t_pred) ** 2)
    t_rmse = np.sqrt(t_mse)
    t_mae = np.mean(np.abs(t - t_pred))

    return train_total_loss / cnt, r_rmse, r_mae, t_rmse, t_mae


def test_one_epoch(args, net, test_loader):
    net.eval()

    R_list = []
    t_list = []
    R_pred_list = []
    t_pred_list = []
    euler_list = []

    cnt = 0
    test_total_loss = 0.

    for src, tgt, euler, T in tqdm(test_loader):
        src = src.cuda()
        tgt = tgt.cuda()
        R = T[:, :3, :3].cuda()
        t = T[:, :3, 3].squeeze(-1).cuda()

        R_pred, t_pred, loss = net(src, tgt)

        cnt += 1
        test_total_loss += loss

        R_list.append(R.detach().cpu().numpy())
        t_list.append(t.detach().cpu().numpy())
        R_pred_list.append(R_pred.detach().cpu().numpy())
        t_pred_list.append(t_pred.detach().cpu().numpy())
        euler_list.append(euler.numpy())

    #
    R = np.concatenate(R_list, axis=0)
    t = np.concatenate(t_list, axis=0)
    R_pred = np.concatenate(R_pred_list, axis=0)
    t_pred = np.concatenate(t_pred_list, axis=0)
    euler = np.concatenate(euler_list, axis=0)

    euler_pred = npmat2euler(R_pred)
    r_mse = np.mean((euler_pred - np.degrees(euler)) ** 2)
    r_rmse = np.sqrt(r_mse)
    r_mae = np.mean(np.abs(euler_pred - np.degrees(euler)))
    t_mse = np.mean((t - t_pred) ** 2)
    t_rmse = np.sqrt(t_mse)
    t_mae = np.mean(np.abs(t - t_pred))

    return test_total_loss / cnt, r_rmse, r_mae, t_rmse, t_mae


def train(args, net, train_loader, test_loader, textio, opt, scheduler):
    best_test_loss = 1e6
    best_test_R_RMSE = 1e6
    best_test_R_MAE = 1e6
    best_test_t_RMSE = 1e6
    best_test_t_MAE = 1e6

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(args, net, train_loader, opt)
        test_stats = test_one_epoch(args, net, test_loader)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            snap = {'epoch': epoch + 1,
                    'model': net.state_dict(),
                    'min_loss': best_test_loss,
                    'optimizer': opt.state_dict(),
                    'scheduler': scheduler.state_dict()}
            torch.save(snap, args.checkpoints + '/%s/models/model_snap.%d.t7' % (args.exp_name, epoch))

        if test_stats[1] < best_test_R_RMSE and test_stats[1] < best_test_t_RMSE:
            snap = {'epoch': epoch + 1,
                    'model': net.state_dict(),
                    'min_loss': best_test_loss,
                    'optimizer': opt.state_dict(),
                    'scheduler': scheduler.state_dict()}
            torch.save(snap, args.checkpoints + '/%s/models/best_model_snap.t7' % args.exp_name)
            best_test_R_RMSE = test_stats[1]
            best_test_R_MAE = test_stats[2]
            best_test_t_RMSE = test_stats[3]
            best_test_t_MAE = test_stats[4]

        # save log
        textio.cprint('=====  EPOCH %d  Time: %s=====' % (epoch + 1, str(datetime.datetime.now().year) + ' ' +
                          str(datetime.datetime.now().month) + ' ' +
                          str(datetime.datetime.now().day) + ' ' +
                          str(datetime.datetime.now().hour) + ' ' +
                          str(datetime.datetime.now().minute)))
        textio.cprint('TRAIN: loss: %f, rot_RMSE: %f, rot_MAE: %f, trans_RMSE: %f, trans_MAE: %f' % train_stats)
        textio.cprint('TEST: loss: %f, rot_RMSE: %f, rot_MAE: %f, trans_RMSE: %f, trans_MAE: %f'% test_stats)


def load_args():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    # train and test settings
    parser.add_argument('--exp_name', type=str,default='exp')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--checkpoints', default="checkpoints", type=str)
    # dataset
    parser.add_argument('--noise_type', default='crop', type=str)
    parser.add_argument('--num_points', default=1024, type=int)
    parser.add_argument('--rot_mag', default=45.0, type=float)
    parser.add_argument('--trans_mag', default=0.5, type=float)
    parser.add_argument('--partial_p_keep', default=[0.7, 0.7], type=list)
    parser.add_argument('--unseen', type=bool, default=False)
    parser.add_argument('--dataset_path', type=str)
    # model settings
    parser.add_argument('--N_hat', type=float, default=0.75)
    parser.add_argument('--K', type=float, default=0.5)
    parser.add_argument('--group_nums', type=int, default=30)
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument('--t', type=int, default=10)
    parser.add_argument('--distance_threshold', type=float, default=2e-6)
    parser.add_argument('--seed_threshold', type=float, default=0.05)

    args = parser.parse_args()
    return args


def main():
    args = load_args()
    textio = IOStream(args.checkpoints + '/' + args.exp_name + '/run.log')
    textio.cprint(str(args))


    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)
    if not os.path.exists(args.checkpoints + '/' + args.exp_name):
        os.makedirs(args.checkpoints + '/' + args.exp_name)
    if not os.path.exists(args.checkpoints + '/' + args.exp_name + '/' + 'models'):
        os.makedirs(args.checkpoints + '/' + args.exp_name + '/' + 'models')


    from get_dataloader import get_modelnet40_dataloader
    train_loader, test_loader = get_modelnet40_dataloader(
        noise_type=args.noise_type,
        root=args.modelnet40_dataset_path,
        rot_mag=args.rot_mag,
        trans_mag=args.trans_mag,
        num_points=args.num_points,
        partial_p_keep=args.partial_p_keep,
        unseen=args.unseen,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size
    )

    net = MCLNet(args).cuda()

    trainable = filter(lambda x: x.requires_grad, net.parameters())
    opt = optim.Adam(trainable, lr=0.0001, weight_decay=0.001)
    scheduler = CosineAnnealingLR(opt, T_max=30)

    if args.resume:
        assert os.path.isfile(args.resume)
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['schedulerx'])

    train(args, net, train_loader, test_loader, textio, opt, scheduler)


if __name__ == '__main__':
    main()
