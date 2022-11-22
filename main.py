import argparse
import random
import pprint
import time
import sys
import os
import math

from datetime import timedelta
from workspace import Workspace

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from models import resnet
from models import resnet_ARM

from torch.utils.tensorboard import SummaryWriter
from utils import Logger, AverageMeter, accuracy, calc_metrics, RandomFiveCrop

from tqdm import tqdm

# centerloss module
from loss import SparseCenterLoss, TripletLoss
from dataset import *

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np

from apex import amp
import seaborn as sns

parser = argparse.ArgumentParser(description='DACL for FER in the wild')
parser.add_argument('--arch', type=str)
parser.add_argument('--lr', type=float)
parser.add_argument('--wd', type=float)
parser.add_argument('--bs', type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--alpha', type=float)
parser.add_argument('--lamb', type=float)
parser.add_argument('--pretrained', type=str) #, default='msceleb'
parser.add_argument('--deterministic', default=False, action='store_true')

best_epoch = 0

def main(cfg):

    global device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    if cfg['deterministic']:
        random.seed(cfg['seed'])
        torch.manual_seed(cfg['seed'])
        cudnn.deterministic = True
        cudnn.benchmark = False

    # Loading RAF-DB
    # -----------------
    print('[>] Loading dataset '.ljust(64, '-'))
    normalize = transforms.Normalize(mean=[0.5752, 0.4495, 0.4012], std=[0.2086, 0.1911, 0.1827])
    normalize_RAF_Aff = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize_FER = transforms.Normalize(mean=[0.485,], std=[0.229,])

    # train set
    
    train_set = datasets.ImageFolder(
        root=os.path.join(cfg['root_dir'], 'train'),
        transform=transforms.Compose([
            transforms.Resize(256),
            RandomFiveCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_RAF_Aff,
            transforms.RandomErasing(scale=(0.02, 0.1))
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg['batch_size'], shuffle=True,
        num_workers=cfg['workers'], pin_memory=True)
    
    # validation set
    
    val_loader = torch.utils.data.DataLoader(
        dataset=datasets.ImageFolder(
            root=os.path.join(cfg['root_dir'], 'test'),
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize_RAF_Aff
            ])
        ),
        batch_size=cfg['valid_batch_size'], shuffle=False,
        num_workers=cfg['workers'], pin_memory=True
    )

    print('[*] Loaded dataset!')

    # Create Model
    # ------------
    print('[>] Model '.ljust(64, '-'))
    if cfg['arch'] == 'resnet18':
        feat_size = 512 #121
        if not cfg['pretrained'] == '':
            model = resnet.resnet18(pretrained=cfg['pretrained'])
            model.fc = nn.Linear(feat_size, 7)
        else:
            print('[!] model is trained from scratch!')
            model = resnet.resnet18(num_classes=7, pretrained=cfg['pretrained'])
    else:
        raise NotImplementedError('only working with "resnet18" now! check cfg["arch"]')
    #model = torch.nn.DataParallel(model).to(device)
    model = model.to(device)
    print('[*] Model initialized!')

    #learnable loss parameters (center, triplet)
    beta = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
    gamma = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)

    # define loss function (criterion) and optimizer
    # ----------------------------------------------
    criterion = {
        'softmax': nn.CrossEntropyLoss().to(device),
        'center': SparseCenterLoss(7, feat_size).to(device),
        'triplet': TripletLoss(margin = 0.5).to(device)
    }
    optimizer = {
        'softmax': torch.optim.Adam(model.parameters(), #cfg['lr'],
                                   #momentum=cfg['momentum'],
                                   weight_decay=cfg['weight_decay']),
        'center': torch.optim.Adam(criterion['center'].parameters(), cfg['alpha']), #
        'weight': torch.optim.Adam([beta, gamma], 0.001) # learnable loss parameters (center, triplet)
    }
    
    # lr scheduler
    #scheduler_s = torch.optim.lr_scheduler.StepLR(optimizer['softmax'], step_size=20, gamma=0.1)
    scheduler_s = torch.optim.lr_scheduler.ExponentialLR(optimizer['softmax'], gamma=0.9)

    model, [optimizer['softmax'], optimizer['center'], optimizer['weight']] = amp.initialize(model, [optimizer['softmax'], optimizer['center'], optimizer['weight']], opt_level="O1", verbosity=0)

    # training and evaluation
    # -----------------------
    global best_valid
    best_valid = dict.fromkeys(['acc', 'rec', 'f1', 'aucpr', 'aucroc'], 0.0)

    print('[>] Begin Training '.ljust(64, '-'))

    epoch_list = []
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []

    global best_epoch

    for epoch in range(1, cfg['epochs'] + 1):

        start = time.time()

        # train for one epoch
        #train(train_loader, pos_loader, neg_loader, model, criterion, optimizer, epoch, cfg)
        train(train_loader, model, criterion, optimizer, epoch, cfg, train_loss_list, train_acc_list, epoch_list, beta, gamma)

        # validate for one epoch
        val_loss = validate(val_loader, model, criterion, epoch, cfg, valid_loss_list, valid_acc_list, beta, gamma)

        # progress
        end = time.time()
        progress = (
            f'[*] epoch time = {end - start:.2f}s | '
            f'lr = {optimizer["softmax"].param_groups[0]["lr"]}\n'
        )
        print(progress)

        # lr step
        scheduler_s.step()
        #scheduler_c.step()
        #scheduler_w.step()
    
    print("best valid acc is in epoch: ", best_epoch)

    #save loss & acc image
    plot_loss(train_loss_list, epoch_list, 'Train')
    plot_loss(valid_loss_list, epoch_list, 'Valid')
    plot_acc(train_acc_list, epoch_list, 'Train')
    plot_acc(valid_acc_list, epoch_list, 'Valid')
    
    # best valid info
    # ---------------
    """
    print('[>] Best Valid '.ljust(64, '-'))
    stat = (
        f'[+] acc={best_valid["acc"]:.4f}\n'
        f'[+] rec={best_valid["rec"]:.4f}\n'
        f'[+] f1={best_valid["f1"]:.4f}\n'
        f'[+] aucpr={best_valid["aucpr"]:.4f}\n'
        f'[+] aucroc={best_valid["aucroc"]:.4f}'
    )
    print(stat)
    """

#plot & save loss, acc png
def plot_loss(loss_list, epoch_list, mode):
    fig = plt.figure(figsize=(16, 8))
    plt.plot(epoch_list, loss_list, linewidth=3, label = mode + ' loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(mode + ' Loss')
    plt.grid()
    plt.legend()
    file_name = mode + ' Loss.png'
    plt.savefig(os.path.join(cfg['save_path'], file_name))
    print('Save file: ' + file_name)

def plot_acc(acc_list, epoch_list, mode):
    fig = plt.figure(figsize=(16, 8))
    plt.plot(epoch_list, acc_list, linewidth=3, label = mode + ' acc')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.title(mode + ' Acc')
    plt.grid()
    plt.legend()
    file_name = mode + ' Acc.png'
    plt.savefig(os.path.join(cfg['save_path'], file_name))
    print('Save file: ' + file_name)
    

def train(train_loader, model, criterion, optimizer, epoch, cfg, train_loss_list, train_acc_List, epoch_list, beta, gamma):
    losses = {
        'softmax': AverageMeter(),
        'center': AverageMeter(),
        'triplet': AverageMeter(),
        'total': AverageMeter()
    }
    accs = AverageMeter()
    y_pred, y_true, y_scores = [], [], []

    # switch to train mode
    model.train()

    with tqdm(total=int(len(train_loader.dataset) / cfg['batch_size'])) as pbar:
        for i, (images, target) in enumerate(train_loader):
            images = images.to(device)
            target = target.to(device)
            #return f_cen, f_trip, out, A_cen, A_tripPos, A_tripNeg
            # compute output
            feat, output, A = model(images)
            l_softmax = criterion['softmax'](output, target)
            l_center = criterion['center'](feat, A, target)
            l_triplet = criterion['triplet'](feat, target, A) 
            l_total = l_softmax + cfg['lamb'] * l_center + cfg['lamb2'] *l_triplet
            #l_total = l_softmax + torch.exp(beta).to(device) * l_center + torch.exp(gamma).to(device) * l_triplet # torch.exp(beta).to(device)

            # compute grads + opt step
            optimizer['softmax'].zero_grad()
            optimizer['center'].zero_grad()
            #optimizer['weight'].zero_grad()

            with amp.scale_loss(l_total, [optimizer['softmax'], optimizer['center']]) as scaled_loss:
                scaled_loss.backward()

            optimizer['softmax'].step()
            optimizer['center'].step()
            optimizer['weight'].step()                

            # measure accuracy and record loss
            acc, pred = accuracy(output, target)
            losses['softmax'].update(l_softmax.item(), images.size(0))
            losses['center'].update(l_center.item(), images.size(0))
            losses['triplet'].update(l_triplet.item(), images.size(0))
            losses['total'].update(l_total.item(), images.size(0))
            accs.update(acc.item(), images.size(0))

            # collect for metrics
            y_pred.append(pred)
            y_true.append(target)
            y_scores.append(output.data)

            #l_total.backward()

            # progressbar
            pbar.set_description(f'TRAINING [{epoch:03d}/{cfg["epochs"]}]')
            pbar.set_postfix({'L': losses["total"].avg,
                              'Ls': losses["softmax"].avg,
                              'Lsc': losses["center"].avg,
                              'Lt': losses["triplet"].avg,
                              'acc': accs.avg})
            pbar.update(1)
        #print("beta: ", beta, "EXPbeta: ", torch.exp(beta), "gamma: ", gamma, "EXPgamma: ", torch.exp(gamma))

    metrics = calc_metrics(y_pred, y_true, y_scores)
    progress = (
        f'[-] TRAIN [{epoch:03d}/{cfg["epochs"]}] | '
        f'L={losses["total"].avg:.4f} | '
        f'Ls={losses["softmax"].avg:.4f} | '
        f'Lsc={losses["center"].avg:.4f} | '
        f'Lt={losses["triplet"].avg:.4f} | '
        f'acc={accs.avg:.4f} | '
        f'rec={metrics["rec"]:.4f} | '
        f'f1={metrics["f1"]:.4f} | '
        f'aucpr={metrics["aucpr"]:.4f} | '
        f'aucroc={metrics["aucroc"]:.4f}'
    )
    print(progress)
    write_log(losses, accs.avg, metrics, epoch, tag='train')

    #save loss & acc png
    epoch_list.append(epoch)
    train_loss_list.append(losses['total'].avg)
    train_acc_List.append(accs.avg)
    

def plot_embedding(X, y, epoch, mode):
    plt.style.use('seaborn-paper')
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['figure.dpi'] = 300
    theme_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    fig1 = plt.figure(figsize=(10, 10))
    for i in range(len(X)): 
        #colors = COLOR_POOL[y[i]]
        colors = theme_colors[y[i]]
        plt.scatter(X[i, 0], X[i, 1], color=colors)
    fig_name = 'epoch_' + str(epoch) + '.png'
    if mode == 'tsne':
        path = os.path.join(cfg['save_path'], 'tsne')
        if not os.path.isdir(path):
            os.mkdir(path)
    else:
        path = os.path.join(cfg['save_path'], 'pca')
        if not os.path.isdir(path):
            os.mkdir(path)
    plt.savefig(path + '/' + fig_name)
    print('{} is saved'.format(fig_name))


def validate(valid_loader, model, criterion, epoch, cfg, valid_loss_list, valid_acc_List, beta, gamma):
    losses = {
        'softmax': AverageMeter(),
        'center': AverageMeter(),
        'triplet': AverageMeter(),
        'total': AverageMeter()
    }
    accs = AverageMeter()
    y_pred, y_true, y_scores = [], [], []

    # switch to evaluate mode
    model.eval()
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    pca = PCA(n_components=2, iterated_power=3000)
    features_list = []
    label_list = []

    with tqdm(total=int(len(valid_loader.dataset) / cfg['valid_batch_size'])) as pbar:
        with torch.no_grad():
            for i, (images, target) in enumerate(valid_loader):

                images = images.to(device)
                target = target.to(device)

                # compute output
                feat, output, A = model(images)
                l_softmax = criterion['softmax'](output, target)
                l_center = criterion['center'](feat, A, target)
                #l_triplet = criterion['triplet'](feat_trip, target, A_tripPos, A_tripNeg)
                l_total = l_softmax + cfg['lamb'] * l_center #+ cfg['lamb2'] * l_triplet
                #l_total = l_softmax + torch.exp(beta).to(device) * l_center #torch.exp(beta).to(device)

                #get tsne input
                features_list.append(feat[0])
                label_list.append(target[0])

                # measure accuracy and record loss
                acc, pred = accuracy(output, target)
                losses['softmax'].update(l_softmax.item(), images.size(0))
                losses['center'].update(l_center.item(), images.size(0))
                #losses['triplet'].update(l_triplet.item(), images.size(0))
                losses['total'].update(l_total.item(), images.size(0))
                accs.update(acc.item(), images.size(0))

                # collect for metrics
                y_pred.append(pred)
                y_true.append(target)
                y_scores.append(output.data)

                # progressbar
                pbar.set_description(f'VALIDATING [{epoch:03d}/{cfg["epochs"]}]')
                pbar.update(1)

    if epoch % 5 == 0:
        features_list = torch.stack(features_list)
        label_list = torch.stack(label_list)
        features_tsne = tsne.fit_transform(features_list.detach().cpu().numpy())
        plot_embedding(features_tsne, label_list.detach().cpu().numpy(), epoch, 'tsne')
        pca_result = pca.fit_transform(features_list.detach().cpu().numpy())
        plot_embedding(pca_result, label_list.detach().cpu().numpy(), epoch, 'pca')


    metrics = calc_metrics(y_pred, y_true, y_scores)
    progress = (
        f'[-] VALID [{epoch:03d}/{cfg["epochs"]}] | '
        f'L={losses["total"].avg:.4f} | '
        f'Ls={losses["softmax"].avg:.4f} | '
        f'Lsc={losses["center"].avg:.4f} | '
        f'acc={accs.avg:.4f} | '
        f'rec={metrics["rec"]:.4f} | '
        f'f1={metrics["f1"]:.4f} | '
        f'aucpr={metrics["aucpr"]:.4f} | '
        f'aucroc={metrics["aucroc"]:.4f}'
    )
    print(progress)

    #save loss & acc png
    valid_loss_list.append(losses['total'].avg)
    valid_acc_List.append(accs.avg)

    global best_epoch

    # save model checkpoints for best valid
    if accs.avg > best_valid['acc']:
        save_checkpoint(epoch, model, cfg, tag='best_valid_acc.pth')
        best_epoch = epoch
    if metrics['rec'] > best_valid['rec']:
        save_checkpoint(epoch, model, cfg, tag='best_valid_rec.pth')

    best_valid['acc'] = max(best_valid['acc'], accs.avg)
    best_valid['rec'] = max(best_valid['rec'], metrics['rec'])
    best_valid['f1'] = max(best_valid['f1'], metrics['f1'])
    best_valid['aucpr'] = max(best_valid['aucpr'], metrics['aucpr'])
    best_valid['aucroc'] = max(best_valid['aucroc'], metrics['aucroc'])
    write_log(losses, accs.avg, metrics, epoch, tag='valid')

    return losses["total"].avg # for 


def write_log(losses, acc, metrics, e, tag='set'):
    # tensorboard
    writer.add_scalar(f'L_softmax/{tag}', losses['softmax'].avg, e)
    writer.add_scalar(f'L_center/{tag}', losses['center'].avg, e)
    writer.add_scalar(f'L_total/{tag}', losses['total'].avg, e)
    writer.add_scalar(f'acc/{tag}', acc, e)
    writer.add_scalar(f'rec/{tag}', metrics['rec'], e)
    writer.add_scalar(f'f1/{tag}', metrics['f1'], e)
    writer.add_scalar(f'aucpr/{tag}', metrics['aucpr'], e)
    writer.add_scalar(f'aucroc/{tag}', metrics['aucroc'], e)


def save_checkpoint(epoch, model, cfg, tag):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, os.path.join(cfg['save_path'], tag))


if __name__ == '__main__':

    # setting up workspace
    args = parser.parse_args()
    workspace = Workspace(args)
    cfg = workspace.config

    # setting up writers
    global writer
    writer = SummaryWriter(cfg['save_path'])
    sys.stdout = Logger(os.path.join(cfg['save_path'], 'log.log'))

    # print finalized parameter config
    print('[>] Configuration '.ljust(64, '-'))
    pp = pprint.PrettyPrinter(indent=2)
    print(pp.pformat(cfg))

    # -----------------
    start = time.time()
    main(cfg)
    end = time.time()
    # -----------------

    print('The best valid acc is: ', best_valid)
    print('\n[*] Fini! '.ljust(64, '-'))
    print(f'[!] total time = {timedelta(seconds=end - start)}s')
    sys.stdout.flush()
