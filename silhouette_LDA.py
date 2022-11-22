import os
import torch
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from torch.utils.data import DataLoader
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import ast
from workspace import Workspace
import argparse
import torch.backends.cudnn as cudnn
from models import resnet
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 模型參數路徑
PATH_TO_WEIGHTS = 'D:\\Master_Thesis\\DACL\\logs\\FER_Lce\\best_valid_acc.pth'
Dataset = "FER-2013"
Method = "Lce"

parser = argparse.ArgumentParser(description='silhouette score')
parser.add_argument('--arch', type=str)
parser.add_argument('--lr', type=float)
parser.add_argument('--wd', type=float)
parser.add_argument('--bs', type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--alpha', type=float)
parser.add_argument('--lamb', type=float)
parser.add_argument('--pretrained', type=str) #, default='msceleb'
parser.add_argument('--deterministic', default=False, action='store_true')

def silhouette_plot(cfg):
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


    # Load model
    model = resnet.resnet18(pretrained=cfg['pretrained'])
    model.fc = nn.Linear(512, 7)
    model.load_state_dict(torch.load(PATH_TO_WEIGHTS)['model_state_dict'])
    model = model.to(device)
    model.eval()

    normalize_RAF_Aff = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

    features_list = []
    label_list = []
    
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            feat, output, A = model(images)

            feat = feat * A
            #get tsne input
            features_list.append(feat[0].data.cpu().numpy())
            label_list.append(target[0].data.cpu().numpy())

    silhouette_vals = silhouette_samples(features_list, label_list)

    LDA=LinearDiscriminantAnalysis(n_components=6)
    lda=LDA.fit(features_list, label_list) 
    Lda_score = lda.score(features_list, label_list)

    # Get the average silhouette score
    avg_score = np.mean(silhouette_vals)
    print(Dataset + ": " + Method)
    print('Average silhouette score = %f' % avg_score)
    print("scores of LDA: ", Lda_score)
    
if __name__ == '__main__':
    # setting up workspace
    args = parser.parse_args()
    workspace = Workspace(args)
    cfg = workspace.config

    silhouette_plot(cfg)
