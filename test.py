import torch
import torch.nn as nn
from dataset import test_dataset
from model import reverse_attention_net
from torch.utils.data import DataLoader
from torchvision import transforms
from copy import deepcopy, copy
import numpy as np
from tqdm import tqdm
from itertools import chain
from random import randint
import matplotlib.pyplot as plt
import signal
from functools import partial
from itertools import cycle


def test(dataset_root, save_test, weight_pth, CUDA=None, batch_size=1, num_workers=1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_set = test_dataset(dataset_root["test_dataset_root"], transform)
    dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # model_RA = reverse_attention_net()
    # model_RA.load_state_dict(torch.load('./model_RA'))
    model_RA = torch.load(weight_pth["ra_net"])
    if CUDA is None:
        model_RA = model_RA.cpu()
    print("testing")
    model_RA.eval()

    for i, (golden, defect) in tqdm(enumerate(dataloader), total=len(dataloader)):
        if CUDA is not None:
            golden = golden.cuda(CUDA)
            defect = defect.cuda(CUDA)
        orginal_predict, reverse_predict, combine_predict = model_RA(torch.cat((golden, defect), 1))

        plt.imshow(golden.cpu().squeeze(0).squeeze(0).numpy(), cmap=plt.cm.gray)
        plt.savefig(save_test + '/%d_golden.png' % i)
        plt.imshow(defect.cpu().squeeze(0).squeeze(0).numpy(), cmap=plt.cm.gray)
        plt.savefig(save_test + '/%d_defect.png' % i)
        plt.imshow(orginal_predict.detach().cpu().squeeze(0).squeeze(0).numpy(), cmap=plt.cm.gray)
        plt.savefig(save_test + '/%d_orginal_diff.png' % i)
        plt.imshow(reverse_predict.detach().cpu().squeeze(0).squeeze(0).numpy(), cmap=plt.cm.gray)
        plt.savefig(save_test + '/%d_reverse_diff.png' % i)
        plt.imshow(combine_predict.detach().cpu().squeeze(0).squeeze(0).numpy(), cmap=plt.cm.gray)
        plt.savefig(save_test + '/%d_combine_diff.png' % i)


if __name__ == "__main__":
    weight_pth = {"ra_net": "../weight/model_RA_14.pth"}

    dataset_root = {"test_dataset_root": "../dataset/try"}  # try

    test(dataset_root=dataset_root,
         save_test="../saves/save_test",
         weight_pth=weight_pth,
         CUDA=None)
