import torch
import torch.nn as nn
from model import reverse_attention_net
from dataset import real_dataset, sim_dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from copy import deepcopy, copy
import numpy as np
from torch.nn import MSELoss, BCELoss, L1Loss
from torch.optim import Adam
from tqdm import tqdm
from itertools import chain
from random import randint
import matplotlib.pyplot as plt
import signal
from functools import partial
from itertools import cycle


def dataloader_init(x):
    # signal.signal(signal.SIGINT, signal.SIG_IGN)
    return


def sigINT_handler(model_RA, save_model, sig, frame):
    torch.save(model_RA.state_dict(), save_model + '/model_RA')
    torch.save(model_RA, save_model + '/model_RA.pth')
    torch.cuda.empty_cache()
    exit()


def test_save_imgs_process(sim_set, model_RA, CUDA, epoch, save_img):
    golden, defect, target = sim_set[randint(0, len(sim_set)-1)]

    plt.imshow(golden.squeeze(0), cmap=plt.cm.gray)
    plt.savefig(save_img + '/%d_golden.png' % epoch)
    plt.imshow(defect.squeeze(0), cmap=plt.cm.gray)
    plt.savefig(save_img + '/%d_defect.png' % epoch)
    plt.imshow(target.squeeze(0), cmap=plt.cm.gray)
    plt.savefig(save_img + '/%d_target.png' % epoch)

    golden, defect, target = golden.unsqueeze(0), defect.unsqueeze(0), target.unsqueeze(0)
    if CUDA is not None:
        golden = golden.cuda(CUDA)
        defect = defect.cuda(CUDA)

    model_RA.eval()
    orginal_predict, reverse_predict, combine_predict = model_RA(torch.cat((golden, defect), 1))
    plt.imshow(orginal_predict.detach().cpu().squeeze(0).squeeze(0).numpy(), cmap=plt.cm.gray)
    plt.savefig(save_img + '/%d_orginal_diff.png' % epoch)
    plt.imshow(reverse_predict.detach().cpu().squeeze(0).squeeze(0).numpy(), cmap=plt.cm.gray)
    plt.savefig(save_img + '/%d_reverse_diff.png' % epoch)
    plt.imshow(combine_predict.detach().cpu().squeeze(0).squeeze(0).numpy(), cmap=plt.cm.gray)
    plt.savefig(save_img + '/%d_combine_diff.png' % epoch)
    model_RA.train()


def train(dataset_root, save_model, save_img, save_log, weight_pth=None, max_epoch=0, CUDA=None, batch_size=8, num_workers=1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    sim_set = sim_dataset(dataset_root["sim_dataset_root"], transform)
    sim_dataloader = DataLoader(dataset=sim_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=dataloader_init)

    if weight_pth is None:
        model_RA = reverse_attention_net()

    else:
        model_RA = torch.load(weight_pth["ra_net"])

    if CUDA is not None:
        model_RA = model_RA.cuda(CUDA)

    criterion = MSELoss()
    optimizer = Adam(params=model_RA.parameters(), lr=2e-4)

    logger = SummaryWriter(save_log)
    model_RA.train()
    print("training")
    signal.signal(signal.SIGINT, partial(sigINT_handler, model_RA, save_model))
    for epoch in range(0, max_epoch):
        print("epoch: ", epoch)
        loss_RA = []

        for i, (golden, defect, target) in tqdm(enumerate(sim_dataloader), total=len(sim_dataloader)):
            if CUDA is not None:
                golden = golden.cuda(CUDA)
                defect = defect.cuda(CUDA)
                target = target.cuda(CUDA)

            orginal_predict, reverse_predict, combine_predict = model_RA(torch.cat((golden, defect), 1))
            loss = criterion(torch.cat((orginal_predict, reverse_predict, combine_predict), 1), torch.cat((target, target, target), 1))
            loss_RA.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("loss_RA: ", np.mean(loss_RA))
        logger.add_scalar("loss_RA", np.mean(loss_RA), global_step=epoch)
        logger.flush()

        test_save_imgs_process(sim_set, model_RA, CUDA, epoch, save_img)
        if (epoch+1) % 5 == 0:
            torch.save(model_RA, save_model + '/model_RA_' + str(epoch) + '.pth')

    torch.save(model_RA, save_model + '/model_RA.pth')


if __name__ == "__main__":
    weight_pth = {"ra_net": "../weight/model_RA.pth"}

    dataset_root = {"sim_dataset_root": "../dataset/real_sim"}

    train(dataset_root=dataset_root,
          save_model="../saves/save_model",
          save_img="../saves/save_img",
          save_log="../saves/save_log",
          #   weight_pth=weight_pth,
          max_epoch=100,
          CUDA=0)
