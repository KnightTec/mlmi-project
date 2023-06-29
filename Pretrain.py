import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from Contrastiveloss import ContrastiveLoss
from Dataset_Pretrain import EEGDataset
from torchvision import models
from Augmentation import Augment
import matplotlib.pyplot as plt

def pretrain(dataloader, model, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for X, y in dataloader:
        trans1, res, ll = Augment(X).sselct()
        X1 = trans1
        trans2, res2, ll = Augment(X, res).sselct()
        X2 = trans2
        y_pred1 = model(X1)
        y_pred2 = model(X2)
        c_loss = criterion(y_pred1, y_pred2)
        optimizer.zero_grad()
        c_loss.backward()
        optimizer.step()
        print("-------------------------------")
        print("In Pretraining,  Contrastive Loss for this Batch: ", c_loss)
        print("-------------------------------")
        running_loss += c_loss.item()

    nb_batches = len(dataloader)
    running_loss /= nb_batches
    print(f"PreTrain loss for epoch : {running_loss:>8f}")
    return running_loss


def change_layers(model):
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(512, 512, bias=True)
    return model

def Draw_Contrastive_Loss(epoch, loss_list):
    plt.figure()
    plt.plot(epoch, loss_list, color='g')
    plt.savefig('Contrastive_Loss.png')
    plt.show()

def early_stopping(min_loss,loss_list):
    if (loss_list[-1] > min_loss
            and loss_list[-2] > min_loss
            and loss_list[-3] > min_loss
            and loss_list[-4] > min_loss
            and loss_list[-5] > min_loss
            and loss_list[-6] > min_loss
            and loss_list[-7] > min_loss
            and loss_list[-8] > min_loss
            and loss_list[-9] > min_loss
    ):
        return True

if __name__ == "__main__":

    dataset = EEGDataset(augment=True)
    print("Pretrain dataset length", len(dataset))
    pretrain_dataloader = DataLoader(dataset, batch_size=10, shuffle=True, drop_last=True)

    model = models.resnet18(pretrained=True)
    model = change_layers(model)
    model_Pretrain = nn.Sequential(
        model,
        nn.Linear(512, 4, bias=True)
    )

    optimizer = torch.optim.Adam(model_Pretrain.parameters(), lr=3e-2, weight_decay=1e-4)
    contrastive = ContrastiveLoss(batch_size=10, temperature=0.07, verbose=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    epochs = []
    contrastive_loss=[]
    min_loss = np.inf


    for t in range(200):

        pretrain_loss=pretrain(pretrain_dataloader, model_Pretrain, contrastive,optimizer)
        epochs.append(t)
        contrastive_loss.append(pretrain_loss)

        if min_loss > pretrain_loss:
            print("Contrastive loss Decreased!Saving The Model...")
            min_loss = pretrain_loss
            torch.save(model.state_dict(), 'Pretrain.pth')
            print("Model updated!")

        if t > 20 :
            if early_stopping(min_loss, contrastive_loss):
                torch.save(contrastive_loss, 'Contrastive_loss.pt')
                print("We are at epoch:", t)
                break

        torch.save(contrastive_loss, 'Contrastive_loss.pt')

    Draw_Contrastive_Loss(epochs, contrastive_loss)