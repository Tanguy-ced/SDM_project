import pandas as pd
import numpy as np
import torch 
import wandb
import os
import rasterio
from torch import nn
from tqdm import tqdm
from models import twoBranchCNN
from sklearn.metrics import f1_score, precision_score, recall_score
from data import GLC23Datasets , GLC23PatchesProviders 
from data.GLC23Datasets import RGBNIR_env_Dataset
from models import twoBranchCNN, Two_branch_Inception
from utils import seed_everything, sample_data
import matplotlib.pyplot as plt
from PIL import Image
import utils
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import MultiStepLR

data_path = 'data/full_data/'
presence_only_path = data_path+'Presence_only_occurrences/Presences_only_train.csv'
presence_absence_path = data_path+'Presence_Absence_surveys/Presences_Absences_train.csv'

seed_everything(42)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = (torch.cuda.device_count() if torch.cuda.is_available() else os.cpu_count())
print(f"Device used: {device}")

BATCH_SIZE = 32
LEARNING_RATE=0.0005
N_EPOCHS = 50
BIN_TRESH = 0.6
TRESH_FREQ_TRAIN = 500
TRESH_FREQ_VAL = 250
PATCH_ENV = 10
PATCH_RGB = 100
MOMENTUM = 0.9
GAMMA = 0.2
ITERATIONS = (5, 10, 15, 20, 25)
print(NUM_WORKERS)

if __name__ == "__main__":  
    presence_only_df = pd.read_csv(presence_only_path, sep=";", header='infer', low_memory=False)
    presence_absence_df = pd.read_csv(presence_absence_path, sep=";", header='infer', low_memory=False)
    presence_only_df = sample_data(presence_only_df, TRESH_FREQ_TRAIN,True)
    presence_absence_df = sample_data(presence_absence_df, TRESH_FREQ_VAL,True)
    print(presence_only_df.shape , presence_absence_df )
    presence_only_df, presence_absence_df = utils.keep_common(presence_only_df, presence_absence_df)
    n_spe = len(presence_absence_df["speciesId"].value_counts())
    print(f"There are {n_spe} species")

    presence_only_df_orleans = presence_only_df[(presence_only_df["lon"] > 6.5) & (presence_only_df["lon"] < 7) &
                                                (presence_only_df["lat"] > 45) & (presence_only_df["lat"] <55)]
    #print(presence_only_df_orleans.shape)

    presence_absence_df_orleans = presence_absence_df[(presence_absence_df["lon"] > 6.5) & (presence_absence_df["lon"] < 7) &
                                                (presence_absence_df["lat"] > 45) & (presence_absence_df["lat"] <55)]

    print(presence_only_df_orleans.shape, presence_absence_df_orleans.shape)
    
    
    

    print(f"The size of the two datasets are the followings: {presence_only_df_orleans.shape}, {presence_absence_df_orleans.shape}")

    train_dataset = RGBNIR_env_Dataset(presence_only_df_orleans, env_patch_size=PATCH_ENV, rgbnir_patch_size=PATCH_RGB)
    val_dataset = RGBNIR_env_Dataset(presence_absence_df_orleans, species=train_dataset.species, env_patch_size=PATCH_ENV, rgbnir_patch_size=PATCH_RGB)
    n_species = len(train_dataset.species)
    print(f"Training set: {len(train_dataset)} sites, {n_species} sites")
    print(f"Validation set: {len(val_dataset)} sites, {len(val_dataset.species)} sites")
    print(f"Number of species : {n_species}")

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS) ## Maybe use a transform argument 
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    model = twoBranchCNN(n_species).to(device)
    #optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)#, momentum=0.9)
    loss_fn = torch.nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    #scheduler = MultiStepLR(optimizer, milestones=list(ITERATIONS), gamma=GAMMA)


    run_name = 'Two_Branch_CNN_PO'
    if not os.path.exists(f"models/{run_name}"): 
        os.makedirs(f"models/{run_name}")
        

    
    run = wandb.init(project='SDM_project', name=run_name, resume='allow', config={
        'epochs': N_EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LEARNING_RATE, 'n_species': n_species, 
        'optimizer':'SGD', 'model': 'Two Branch CNN', 'loss': 'BCEWithLogitsLoss', 
        'env_patch_size': PATCH_ENV, 'rgb_patch_size':PATCH_RGB, 'train_data': 'PA',
        "treshold":BIN_TRESH
    }) #resume='never',
    print('before epochs')

    for epoch in range(0, N_EPOCHS):
            print(f"EPOCH {epoch}")
        
            model.train()
            train_loss_list = []
            for rgb, env, labels in tqdm(train_loader):
                y_pred = model(rgb.to(torch.float32).to(device),env.to(torch.float32).to(device))# env.to(torch.float32).to(device))
                print(y_pred.shape , labels.shape)
                loss = loss_fn(y_pred, labels.to(torch.float32).to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_list.append(loss.cpu().detach())
            avg_train_loss = np.array(train_loss_list).mean()
            print(f"\tTRAIN LOSS={avg_train_loss}")
            #scheduler.step(epoch + 1)
            model.eval()
            val_loss_list, val_precision_list, val_recall_list, val_f1_list = [], [], [], []
            for rgb, env, labels in tqdm(val_loader):
                print(y_pred.shape , labels.shape)
                y_pred = model(rgb.to(torch.float32).to(device),env.to(torch.float32).to(device))#, env.to(torch.float32).to(device))#, val=True)
                #print(f"Here are the predicted species {y_pred}")
                #print(f"Here are the real species {labels.to(torch.float32)}")
                val_loss = loss_fn(y_pred, labels.to(torch.float32).to(device)).cpu().detach()
                #print(f"here is the loss associated to this : {val_loss}")
                val_loss_list.append(val_loss)

                y_pred = torch.sigmoid(y_pred).cpu().detach().numpy()
                y_bin = np.where(y_pred > BIN_TRESH, 1, 0)
                val_precision_list.append(precision_score(labels.T, y_bin.T, average='macro', zero_division=0))
                val_recall_list.append(recall_score(labels.T, y_bin.T, average='macro', zero_division=0))
                val_f1_list.append(f1_score(labels.T, y_bin.T, average='macro', zero_division=0)) 

            avg_val_loss = np.array(val_loss_list).mean()
            avg_val_precision = np.array(val_precision_list).mean()
            avg_val_recall = np.array(val_recall_list).mean()
            avg_val_f1 = np.array(val_f1_list).mean()
            print(f"\tVALIDATION LOSS={avg_val_loss}\tPRECISION={avg_val_precision}, RECALL={avg_val_recall}, F1-SCORE={avg_val_f1} (threshold={BIN_TRESH})")
            
            wandb.log({
                "train_loss": avg_train_loss, "val_loss": avg_val_loss, 
                "val_prec": avg_val_precision, "val_recall": avg_val_recall, "val_f1": avg_val_f1,
                "learning_rate" : LEARNING_RATE
            })
            
