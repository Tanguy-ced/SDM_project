import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd


def sample_data(data ,min_freq,val = False):
    
    Grouped_name = data.groupby(by = ["speciesId"]).size()
    print(data.shape)
    data['GroupSize'] = data['speciesId'].transform(lambda x: Grouped_name[x])

    # Filter the DataFrame based on the group size criteria
    filtered_df = data[data['GroupSize'] >= min_freq]
    selected_species = filtered_df["speciesId"].value_counts()
    print(f"{filtered_df.shape},{selected_species}, {data.shape}")
    return filtered_df

def keep_common(presence_only,
                presence_absence):
    print("entering in filter")
    print(presence_absence.shape, presence_only.shape)
    #print(presence_absence["speciesId"].value_counts(),presence_only["speciesId"].value_counts())
    presence_only_df = presence_only[presence_only["speciesId"].isin((presence_absence["speciesId"]))]
    presence_absence_df = presence_absence[presence_absence["speciesId"].isin((presence_only["speciesId"]))]
    print("exiting of it ")
    print(presence_absence_df.shape, presence_only_df.shape)
    print(len(presence_only_df["speciesId"].value_counts()), len(presence_absence_df["speciesId"].value_counts()))
    
    return presence_only_df, presence_absence_df
    
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed) # Numpy seed also uses by Scikit Learn
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("papon")
    
def show_n_sample(train_loader, nb_sample)  :
    rgb_dl , env_dl ,train_label_dl = next(iter(train_loader))
    print(f"Here are the species labels for this location : {train_label_dl}")
    # Initialize lists to store data for the plots
    image_rgb_list = []
    image_nir_list = []
    bio_1_list = []
    bio_2_list = []
    bio_3_list = []
    bio_4_list = []

    # Loop through the data
    for j in range(nb_sample):
        rgb, env, labels = rgb_dl[j], env_dl[j], train_label_dl[j]
        image_rgb = rgb[:3].permute(1, 2, 0)
        image_nir = rgb[3]

        image_rgb_list.append(image_rgb)
        image_nir_list.append(image_nir)

        bio_1, bio_2, bio_3, bio_4 = env[0], env[1], env[2], env[3]

        bio_1_list.append(bio_1.numpy())
        bio_2_list.append(bio_2.numpy())
        bio_3_list.append(bio_3.numpy())
        bio_4_list.append(bio_4.numpy())

    # Create subplots and display all the plots at the end
    fig, ax = plt.subplots(nb_sample, 2)
    fig_2, ax_env = plt.subplots(2 * nb_sample, 2)

    for i in range(nb_sample):
        ax[i][0].imshow(image_rgb_list[i])
        ax[i][0].set_title(f"RGB_image of size {image_rgb_list[i].shape[0]} , {image_rgb_list[i].shape[1]}", fontsize=5)
        ax[i][1].imshow(image_nir_list[i])
        ax[i][1].set_title(f"Infrared image of size {image_nir_list[i].shape[0]} , {image_nir_list[i].shape[1]}", fontsize=5)

        ax_env[2 * i][0].imshow(bio_1_list[i])
        ax_env[2 * i][1].imshow(bio_2_list[i])
        ax_env[2 * i + 1][0].imshow(bio_3_list[i])
        ax_env[2 * i + 1][1].imshow(bio_4_list[i])

    # Show the plots
    print("Showing the samples")
    
    plt.show()

        
