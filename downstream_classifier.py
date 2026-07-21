#!/usr/bin/env python
# coding: utf-8

# In[2]:
import numpy as np
import random
import re
import sklearn
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputClassifier
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from sklearn.ensemble import RandomForestClassifier
import os.path as osp
import sys
from functools import partial
import argparse 
import warnings
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

random.seed(4)
torch.manual_seed(4)
np.random.seed(4)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)

parser.add_argument('--emb_file', type=str)#embeddings File path, e.g.: all_emb_supbs560selfsup_bs560EUOS_4CJ_multisupcon_BMT_R50_1-5_pixupd_real_best.csv
parser.add_argument('--aoi', type=str)#annotation of interest; label source ["bmoa_upd","btarget","mesh"]
parser.add_argument('--kf', type=int) # Cross validation; integer number of the k-fold (0-4)


parser.add_argument('--b22', type=bool, default=False)
parser.add_argument('--b36', type=bool, default=False)
parser.add_argument('--dino', type=bool, default=False)
parser.add_argument('--euos', type=bool, default=False)#
parser.add_argument('--cellprof', type=bool, default=False)
parser.add_argument('--best_choice', type=str, default="acc")# or precision



args = parser.parse_args()

device_id = args.device
kf = args.kf

b22=args.b22
b36=args.b36
dino=args.dino
euos=args.euos
cellprof=args.cellprof
best_choice=args.best_choice
aoiX=str(args.aoi)




def rowN_remover(x):
    return x.split("_")[0]

emb_type2=False
emb_file=str(args.emb_file)
if b22:
    mesh_ohe_matrix=pd.read_csv("bbbc22//b22mesh_ohe_matrix.csv",index_col=0)
    bmoa_ohe_matrix=pd.read_csv("bbbc22/b22_bmoa_ohematrix_ID2Name.csv",index_col=0,delimiter=";")
    btarget_ohe_matrix=pd.read_csv("bbbc22/b22_btarget_ohematrix_ID2Name.csv",index_col=0,delimiter=";")
    emb_type2=True
    if dino:
        emb_file="embeddings/dino/emb_bs192_b22_5Ctiff_DINO+_nobrightcont.csv"
        emb_type2=False
    if cellprof:
            emb_file="embeddings/cellprof/full_cellprofb22_nucnanfilt.csv"
            emb_type2=False
if b36:
    mesh_ohe_matrix=pd.read_csv("b36//b36mesh_ohe_matrix.csv",index_col=0,delimiter=";")
    pnd_ohe_matrix=pd.read_csv("b36/b36pnd_ohe_matrix.csv",index_col=0,delimiter=";")
    bmoa_ohe_matrix=pd.read_csv("b36/bmoa_ohe_matrix36.csv",index_col=0,delimiter=";")
    btarget_ohe_matrix=pd.read_csv("b36/btarget_ohe_matrix36.csv",index_col=0,delimiter=";")
    if dino:
        emb_file="embeddings/emb_bs192_36_5CJ_DINO_OGnowarpII_final.csv"
if euos:
    mesh_ohe_matrix=pd.read_csv("euos/mesh_ohe_matrix_euos.csv",index_col=0,delimiter=";")
    bmoa_ohe_matrix=pd.read_csv("euos/bmoa_ohe_matrix_euos.csv",index_col=0,delimiter=";")
    btarget_ohe_matrix=pd.read_csv("euos/btarget_ohe_matrix_euos.csv",index_col=0,delimiter=";")
    if cellprof:
        dfcpX=pd.read_parquet(emb_file)

res_dict={}
print(emb_file)
print(aoiX)
embis=[emb_file]
for emb_file in embis:
    delimito=";"
    if emb_type2:
            test_dfX0=pd.read_csv(emb_file,index_col=0,delimiter=delimito)
    else:
        if cellprof:
            test_dfX0=pd.read_csv(emb_file,index_col=0,delimiter=";")
        else:
            test_dfX0=pd.read_csv(emb_file,header=None,index_col=0,delimiter=";")
    print(len(test_dfX0))
    idkey="Molecules"
    missing=[]

    test_dfX0_=test_dfX0.copy()
    aoi_list=[aoiX]
    for aoi in aoi_list:
        test_dfX0=test_dfX0_.copy()
        if aoi=="mesh":
            moi=mesh_ohe_matrix.copy()
        if aoi=="bmoa_upd":
            moi=bmoa_ohe_matrix.copy()
        if aoi=="btarget":
            moi=btarget_ohe_matrix.copy()
        if cellprof:
            test_dfX0.index=test_dfX0.Image_Metadata_SOURCE_COMPOUND_NAME    
        
        if emb_type2:
            new_index=[rowN_remover(i) for i in test_dfX0.index]
            test_dfX0["Molecules"]=new_index#dft.index
        else:
            test_dfX0["Molecules"]=test_dfX0.index
        groups = [df for _, df in test_dfX0.groupby('Molecules')]
        random.shuffle(groups)
        test_dfX0=pd.concat(groups).reset_index(drop=True)
        t0=time.time()
        print((test_dfX0.Molecules.isin(moi.index)).sum())
        
        skf = GroupKFold(n_splits=5)
        dX=test_dfX0.copy()
        start_feat,end_feat=2,-1
        if emb_type2:
            start_feat,end_feat=0,224
        

        results=[]
        #print(emb_file)
        #print(aoi)
        FullReporter={}
        #moi=mesh_ohe_matrix.copy()
        test_dfX_2=test_dfX0.copy()
        test_dfX_2=test_dfX_2[test_dfX_2[idkey].isin(moi.index)]
        moi=moi[moi.index.isin(test_dfX_2[idkey])]
        print(len(test_dfX_2[idkey].unique()),len(test_dfX_2.index))
        dX=test_dfX_2.copy()
        if cellprof:
            start_feat,end_feat=10,10+824
            scaler = StandardScaler()
            normalized_dX= scaler.fit_transform(dX.iloc[:,start_feat:end_feat])
            dX.iloc[:, start_feat:end_feat] = normalized_dX

        for i, (train_index, test_index) in enumerate(skf.split(dX, groups=dX["Molecules"])):
            
            trainX=dX.iloc[train_index].copy()
            skX2=trainX.iloc[:,start_feat:end_feat]
            skY2=moi.loc[trainX.Molecules]

            testX=dX.iloc[test_index].copy()
            if i!=kf:
                continue
 
            

            class CustomDataset(Dataset):
                def __init__(self, dataframe, transform=None):
                    self.dataframe = dataframe
                    self.transform = transform

                def __len__(self):
                    return len(self.dataframe)

                def __getitem__(self, idx):
                    sample = self.dataframe.iloc[idx]

                    # Extract features and label from the sample
                    features = sample.iloc[start_feat:end_feat].values.astype(float)  # Adjust 'your_feature_column_name'
                    label = moi.loc[sample.Molecules].values  # Adjust 'your_label_column_name'

                    # Apply transformations if specified
                    if self.transform:
                        features,label = self.transform(features,label)

                    return features, label


            
            class ToTensor(object):
                def __call__(self, features, label):
                    # Convert features and label to PyTorch tensors
                    features = torch.tensor(features, dtype=torch.float32)
                    label = torch.tensor(label, dtype=torch.float32)  # Adjust dtype if needed

                    return features, label

            
            transform = ToTensor()
            train_dataset = CustomDataset(trainX,transform=transform)
            test_dataset = CustomDataset(testX,transform=transform)
            sample_index = 0
            features, label = train_dataset[sample_index]
            bs = 32
            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
            test_loader = DataLoader(test_dataset, shuffle=False)
            
            device = torch.device('cuda:'+str(device_id) if torch.cuda.is_available() else 'cpu')
            num_feat=features.shape[0]
            num_classes=label.shape[0]
            class Net(nn.Module):
                def __init__(self, input_size, hidden_size1, hidden_size2, num_classes, dropout_prob=0.5):
                    super(Net, self).__init__()

                    self.layer1 = nn.Sequential(
                        nn.Linear(input_size, hidden_size1),
                        nn.ReLU(),
                        nn.Dropout(dropout_prob)
                    )
                    self.layer2 = nn.Sequential(
                        nn.Linear(hidden_size1, hidden_size2),
                        nn.ReLU(),
                        nn.Dropout(dropout_prob)
                    )
                    self.classification_layer = nn.Linear(hidden_size2, num_classes)

                def forward(self, x):
                    x = self.layer1(x)
                    x = self.layer2(x)
                    x = self.classification_layer(x)
                    return x

            
            input_size = num_feat  
            hidden_size1 = 512
            hidden_size2 = 256
            model = Net(input_size, hidden_size1, hidden_size2, num_classes)

            
            save,save_all_ep=True,False
            multi_label=True
            full_train=False
            if multi_label:
                multi_string="multi"
            else:
                multi_string=""
            if full_train:
                split="full"
            else:
                split="split"
            if cellprof:
                PATHx_save= "models/b22/model_dsnn_myriad{0}_".format(str(i))+aoi+"_"+str(bs)+emb_file.split("/")[-1].split(".csv")[0]
            else:
                if best_choice=="precision":
                    PATHx_save="models/myriad/model_dsnn_X3myriad{0}_".format(str(i))+aoi+"_"+str(bs)+emb_file.split("emb_")[1]+"_precision"
                if best_choice=="acc":#Multi-label accuracy:
                    PATHx_save="models/myriad/model_dsnn_X3myriad{0}_".format(str(i))+aoi+"_"+str(bs)+emb_file.split("emb_")[1]
            outfile="embeddings/emb"+PATHx_save.split("/")[-1][5:-4]+".csv"
            logfile="logs/log_"+str(i)+"_"+PATHx_save.split("/")[-1][5:-4]

            optimizer = torch.optim.Adam(model.parameters(), lr=10**-2.5)#10**-2.5
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.7, patience=5,min_lr=10**-5)
            criterion = torch.nn.CrossEntropyLoss()
            model.to(device)
            def test(loader=test_loader,debug=False):
                model.eval()
                prec, rec,f1s=[],[],[]
                matches=[]
                praucs=[]
                accs=[]
                error = 0
                error_acc = 0
                threshold = 0.5
                with torch.no_grad():
                    for data in loader:
                        #data = [im for im in data.values()]#data = [im.cuda(device=device,non_blocking=True) for im in data.values()]
                        features=data[0].cuda(device=device,non_blocking=True).float()
                        #labels=data[3].cuda(device=device,non_blocking=True)#.int()
                        preds=model(features).cpu()#.clip(0, 1).round().int()#.cuda()
                        one_hot_encoding=data[1].cpu()#.float()
                        #nan_mask = torch.isnan(one_hot_encoding)

                        flat_y_true = one_hot_encoding.view(-1)

                        valid_indices = ~torch.isnan(flat_y_true)
                        flat_y_true_valid = flat_y_true[valid_indices]

                        #preds[nan_mask] = float('nan') 
                        flat_preds = preds.view(-1)
                        flat_pr_valid = flat_preds[valid_indices]
                        flat_pr_valid_float=flat_pr_valid.clone()
                        flat_pr_valid=flat_pr_valid.clip(0, 1).round().int()
                        correct_predictions = torch.sum(flat_pr_valid == flat_y_true_valid)#acc=accuracy_score(preds, target)
                        subset_accuracy = correct_predictions.item() / num_classes
                        error+=subset_accuracy
                        maskP = (flat_pr_valid == 1) & (flat_y_true_valid == 1)

                        # Use boolean indexing to get the values where both are equal to 1
                        matching_values = flat_pr_valid[maskP]

                        # Calculate the sum of matching values
                        sum_of_matching_values = torch.sum(matching_values)
                        matches.append(sum_of_matching_values)
                        
                        #soft = nn.Softmax(dim=0)
                        #flat_pred_prob=soft(flat_pr_valid_float)
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            p,r,f1,sup=sklearn.metrics.precision_recall_fscore_support(flat_y_true_valid,flat_pr_valid,average="macro",
                                                                                       zero_division=0)
                            prec.append(p), rec.append(r),f1s.append(f1)
                            prauc=sklearn.metrics.average_precision_score(flat_y_true_valid,flat_pr_valid_float.clip(0, 1),average="micro")
                        praucs.append(prauc)
                        accs.append(np.mean(np.all(flat_y_true_valid.numpy() == flat_pr_valid.numpy(), axis=0)))
                        if debug:
                            break
                #return error / len(loader),np.mean(prec),np.mean(rec),np.mean(f1s),np.mean(praucs),np.mean(matches),np.mean(accs)
                return np.mean(prec),np.mean(rec),np.mean(f1s),np.mean(praucs),np.mean(accs)
            
            
            #test(debug=True)     
            def train(loader=train_loader,debug=False):
                model.train()
                prec, rec,f1s=[],[],[]
                error = 0
                loss_all = 0
                threshold = 0.5
                matches=[]
                praucs=[]
                for data in train_loader:
                    #data = [im for im in data.values()]#data = [im.cuda(device=device,non_blocking=True) for im in data.values()]
                    features=data[0].cuda(device=device,non_blocking=True).float()
                    optimizer.zero_grad()
                    #labels=data[3].cuda(device=device,non_blocking=True)#.int()
                    preds=model(features)#.cuda()
                    one_hot_encoding=data[1].to(device)#.float()
                    #nan_mask = torch.isnan(one_hot_encoding)


                    flat_y_true = one_hot_encoding.view(-1)

                    valid_indices = ~torch.isnan(flat_y_true)
                    flat_y_true_valid = flat_y_true[valid_indices]

                    #preds[nan_mask] = float('nan') 
                    flat_preds = preds.view(-1)
                    flat_pr_valid = flat_preds[valid_indices]
                    loss = criterion(flat_pr_valid.float(), flat_y_true_valid.float())
                    loss.backward()
                    loss_all += loss.item() #* data.num_graphs
                    optimizer.step()
                    correct_predictions = torch.sum(flat_pr_valid == flat_y_true_valid)#acc=accuracy_score(preds, target)
                    subset_accuracy = correct_predictions.item() / num_classes
                    error+=subset_accuracy

                    flat_pr_valid_float=flat_pr_valid.clone().float().cpu().detach()
                    flat_pr_valid=flat_pr_valid.clip(0, 1).round().int().cpu()
                    flat_y_true_valid=flat_y_true_valid.cpu()
                    maskP = (flat_pr_valid == 1) & (flat_y_true_valid == 1)

                    # Use boolean indexing to get the values where both are equal to 1
                    matching_values = flat_pr_valid[maskP]
                    #soft = nn.Softmax(dim=0)
                    ##flat_pred_prob=soft(flat_pr_valid_float)
                    # Calculate the sum of matching values
                    sum_of_matching_values = torch.sum(matching_values)
                    matches.append(sum_of_matching_values)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        p,r,f1,sup=sklearn.metrics.precision_recall_fscore_support(flat_y_true_valid,flat_pr_valid,average="macro",zero_division=0)
                        prec.append(p), rec.append(r),f1s.append(f1)
                        prauc=sklearn.metrics.average_precision_score(flat_y_true_valid,flat_pr_valid_float.clip(0, 1),average="micro")
                    praucs.append(prauc)
                    if debug:
                        break
                return loss_all / len(loader),np.mean(prec),np.mean(rec),np.mean(f1s),np.mean(praucs),np.mean(matches)

            def calc_precision(loader,pfilename="precision"+logfile.split("/")[-1][3:]+".csv"):
                model.eval()
                prec, rec, f1s = [], [], []
                praucs, accs = [], []
                threshold = 0.5
                true_positives = np.zeros(num_classes)
                false_positives = np.zeros(num_classes)
                total_preds=np.zeros(num_classes)
                with torch.no_grad():
                    for data in loader:
                        features = data[0].cuda(device=device, non_blocking=True).float()
                        labels = data[1].cpu().numpy().flatten()  # True labels
            
                        preds = model(features).cpu().numpy()
                        preds_binary = (preds >= threshold).astype(int).flatten()
            
                        # Update TP and FP counts for each class
                        for i in range(num_classes):
                            if preds_binary[i] < 1:
                                continue
                            if preds_binary[i] == 1 and labels[i] == 1:
                                true_positives[i] += 1  # Increment TP
                            elif preds_binary[i] == 1 and labels[i] == 0:
                                false_positives[i] += 1  # Increment FP
                        
                    # Calculate precision for each class
                    precisions = np.zeros(num_classes)
                    for i in range(num_classes):
                        totals=true_positives[i] + false_positives[i]
                        if totals > 0:
                            precisions[i] = true_positives[i] / (true_positives[i] + false_positives[i]) 
                            total_preds[i]=totals
                        else:
                            precisions[i] = np.nan  # Handle division by zero
                            total_preds[i]=0
                    precision_df = pd.DataFrame(precisions, columns=['Precision'], index=[moi.columns])#.transpose()
                    precision_df["Support"]=total_preds
                    precision_df.sort_values(by=["Precision","Support"],ascending=False,inplace=True)
                    precision_df.to_csv("precisions/"+pfilename,sep=";")
                    return np.nanmean(precisions)
            
            best_score = None
            
            tr_mae,timo=[],[]
            ep=100
            debug=False
            if os.path.exists(logfile):
                print("Logfile exists already!")
                model.load_state_dict(torch.load(PATHx_save[:-3]+"_best"+PATHx_save[-3:]))  
                model.to(device);  
                print("Recalculate precision anyway")
                calc_precision(test_loader,pfilename="best_precision"+logfile.split("/")[-1][3:]+".csv")# if precision as best_choce there should be 2x precision in the name
                break
            with open(logfile,"w+") as g:
                g.write("")            
            print(logfile)
            for epoch in range(1,ep):
                loss=0
                start=time.time()
                loss,train_precision,train_recall,train_f1score,train_prauc,train_pos_match=train()
                pp,rr,f1,prauc,accu=test(test_loader)
                prec=calc_precision(test_loader)
                if save:
                  if save_all_ep:
                      PATHx_save=PATHx_save[:-3]+str(epoch)+PATHx_save[-3:]
                  torch.save(model.state_dict(), PATHx_save)
                if best_choice=="precision":
                    best_metric=prec
                if best_choice=="acc":
                    best_metric=accu

                if best_score is None or best_metric > best_score:
                    best_score = prec
                    best_acc=accu
                    best_prauc=prauc
                    top_ep=epoch
                    if save:
                        torch.save(model.state_dict(), PATHx_save[:-3]+"_best"+PATHx_save[-3:])
                    calc_precision(test_loader,pfilename="best_precision"+logfile.split("/")[-1][3:]+".csv")# if precision as best_choce there should be 2x precision in the name
                end=time.time()
         
                diff_time=abs(start-end)
                
                with open(logfile,"a+") as g:
                        g.writelines("Supervised loss "+ str(loss)+"\n")
                        g.writelines(str(diff_time)+"\n")
                        g.writelines('Epoch: {:03d}, Loss: {:.7f}'.format(epoch, loss)+"\n")
                        g.writelines(str(prec)+" precision \n")
                        g.writelines(str(prauc)+" prauc \n")
                        g.writelines(str(accu)+" ACC \n")
            top_ep,best_score       
            res_dict[logfile]=[top_ep,best_score,best_acc]
            print(top_ep,best_score,best_acc,best_prauc,logfile)  



