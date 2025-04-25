## imports
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from os.path import join
from core.utils_copy5 import standardize, unnormalized, CustomModule
from core.datasets_copy import SeismicDataset1D
from torch.utils.data import DataLoader
from core.model1D_copy import MustafaNet
from core.utils_DANN import GradientReversal
import tqdm
import torch.nn.functional as F
from tqdm import tqdm
import copy

import errno
import argparse
torch.autograd.set_detect_anomaly(True)
#from core.models import inverse_model

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def set_seed(x):
    torch.manual_seed(x)
    torch.cuda.manual_seed_all(x)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(x)
    #random.seed(3)
    os.environ['PYTHONHASHSEED'] = str(x)

def main(args):
    
    model = MustafaNet().to(device)
    torch.manual_seed(40)

    feature_extractor= model.tcn_local
    predictor = model.regression = CustomModule().to(device)
    #print(model.eval())

    discriminator =nn.Sequential(GradientReversal(),
          nn.Linear(3505, 1000),
          #nn.Dropout(0.2),
          nn.ELU(),
          nn.Linear(1000, 100),
          #nn.Dropout(0.1),
          nn.ELU(),
          nn.Linear(100, 10),
          #nn.Dropout(0.1),
          nn.ReLU(),
          nn.Linear(10,1)
    ).to(device)

    seismic_m2 = np.load('data/input_LFnew_seismic_3c_DA.npy').squeeze()
    model_m2= np.load('data/input_LFnew_imp_3c_DA.npy').squeeze()

    seismic_n_m2, model_n_m2 = standardize(seismic_m2, seismic_m2, model_m2,model_m2, no_wells=12000)   


  # specify pseudolog positions for training and validation
    traces_m2_train = np.linspace(0, len(model_m2)-1, 12000, dtype=int)
    traces_m2_validation = np.linspace(0, len(model_m2)-1, 2000, dtype=int)
      
    seam_train_dataset_source = SeismicDataset1D(seismic_n_m2, model_n_m2, traces_m2_train)
    loader_source = DataLoader(seam_train_dataset_source, batch_size = args.batch_size, shuffle=True)


    seismic_s = np.load('data/seam_input_3c_DA_minip.npy')[:,:,::2][:, :, 50:]
    model_s = np.load('data/seam_model_DA_5to120.npy')[:,::2][:, 50:]

    seismic_n, model_n = standardize(seismic_s,seismic_s, model_s,model_s, no_wells=150)
        
    #return seismic, model

    # specify pseudolog positions for training and validation
    traces_seam_train = np.linspace(0, len(model_s)-1, 150, dtype=int)
    #traces_seam_validation = np.linspace(0, len(model_s)-1, 50, dtype=int)
        
    seam_train_dataset_target = SeismicDataset1D(seismic_n, model_n, traces_seam_train)
    loader_target = DataLoader(seam_train_dataset_target, batch_size = args.batch_size, shuffle=True)
    
    #unlabled data_target
    seismic_s1 = np.load('data/seam_input_LFnew_3c_minip.npy')[:,:,::2][:, :, 50:]
    model_s1 = np.load('data/Seam_model_full.npy')[:,::2][:, 50:]
   
    
    traces_seam_validation = np.linspace(0, len(seismic_s1)-1, 1502, dtype=int)
    
    seismic_n1, model_n1 = standardize(seismic_s1,seismic_s1, model_s1,model_s1, no_wells=1502)
    seam_val_dataset_target = SeismicDataset1D(seismic_n1, model_n1, traces_seam_validation)
    loader_target_val = DataLoader(seam_val_dataset_target, batch_size = args.batch_size, shuffle=True)
     
    optimizer_seam = torch.optim.Adam(list(model.parameters())+list(discriminator.parameters()),
                                          weight_decay=0.00001,
                                          lr=0.001)

    import time
    start_time=time.time()
    dl=[]
    ll=[]

    for epoch in range(args.epochs):
      set_seed(5)
      batches = zip(loader_source, loader_target, loader_target_val)
      n_batches = min(len(loader_source), len(loader_target), len(loader_target_val))
      p=epoch/args.epochs
      total_domain_loss = total_label_loss = 0

      for (source_x, source_labels), (target_x, target_labels), (target_x_val, _) in tqdm(batches, leave=False, total=n_batches):
        x = torch.cat([source_x, target_x, target_x_val])
        
        x = x.to(device)
        domain_y = torch.cat([torch.ones(source_x.shape[0]),
                                        torch.zeros(target_x.shape[0]+target_x_val.shape[0])])
        domain_y = domain_y.to(device)
        label_y = torch.cat([(source_labels),(target_labels)]).to(device)
        features= feature_extractor(x)
        features1 = features.view(x.shape[0], -1)

        #print(features1.shape)
        #print(features.shape)
        
        domain_preds = discriminator(features1).squeeze()
        index = source_x.shape[0]+target_labels.shape[0]
        label_preds = predictor(features[:index])
                 
        domain_loss = F.binary_cross_entropy_with_logits(domain_preds, domain_y)
        label_loss = F.mse_loss(label_preds, label_y)
        r = label_loss.item()/(domain_loss.item()+00000000.1)
        loss = r*domain_loss + label_loss
        optimizer_seam.zero_grad()
        loss.backward()
        optimizer_seam.step()

        total_domain_loss += domain_loss.item()
        total_label_loss += label_loss.item()

      mean_loss = total_domain_loss/n_batches
      mean_label = total_label_loss/n_batches
      tqdm.write(f'EPOCH {epoch:03d}: domain_loss={mean_loss:.4f}: label_loss={mean_label:0.4f}')
                
      dl.append(mean_loss)
      ll.append(mean_label)
      #print(len(dl))

      torch.save(model.state_dict(), 'saved_models/seam_revgrad_syn.pth')
      torch.save(model,'saved_models/seam_revgrad_model_syn.pth')
      
      domain_label1 = domain_y.detach().cpu().numpy()
      feat = features1.detach().cpu().numpy()
      feats= np.column_stack((feat,domain_label1))
      #np.save('feat_mod100.npy', feats, allow_pickle=True)

    domain_loss2 = np.array(dl, dtype='float32')
    label_loss2 = np.array(ll, dtype='float32')
    print(domain_loss2.shape)

    np.save('output/domain_loss.npy', domain_loss2, allow_pickle=True)
    np.save('output/label_loss.npy', label_loss2, allow_pickle=True)
    print(f'\nDuration:{time.time()-start_time:.0f} seconds')

def test(args):
    
    print('model testing')
    
    seismic_s = np.load('data/seam_input_LFnew_3c_minip.npy')[:,:,::2][:, :, 50:]
    #print(seismic_s.shape)
    model_s= np.load('data/Seam_model_full.npy')[:,::2][:, 50:]
   
    seismic_train = np.load('data/seam_input_3c_DA_minip.npy')[:,:,::2][:, :, 50:]
    model_train = np.load('data/seam_model_DA_5to120.npy')[:,::2][:, 50:]
    
    seismic_n, model_n = standardize(seismic_s, seismic_train,model_s,model_train,args.no_wells)                                       
    
    # define device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    traces_seam_test = np.arange(len(model_s), dtype=int)
    
    seam_test_dataset = SeismicDataset1D(seismic_n, model_n, traces_seam_test)
    seam_test_loader = DataLoader(seam_test_dataset, batch_size = args.batch_size)
    
    #setup model
    model_seam = torch.load('saved_models/seam_revgrad_model_syn.pth', weights_only=False).to(device)
    model_seam.load_state_dict(torch.load('saved_models_g/seam_revgrad_syn.pth', weights_only=False))

    # infer on SEAM
    print("\nInferring ...")
    x, y = seam_test_dataset[0] # get a sample
    AI_pred = torch.zeros((len(seam_test_dataset), y.shape[-1])).float().to(device)
    AI_act = torch.zeros((len(seam_test_dataset), y.shape[-1])).float().to(device)
    
    mem = 0
    with torch.no_grad():
        for i, (x,y) in enumerate(seam_test_loader):
          model_seam.eval()
          y_pred  = model_seam(x)
          AI_pred[mem:mem+len(x)] = y_pred.squeeze().data
          AI_act[mem:mem+len(x)] = y.squeeze().data
          mem += len(x)
          del x, y, y_pred
    
    
    #vmin_act, vmax_act = AI_act.min(), AI_act.max()
    AI_pred = AI_pred.detach().cpu().numpy()
    AI_act = AI_act.detach().cpu().numpy()

    AI_pred_un = unnormalized(AI_pred, model_train, args.no_wells)
    AI_act_un = unnormalized(AI_act,model_train, args.no_wells)

    np.save('outputs/seam_AI_pred_revgrad.npy', AI_pred_un, allow_pickle=True)
    

    print(AI_pred.shape)
    print(AI_pred.min(),AI_pred.max())
    print(AI_pred_un.min(),AI_pred_un.max())
    
    vmin_n, vmax_n = AI_pred_un.min(),AI_pred_un.max()
    vmin_act_n, vmax_act_n = AI_act_un.min(), AI_act_un.max()
    print(vmin_act_n,vmax_act_n)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Domain adaptation using RevGrad')
    #arg_parser.add_argument('MODEL_FILE', help='A model in trained_models')
    arg_parser.add_argument('--batch_size', type=int, default=32)
    arg_parser.add_argument('--epochs', type=int, default=500)
    arg_parser.add_argument('--no_wells', type=int, default=150)
    arg_parser.add_argument('-alpha', type=float, default=1, help="weight of domain loss term")
    args = arg_parser.parse_args()
    main(args)
    test(args)

