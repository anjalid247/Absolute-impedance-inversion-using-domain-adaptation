# contains utility functions
import zipfile
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from core.utils_DANN import GradientReversal

def extract(source_path, destination_path):
    """function extracts all files from a zip file at a given source path
    to the provided destination path"""
    
    with zipfile.ZipFile(source_path, 'r') as zip_ref:
        zip_ref.extractall(destination_path)
    
    
def standardize(seismic, seismic_train, model,model_train,no_wells):
    """function standardizes data using statistics extracted from training 
    wells
    
    Parameters
    ----------
    seismic : array_like, shape(num_traces, depth samples)
        2-D array containing seismic section
        
    model : array_like, shape(num_wells, depth samples)
        2-D array containing model section
        
    no_wells : int,
        no of wells (and corresponding seismic traces) to extract.
    """
    torch.manual_seed(44)
    train_indices = (np.linspace(0, len(model_train)-1, no_wells, dtype=int))  
    seismic_org= seismic[:,0,:]
    seismic_env= seismic[:,1,:]
    seismic_ip=seismic[:,2,:]
    seismic_org_train = seismic_train[:,0,:]
    seismic_env_train = seismic_train[:,1,:]
    seismic_ip_train = seismic_train[:,2,:]
    seismic_normalized_org  = (seismic_org - seismic_org_train[train_indices].mean())/ seismic_org_train[train_indices].std()
    seismic_normalized_env  = (seismic_env - seismic_env_train[train_indices].mean())/ seismic_env_train[train_indices].std()
    seismic_normalized_ip  = (seismic_ip - seismic_ip_train[train_indices].mean())/ seismic_ip_train[train_indices].std()
    seismic_normalized = np.dstack((seismic_normalized_org,seismic_normalized_env, seismic_normalized_ip))
    seismic_normalized = seismic_normalized.transpose((0,2,1))
   
    #model_normalized = (model_split - model[train_indices].mean()) / model[train_indices].std()
    model_normalized = (model - model_train[train_indices].mean()) / model_train[train_indices].std()
    return seismic_normalized, model_normalized

#function for unnormalization_my
def unnormalized(AI_pred,model_train,no_wells):
    train_indices = (np.linspace(0, len(model_train)-1, no_wells, dtype=int))
    y_pred_un = (AI_pred * model_train[train_indices].std()) + model_train[train_indices].mean()
    #y_pred_un = (AI_pred* model.std()) + model.mean()
    return y_pred_un

def normalization(seismic, model):
    seismic_org= seismic[:,0,:]
    seismic_env= seismic[:,1,:]
    seismic_normalized_org  = (seismic_org - seismic_org.min())/ (seismic_org.max() - seismic_org.min())
    seismic_normalized_env  = (seismic_env - seismic_env.min())/ (seismic_env.max() - seismic_env.min())
    seismic_normalized = np.dstack((seismic_normalized_org,seismic_normalized_env))
    seismic_normalized = seismic_normalized.transpose((0,2,1))
    #train_indices = (np.linspace(0, len(model)-1, no_wells, dtype=int))
    model_normalized = (model - model.min()) / (model.max() - model.min())
    return seismic_normalized, model_normalized

def unnormalized_n(AI_pred,model):
    #train_indices = (np.linspace(0, len(model)-1, no_wells, dtype=int))
    #model_normalized = (model - model[train_indices].mean()) / model[train_indices].std()
    y_pred_un = (AI_pred* (model.max()-model.min())) + model.min()

    return y_pred_un

class CustomModule(nn.Module):
    def __init__(self):
        super(CustomModule,self).__init__()
        self.gru1=nn.GRU(5, 200, 3,batch_first=True, bidirectional=True, dropout=0.1)
        #self.gru1=nn.GRU(5, 30, 3,batch_first=True, bidirectional=True, dropout=0.1)
        #self.elu() = nn.ELU()
        #self.gru2=nn.GRU(60, 30, 3,batch_first=True, bidirectional=True, dropout=0.2)
        #self.gru3=nn.GRU(20,20,2, batch_first=True, bidirectional=False, dropout=0.2)
        #self.gru4=nn.GRU(20,20, 2,batch_first=True, bidirectional=False, dropout=0.2 )
        self.elu = nn.ELU()
        #self.dropout = nn.Dropout(0.2)
        self.conv1d=nn.Conv1d(in_channels=400, out_channels=1, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight.data)
                m.bias.data.zero_()

    def forward(self,x):
        x = x.view(x.shape[0], 5,-1)
        x=x.permute(0,2,1)
        x,_=self.gru1(x)
        #x,_=self.gru2(x,h)
        #x,h=self.gru3(x,h)
        #x,_=self.gru4(x,h)
        x = x.permute(0,2,1)
        x = self.elu(x)
        #x = self.dropout(x)
        x = self.conv1d(x)
        return x
   
