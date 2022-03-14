import numpy as np
import pandas as pd 
import json
import pickle
#import cPickle
import MetaFeaturesCalculation_copy as MFC
import torch
#from DualAutoEncoder import DualAutoEncoder
#from sklearn.preprocessing import MinMaxScaler
import heapq
#net=torch.load('/media/sia1/Elements SE/TRAIN/DualAutoEncoder_model_0.pt')
class get_model:
    def __init__(self, X, y, k, N, add_model_metafeatures, add_data_samples, add_pipeline, address_dmn,verbose):
        #super(get_model, self).__init__(dataset_feature, model_feature, n_hidden, dataset_output, model_output)
        self.model_metafeatures = pd.read_csv(add_model_metafeatures)
        with open(add_pipeline, 'r') as jsonfile:
            self.pipelines = json.load(jsonfile)
        self.scaler_data=[]
        self.scaler_model=[]
        self.network=[]
        self.verbose = verbose
        for i in range(15):
            F=open(address_dmn+'ex_scaler01_data1'+str(i)+'.pkl','rb')
            self.scaler_data.append(pickle.load(F))
            F=open(address_dmn+'ex_scaler01_model1'+str(i)+'.pkl','rb')
            self.scaler_model.append(pickle.load(F))
            self.network.append(torch.load(address_dmn+'6SubDualAutoEncoder'+str(i)+'.pt', map_location=lambda storage, loc: storage))         
        #print(add_data_samples)
        self.data_feats_featurized = pd.read_csv(add_data_samples)#.iloc[:, 46:]
        self.dff=self.data_feats_featurized.iloc[:, :46]
        self.data_feats_featurized = self.data_feats_featurized.iloc[:, 46:]
        self.metadatas=MFC.metafeature_calculate(X, y,verbose).newdataset_metafeas.tolist()
        self.knn=self.kn_dist(k)
        self.Model = self.choose_model(X, y, N)    
        #self.lda_help = self.metadatas[0]
    def kn_dist(self, k):
        Weights = [
            4.284254337774223, 1.0884829057306453, 4.60797648353123,
            8.346083710685244, 0.5216970771922169, 6.064006338779894,
            2.9732120470445165, 1.7402763881073966
        ]  #from offline stage2
        w1, w2, w3, w4, w5, w6, w7, w8 = np.array(Weights)  # / sum(Weights)
        #dff = self.data_feats_featurized.dropna()
#         dff = self.data_feats_featurized.dropna().iloc[:,
#                                                        list(range(0, 5)) +
#                                                        list(range(7, 18)) +
#                                                        list(range(19, 21)) +
#                                                        [29, 31, 35] +
#                                                        list(range(37, 46))]
        process1 = abs(self.metadatas[1:] - self.dff)
        process1 = (process1 - process1.min()) / (process1.max() -
                                                  process1.min())
        
        dist_ = np.sum(process1.iloc[:, :5], axis=1) * w1 + np.sum(
           process1.iloc[:, 7:11], axis=1) * w2 + np.sum(
               process1.iloc[:, 11:17], axis=1) * w3 + np.sum(
                   process1.iloc[:, [17, 19, 20]], axis=1) * w4 + np.sum(
                       process1.iloc[:, [31, 29]],
                       axis=1) * w5 + process1.iloc[:, 35] * w6 + np.sum(
                           process1.iloc[:, 37:41], axis=1) * w7 + np.sum(
                               process1.iloc[:, 41:46], axis=1) * w8
        knd = list(np.argsort(dist_)[:k]) 
        return knd
    
    def choose_model(self, X, y, N):
        device = torch.device('cpu')#torch.device('cuda' if torch.cuda.is_available() else 'cpu')#
        #dataset_mf=MFC.metafeature_calculate(X, y).newdataset_metafeas.tolist()
        #IN2= pd.read_csv(self.model_metafeatures)
        input_train2=np.array(self.model_metafeatures)
        Predictions=[]
        rec_models=[]
        mean_knn=list(np.mean(self.data_feats_featurized.loc[self.knn,:]))
        for i in range(15):
            inputn2 = self.scaler_model[i].transform(input_train2[1390*i:1390*(i+1),:])
            inputn2 = torch.Tensor(inputn2).to(device)
            if i<14:
                Train=[self.metadatas[1:]+mean_knn[1390*i:1390*(i+1)]]
            else:
                Train=[self.metadatas[1:]+mean_knn[1390*i:]+[mean_knn[0]]]
            #Train=[self.metadatas[1:]+[0]*1807]
            inputn_test = self.scaler_data[i].transform(Train)
            inputn_test = torch.Tensor(inputn_test).to(device)
            self.network[i].eval()
            pre_model = self.network[i](inputn_test, inputn2)[1]
            pre_model = pre_model.data.cpu().numpy()#numpy()#.cpu().numpy()
            pre_model=pre_model.reshape(-1)
            Predictions=np.array(pre_model)
            if self.verbose:
                print(Predictions)
            rec_models.extend(np.array(heapq.nlargest(N//15, range(len(Predictions)), Predictions.take))+1390*i)
            #Predictions.extend(pre_model)
#         inputn2 = self.scaler_model.transform(input_train2)
#         inputn2 = torch.Tensor(inputn2).to(device)
#         Train=[list(np.mean(self.data_feats_featurized.loc[self.knn,:]))+self.metadatas[1:]]
        
#         inputn_test = self.scaler_data.transform(Train)
#         inputn_test = torch.Tensor(inputn_test).to(device)
        #加载训练好的模型
  
#         self.network.eval()
#         pre_model = self.network(inputn_test, inputn2)[1]
#         pre_model = pre_model.data.cpu().numpy()#numpy()#.cpu().numpy()
        
#         pre_model=pre_model.reshape(-1)
#         Predictions=np.array(Predictions)
#         print(Predictions)
#         rec_models=heapq.nlargest(N, range(len(Predictions)), Predictions.take)
        if self.verbose:
            print(rec_models)
        Candi_models=[self.metadatas[0]]
        for i in rec_models:
            Candi_models.append(self.pipelines[i])
        return Candi_models
