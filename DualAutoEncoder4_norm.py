import torch 
import torch.utils.data as Data
# from torchviz import make_dot
from torch.autograd import Variable 
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
device = torch.device('cpu')#torch.device('cuda' if torch.cuda.is_available() else 'cpu')#

class DualAutoEncoder(torch.nn.Module):
    def __init__(self, p, dataset_feature, model_feature, n_hidden1, n_hidden2,
                 n_hidden3, n_hidden1_, dataset_output,
                 model_output):
        super(DualAutoEncoder, self).__init__()
        self.layer1_1 = torch.nn.Linear(dataset_feature, n_hidden1)
        self.layer1_2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.layer1_3 = torch.nn.Linear(n_hidden2, n_hidden3)
        self.layer1_4 = torch.nn.Linear(n_hidden3, n_hidden2)
        self.layer1_5 = torch.nn.Linear(n_hidden2, n_hidden1)
        self.predict_1 = torch.nn.Linear(n_hidden1, dataset_output)

        self.layer2_1 = torch.nn.Linear(model_feature, n_hidden1_)
        self.layer2_2 = torch.nn.Linear(n_hidden1_, n_hidden3)
        self.layer2_3 = torch.nn.Linear(n_hidden3, n_hidden1_)
        self.predict_2 = torch.nn.Linear(n_hidden1_, model_output)
        self.dropout = torch.nn.Dropout(p=p)  # dropout训练

    def forward(self, x1, x2):
        x1 = self.dropout(x1)
        x1 = torch.relu(self.layer1_1(x1))
        x1 = self.dropout(x1)
        x1 = torch.relu(self.layer1_2(x1))    
        x1 = torch.sigmoid(self.layer1_3(x1))
        x_1 = x1
        #x1 = self.dropout(x1)
        
        x1 = torch.relu(self.layer1_4(x1))
        x1 = self.dropout(x1)
        x1 = torch.relu(self.layer1_5(x1))
        x1 = self.dropout(x1)
        x2_1 = torch.sigmoid(self.predict_1(x1))

        #x1 = self.filter_x2*x
        
        x2 = torch.relu(self.layer2_1(x2))
        #x2 = self.dropout(x2)
        x2 = torch.sigmoid(self.layer2_2(x2))
        x_2 = x2
        x2 = torch.relu(self.layer2_3(x2))
        #x2 = self.dropout(x2)
        x2_2 = torch.sigmoid(self.predict_2(x2))
        #print(x2_1)
        #print(x2_2)
        c = torch.Tensor().to(device)
        for eachdata in x_1:
            c = torch.cat((c, x_2 * eachdata), 0)
        pre_data = x2_1.flatten()
        #a=x2_1*x2_2
        # print(a.shape)
        #pre_dm = torch.sum(c, dim=1)
        scaler_dm=MinMaxScaler()
        pre_dm = torch.sum(c, dim=1).data.cpu().numpy().reshape(-1, 1) 
       # pre_model.data.cpu().numpy()
        
        pre_dm=scaler_dm.fit_transform(pre_dm)
        pre_dm=torch.Tensor(pre_dm).to(device).flatten()
        #print(pre_dm)
        pre_model = x2_2.T.flatten()
        # print(predict_res)
        return pre_data, pre_dm, pre_model  #,x_1