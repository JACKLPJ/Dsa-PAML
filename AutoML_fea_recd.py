from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
#from sklearn.ensemble import VotingClassifier
#from sklearn.ensemble import StackingClassifier
import time
import SklearnModels_copy as sm
import DataModeling_sub as dm
from multiprocessing import Pool
import multiprocessing
from functools import partial
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV


'''
X:数据集的特征
y:数据集的标签
N:前N个best model
data_pre_processing:是否需要数据预处理 
'''
class Automl:
    #time_per_model=360

    def __init__(self, N=200, 
                 verbose=False,
                 #DoEnsembel=True, 
                 #data_pre_processing=False, 
                 time_per_model=360,                
                 N_jobs=-1,
                 system='linux'
                 #address='./metaKB/'#'/media/sia1/Elements SE/TRAIN/'
#                  address_model_metafeatures='/media/sia1/Elements SE/TRAIN/model_samples.csv', 
#                  address_data_metafeatures='/media/sia1/Elements SE/TRAIN/data_samples.csv',
#                  address_pipelines='/media/sia1/Elements SE/TRAIN/New_pipelines.json', 
                 ):
        #DualAutoEncoder_model_4, gpuStackedDualAutoEncoder_model7, DSDAE_model,DualAutoEncoder_new
        self.address='./RequiredBases/'#address
        self.address_model_metafeatures = self.address+'EX_model_samples1.csv'#address_model_metafeatures
        self.address_data_metafeatures=self.address+'EX_data_samples1.csv'#_data_metafeatures
        self.address_pipeline = self.address+'New_pipelines1.json'#address_pipelines
        self.verbose=verbose
        self.DoEnsembel = True#DoEnsembel
        self.y = []
        #sm.time_per_model = time1
        self.ensemble_clf = []
        self.k=20
        
        self.N = N
        self.n_estimators = 200
        self.scaler = StandardScaler()
        self.data_pre_processing = False#data_pre_processing  
        self.time_per_model = time_per_model
        self.N_jobs = N_jobs
        if system=='linux':
            multiprocessing.set_start_method('forkserver',force=True)
    
        self.enc = OrdinalEncoder()
        
    def fit(self, Xtrain, ytrain):
        X = Xtrain.copy(deep=True)
        y = ytrain.copy(deep=True)
        X_ = Xtrain.copy(deep=True)
        y_ = ytrain.copy(deep=True)
        if self.N < 15:
            raise ValueError('N must be more than 15')
        self.y = ytrain.copy(deep=True)
#         preprocessing_dics, model_dics = dm.data_modeling(
#             X, y, self.N).result
    
        preprocessing_dics, model_dics = dm.data_modeling(
            X_, y_, self.k, self.N, self.address_model_metafeatures, self.address_data_metafeatures, self.address_pipeline, self.address, self.verbose).result  #_preprocessor
        self.cats, self.nums = [], []
        Xtype=list(X.dtypes)
        col_num = len(Xtype)
        #self.flag = False
        for i in range(col_num):
            if Xtype[i]=='O' or Xtype[i]=='object':
                self.cats.append(i)
            else:
                self.nums.append(i)
        if self.cats and self.nums:
            X_train_some = X.iloc[:, self.cats]
            X_train_some = pd.DataFrame(self.enc.fit_transform(X_train_some))#.toarray())
            X_train_others = X.iloc[:, self.nums]
            X_train_others=X_train_others.reset_index(drop=True)# = pd.DataFrame(X_train.iloc[:,nums]) 
            X = pd.concat([X_train_some, X_train_others],axis=1)
        elif self.cats:
            X = pd.DataFrame(self.enc.fit_transform(X))
        X = pd.DataFrame(self.scaler.fit_transform(X))
        self.poly = None
        if col_num <= 5:
            self.poly = PolynomialFeatures(3)
            X=pd.DataFrame(self.poly.fit_transform(X))
        elif col_num <= 15:
            self.poly = PolynomialFeatures(2)
            X=pd.DataFrame(self.poly.fit_transform(X))
        t_fs = time.perf_counter()
        self.rfecv = RFECV(estimator=RandomForestClassifier(n_jobs = -1),          # 学习器
              min_features_to_select=X.shape[1]//5*3, # 最小选择的特征数量
              step=max(1, X.shape[1]//25),                 # 移除特征个数
              cv=StratifiedKFold(2),  # 交叉验证次数
              scoring='accuracy',     # 学习器的评价标准
              verbose = 1 if self.verbose else 0,
              n_jobs = -1
              ).fit(X, y)
        X = pd.DataFrame(self.rfecv.transform(X))
        if self.verbose:
            print("The time for features selection is: {}".format(time.perf_counter() -
                                                      t_fs))
            print('#######################################')
        n = len(preprocessing_dics)
        if self.y.dtypes == 'object' or self.y.dtypes == 'O':
            labels = list(y.unique())
            y = y.replace(labels, list(range(len(labels))))
        y = y.astype('int')
        accuracy = []
        great_models = []
        pool=Pool()
        all_results = []
       # self.rxc = X.shape[0] * X.shape[1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                        test_size=0.25,
                                                        random_state=0)

        td = time.perf_counter()
        for i in range(n):
           # t_m=time.perf_counter()
            if model_dics[i][0] == 'xgradient_boosting':
                if X.shape[0] >= 10 ** 5:#self.rxc > 10 ** 7:
                    worker = sm.LGB
                else:
                    worker = sm.XGB                
                    
            elif model_dics[i][0] == 'gradient_boosting':
                if X.shape[0] < 10 ** 5:
                    worker = sm.GradientBoosting    
                else:
                    continue
                 #   worker = sm.XGB
                

            elif model_dics[i][0] == 'lda':
                worker = sm.LDA
                
            elif model_dics[i][0] == 'extra_trees':
                worker = sm.ExtraTrees

            elif model_dics[i][0] == 'random_forest':
                worker = sm.RandomForest

            elif model_dics[i][0] == 'decision_tree':
                worker = sm.DecisionTree
                
            elif model_dics[i][0] == 'libsvm_svc':
                worker = sm.SVM

            elif model_dics[i][0] == 'k_nearest_neighbors':
                worker = sm.KNN

            elif model_dics[i][0] == 'bernoulli_nb':
                worker = sm.BernoulliNB

            elif model_dics[i][0] == 'multinomial_nb':
                worker = sm.MultinomialNB

            elif model_dics[i][0] == 'qda':
                worker = sm.QDA

            else:
                worker = sm.GaussianNB
             
            abortable_func = partial(sm.abortable_worker, worker, timeout=self.time_per_model) 
#             all_results.append(
#                         pool.apply_async(worker,
#                                          args=(
#                                              X_train,
#                                              X_test,
#                                              y_train,
#                                              y_test,
#                                              model_dics[i],
#                                              self.data_pre_processing,
#                                              preprocessing_dics[i],
#                                          )))
            all_results.append(
                        pool.apply_async(abortable_func,
                                         args=(
                                             X_train,
                                             X_test,
                                             y_train,
                                             y_test,
                                             model_dics[i],
                                             self.data_pre_processing,
                                             preprocessing_dics[i],
                                         )))
                
        pool.close()
        pool.join()
        if self.verbose:
            print('The time of pools is: {}'.format(time.perf_counter() -
                                                      td))
            
        td = time.perf_counter()
        model_name = []
#         All_results=[]
#         for sub_res in all_results:
#             try:
#                 All_results.append(sub_res.get())
#             except:
#                 continue
        #print(all_results)
        all_results = np.array([
            sub_res.get() for sub_res in all_results
        ])
        #print(all_results)
        #all_results=All_results
        if None in all_results:
            all_results = all_results[all_results != None]
        #print(all_results)
        Y_hat = []
        for a in all_results:
            #a=sub_res.get()
            Y_hat.append(a[3])
            model_name.append(a[0])
            accuracy.append(a[2])
            great_models.append(a[1])
        if self.verbose:
            print('The time of individuals is: {}'.format(time.perf_counter() -
                                                      td))
        Y_hat=np.array(Y_hat)
        sort_id = sorted(range(len(accuracy)),
                         key=lambda m: accuracy[m],
                         reverse=True)
        mean_acc = np.mean(accuracy)#np.median(accuracy)#
        #mean_f1 = np.mean(f1_scores)
        estimators_stacking = []  #[great_models[sort_id[0]]]
        #X_val_predictions = [all_results[sort_id[0]][-1]]
        id_n = len(sort_id)
        id_i = 0
        base_acc_s = [] 
        pre=[]
        while accuracy[sort_id[id_i]] > mean_acc: 
            pre.append(sort_id[id_i])
            id_i += 1
        
        Y_hat=Y_hat[pre]
        n_pre=len(Y_hat)
         
        Res_=[] 
        
        td = time.perf_counter()
        pool = Pool()  
        for i in range(n_pre):
            Res_.append(pool.apply_async(self.Sum_diff,args=(i,n_pre,Y_hat,)))   
        pool.close()
        pool.join()
        res_=[] 
        Sort=[]
        #fa=0
        for s in Res_: 
            aa=s.get()
#             if aa[0]:
#                 fa=max(fa,min(aa[0]))
            res_.append(aa[0])
            Sort.append(aa[1])
        if self.verbose:
            print('The time of pools2 is: {}'.format(time.perf_counter() -
                                                      td))
        c = sorted(range(len(Sort)), key=lambda k: Sort[k])
        res_ = np.array(res_)[c]
        
        Rubbish=set()
        
        final=[]
        for i in range(n_pre):
            if i not in Rubbish:
                final.append(pre[i])
                for k in range(len(res_[i])):
                    if res_[i][k] == 0: 
                        Rubbish.add(i+k+1)
        
        #print(final)
        if len(final)==1:
            self.DoEnsembel=False
        estimators_stacking=[great_models[i] for i in final]#.append(great_models[sort_id0[id_i]])
        base_acc_s=[accuracy[i] for i in final]#.append(accuracy[sort_id0[id_i]])
        
       # print(self.imbalance)#, fa)
        if self.verbose:
            print(id_n, len(base_acc_s))
            print(base_acc_s, mean_acc)
        if self.DoEnsembel:
            te = time.perf_counter()
            meta_clf = RandomForestClassifier(n_jobs=-1,
                                              n_estimators=self.n_estimators)
            
            eclf_stacking = StackingClassifier(classifiers=estimators_stacking,
                                               meta_classifier=meta_clf,
                                               use_probas=True,
                                               #preprocessing=self.data_pre_processing,
                                               fit_base_estimators=False)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            #accuracy.append(
            eclf_stacking = eclf_stacking.fit(X_train, y_train)
            if self.verbose:
                print('Ensemble val score:',
                  accuracy_score(y_test, eclf_stacking.predict(X_test)))
            self.ensemble_clf = [estimators_stacking, eclf_stacking]
            if self.verbose:
                print('The time of ensemble is: {}'.format(time.perf_counter() -
                                                       te))
            #print(self.ensemble_clf)
            #return meta_clf
        else:

            self.clf = [model_name[sort_id[0]], great_models[sort_id[0]]]
            if self.verbose:
                print(self.clf)
            #allresult = [great_models[sort_id[0]], accuracy[sort_id[0]]]
            return self
        
    def Sum_diff(self,i,n,Y_hat):
        res=[]
        for j in range(i+1,n):
            res.append(np.sum(Y_hat[i]!=Y_hat[j])) 
        return [res,i]

    def predict(self, Xtest):
        X_Test = Xtest.copy(deep=True)    
        if self.cats and self.nums:
            X_test_some = X_Test.iloc[:,self.cats]
            X_test_some = pd.DataFrame(self.enc.transform(X_test_some))#.toarray())
            X_test_others = X_Test.iloc[:, self.nums]
            X_test_others=X_test_others.reset_index(drop=True)
            X_Test = pd.concat([X_test_some, X_test_others],axis=1) 
        elif self.cats:
            X_Test = pd.DataFrame(self.enc.transform(X_Test))
        X_Test=pd.DataFrame(self.scaler.transform(X_Test))
        if self.poly:
            X_Test=pd.DataFrame(self.poly.transform(X_Test)) 
        #X_Test = self.select_fea.transform(X_Test)
        X_Test = pd.DataFrame(self.rfecv.transform(X_Test))
        #X_Test = self.pre_processing_X(X_Test)
        if self.DoEnsembel:
            # X_test_predictions = self.scaler.transform(X_test_predictions)
            ypre = self.ensemble_clf[1].predict(X_Test)
        else:
            if self.clf[0] == 'mnb':
                from sklearn import preprocessing
                min_max_scaler = preprocessing.MinMaxScaler()
                X_Test = min_max_scaler.fit_transform(X_Test)
            if self.data_pre_processing:
                X_Test=sm.Preprocessing(X_Test, self.clf[1][1])
           # t = time.perf_counter()            
            ypre = self.clf[1][0].predict(X_Test)
        if self.y.dtypes == 'object' or self.y.dtypes == 'O':
            b = self.y.unique()
            return [b[i] for i in ypre]
        return ypre
    
    def predict_proba(self, Xtest):
        X_Test = Xtest.copy(deep=True)
        X_Test = self.pre_processing_X(X_Test)
        if self.DoEnsembel:
            # X_test_predictions = self.scaler.transform(X_test_predictions)
            ypre = self.ensemble_clf[1].predict_proba(X_Test)
        else:
            if self.clf[0] == 'mnb':
                from sklearn import preprocessing
                min_max_scaler = preprocessing.MinMaxScaler()
                X_Test = min_max_scaler.fit_transform(X_Test)
            if self.data_pre_processing:
                X_Test=sm.Preprocessing(X_Test, self.clf[1][1])
           # t = time.perf_counter()
            
            ypre = self.clf[1][0].predict_proba(X_Test)
        
        return ypre
