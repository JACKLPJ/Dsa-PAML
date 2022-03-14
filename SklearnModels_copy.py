#@timeout_decorator.timeout(time_per_model, use_signals=False)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool

def abortable_worker(func, *args, **kwargs):
    timeout = kwargs.get('timeout', None)
    p = ThreadPool()
    res = p.apply_async(func, args=args)
    try:
        out = res.get(timeout)  
        # Wait timeout seconds for func to complete.
        return out
    except multiprocessing.TimeoutError:
        pass

def Preprocessing(X, preprocessing_dic):
    if preprocessing_dic[0] == 'polynomial':
       # return X
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(
            degree=preprocessing_dic[1]['degree'],
            interaction_only=preprocessing_dic[1]['interaction_only'],
            include_bias=preprocessing_dic[1]['include_bias'])
        return poly.fit_transform(X)
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=preprocessing_dic[1]['n_components'],
                  whiten=preprocessing_dic[1]['whiten'])
       # pca.fit(X_train)
        return pca.fit_transform(X)
     
# def polynomial(X, preprocessing_dic):
#     poly = PolynomialFeatures(
#         degree=preprocessing_dic[1]['degree'],
#         interaction_only=preprocessing_dic[1]['interaction_only'],
#         include_bias=preprocessing_dic[1]['include_bias'])
#     return poly.fit_transform(X)

# def PCA(X, preprocessing_dic):
#     pca = PCA(n_components=preprocessing_dic[1]['n_components'],
#               whiten=preprocessing_dic[1]['whiten'])
#     #pca.fit(X)
#     return pca.fit_transform(X)
# y_pro = MIAMRS.predict_proba(X_test)
# print(time.perf_counter() - t0)
# print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))
# print("Auc score", sklearn.metrics.roc_auc_score(y_test, y_pro, average='macro', multi_class='ovr'))


def LDA(X_train, X_test, y_train, y_test, model_dic, data_pre_processing, preprocessing_dic):
    if data_pre_processing:
        if preprocessing_dic[0] == 'polynomial':
           # return X
            from sklearn.preprocessing import PolynomialFeatures
            #preprocessing_dic[1]['degree']=min(2, preprocessing_dic[1]['degree'])
            poly = PolynomialFeatures(
                degree=preprocessing_dic[1]['degree'],
                interaction_only=preprocessing_dic[1]['interaction_only'],
                include_bias=preprocessing_dic[1]['include_bias'])
            X_train, X_test = poly.fit_transform(X_train), poly.fit_transform(X_test)
        if preprocessing_dic[0] == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(preprocessing_dic[1]['n_components'], X_train.shape[1]),
                      whiten=preprocessing_dic[1]['whiten'])
           # pca.fit(X_train)
            X_train = pca.fit_transform(X_train)#, pca.fit_transform(X_test)
            preprocessing_dic[1]['n_components'] = X_train.shape[1]
            X_test = PCA(n_components=preprocessing_dic[1]['n_components'],
                      whiten=preprocessing_dic[1]['whiten']).fit_transform(X_test)
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import accuracy_score
    #         X_train, X_test, y_train, y_test = train_test_split(X,
    #                                                             y,
    #                                                             test_size=0.2)
    #shrinkage = None  #if model_dic[1]['shrinkage'] == -1 else model_dic[1][
    # 'shrinkage']
    #n_components = model_dic[1]['n_components']
    #tol = model_dic[1]['tol']
    #if 'shrinkage_factor' in model_dic[1]:
    #    clf = LinearDiscriminantAnalysis(
    #        shrinkage=shrinkage,
    #        n_components=n_components,
    #        tol=tol,
    #        shrinkage_factor=model_dic[1]['shrinkage_factor'])

    #clf = LinearDiscriminantAnalysis(shrinkage=None,
     #                                n_components=min(model_dic[1]['n_components'], X_train.shape[1]),
      #                               tol=model_dic[1]['tol'])
    try:
        clf = LinearDiscriminantAnalysis(shrinkage=None,
                                     n_components=min(model_dic[1]['n_components'], X_train.shape[1]),
                                     tol=model_dic[1]['tol'])
        clf = clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test) 
        acc = accuracy_score(y_test, y_hat)

    #cks = cohen_kappa_score(y_test, y_hat)#, average='weighted')
        print('lda finished!')
    except:
        return None
    #print('svm finished!')
    return 'lda', [clf, preprocessing_dic], acc, y_hat

#@timeout_decorator.timeout(time_per_model, use_signals=False)
def QDA(X_train, X_test, y_train, y_test, model_dic, data_pre_processing,
        preprocessing_dic):
    if data_pre_processing:
        if preprocessing_dic[0] == 'polynomial':
           # return X
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(
                degree=preprocessing_dic[1]['degree'],
                interaction_only=preprocessing_dic[1]['interaction_only'],
                include_bias=preprocessing_dic[1]['include_bias'])
            X_train, X_test = poly.fit_transform(X_train), poly.fit_transform(
                X_test)
        if preprocessing_dic[0] == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=preprocessing_dic[1]['n_components'],
                      whiten=preprocessing_dic[1]['whiten'])
            #pca.fit(X_train)
            X_train = pca.fit_transform(X_train)#, pca.fit_transform(X_test)
            preprocessing_dic[1]['n_components'] = X_train.shape[1]
            X_test = PCA(n_components=preprocessing_dic[1]['n_components'],
                      whiten=preprocessing_dic[1]['whiten']).fit_transform(X_test)
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import accuracy_score
    reg_param = model_dic[1]['reg_param']
    clf = QDA(reg_param=reg_param)
    try:
        
        clf = clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test) 
        acc = accuracy_score(y_test, y_hat)

        #cks = cohen_kappa_score(y_test, y_hat)#, average='weighted')
        print('qda finished!')
    except:
        return None
    
    return 'qda', [clf, preprocessing_dic], acc, y_hat

#@timeout_decorator.timeout(time_per_model, use_signals=False)
def SVM(X_train, X_test, y_train, y_test, model_dic, data_pre_processing, preprocessing_dic):
    if data_pre_processing:
        if preprocessing_dic[0] == 'polynomial':
           # return X
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(
                degree=preprocessing_dic[1]['degree'],
                interaction_only=preprocessing_dic[1]['interaction_only'],
                include_bias=preprocessing_dic[1]['include_bias'])
            X_train, X_test = poly.fit_transform(X_train), poly.fit_transform(X_test)
        if preprocessing_dic[0] == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=preprocessing_dic[1]['n_components'],
                      whiten=preprocessing_dic[1]['whiten'])
            #pca.fit(X_train)
            X_train = pca.fit_transform(X_train)#, pca.fit_transform(X_test)
            preprocessing_dic[1]['n_components'] = X_train.shape[1]
            X_test = PCA(n_components=preprocessing_dic[1]['n_components'],
                      whiten=preprocessing_dic[1]['whiten']).fit_transform(X_test)
    from sklearn import svm
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import accuracy_score
    kernel = model_dic[1]['kernel']
    C = model_dic[1]['C']
    max_iter = model_dic[1]['max_iter']
    tol = model_dic[1]['tol']
    shrinking = True if model_dic[1]['shrinking'] else False
    gamma = model_dic[1]['gamma']
    f, k = 0, 0
    if 'degree' in model_dic[1]:
        degree = model_dic[1]['degree']
        f = 1
    if 'coef0' in model_dic[1]:
        coef0 = model_dic[1]['coef0']
        k = 1
    if f == 0 and k == 0:
        clf = svm.SVC(kernel=kernel,
                      C=C,
                      max_iter=max_iter,
                      tol=tol,
                      shrinking=shrinking,
                      gamma=gamma)
    elif f == 0 and k == 1:
        clf = svm.SVC(kernel=kernel,
                      C=C,
                      max_iter=max_iter,
                      tol=tol,
                      shrinking=shrinking,
                      gamma=gamma,
                      coef0=coef0)
    elif f == 1 and k == 1:
        clf = svm.SVC(kernel=kernel,
                      C=C,
                      max_iter=max_iter,
                      tol=tol,
                      shrinking=shrinking,
                      gamma=gamma,
                      coef0=coef0,
                      degree=degree)
    else:
        clf = svm.SVC(kernel=kernel,
                      C=C,
                      max_iter=max_iter,
                      tol=tol,
                      shrinking=shrinking,
                      gamma=gamma,
                      degree=degree)
    
    try:
        
        clf = clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test) 
        acc = accuracy_score(y_test, y_hat)

        #cks = cohen_kappa_score(y_test, y_hat)#, average='weighted')
        print('svm finished!')
    except:
        
        return None
   
    return 'svm', [clf, preprocessing_dic], acc, y_hat

#@timeout_decorator.timeout(time_per_model, use_signals=False)
def KNN(X_train, X_test, y_train, y_test, model_dic, data_pre_processing, preprocessing_dic):
    if data_pre_processing:
        if preprocessing_dic[0] == 'polynomial':
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(
                degree=preprocessing_dic[1]['degree'],
                interaction_only=preprocessing_dic[1]['interaction_only'],
                include_bias=preprocessing_dic[1]['include_bias'])
            
        if preprocessing_dic[0] == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=preprocessing_dic[1]['n_components'],
                      whiten=preprocessing_dic[1]['whiten'])
            #pca.fit(X_train)
            X_train = pca.fit_transform(X_train)#, pca.fit_transform(X_test)
            preprocessing_dic[1]['n_components'] = X_train.shape[1]
            X_test = PCA(n_components=preprocessing_dic[1]['n_components'],
                      whiten=preprocessing_dic[1]['whiten']).fit_transform(X_test)
    from sklearn.neighbors import KNeighborsClassifier
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import accuracy_score
    print(model_dic)
    clf = KNeighborsClassifier(p=model_dic[1]['p'],n_jobs=1,
                               weights=model_dic[1]['weights'],
                               n_neighbors=min(len(y_test), model_dic[1]['n_neighbors']))
                                               
   
    try:
        clf = clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test) 
        acc = accuracy_score(y_test, y_hat)

        #cks = cohen_kappa_score(y_test, y_hat)#, average='weighted')
        print('knn finished!')
    except:
        return None
    
    
    return 'knn', [clf, preprocessing_dic], acc, y_hat

#@timeout_decorator.timeout(time_per_model, use_signals=False)
def GaussianNB(X_train, X_test, y_train, y_test, model_dic, data_pre_processing, preprocessing_dic):
    if data_pre_processing:
        if preprocessing_dic[0] == 'polynomial':
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(
                degree=preprocessing_dic[1]['degree'],
                interaction_only=preprocessing_dic[1]['interaction_only'],
                include_bias=preprocessing_dic[1]['include_bias'])
            
        if preprocessing_dic[0] == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=preprocessing_dic[1]['n_components'],
                      whiten=preprocessing_dic[1]['whiten'])
           # pca.fit(X_train)
            X_train = pca.fit_transform(X_train)#, pca.fit_transform(X_test)
            preprocessing_dic[1]['n_components'] = X_train.shape[1]
            X_test = PCA(n_components=preprocessing_dic[1]['n_components'],
                      whiten=preprocessing_dic[1]['whiten']).fit_transform(X_test)
    from sklearn.naive_bayes import GaussianNB
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import accuracy_score
    clf = GaussianNB()#alpha=model_dic[1]['alpha'],
                        #priors=model_dic[1]['fit_prior'])
   
    try:
        clf = clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test) 
        acc = accuracy_score(y_test, y_hat)

        #cks = cohen_kappa_score(y_test, y_hat)#, average='weighted')
        print('gnb finished!')
    except:
        return None
      
    return 'gnb', [clf, preprocessing_dic], acc, y_hat

def MultinomialNB(X_train, X_test, y_train, y_test, model_dic, data_pre_processing, preprocessing_dic):
    if data_pre_processing:
        if preprocessing_dic[0] == 'polynomial':
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(
                degree=preprocessing_dic[1]['degree'],
                interaction_only=preprocessing_dic[1]['interaction_only'],
                include_bias=preprocessing_dic[1]['include_bias'])
            
        if preprocessing_dic[0] == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=preprocessing_dic[1]['n_components'],
                      whiten=preprocessing_dic[1]['whiten'])
            #pca.fit(X_train)
            X_train = pca.fit_transform(X_train)#, pca.fit_transform(X_test)
            preprocessing_dic[1]['n_components'] = X_train.shape[1]
            X_test = PCA(n_components=preprocessing_dic[1]['n_components'],
                      whiten=preprocessing_dic[1]['whiten']).fit_transform(X_test)
    from sklearn import preprocessing 
    from sklearn.naive_bayes import MultinomialNB
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import accuracy_score
    clf = MultinomialNB(alpha=model_dic[1]['alpha'],
                        fit_prior=model_dic[1]['fit_prior'])
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.fit_transform(X_test)
    
    
    try:
        clf = clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test) 
        acc = accuracy_score(y_test, y_hat)

        #cks = cohen_kappa_score(y_test, y_hat)#, average='weighted')
        print('mnb finished!')
    except:
        return None
      
    
    return 'mnb', [clf, preprocessing_dic], acc, y_hat


#@timeout_decorator.timeout(time_per_model, use_signals=False)
def BernoulliNB(X_train, X_test, y_train, y_test, model_dic, data_pre_processing, preprocessing_dic):
    if data_pre_processing:
        if preprocessing_dic[0] == 'polynomial':
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(
                degree=preprocessing_dic[1]['degree'],
                interaction_only=preprocessing_dic[1]['interaction_only'],
                include_bias=preprocessing_dic[1]['include_bias'])
            X_train, X_test = poly.fit_transform(X_train), poly.fit_transform(X_test)
        if preprocessing_dic[0] == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=preprocessing_dic[1]['n_components'],
                      whiten=preprocessing_dic[1]['whiten'])
            #pca.fit(X_train)
            X_train = pca.fit_transform(X_train)#, pca.fit_transform(X_test)
            preprocessing_dic[1]['n_components'] = X_train.shape[1]
            X_test = PCA(n_components=preprocessing_dic[1]['n_components'],
                      whiten=preprocessing_dic[1]['whiten']).fit_transform(X_test)
    from sklearn.naive_bayes import BernoulliNB
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import accuracy_score

    clf = BernoulliNB(alpha=model_dic[1]['alpha'],
                      fit_prior=model_dic[1]['fit_prior'])
    #         clf.fit(X_train, y_train)
    #         y_pred = clf.predict(X_test)
    
    try:
        clf = clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test) 
        acc = accuracy_score(y_test, y_hat)

        #cks = cohen_kappa_score(y_test, y_hat)#, average='weighted')
        print('bnb finished!')
    except:
        return None
       
    
    return 'bnb', [clf, preprocessing_dic], acc, y_hat

#@timeout_decorator.timeout(time_per_model, use_signals=False)
def DecisionTree(X_train, X_test, y_train, y_test, model_dic, data_pre_processing, preprocessing_dic):
    if data_pre_processing:
        if preprocessing_dic[0] == 'polynomial':
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(
                degree=preprocessing_dic[1]['degree'],
                interaction_only=preprocessing_dic[1]['interaction_only'],
                include_bias=preprocessing_dic[1]['include_bias'])
            X_train, X_test = poly.fit_transform(X_train), poly.fit_transform(X_test)
        if preprocessing_dic[0] == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=preprocessing_dic[1]['n_components'],
                      whiten=preprocessing_dic[1]['whiten'])
            #pca.fit(X_train)
            X_train = pca.fit_transform(X_train)#, pca.fit_transform(X_test)
            preprocessing_dic[1]['n_components'] = X_train.shape[1]
            X_test = PCA(n_components=preprocessing_dic[1]['n_components'],
                      whiten=preprocessing_dic[1]['whiten']).fit_transform(X_test)
    from sklearn.tree import DecisionTreeClassifier
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import accuracy_score
    if model_dic[1]['max_features'] > 0 and X_train.shape[1] < model_dic[1][
            'max_features']:
        max_features_ = X_train.shape[1]
    elif model_dic[1]['max_features'] <= 0:
        max_features_ = None
    else:
        max_features_ = int(model_dic[1]['max_features']) + 1
    clf = DecisionTreeClassifier(
        splitter=model_dic[1]['splitter'],
        min_samples_leaf=model_dic[1]['min_samples_leaf'],
        max_features=max_features_,  #int(model_dic[1]['max_features']) +
        #1 if model_dic[1]['max_features'] > 0 else None,
        min_weight_fraction_leaf=model_dic[1]['min_weight_fraction_leaf'],
        criterion=model_dic[1]['criterion'],
        min_samples_split=model_dic[1]['min_samples_split'],
        max_depth=int(model_dic[1]['max_depth']) +
        1 if model_dic[1]['max_depth'] > 0 else None,
        max_leaf_nodes=model_dic[1]['max_leaf_nodes']
        if model_dic[1]['max_leaf_nodes'] > 0 else None)
    
    try:
        clf = clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test) 
        acc = accuracy_score(y_test, y_hat)

        #cks = cohen_kappa_score(y_test, y_hat)#, average='weighted')
        print('dt finished!')
    except:
        return None
      
    return 'dt', [clf, preprocessing_dic], acc, y_hat

#@timeout_decorator.timeout(time_per_model, use_signals=False)
def RandomForest(X_train, X_test, y_train, y_test, model_dic, data_pre_processing, preprocessing_dic):
    if data_pre_processing:
        if preprocessing_dic[0] == 'polynomial':
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(
                degree=preprocessing_dic[1]['degree'],
                interaction_only=preprocessing_dic[1]['interaction_only'],
                include_bias=preprocessing_dic[1]['include_bias'])
            X_train, X_test = poly.fit_transform(X_train), poly.fit_transform(X_test)
        if preprocessing_dic[0] == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=preprocessing_dic[1]['n_components'],
                      whiten=preprocessing_dic[1]['whiten'])
            #pca.fit(X_train)
            X_train = pca.fit_transform(X_train)#, pca.fit_transform(X_test)
            preprocessing_dic[1]['n_components'] = X_train.shape[1]
            X_test = PCA(n_components=preprocessing_dic[1]['n_components'],
                      whiten=preprocessing_dic[1]['whiten']).fit_transform(X_test)
    from sklearn.ensemble import RandomForestClassifier
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import accuracy_score
    if model_dic[1]['max_features'] > 0 and X_train.shape[1] < model_dic[1][
            'max_features']:
        max_features_ = X_train.shape[1]
    elif model_dic[1]['max_features'] <= 0:
        max_features_ = None
    else:
        max_features_ = int(model_dic[1]['max_features']) + 1
    clf = RandomForestClassifier(
        n_jobs=1,
        bootstrap=model_dic[1]['bootstrap'],
        n_estimators=model_dic[1]['n_estimators'],
        min_samples_leaf=model_dic[1]['min_samples_leaf'],
        max_features=max_features_,  #int(model_dic[1]['max_features']) +
        # 1 if model_dic[1]['max_features'] > 0 else None,
        min_weight_fraction_leaf=model_dic[1]['min_weight_fraction_leaf'],
        criterion=model_dic[1]['criterion'],
        min_samples_split=model_dic[1]['min_samples_split'],
        max_depth=int(model_dic[1]['max_depth']) +
        1 if model_dic[1]['max_depth'] > 0 else None,
        max_leaf_nodes=model_dic[1]['max_leaf_nodes']
        if model_dic[1]['max_leaf_nodes'] > 0 else None)
    
    try:
        clf = clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test) 
        acc = accuracy_score(y_test, y_hat)

        #cks = cohen_kappa_score(y_test, y_hat)#, average='weighted')
        print('rf finished!')
    except:
        return None   
    return 'rf', [clf, preprocessing_dic], acc, y_hat

#@timeout_decorator.timeout(time_per_model, use_signals=False)
def GradientBoosting(X_train, X_test, y_train, y_test, model_dic, data_pre_processing, preprocessing_dic):
    if data_pre_processing:
        if preprocessing_dic[0] == 'polynomial':
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(
                degree=preprocessing_dic[1]['degree'],
                interaction_only=preprocessing_dic[1]['interaction_only'],
                include_bias=preprocessing_dic[1]['include_bias'])
            X_train, X_test = poly.fit_transform(X_train), poly.fit_transform(X_test)
        if preprocessing_dic[0] == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=preprocessing_dic[1]['n_components'],
                      whiten=preprocessing_dic[1]['whiten'])
           # pca.fit(X_train)
            X_train = pca.fit_transform(X_train)#, pca.fit_transform(X_test)
            preprocessing_dic[1]['n_components'] = X_train.shape[1]
            X_test = PCA(n_components=preprocessing_dic[1]['n_components'],
                      whiten=preprocessing_dic[1]['whiten']).fit_transform(X_test)
    from sklearn.ensemble import GradientBoostingClassifier
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import accuracy_score
    if model_dic[1]['max_features'] > 0 and X_train.shape[1] < model_dic[1][
            'max_features']:
        max_features_ = X_train.shape[1]
    elif model_dic[1]['max_features'] <= 0:
        max_features_ = None
    else:
        max_features_ = int(model_dic[1]['max_features']) + 1
    clf = GradientBoostingClassifier(
        subsample=model_dic[1]['subsample'],
        loss=model_dic[1]['loss'],
        n_estimators=model_dic[1]['n_estimators'],
        min_samples_leaf=model_dic[1]['min_samples_leaf'],
        max_features=max_features_,  #int(model_dic[1]['max_features']) +
        #1 if model_dic[1]['max_features'] > 0 else None,
        min_weight_fraction_leaf=model_dic[1]['min_weight_fraction_leaf'],
        learning_rate=model_dic[1]['learning_rate'],
        min_samples_split=model_dic[1]['min_samples_split'],
        max_depth=int(model_dic[1]['max_depth']) +
        1 if model_dic[1]['max_depth'] > 0 else None,
        max_leaf_nodes=model_dic[1]['max_leaf_nodes']
        if model_dic[1]['max_leaf_nodes'] > 0 else None)
   
    try:
        clf = clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test) 
        acc = accuracy_score(y_test, y_hat)
        #cks = cohen_kappa_score(y_test, y_hat)#, average='weighted')
        print('gb finished!')
    except:
        return None
  
    
    return 'gb', [clf, preprocessing_dic], acc, y_hat       

#@timeout_decorator.timeout(time_per_model, use_signals=False)
def ExtraTrees(X_train, X_test, y_train, y_test, model_dic, data_pre_processing, preprocessing_dic):
    if data_pre_processing:
        if preprocessing_dic[0] == 'polynomial':
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(
                degree=preprocessing_dic[1]['degree'],
                interaction_only=preprocessing_dic[1]['interaction_only'],
                include_bias=preprocessing_dic[1]['include_bias'])
            X_train, X_test = poly.fit_transform(X_train), poly.fit_transform(X_test)
        if preprocessing_dic[0] == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=preprocessing_dic[1]['n_components'],
                      whiten=preprocessing_dic[1]['whiten'])
            #pca.fit(X_train)
            X_train = pca.fit_transform(X_train)#, pca.fit_transform(X_test)
            preprocessing_dic[1]['n_components'] = X_train.shape[1]
            X_test = PCA(n_components=preprocessing_dic[1]['n_components'],
                      whiten=preprocessing_dic[1]['whiten']).fit_transform(X_test)
    from sklearn.ensemble import ExtraTreesClassifier
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import accuracy_score
    if model_dic[1]['max_features'] > 0 and X_train.shape[1] < model_dic[1][
            'max_features']:
        max_features_ = X_train.shape[1]
    elif model_dic[1]['max_features'] <= 0:
        max_features_ = None
    else:
        max_features_ = int(model_dic[1]['max_features']) + 1
    clf = ExtraTreesClassifier(
        n_jobs=1,
        bootstrap=model_dic[1]['bootstrap'],
        n_estimators=model_dic[1]['n_estimators'],
        min_samples_leaf=model_dic[1]['min_samples_leaf'],
        max_features=max_features_,  #int(model_dic[1]['max_features']) +
        #1 if model_dic[1]['max_features'] > 0 else None,
        min_weight_fraction_leaf=model_dic[1]['min_weight_fraction_leaf'],
        criterion=model_dic[1]['criterion'],
        min_samples_split=model_dic[1]['min_samples_split'],
        max_depth=int(model_dic[1]['max_depth']) +
        1 if model_dic[1]['max_depth'] > 0 else None)
    
    try:
        clf = clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test) 
        acc = accuracy_score(y_test, y_hat)

        #cks = cohen_kappa_score(y_test, y_hat)#, average='weighted')
        print('et finished!')
    except:
        return None
    
    return 'etree', [clf, preprocessing_dic], acc, y_hat

#@timeout_decorator.timeout(time_per_model, use_signals=False)
def LGB(X_train, X_test, y_train, y_test, model_dic, data_pre_processing, preprocessing_dic):
    if data_pre_processing:
        if preprocessing_dic[0] == 'polynomial':
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(
                degree=preprocessing_dic[1]['degree'],
                interaction_only=preprocessing_dic[1]['interaction_only'],
                include_bias=preprocessing_dic[1]['include_bias'])
            X_train, X_test = poly.fit_transform(X_train), poly.fit_transform(X_test)
        if preprocessing_dic[0] == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=preprocessing_dic[1]['n_components'],
                      whiten=preprocessing_dic[1]['whiten'])
           # pca.fit(X_train)
            X_train = pca.fit_transform(X_train)#, pca.fit_transform(X_test)
            preprocessing_dic[1]['n_components'] = X_train.shape[1]
            X_test = PCA(n_components=preprocessing_dic[1]['n_components'],
                      whiten=preprocessing_dic[1]['whiten']).fit_transform(X_test)
    #from lightgbm import LGBMClassifier
    import lightgbm as lgb
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import accuracy_score
    
    m_d = int(model_dic[1]
              ['max_depth']) + 1 if model_dic[1]['max_depth'] > 0 else None
    #print('*************')
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt',
        verbose=-1,
        force_col_wise=True,
       # num_threads=1, 
        n_jobs=1, 
       # num_leaves=min(31, 2**m_d - 1) if (m_d and m_d > 1) else 31,
        reg_alpha=model_dic[1]['alpha'],
        reg_lambda=model_dic[1]['reg_lambda'],
        max_depth=m_d,
        n_estimators=model_dic[1]['n_estimators'],
        subsample=model_dic[1]['subsample'],
        colsample_bytree=model_dic[1]['colsample_bytree'],
        #subsample_freq=1,
        learning_rate=model_dic[1]['learning_rate'],
        min_child_weight=model_dic[1]['min_child_weight'])
    
    try:
        clf = clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test) 
        acc = accuracy_score(y_test, y_hat)

        #cks = cohen_kappa_score(y_test, y_hat)#, average='weighted')
        print('lgb finished!')
    except:
        return None
    return 'lgb', [clf, preprocessing_dic], acc, y_hat


#@timeout_decorator.timeout(time_per_model, use_signals=False)
def XGB(X_train, X_test, y_train, y_test, model_dic, data_pre_processing, preprocessing_dic):
    if data_pre_processing:
        if preprocessing_dic[0] == 'polynomial':
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(
                degree=preprocessing_dic[1]['degree'],
                interaction_only=preprocessing_dic[1]['interaction_only'],
                include_bias=preprocessing_dic[1]['include_bias'])
            X_train, X_test = poly.fit_transform(X_train), poly.fit_transform(X_test)
        if preprocessing_dic[0] == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=preprocessing_dic[1]['n_components'],
                      whiten=preprocessing_dic[1]['whiten'])
            #pca.fit(X_train)
            X_train = pca.fit_transform(X_train)#, pca.fit_transform(X_test)
            preprocessing_dic[1]['n_components'] = X_train.shape[1]
            X_test = PCA(n_components=preprocessing_dic[1]['n_components'],
                      whiten=preprocessing_dic[1]['whiten']).fit_transform(X_test)
    #from xgboost.sklearn import XGBClassifier
    import xgboost as xgb
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import accuracy_score
#     xgb_model = xgb.XGBClassifier(n_jobs=1).fit(X[train_index], y[train_index])
#     predictions = xgb_model.predict(X[test_index])
#     actuals = y[test_index]
#     print(confusion_matrix(actuals, predictions))
    clf = xgb.XGBClassifier(
        n_jobs=1,  
        #verbosity=3,
        colsample_bytree=model_dic[1]['colsample_bytree'],
        colsample_bylevel=model_dic[1]['colsample_bylevel'],
        alpha=model_dic[1]['alpha'],
        # scale_pos_weight=model_dic[1]['scale_pos_weight'],
        learning_rate=model_dic[1]['learning_rate'],
        max_delta_step=model_dic[1]['max_delta_step'],
        base_score=model_dic[1]['base_score'],
        n_estimators=model_dic[1]['n_estimators'],
        subsample=model_dic[1]['subsample'],
        reg_lambda=model_dic[1]['reg_lambda'],
        min_child_weight=model_dic[1]['min_child_weight'],
        max_depth=int(model_dic[1]['max_depth']) + 1 if model_dic[1]['max_depth'] > 0 else None,        
        gamma=model_dic[1]['gamma'])
    #X_train, X_test, y_train, y_test = train_test_split(X, y, \
    #                                                    test_size=0.25, random_state=42)
#
    ##         accu = np.mean(cross_val_score(clf, X, y, cv=5))
    #         print("prediction accuracy: {}".format(accu))
    #         return accu
    
    try:
        clf = clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test) 
        acc = accuracy_score(y_test, y_hat)

        #cks = cohen_kappa_score(y_test, y_hat)#, average='weighted')
        print('xgb finished!')
    except:
        return None
  
    
    return 'xgb', [clf, preprocessing_dic], acc, y_hat

