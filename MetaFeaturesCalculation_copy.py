## python 3.8
import numpy as np
import pandas as pd 
from collections import defaultdict, OrderedDict, deque
import scipy.stats
from scipy.linalg import LinAlgError
import scipy.sparse
import sklearn.tree
import sklearn.neighbors
import sklearn.discriminant_analysis
import sklearn.naive_bayes
import sklearn.decomposition
from concurrent.futures import ThreadPoolExecutor, as_completed
#from multiprocessing import Pool
import time
#import multiprocessing
# +
class metafeature_calculate:
    def __init__(self, X, y,verbose):
        self.verbose=verbose
        if self.verbose:
            print('#######################################')
        self.Missing = X.isnull()  #~np.isfinite(X)
        self.r, self.c = X.shape[0], X.shape[1]
        self.yrc = y.shape
        #print(self.yrc)
        if self.yrc == 2:
            self.yr, self.yc = y.shape[0], y.shape[1]
        else:
            self.yr, self.yc = y.shape[0], None
        #multiprocessing.set_start_method('forkserver',force=True)
        num_missing = self.Missing.sum(axis=0)
        self.NOFWMV = float(
            np.sum([1 if num > 0 else 0 for num in num_missing]))

        num_missing_1 = self.Missing.sum(axis=1)
        self.NOIWMV = float(
            np.sum([1 if num > 0 else 0 for num in num_missing_1]))

        self.NOMV = float(np.sum(num_missing_1))

        self.DR = float(self.c) / float(self.r)
        self.nominal = 0
        for i in range(self.c):
            if X.iloc[:, i].dtypes == 'object' or X.iloc[:, i].dtypes == 'O':
                self.nominal += 1
        self.numerical = self.c - self.nominal

        #t0 = time.perf_counter()
        labels = 1 if len(self.yrc) == 1 else self.yc
        if labels == 1:
            y = y.values.reshape((-1, 1))
        self.all_occurence_dict = {}
        self.Class = 0
        for i in range(labels):
            occurence_dict = defaultdict(float)
            for value in y[:, i]:
                occurence_dict[value] += 1
            self.Class = max(self.Class, len(occurence_dict))
            self.all_occurence_dict[i] = occurence_dict
        #print('The time for calculating all_occurence_dict is {}'.format(
        #  time.process_time() - t0))
        y = pd.core.series.Series(y.reshape(1, -1)[0])
        # t1 = time.process_time()
        self.kurts = []
        self.skews = []
        for i in range(self.c):
            if X.iloc[:, i].dtypes != 'object' and X.iloc[:, i].dtypes != 'O':
                self.kurts.append(scipy.stats.kurtosis(X.iloc[:, i]))
                self.skews.append(scipy.stats.skew(X.iloc[:, i]))

        #return skews
        self.col = list(X.columns)
        #self.fea_len=len(self.col)
        self.lda_n = self.Class - 1#min(len(self.col), self.Class - 1)
        self.symbols_per_column = []
        for j in range(self.c):#self.col:
            if X.iloc[:,j].dtypes == 'object' or X.iloc[:,j].dtypes == 'O':
                b = X.iloc[:,j].unique()
                for i in range(len(b)):
                    X.iloc[:,j].loc[X.iloc[:,j] == b[i]] = i
                X.iloc[:,j] = X.iloc[:,j].astype("int")
                self.symbols_per_column.append(len(b))
    # t4 = time.process_time()
        if y.dtypes == 'object' or y.dtypes == 'O':
            b = y.unique()
            for i in range(len(b)):
                y.loc[y == b[i]] = i
            y = y.astype("int")
        y = y.astype("int")
        ## PCA
        self.pca = sklearn.decomposition.PCA(copy=True)
        rs = np.random.RandomState(42)
        indices = np.arange(X.shape[0])
        for i in range(10):
            try:
                rs.shuffle(indices)
                self.pca.fit(X.iloc[indices])
            except LinAlgError:
                pass

    # print('The time for calculating pca is {}'.format(time.process_time() -
    #                                                   t4))
    #return None

        self.newdataset_metafeas = [self.lda_n]
        tm = time.perf_counter()
        self.landmarks_acc=self.meta_landmarks(X, y)
        if self.verbose:
            print(time.perf_counter()-tm)
        tm = time.perf_counter()
        #all_task = []

        #pool = Pool()  #(processes=n)
        all_results = []
        all_results.append(self.ClassEntropy(X, y))
        #print(all_results)
        all_results.append(self.ClassProbabilityMax
                                           (X, y))
                        
        all_results.append(self.ClassProbabilityMean( 
                                           X, y))
        
        all_results.append(self.ClassProbabilityMin( 
                                           X, y))
        
        all_results.append(self.ClassProbabilitySTD( 
                                           X, y))
        all_results.append(self.DatasetRatio(X, y))
        all_results.append(self.InverseDatasetRatio(X, y))
        
        all_results.append(self.KurtosisMax( 
                                           X, y))
        
        all_results.append(self.KurtosisMean( 
                                           X, y))
        
        all_results.append(self.KurtosisMin( 
                                           X, y))
        
        all_results.append(self.KurtosisSTD( 
                                           X, y))
        
        all_results.append((self.landmarks_acc[0],11))#self.Landmark1NN( 
                                          # X, y))
        
        all_results.append((self.landmarks_acc[1],12))
            #self.LandmarkDecisionNodeLearner( 
             #               X, y))
        
        all_results.append((self.landmarks_acc[2],13))#self.LandmarkDecisionTree( 
                                         #  X, y))
        
        all_results.append((self.landmarks_acc[3],14))#self.LandmarkLDA( 
                                          # X, y))
        
        all_results.append((self.landmarks_acc[4],15))#self.LandmarkNaiveBayes( 
                                          # X, y))
        
        all_results.append((self.landmarks_acc[5],16))
            #self.LandmarkRandomNodeLearner( 
             #               X, y))
        
        all_results.append(self.LogDatasetRatio( 
                                           X, y))
        all_results.append(self.LogInverseDatasetRatio(X, y))
        
        all_results.append(self.LogNumberOfFeatures( 
                                           X, y))
        
        all_results.append(self.LogNumberOfInstances( 
                                           X, y))
        
        all_results.append(self.NumberOfCategoricalFeatures(X, y))
        all_results.append(self.NumberOfClasses(X, y))
        all_results.append(self.NumberOfFeatures(X, y))
        all_results.append(
            self.NumberOfFeaturesWithMissingValues(X, y))
        all_results.append(self.NumberOfInstances(X, y))
        all_results.append(
            self.NumberOfInstancesWithMissingValues(X, y))
        all_results.append(self.NumberOfMissingValues(X, y))
        all_results.append(self.NumberOfNumericFeatures(X, y))
        
        all_results.append(
                self.PCAFractionOfComponentsFor95PercentVariance(
                            X, y))
        all_results.append(self.PCAKurtosisFirstPC(X, y))
        all_results.append(self.PCASkewnessFirstPC( 
                                           X, y))
        
        all_results.append(
            self.PercentageOfFeaturesWithMissingValues(X, y))
        all_results.append(
            self.PercentageOfInstancesWithMissingValues(X, y))
        all_results.append(self.PercentageOfMissingValues(X, y))
        all_results.append(self.RatioNumericalToNominal(X, y))
        all_results.append(self.RatioNominalToNumerical(X, y))
            
        
        all_results.append(self.SkewnessMax(X, y))

        all_results.append(self.SkewnessMean(X, y))

        all_results.append(self.SkewnessMin(X, y))

        all_results.append(self.SkewnessSTD(X, y))
        
        all_results.append(self.SymbolsMax(X, y))

        all_results.append(self.SymbolsMean(X, y))

        all_results.append(self.SymbolsMin(X, y))

        all_results.append(self.SymbolsSTD(X, y))
       
        all_results.append(self.SymbolsSum(X, y))
        

        Orders = [-1]

        for srg in all_results:  
            if self.verbose:
                print(srg)
            self.newdataset_metafeas.append(srg[0])
            Orders.append(srg[1])
        #for sub_res in all_results:
        #    srg = sub_res.get()
        #    self.newdataset_metafeas.append(srg[0])
        #    Orders.append(srg[1])
        c = sorted(range(len(Orders)), key=lambda k: Orders[k])
        self.newdataset_metafeas = np.array(self.newdataset_metafeas)[c]
        if self.verbose:
            print('The time of metafeas calculation is: {}'.format(
            time.perf_counter() - tm))
        
            print('The metafeatures for the dataset are {}'.format(
            self.newdataset_metafeas[1:]))
            print('#######################################')

    def ClassEntropy(self, X, y):
        # def _calculate(self, X, y, categorical):
        all__occurence_dict = self.all_occurence_dict
        entropies = []
        for occurence_dict in all__occurence_dict.values():
            entropies.append(
                scipy.stats.entropy(
                    [occurence_dict[key] for key in occurence_dict], base=2))
        return np.mean(entropies), 0

    def ClassProbabilityMax(self, X, y):
        #def _calculate(self, X, y, categorical):
        occurences = self.all_occurence_dict
        max_value = -1
        if len(self.yrc) == 2:
            for i in range(self.yc):
                max_value = max(max_value, max(occurences[i].values()))
        else:
            max_value = max(occurences[0].values())
        return max_value / float(self.yr), 1

    def ClassProbabilityMean(self, X, y):
        occurence_dict = self.all_occurence_dict
        if len(self.yrc) == 2:
            occurences = []
            for i in range(self.yc):
                occurences.extend(
                    [occurrence for occurrence in occurence_dict[i].values()])
            occurences = np.array(occurences)
        else:
            occurences = np.array(
                [occurrence for occurrence in occurence_dict[0].values()],
                dtype=np.float64)
        return (occurences / self.yr).mean(), 2

    def ClassProbabilityMin(self, X, y):
        #def _calculate(self, X, y, categorical):
        occurences = self.all_occurence_dict
        min_value = self.yr
        if len(y.shape) == 2:
            for i in range(self.yc):
                min_value = min(min_value, min(occurences[i].values()))
        else:
            min_value = min(occurences[0].values())
        return min_value / float(self.yr), 3

    def ClassProbabilitySTD(self, X, y):
        #def _calculate(self, X, y, categorical):
        occurence_dict = self.all_occurence_dict

        if len(y.shape) == 2:
            stds = []
            for i in range(self.yc):
                std = np.array(
                    [occurrence for occurrence in occurence_dict[i].values()],
                    dtype=np.float64)
                std = (std / self.yr).std()
                stds.append(std)
            return np.mean(stds), 4
        else:
            occurences = np.array(
                [occurrence for occurrence in occurence_dict[0].values()],
                dtype=np.float64)
            return (occurences / self.yr).std(), 4

    def DatasetRatio(self, X, y):
        return self.DR, 5

    def InverseDatasetRatio(self, X, y):
        return 1 / self.DR, 6

    def KurtosisMax(self, X, y):
        Kurtosis = self.kurts
        maximum = np.nanmax(Kurtosis) if len(Kurtosis) > 0 else 0
        return maximum if np.isfinite(maximum) else 0, 7

    def KurtosisMean(self, X, y):
        Kurtosis = self.kurts
        mean = np.nanmean(Kurtosis) if len(Kurtosis) > 0 else 0
        return mean if np.isfinite(mean) else 0, 8

    def KurtosisMin(self, X, y):
        Kurtosis = self.kurts
        minimum = np.nanmin(Kurtosis) if len(Kurtosis) > 0 else 0
        return minimum if np.isfinite(minimum) else 0, 9

    def KurtosisSTD(self, X, y):
        Kurtosis = self.kurts
        std = np.nanstd(Kurtosis) if len(Kurtosis) > 0 else 0
        return std if np.isfinite(std) else 0, 10
########################################################################################################### 
    def mkNN(self,X_tr, y_tr, X_te, y_te,yrc,yc):
        kNN = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1,n_jobs=-1)
        if len(yrc) == 1 or yc == 1:
            kNN.fit(X_tr, y_tr)
        else:
            kNN = OneVsRestClassifier(kNN)
            kNN.fit(X_tr, y_tr)
        predictions = kNN.predict(X_te)
        return sklearn.metrics.accuracy_score(predictions,
                                                   y_te),11
    def mDecisionTree(self,X_tr, y_tr, X_te, y_te,yrc,yc):
        random_state = sklearn.utils.check_random_state(42)
        tree = sklearn.tree.DecisionTreeClassifier(
                random_state=random_state)
        if len(yrc) == 1 or yc == 1:
            tree.fit(X_tr, y_tr)
        else:
            tree = OneVsRestClassifier(tree)
            tree.fit(X_tr, y_tr)
        predictions = tree.predict(X_te)
        return sklearn.metrics.accuracy_score(predictions,
                                                   y_te),13
    def mDecisionNodeLearner(self,X_tr, y_tr, X_te, y_te,yrc,yc):
        random_state = sklearn.utils.check_random_state(42)
        node = sklearn.tree.DecisionTreeClassifier(
                criterion="entropy",
                max_depth=1,
                random_state=random_state,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features=None)
        if len(yrc) == 1 or yc == 1:
            node.fit(X_tr, y_tr)
        else:
            node = OneVsRestClassifier(node)
            node.fit(X_tr, y_tr)
        predictions = node.predict(X_te)
        return sklearn.metrics.accuracy_score(predictions,
                                                   y_te),12
    def mLDA(self, X_tr, y_tr, X_te, y_te, yrc, yc):
        try:
            lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
           
            if len(yrc) == 1 or yc == 1:
                lda.fit(X_tr, y_tr)
            else:
                lda = OneVsRestClassifier(lda)
                lda.fit(X_tr, y_tr)
            predictions = lda.predict(X_te)
            return sklearn.metrics.accuracy_score(predictions,
                                                   y_te),14
        except:
            #self.logger.warning("LDA failed: %s Returned 0 instead!" % e)
            return 0,14
    
    def mNaiveBayes(self, X_tr, y_tr, X_te, y_te, yrc, yc):
        node = sklearn.naive_bayes.GaussianNB()
        if len(yrc) == 1 or yc == 1:
            node.fit(X_tr, y_tr)
        else:
            node = OneVsRestClassifier(node)
            node.fit(X_tr, y_tr)
        predictions = node.predict(X_te)
        return sklearn.metrics.accuracy_score(predictions,
                                                   y_te),15
    def mRandomNodeLearner(self, X_tr, y_tr, X_te, y_te, yrc, yc):
        random_state = sklearn.utils.check_random_state(42)
        node = sklearn.tree.DecisionTreeClassifier(
                criterion="entropy",
                max_depth=1,
                random_state=random_state,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features=1)
        
        node.fit(X_tr, y_tr)
        predictions = node.predict(X_te)
        return sklearn.metrics.accuracy_score(predictions,
                                                   y_te),16
    def meta_landmarks(self, X, y):
        kf = sklearn.model_selection.StratifiedKFold(n_splits=5)
        accuracy = [0,0,0,0,0,0]
       # pool = Pool()  #(processes=n)
        Acc_results = []
        executor = ThreadPoolExecutor()
        for train, test in kf.split(X, y):
            for func in [self.mDecisionTree, self.mRandomNodeLearner, self.mNaiveBayes, self.mkNN, self.mDecisionNodeLearner, self.mLDA]:
                Acc_results.append(executor.submit(func, 
                                                   X.iloc[train], 
                                                  y.iloc[train],
                                                  X.iloc[test],
                                                  y.iloc[test],
                                                  self.yrc,
                                                  self.yc))

        for sub_acc in as_completed(Acc_results):
            a=sub_acc.result() 
            accuracy[a[1]-11]+=a[0]
        
        return [it/5 for it in accuracy]   

########################################################################################################### 
    def Landmark1NN(self, X, y):
        return self.landmarks_acc[0],11
        kf = sklearn.model_selection.StratifiedKFold(n_splits=5)
        accuracy = 0.
        for train, test in kf.split(X, y):
            kNN = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1,
                                                         n_jobs=-1)
            if len(self.yrc) == 1 or self.yc == 1:
                kNN.fit(X.iloc[train], y.iloc[train])
            else:
                kNN = OneVsRestClassifier(kNN)
                kNN.fit(X.iloc[train], y.iloc[train])
            predictions = kNN.predict(X.iloc[test])
            accuracy += sklearn.metrics.accuracy_score(predictions,
                                                       y.iloc[test])
        return accuracy / 5, 11

    def LandmarkDecisionNodeLearner(self, X, y):
        return self.landmarks_acc[1],12
        kf = sklearn.model_selection.StratifiedKFold(n_splits=5)
        accuracy = 0.
        for train, test in kf.split(X, y):
            random_state = sklearn.utils.check_random_state(42)
            node = sklearn.tree.DecisionTreeClassifier(
                criterion="entropy",
                max_depth=1,
                random_state=random_state,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features=None)
            if len(self.yrc) == 1 or self.yc == 1:
                node.fit(X.iloc[train], y.iloc[train])
            else:
                node = OneVsRestClassifier(node)
                node.fit(X.iloc[train], y.iloc[train])
            predictions = node.predict(X.iloc[test])
            accuracy += sklearn.metrics.accuracy_score(predictions,
                                                       y.iloc[test])
        return accuracy / 5, 12

    def LandmarkDecisionTree(self, X, y):
        return self.landmarks_acc[2],13
        kf = sklearn.model_selection.StratifiedKFold(n_splits=5)
        accuracy = 0.
        for train, test in kf.split(X, y):
            random_state = sklearn.utils.check_random_state(42)
            tree = sklearn.tree.DecisionTreeClassifier(
                random_state=random_state)

            if len(self.yrc) == 1 or self.yc == 1:
                tree.fit(X.iloc[train], y.iloc[train])
            else:
                tree = OneVsRestClassifier(tree)
                tree.fit(X.iloc[train], y.iloc[train])

            predictions = tree.predict(X.iloc[test])
            accuracy += sklearn.metrics.accuracy_score(predictions,
                                                       y.iloc[test])
        return accuracy / 5, 13

    def LandmarkLDA(self, X, y):
        return self.landmarks_acc[3],14
        kf = sklearn.model_selection.StratifiedKFold(n_splits=5)
        accuracy = 0.
        try:
            for train, test in kf.split(X, y):
                lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(
                )
                if len(self.yrc) == 1 or self.yc == 1:
                    lda.fit(X.iloc[train], y.iloc[train])
                else:
                    lda = OneVsRestClassifier(lda)
                    lda.fit(X.iloc[train], y.iloc[train])

                predictions = lda.predict(X.iloc[test])
                accuracy += sklearn.metrics.accuracy_score(
                    predictions, y.iloc[test])
            return accuracy / 5, 14
        except scipy.linalg.LinAlgError as e:
            #self.logger.warning("LDA failed: %s Returned 0 instead!" % e)
            return np.NaN, 14
        except ValueError as e:
            #self.logger.warning("LDA failed: %s Returned 0 instead!" % e)
            return np.NaN, 14

    def LandmarkNaiveBayes(self, X, y):
        return self.landmarks_acc[4],15
        kf = sklearn.model_selection.StratifiedKFold(n_splits=5)
        accuracy = 0.
        for train, test in kf.split(X, y):
            nb = sklearn.naive_bayes.GaussianNB()
            if len(self.yrc) == 1 or self.yc == 1:
                nb.fit(X.iloc[train], y.iloc[train])
            else:
                nb = OneVsRestClassifier(nb)
                nb.fit(X.iloc[train], y.iloc[train])
            predictions = nb.predict(X.iloc[test])
            accuracy += sklearn.metrics.accuracy_score(predictions,
                                                       y.iloc[test])
        return accuracy / 5, 15

    def LandmarkRandomNodeLearner(self, X, y):
        return self.landmarks_acc[5],16
        kf = sklearn.model_selection.StratifiedKFold(n_splits=5)
        accuracy = 0.
        for train, test in kf.split(X, y):
            random_state = sklearn.utils.check_random_state(42)
            node = sklearn.tree.DecisionTreeClassifier(
                criterion="entropy",
                max_depth=1,
                random_state=random_state,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features=1)
            node.fit(X.iloc[train], y.iloc[train])
            predictions = node.predict(X.iloc[test])
            accuracy += sklearn.metrics.accuracy_score(predictions,
                                                       y.iloc[test])
        return accuracy / 5, 16

    def LogDatasetRatio(self, X, y):
        return np.log(self.DR), 17

    def LogInverseDatasetRatio(self, X, y):
        return np.log(1 / self.DR), 18

    def LogNumberOfFeatures(self, X, y):
        return np.log(self.c), 19

    def LogNumberOfInstances(self, X, y):
        return np.log(self.r), 20

    def NumberOfCategoricalFeatures(self, X, y):
        return self.nominal, 21

    def NumberOfClasses(self, X, y):
        res = []
        for i in self.all_occurence_dict.values():
            res.extend(i.keys())
        return len(set(res)), 22

    def NumberOfFeatures(self, X, y):
        return self.c, 23

    def NumberOfFeaturesWithMissingValues(self, X, y):

        return self.NOFWMV, 24

    def NumberOfInstances(self, X, y):
        return self.r, 25

    def NumberOfInstancesWithMissingValues(self, X, y):

        return self.NOIWMV, 26

    def NumberOfMissingValues(self, X, y):

        return self.NOMV, 27

    def NumberOfNumericFeatures(self, X, y):
        return self.numerical, 28

    def PCAFractionOfComponentsFor95PercentVariance(self, X, y):
        pca_ = self.pca
        if pca_ is None:
            return np.NaN, 29
        sum_ = 0.
        idx = 0
        while sum_ < 0.95 and idx < len(pca_.explained_variance_ratio_):
            sum_ += pca_.explained_variance_ratio_[idx]
            idx += 1
        return float(idx) / float(self.c), 29

    def PCAKurtosisFirstPC(self, X, y):
        pca_ = self.pca
        if pca_ is None:
            return np.NaN
        components = pca_.components_
        pca_.components_ = components[:1]
        transformed = pca_.transform(X)
        pca_.components_ = components
        kurtosis = scipy.stats.kurtosis(transformed)
        return kurtosis[0], 30

    def PCASkewnessFirstPC(self, X, y):
        pca_ = self.pca
        if pca_ is None:
            return np.NaN, 31
        components = pca_.components_
        pca_.components_ = components[:1]
        transformed = pca_.transform(X)
        pca_.components_ = components
        skewness = scipy.stats.skew(transformed)
        return skewness[0], 31

    def PercentageOfFeaturesWithMissingValues(self, X, y):
        return self.NOFWMV / self.c, 32

    def PercentageOfInstancesWithMissingValues(self, X, y):
        return self.NOIWMV / self.r, 33

    def PercentageOfMissingValues(self, X, y):
        return self.NOMV / (self.r * self.c), 34

    def RatioNominalToNumerical(self, X, y):
        if self.numerical == 0:
            return 0., 35
        else:
            return self.nominal / self.numerical, 35

    def RatioNumericalToNominal(self, X, y):
        if self.nominal == 0:
            return 0., 36
        else:
            return self.numerical / self.nominal, 36

    def SkewnessMax(self, X, y):
        skews = self.skews
        maximum = np.nanmax(skews) if len(skews) > 0 else 0
        return maximum if np.isfinite(maximum) else 0, 37

    def SkewnessMean(self, X, y):
        skews = self.skews
        mean = np.nanmean(skews) if len(skews) > 0 else 0
        return mean if np.isfinite(mean) else 0, 38

    def SkewnessMin(self, X, y):
        skews = self.skews
        minimum = np.nanmin(skews) if len(skews) > 0 else 0
        return minimum if np.isfinite(minimum) else 0, 39

    def SkewnessSTD(self, X, y):
        skews = self.skews
        std = np.nanstd(skews) if len(skews) > 0 else 0
        return std if np.isfinite(std) else 0, 40

    def SymbolsMax(self, X, y):
        values = self.symbols_per_column
        if len(values) == 0:
            return 0, 41
        return max(max(values), 0), 41

    def SymbolsMean(self, X, y):
        values = [val for val in self.symbols_per_column if val > 0]
        mean = np.nanmean(values)
        return mean if np.isfinite(mean) else 0, 42

    def SymbolsMin(self, X, y):
        help_ = [i for i in self.symbols_per_column if i > 0]
        return min(help_) if help_ else 0, 43

    def SymbolsSTD(self, X, y):
        values = [val for val in self.symbols_per_column if val > 0]
        std = np.nanstd(values)
        return std if np.isfinite(std) else 0, 44

    def SymbolsSum(self, X, y):
        return np.nansum(self.symbols_per_column) if np.isfinite(
            np.nansum(self.symbols_per_column)) else 0, 45
