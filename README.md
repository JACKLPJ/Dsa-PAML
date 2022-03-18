# Dsa-PAML
An AutoML software based on dual-stacked autoencoder.    
1.安装依赖包  
python                    3.8  
xgboost                   1.4.2     
scikit-learn              0.23.2                     
scipy                     1.6.2   
pandas                    1.2.4   
numpy                     1.20.1   
torch                             1.7.1  
2.下载安装说明  
将DsaPAML文件夹所有内容下载到本地，并从  https://drive.google.com/file/d/1Cr1bfTWXmwlxYK3o_QQx8o_ZcF7_0VIy/view?usp=sharing,   
https://drive.google.com/file/d/1v24KBTMpuYoqvkwgvF6sQy2D_qVIYnh9/view?usp=sharing 处下载EX_data_samples1.csv和EX_model_samples1.csv,
将这两个csv文件放置在RequiredBases文件夹中。  
3.使用说明  
在软件文件夹里建立测试文件，比如下图中的test_Dsa-PAML.ipynb，也可以是.py文件，但要求必须在软件文件夹中测试。  
![image](https://user-images.githubusercontent.com/42956088/158143533-e3f20206-e58d-47c3-96eb-620e5ab411d0.png)


（1）导入函数库  
import pandas as pd  
import numpy as np  
import AutoML8w_stack as automl  
import sklearn  
import time   
（2）软件参数的设置  
clf = aml.Automl(  
N=100, # 为测试数据集推荐的模型个数(默认：200，建议N>=50)  
    verbose=False, # 是否显示软件运行的中间结果 (可选：True,False；默认：False)  
time_per_model=360, # 训练单个模型管道的时间上限（默认：360（秒））                
N_jobs=-1, # 并行运行的CPU核数 (默认：-1（表示使用机器所有CPU核）)  
system='linux'# 系统型号（可选：'linux','windows','mac'；默认：'linux'）  
)  
（3）读取待测数据集并分裂训练集和测试集  
publishing_data = pd.read_csv(  
'/media/sia1/Elements SE/AutoML测试数据集/DataSets/publishing_data.csv',  
sep=',',  
header=None)  
X, y = publishing_data.iloc[:, :9], publishing_data.iloc[:, 9]  
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)  
（4）模型的训练和预测  
t0 = time.perf_counter() # 记录训练和测试全过程的时间  
clf.fit(X_train, y_train) # 模型的训练  
y_hat = clf.predict(X_test) # 模型的预测  
print("Runtime: ", time.perf_counter() - t0) # 打印时间开销  
print("Accuracy score: ", accuracy_score(y_test, y_hat)) # 打印测试集上的准确率  
（5）结果打印  
Runtime:  34.77397622299031  
Accuracy score:  0.9056047197640118  
