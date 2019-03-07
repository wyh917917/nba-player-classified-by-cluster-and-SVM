import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz

from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
#读取csv文件，返回Pandas DataFrame
nba = pd.read_csv('nba_players_all.csv',encoding= 'latin1')
nba = nba[(nba['weight'] > 0) & (nba['height'] > 0)]
print(nba[0:3])
# "Position (pos)"是要预测的特征
class_column = 'position'
#这个数据集包含了球员姓名和球队名称这样的数据
#这些数据对分类没有帮助所以去掉
feature_columns = ['height', 'weight']

#Pandas DataFrame允许选择列
#使用列选择将数据分为特征和类别
nba_feature = nba[feature_columns]
nba_class = nba[class_column]

print(nba_feature[0:3])
print(nba_class[0:3])

train_feature, test_feature, train_class, test_class = train_test_split(nba_feature, nba_class, train_size=0.75, test_size=0.25)

svc = SVC().fit(train_feature, train_class)
prediction = svc.predict(test_feature)

# print("Test set predictions:\n{}".format(prediction))
print("\nModel using SVC Classifier using 75/25")
print("Test set accuracy: {:.3f}".format(svc.score(test_feature, test_class)))
print("Confusion matrix:")
print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'],margins =True))
train_class_df = pd.DataFrame(train_class,columns=[class_column])
train_data_df = pd.merge(train_class_df, train_feature, left_index=True, right_index=True)
train_data_df.to_csv('svc_train_data.csv', index=False)
temp_df = pd.DataFrame(test_class,columns=[class_column])
temp_df['Predicted Pos']=pd.Series(prediction, index=temp_df.index)
test_data_df = pd.merge(temp_df, test_feature, left_index=True, right_index=True)
test_data_df.to_csv('svc_test_data.csv', index=False)

#分类前
plt.figure(figsize=(12, 6))
plt.subplot(121)
colors = ['b','g','r']
guard = test_data_df[test_data_df['position'] == 'Guard']
plt.scatter(guard['height'],guard['weight'],c=colors[0])
forward = test_data_df[test_data_df['position'] == 'Forward']
plt.scatter(forward['height'],forward['weight'],c=colors[1])
center = test_data_df[test_data_df['position'] == 'Center']
plt.scatter(center['height'],center['weight'],c=colors[2])
plt.xlabel('height',fontsize=13)
plt.ylabel('weight',fontsize=13)
plt.title('before')
#分类后
plt.subplot(122)
guard = test_data_df[test_data_df['Predicted Pos'] == 'Guard']
plt.scatter(guard['height'],guard['weight'],c=colors[0])
forward = test_data_df[test_data_df['Predicted Pos'] == 'Forward']
plt.scatter(forward['height'],forward['weight'],c=colors[1])
center = test_data_df[test_data_df['Predicted Pos'] == 'Center']
plt.scatter(center['height'],center['weight'],c=colors[2])
plt.xlabel('height',fontsize=13)
plt.ylabel('weight',fontsize=13)
plt.title('after')

plt.show()

scores = cross_val_score(svc, nba_feature, nba_class, cv=10)
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))