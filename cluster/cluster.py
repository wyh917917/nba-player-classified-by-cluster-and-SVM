import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz

from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
#读取csv文件，返回Pandas DataFrame
nba = pd.read_csv('nba_players_all.csv',encoding= 'latin1')

nba = nba[(nba['weight'] > 0) & (nba['height'] > 0)]
print(nba[0:3])
# "Position (pos)"是要预测的特征
class_column = 'position'
feature_columns = ['height', 'weight']

colors = ['b','g','r']
guard = nba[nba['position'] == 'Guard']
plt.scatter(guard['height'],guard['weight'],c=colors[0])
forward = nba[nba['position'] == 'Forward']
plt.scatter(forward['height'],forward['weight'],c=colors[1])
center = nba[nba['position'] == 'Center']
plt.scatter(center['height'],center['weight'],c=colors[2])
plt.xlabel('height',fontsize=13)
plt.ylabel('weight',fontsize=13)
# plt.show()

#Pandas DataFrame允许选择列
#使用列选择将数据分为特征和类别
nba_feature = nba[feature_columns]
nba_class = nba[class_column]

print(nba_feature[0:3])
print(list(nba_class[0:3]))

train_feature, test_feature, train_class, test_class = train_test_split(nba_feature, nba_class, train_size=0.75, test_size=0.25)
km = KMeans(n_clusters = 3, random_state=1).fit(train_feature[['height','weight']])
prediction = km.predict(test_feature)
def calculateAccuracy(prediction, test_class, test_feature):
	prediction_list = prediction.tolist()
	test_class_list = test_class.tolist()
	test_feature_height_list = test_feature.height.tolist()
	#预测正确的数量
	num = 0
	#总数量
	total = len(test_class_list)
	guard = -1
	forward = -1
	center = -1
	listNum = [0,1,2]
	print("Test set predictions:\n{}".format(prediction))
	print("\nModel using KMeans Classifier using 75/25")

	for i in range(total):
		if test_feature_height_list[i] <= 5.8:
			guard = prediction_list[i]
		if test_feature_height_list[i] >= 7.3:
			center = prediction_list[i]

	listNum.remove(guard)
	listNum.remove(center)
	forward = listNum[0]
	for i in range(total):
		if (test_class_list[i] == 'Guard') & (prediction_list[i] == guard):
			num = num + 1
		elif (test_class_list[i] == 'Forward') & (prediction_list[i] == forward):
			num = num + 1
		elif (test_class_list[i] == 'Center') & (prediction_list[i] == center):
			num = num + 1

	print(guard)
	print(center)
	print(forward)

	return num/total

print("Test set accuracy: {:.2f}".format(calculateAccuracy(prediction, test_class, test_feature)))
train_class_df = pd.DataFrame(train_class,columns=[class_column])
train_data_df = pd.merge(train_class_df, train_feature, left_index=True, right_index=True)
train_data_df.to_csv('KMeans_train_data.csv', index=False)
temp_df = pd.DataFrame(test_class,columns=[class_column])
temp_df['Predicted Pos']=pd.Series(prediction, index=temp_df.index)
test_data_df = pd.merge(temp_df, test_feature, left_index=True, right_index=True)
test_data_df.to_csv('KMeans_test_data.csv', index=False)

guard = test_data_df[test_data_df['Predicted Pos'] == 0]
plt.scatter(guard['height'],guard['weight'],c=colors[0])
forward = test_data_df[test_data_df['Predicted Pos'] == 1]
plt.scatter(forward['height'],forward['weight'],c=colors[1])
center = test_data_df[test_data_df['Predicted Pos'] == 2]
plt.scatter(center['height'],center['weight'],c=colors[2])
plt.xlabel('height',fontsize=13)
plt.ylabel('weight',fontsize=13)
plt.show()

# scores = cross_val_score(km, nba_feature, nba_class, cv=10)
# print("Cross-validation scores: {}".format(scores))
# print("Average cross-validation score: {:.2f}".format(scores.mean()))