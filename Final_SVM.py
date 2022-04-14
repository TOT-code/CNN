import matplotlib
import numpy as np
import sklearn
from matplotlib import pyplot as plt
from sklearn import svm


# 数据长度：1981
# define converts(字典)
def Road_label(s):
    it = {b'bus': 0, b'car': 1, b'man': 2}
    return it[s]


# 1.读取数据集
path = r'SVM.data'
data = np.loadtxt(path, dtype=float, delimiter=',', converters={1980: Road_label})
# converters={4:Iris_label}中“4”指的是第5列：将第5列的str转化为label(number)

# 2.划分数据与标签
x, y = np.split(data, indices_or_sections=(1980,), axis=1)  # x为数据，y为标签
x = x[:, 0:200]  # 为便于后边画图显示，只选取前两维度。若不用画图，可选取前四列x[:,0:4]
train_data, test_data, train_label, test_label = \
    sklearn.model_selection.train_test_split \
        (x, y, random_state=0, train_size=0.75, test_size=0.25)

# 3.训练svm分类器
classifier = svm.SVC(C=2, kernel='rbf', gamma=0.9,
                     decision_function_shape='ovr')  # ovr:一对多策略
classifier.fit(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先

# 4.计算svc分类器的准确率
print("训练集：", classifier.score(train_data, train_label))
print("测试集：", classifier.score(test_data, test_label))

"""
#也可直接调用accuracy_score方法计算准确率
from sklearn.metrics import accuracy_score
tra_label=classifier.predict(train_data) #训练集的预测标签
tes_label=classifier.predict(test_data) #测试集的预测标签
print("训练集：", accuracy_score(train_label,tra_label) )
print("测试集：", accuracy_score(test_label,tes_label) )
"""
