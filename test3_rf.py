import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

diabetes = datasets.load_diabetes()

X_train,X_test,y_train,y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.20)

class MetaLearner(object):
    def __init__(self,min_smaples,max_depth):
        self.tree = {}
        self.min_gain = 0
        self.min_smaples = min_smaples
        self.max_depth = max_depth
        self.max_leaves = 100
        self.X = []
        self.y = []

    #计算方差
    def CalculateVar(self, collection):
        y_mean = 0
        y_var = 0
        for i in collection:
            y_mean += self.y[i]
        if len(collection) != 0:
            y_mean = y_mean/len(collection)
        else:
            y_mean = 0
        for i in collection:
            y_var += (self.y[i]-y_mean)**2
        if len(collection) != 0:
            y_var = y_var / len(collection)
        else:
            y_var = 0
        return y_var

    # 计算增益
    def infor_Gain(self, col, threshold, collection):
        L, R = [], []
        for i in collection:
            if self.X[i][col] <= threshold:
                L.append(i)
            else:
                R.append(i)

        old_var = self.CalculateVar(collection)
        var_l = self.CalculateVar(L)
        var_r = self.CalculateVar(R)
        new_Var = old_var - var_l * len(L)/len(collection) - var_r * len(R)/len(collection)
        return new_Var

    # 寻找最适合划分的属性、阈值、信息增益
    def best_col(self, collection):
        best_threshold = 0
        max_var = 0
        best_col = 0
        for i in range(len(self.X[0])):
            new_var, threshold = self.best_point(i, collection)
            if new_var > max_var:
                best_threshold = threshold
                max_var = new_var
                best_col = i
        return best_col, best_threshold, max_var

    # 寻找某一个属性的最佳划分点，返回最大增益和阈值
    def best_point(self, col, collection):
        max_var = 0
        threshold = 0
        x_set = list(set(self.X[i][col] for i in collection))
        for i in x_set:
            Gain = self.infor_Gain(col, i, collection)
            if Gain > max_var:
                threshold = i
                max_var = Gain
        return max_var, threshold

    # 判断集合中的点是否属于同一类
    def is_same(self, cols):
        y_class = self.y[cols[0]]
        for i in cols:
            if self.y[i] != y_class:
                return False
        return True

    def build_tree(self, collection, depth):
        root = {'col': None, # 按哪一列划分
             'value':None, # 划分的值
             'L':None, # 左节点
             'R':None, # 右节点
             }
        l_cols = []
        r_cols = []
        root['col'], root['value'], after_Gain = self.best_col(collection)
        col = root['col']
        threshold = root['value']
        if len(collection) < self.min_smaples:#当样本数小于min_samples时停止划分
            return (collection,self.get_y(collection))
        if self.is_same(collection):#当数据同一个类时停止划分
            return (collection,self.get_y(collection))
        if depth > self.max_depth:#当树的深度大于某个值时停止划分
            return (collection,self.get_y(collection))
        if (2**depth-1) > self.max_leaves:#当叶子节点数目大于max_leaves时停止划分
            return (collection,self.get_y(collection))
        if after_Gain <= self.min_gain:#当Gain小于0时停止划分
            return (collection, self.get_y(collection))

        for i in collection:
            # 划分节点
            if self.X[i][col] <= threshold:
                l_cols.append(i)
            if self.X[i][col] > threshold:
                r_cols.append(i)
        root['L'] = self.build_tree(l_cols, depth+1)
        root['R'] = self.build_tree(r_cols, depth+1)
        return root

    def fit(self, X, y):
        self.X = list(X)
        self.y = list(y)
        collections = range(len(self.X))
        self.tree = self.build_tree(collections, 1)

    def get_y(self, results):
        Y = 0
        for i in results:
            Y += self.y[i]
        y_mean = Y/len(results)
        return y_mean

    def predict(self, X):
        y_predict = []
        for i in X:
            pre_col = self.tree
            while type(pre_col) == dict:
                col = pre_col['col']
                threshold = pre_col['value']
                if i[col] <= threshold:
                    pre_col = pre_col['L']
                else:
                    pre_col = pre_col['R']
            y_predict.append(pre_col[1])
        return np.array(y_predict)

class RandomForestRegressor_(object):
    def __init__(self,N):
        self.N = N
        self.sX = []
        self.sy = []
        self.trees = []

    def fit(self, sX, sy):
        self.sX = list(sX)
        self.sy = list(sy)
        sum_samples = len(self.sX)
        for i in range(self.N):
            X_train, X_test, y_train, y_test = train_test_split(sX, sy, test_size=0.20)
            m = MetaLearner(min_smaples=5,max_depth=20)
            m.fit(X_train,y_train)
            self.trees.append(m.tree)

    def predict(self, sX):
        y_predict = []
        for i in sX:
            y_n_pre = []
            for n in range(self.N):
                pre_col = self.trees[n]
                while type(pre_col) == dict:
                    col = pre_col['col']
                    threshold = pre_col['value']
                    if i[col] <= threshold:
                        pre_col = pre_col['L']
                    else:
                        pre_col = pre_col['R']
                y_n_pre.append(pre_col[1])

            sum = 0
            for n in range(self.N):
                sum += y_n_pre[n]
            y_predict.append(sum/self.N)
        return np.array(y_predict)

m = MetaLearner(min_smaples=5,max_depth=20)
m.fit(X_train,y_train)
y_pre = list(m.predict(X_test))
print("自写决策树(MAE)：", mean_absolute_error(y_pre, list(y_test)))
print("自写决策树(MSE)：", mean_squared_error(y_pre, list(y_test)))


from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import sklearn
rfr = RandomForestRegressor(n_estimators=100, random_state=0)
rfr.fit(X_train, y_train)
sk_pre = list(rfr.predict(X_test))
print("sklearn(MAE): ", mean_absolute_error(sk_pre, list(y_test)))
print("sklearn(MSE): ", mean_squared_error(sk_pre, list(y_test)))

MAE = []
MSE = []
for i in range(1, 30):
    rfr = RandomForestRegressor_(i)
    rfr.fit(X_train, y_train)
    pre=list(rfr.predict(X_test))
    MAE.append(mean_absolute_error(pre, list(y_test)))
    MSE.append(mean_squared_error(pre, list(y_test)))

def show_zx(N, y, y_la):
    plt.plot(N, y, linewidth=5)
    plt.xlabel("N", fontsize=14)
    plt.ylabel(y_la, fontsize=14)
    plt.show()

show_zx(range(1, 30), MAE, "MAE")
show_zx(range(1, 30), MSE, "MSE")
