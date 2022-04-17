import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix

# 第一題

xFleName = "iris_x.txt"
yFleName = "iris_y.txt"

xData = []
yData = []


with open(xFleName, 'r') as f:
    for line in f.readlines():
        xData.append(list(map(float, line.split("\t")[:-1])))

with open(yFleName, 'r') as f:
    for line in f.readlines():
        yData.append(int(line[-2]))

x_train, x_test, y_train, y_test = train_test_split(
    xData, yData, test_size=0.2, random_state=20220413)

# 第二題
mp = LinearRegression()
mp.fit(x_train, y_train)

multipleRegressionResult = mp.predict(x_test)

print("第二題")
print("MSE:" + str(mean_squared_error(y_test, multipleRegressionResult)))


# 第三題
class Gaussian_classifier:
    def ___init__(self):
        self.mu = np.array([])
        self.cov = np.array([])

    def fit(self, data_train, label_train):
        mu, cov = [], []
        data_trainNp = np.array(data_train)
        label_trainNp = np.array(label_train)
        for i in range(np.max(label_trainNp)+1):
            pos = np.where(label_trainNp == i)[0]
            tmp_data = data_trainNp[pos, :]
            tmp_cov = np.cov(np.transpose(tmp_data))
            tmp_mu = np.mean(tmp_data, axis=0)
            mu.append(tmp_mu)
            cov.append(tmp_cov)
        self.mu = np.array(mu)
        self.cov = np.array(cov)

    def predict(self, x_test):
        d_value = []
        for tmp_mu, tmp_cov in zip(self.mu, self.cov):
            zero_center_data = x_test - tmp_mu
            cov = np.dot(zero_center_data.transpose(), np.linalg.inv(tmp_cov))
            cov = np.dot(cov, zero_center_data)
            tmp = np.log(1/3) - 0.5 * np.log(abs(np.linalg.det(tmp_cov))) - 0.5 * cov
            d_value.append(tmp)
        d_value = np.array(d_value)
        return np.argmax(d_value), d_value



gs = Gaussian_classifier()

gs.fit(x_train, y_train)

gs.predict(x_test[0])
gsResult = []

for test in np.array(x_test):
    gsResult.append(gs.predict(test)[0])

cm = confusion_matrix(y_test, gsResult)
acc = np.diag(cm).sum() / cm.sum()

print("第三題")
print('confusion_matrix (QDA):\n{}'.format(cm))
print('confusion_matrix (QDA,acc):{}'.format(acc))
# 第四題
qd = QuadraticDiscriminantAnalysis()
qd.fit(x_train, y_train)

qdaResult = qd.predict(x_test)

cm = confusion_matrix(y_test, qdaResult)
acc = np.diag(cm).sum() / cm.sum()

print("第四題")
print('confusion_matrix (QDA):\n{}'.format(cm))
print('confusion_matrix (QDA,acc):{}'.format(acc))
