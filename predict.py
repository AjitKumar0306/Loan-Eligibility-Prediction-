import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

# Training Model
data = pd.read_csv("data.csv")
# print(data.head())
# print(data.shape)
# print(data.info())
# print(data.describe())
# print(pd.crosstab(data['Credit_History'], data['Loan_Status'], margins=True))
# print(data.boxplot(column='ApplicantIncome'))

# print(data['ApplicantIncome'].hist(bins=20))
# print(data['CoapplicantIncome'].hist(bins=20))

# data.boxplot(column='ApplicantIncome', by='Education')
# data.boxplot(column='LoanAmount')
# data['LoanAmount'].hist()
# plt.show()


# Normalization

data['LoanAmount_log'] = np.log(data['LoanAmount'])
data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']
data['TotalIncome_log'] = np.log((data['TotalIncome']))
data['TotalIncome_log'].hist(bins=20)

# Handling Missing Value
data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
data['Married'].fillna(data['Married'].mode()[0], inplace=True)
data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)
data['Credit_History'].fillna(data['Dependents'].mode()[0], inplace=True)

data.LoanAmount = data.LoanAmount.fillna(data.LoanAmount.mean())
data.LoanAmount_log = data.LoanAmount_log.fillna(data.LoanAmount_log.mean())

# Dividing Datasets in to dependent and independent variables X = independent y = dependent

x = data.iloc[:, np.r_[1:5, 9:11, 13:15]].values
y = data.iloc[:, 12].values

# Splitting dataset into train and test dataset

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Converting into O's & 1's using label encoder

label_encoder_X = LabelEncoder()
for i in range(0, 5):
    x_train[:, i] = label_encoder_X.fit_transform(x_train[:, i])
x_train[:, 7] = label_encoder_X.fit_transform(x_train[:, 7])

for i in range(0, 5):
    x_test[:, i] = label_encoder_X.fit_transform(x_test[:, i])
x_test[:, 7] = label_encoder_X.fit_transform(x_test[:, 7])


label_encoder_Y = LabelEncoder()
y_train = label_encoder_Y.fit_transform(y_train)
y_test = label_encoder_Y.fit_transform(y_test)

# Scaling data
scale = StandardScaler()
x_test = scale.fit_transform(x_test)
x_train = scale.fit_transform(x_train)

# Applying Algorithms

# Decision Tree
DTC = DecisionTreeClassifier(criterion='entropy', random_state=0)
DTC.fit(x_train, y_train)
Predict_y = DTC.predict(x_test)

# Finding Accuracy
# print("The accuracy of decision tree: ", metrics.accuracy_score(Predict_y, y_test))

# Applying naive bayes algorithm
NB = GaussianNB()
NB.fit(x_train, y_train)

Predict_x = NB.predict(x_test)

# print("The accuracy of naive bayes: ", metrics.accuracy_score(Predict_x, y_test))


# Prediction / Testing

test_data = pd.read_csv('test.csv')
# Handling Missing Value
test_data['Gender'].fillna(test_data['Gender'].mode()[0], inplace=True)
test_data['Married'].fillna(test_data['Married'].mode()[0], inplace=True)
test_data['Dependents'].fillna(test_data['Dependents'].mode()[0], inplace=True)
test_data['Self_Employed'].fillna(test_data['Self_Employed'].mode()[0], inplace=True)
test_data['Loan_Amount_Term'].fillna(test_data['Loan_Amount_Term'].mode()[0], inplace=True)
test_data['Credit_History'].fillna(test_data['Dependents'].mode()[0], inplace=True)
test_data.LoanAmount = test_data.LoanAmount.fillna(test_data.LoanAmount.mean())

# Normalization
test_data['LoanAmount_log'] = np.log(test_data['LoanAmount'])
test_data['TotalIncome'] = test_data['ApplicantIncome'] + test_data['CoapplicantIncome']
test_data['TotalIncome_log'] = np.log((test_data['TotalIncome']))

# selecting data in to variable

test_x = test_data.iloc[:, np.r_[1:5, 9:11, 13:15]].values

# Converting into O's & 1's using label encoder


for i in range(0, 5):
    test_x[:, i] = label_encoder_X.fit_transform(test_x[:, i])
test_x[:, 7] = label_encoder_X.fit_transform(test_x[:, 7])


# Scaling data

test_x = scale.fit_transform(test_x)

# Applying naive bayes algorithm

fPredict_x = NB.predict(test_x)
print(fPredict_x)

