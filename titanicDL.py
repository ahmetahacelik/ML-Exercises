import pandas as pd




veriler = pd.read_csv('train.csv')

X= veriler.iloc[:,[2,4,5,6,7,9,11]].values
Y = veriler.iloc[:,1].values
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X[:,1] = le.fit_transform(X[:,1])

le2 = LabelEncoder()
X[:,6] = le2.fit_transform(X[:,6])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[1])

X=ohe.fit_transform(X).toarray()
X = X[:,:]

veriler2=pd.read_csv('test.csv')

X2= veriler2.iloc[:,[1,3,4,5,6,8,10]].values
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X2[:,1] = le.fit_transform(X2[:,1])

le2 = LabelEncoder()
X2[:,6] = le2.fit_transform(X2[:,6])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[1])

X2=ohe.fit_transform(X2).toarray()
X2 = X2[:,:]



x_train=X
x_test=X2
y_train=Y

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(4, init = 'uniform', activation = 'relu' , input_dim = 8))

classifier.add(Dense(4, init = 'uniform', activation = 'relu'))

classifier.add(Dense(1, init = 'uniform', activation = 'sigmoid'))



classifier.compile(optimizer = 'adam', loss =  'binary_crossentropy' , metrics = ['accuracy'] )

classifier.fit(X_train, y_train, epochs=50)

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)
