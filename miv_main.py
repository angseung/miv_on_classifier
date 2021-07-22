from datetime import datetime
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from MIV.miv_dimention_reduction import MIV

X, y = load_wine(return_X_y=True)
mms = MinMaxScaler()
X = mms.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, shuffle=True)

lr = LogisticRegression(max_iter=100, verbose=0, solver='liblinear')

lr_train_pre = datetime.now()
lr.fit(X_train, y_train)
lr_train_af = datetime.now()

print(lr.score(X_train, y_train), lr_train_af - lr_train_pre)
print(lr.score(X_test, y_test))

miv = MIV(Model=lr,
          threshold=0.85,
          zeta=0.1,
          is_clf=True)

miv.fit(X_train, y_train)

X_train_miv = X_train[:, miv.selected_idx]
X_test_miv = X_test[:, miv.selected_idx]

lr_miv = LogisticRegression(max_iter=100, verbose=0, solver='liblinear')

lr_train_pre = datetime.now()
lr_miv.fit(X_train_miv, y_train)
lr_train_af = datetime.now()

print(lr_miv.score(X_train_miv, y_train), lr_train_af - lr_train_pre)
print(lr_miv.score(X_test_miv, y_test))
