##### DRAFT WRITTEN in 2020.03.25
##### KETI SoC Flatform Research Center
##### IM SEUNGWOO

import numpy as np
import scipy
from sklearn.metrics import r2_score, recall_score, f1_score, mean_squared_error


class NotHavePredictMethodError(Exception):
    pass


class NotFittedYetError(Exception):
    pass


class MIV:
    def __init__(self, Model=None, threshold=0.9, zeta=0.1, score=None, is_clf=False):
        self.Model = Model  # This Object MUST HAVE "self.predict" Method
        self.threshold = threshold
        self.zeta = zeta
        self.score = score
        self.is_clf = is_clf  # is Classifier(True) OR Regressor(False)
        self.is_fitted = False

    def fit(self, X, y):
        self.X = X
        self.y = y

        # CHECK "Model" Object Has a "self.predict()" Method
        if "predict" not in dir(self.Model):
            raise NotHavePredictMethodError(
                'Object "Model" MUST HAVE "predict" Method.'
            )

        # Modify Some Data, Calculate Impact Value, and Get Mean Impact Value.
        if self.is_clf:
            ## MIV for Regression Model
            num_features = X.shape[1]
            X_1 = X.copy()
            X_2 = X.copy()
            self.X_1_dbg = np.zeros(X.shape)
            self.X_2_dbg = np.zeros(X.shape)
            self.IV = np.zeros(X.shape)

            for i in range(num_features):
                # Modify Data with "zeta" param
                X_1[:, i] = X_1[:, i] * (1.0 + self.zeta)
                X_2[:, i] = X_2[:, i] * (1.0 - self.zeta)
                self.X_1_dbg[:, i] = X_1[:, i]
                self.X_2_dbg[:, i] = X_2[:, i]

                ## Calculate Impact Value
                Y_1 = self.Model.predict_proba(X_1).max(axis=1)
                Y_2 = self.Model.predict_proba(X_2).max(axis=1)

                ## Inverse sigmoid
                Y_1 = -1 * np.log(1 / Y_1 - 1)
                Y_2 = -1 * np.log(1 / Y_2 - 1)

                self.IV[:, i] = Y_1 - Y_2

                ##Reset Data
                X_1 = X.copy()
                X_2 = X.copy()

            # Calculate MIV from IV
            self.MIV = self.IV.mean(0)

            # Calculate Contribution Factor from MIV
            sel_idx = np.argsort(abs(self.MIV))[::-1]
            sum_MIV = np.sum(abs(self.MIV))
            cum_MIV = 0
            th_idx = 0
            self.explained_MIV = {}

            for i, val in enumerate(sel_idx):
                tmp_MIV = abs(self.MIV)[val] / sum_MIV
                self.explained_MIV[val] = tmp_MIV

            self.explained_variance_ratio_ = list(self.explained_MIV.values())

            for i, val in enumerate(sel_idx):
                tmp_MIV = abs(self.MIV)[val] / sum_MIV
                cum_MIV = cum_MIV + tmp_MIV

                # Cutting off with 'threshold'
                if cum_MIV >= self.threshold:
                    cum_MIV = cum_MIV - tmp_MIV
                    th_idx = i - 1
                    break

            self.selected_idx = sel_idx[0 : th_idx + 1]
            self.selected = np.sort(self.selected_idx)

        else:
            ## MIV for Regression Model
            num_features = X.shape[1]
            X_1 = X.copy()
            X_2 = X.copy()
            self.X_1_dbg = np.zeros(X.shape)
            self.X_2_dbg = np.zeros(X.shape)
            self.IV = np.zeros(X.shape)

            for i in range(num_features):
                # Modify Data with "zeta" param
                X_1[:, i] = X_1[:, i] * (1.0 + self.zeta)
                X_2[:, i] = X_2[:, i] * (1.0 - self.zeta)
                self.X_1_dbg[:, i] = X_1[:, i]
                self.X_2_dbg[:, i] = X_2[:, i]

                # Calculate Impact Value
                Y_1 = self.Model.predict(X_1)
                Y_2 = self.Model.predict(X_2)
                self.IV[:, i] = Y_1 - Y_2

                ##Reset Data
                X_1 = X.copy()
                X_2 = X.copy()

            # Calculate MIV from IV
            self.MIV = self.IV.mean(0)

            # Calculate Contribution Factor from MIV
            sel_idx = np.argsort(abs(self.MIV))[::-1]
            sum_MIV = np.sum(abs(self.MIV))
            cum_MIV = 0
            th_idx = 0
            self.explained_MIV = {}

            for i, val in enumerate(sel_idx):
                tmp_MIV = abs(self.MIV)[val] / sum_MIV
                self.explained_MIV[val] = tmp_MIV

            self.explained_variance_ratio_ = list(self.explained_MIV.values())

            for i, val in enumerate(sel_idx):
                tmp_MIV = abs(self.MIV)[val] / sum_MIV
                cum_MIV = cum_MIV + tmp_MIV

                # Cutting off with 'threshold'
                if cum_MIV >= self.threshold:
                    cum_MIV = cum_MIV - tmp_MIV
                    th_idx = i - 1
                    break

            self.selected_idx = sel_idx[0 : th_idx + 1]
            self.selected = np.sort(self.selected_idx)

        self.is_fitted = True

        return self

    def transform(self, X, y):
        self.X = X
        self.y = y

        if self.is_fitted:
            return X[:, self.selected]

        else:
            raise NotFittedYetError("FIT Model First")

    def fit_transform(self, X, y):
        self.X = X
        self.y = y

        if self.is_fitted:
            return X[:, self.selected]

        else:
            self.fit(X, y)
            return X[:, self.selected]
