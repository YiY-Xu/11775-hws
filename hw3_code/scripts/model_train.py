import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score
from sklearn.svm.classes import SVC
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.model_selection import ParameterGrid
from sklearn.multiclass import OneVsRestClassifier
import copy

class Model_Train:
    def __init__(self, feature, model):
        self.models = []
        self.best_models = []
        self.base_model = model
        self.feature = feature
        self.kf = KFold(n_splits = 5, shuffle=False)

    def train_cv_multiclass(self, grid, balanced):
        best_models = []
        X = np.concatenate((self.feature.train_X, self.feature.val_X), axis=0) if not balanced else self.feature.selected_X.copy()

        for g in ParameterGrid(grid):
            mAP = []
            current_models = []
            best = -1
            for i in range(3):
                y = np.concatenate((self.feature.train_yy, self.feature.val_yy), axis=0) if not balanced else self.feature.selected_yy.copy()
                AP = []
                clf = copy.deepcopy(self.base_model)
                clf.set_params(**g)
                clf =  OneVsRestClassifier(clf)
                for train_index, test_index in self.kf.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    clf.fit(X_train, y_train)
                    if hasattr(clf, 'decision_function'):
                        y_pred = clf.decision_function(X_test)
                    else:
                        y_pred = clf.predict(X_test)
                    AP.append(average_precision_score(y_test, y_pred, average="micro"))
                mAP.append(np.mean(AP))
                current_models.append(clf)
            print(mAP, np.mean(mAP), g)    
            if (np.mean(mAP) > best):
                best = np.mean(mAP)
                best_models = current_models
        self.best_models = best_models

    def train_cv(self, grid, balanced):
        best_models = []
        X = np.concatenate((self.feature.train_X, self.feature.val_X), axis=0) if not balanced else self.feature.selected_X.copy()
        print(X.shape)

        for g in ParameterGrid(grid):
            mAP = []
            current_models = []
            best = -1
            for i in range(3):
                y = np.concatenate((self.feature.train_y, self.feature.val_y), axis=0) if not balanced else self.feature.selected_y.copy()
                y[y!=i+1]=0
                y[y!=0]=1
                AP = []
                clf = copy.deepcopy(self.base_model)
                clf.set_params(**g)
                for train_index, test_index in self.kf.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    clf.fit(X_train, y_train)
                    if hasattr(clf, 'decision_function'):
                        y_pred = clf.decision_function(X_test)
                    else:
                        y_pred = clf.predict(X_test)
                    ap_score = average_precision_score(y_test, y_pred)
                    if np.isnan(ap_score):
                        ap_score = 0
                    AP.append(ap_score)
                mAP.append(np.mean(AP))
                current_models.append(clf)
            print(mAP, np.mean(mAP), g)    
            if (np.mean(mAP) > best):
                best = np.mean(mAP)
                best_models = current_models
        self.best_models = best_models

    def train(self, g):
        self.models = []
        X = self.feature.train_X.copy()
        for i in range(3):
            y = self.feature.train_y.copy()
            y[y!=i+1]=0
            y[y!=0]=1
            clf = copy.deepcopy(self.base_model)
            clf.set_params(**g)
            self.models.append(clf.fit(X, y))

    def validate(self):
        if len(self.models) == 0:
            raise Exception('Model has not been initialized')
        index = 1
        AP = []
        Majority = []
        val_X = self.feature.val_X.copy()

        models = self.models
        for model in models:
            y_true = self.feature.val_y.copy()
            y_true[y_true!=index]=0
            y_true[y_true!=0]=1
            if hasattr(model, 'decision_function'):
                y_pred = model.decision_function(val_X)
            else:
                y_pred = model.predict(val_X)
            average_precision = average_precision_score(y_true, y_pred)
            AP.append(average_precision)
            index += 1


            from sklearn.metrics import precision_recall_curve
            import matplotlib.pyplot as plt
            from sklearn.utils.fixes import signature

            precision, recall, threshold = precision_recall_curve(y_true, y_pred)

            # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
            step_kwargs = ({'step': 'post'}
                           if 'step' in signature(plt.fill_between).parameters
                           else {})
            plt.step(recall, precision, color='b', alpha=0.2,
                     where='post')
            plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

            plt.savefig('AUC' + self.feature.feat_types[0] + str(index) + '.png')
        mAP = np.mean(np.array(AP))

        return AP, mAP

    def test(self, kaggle=False):
        self.test_result = self.apply_best_model(self.feature.test_X, kaggle)

    def process_train_val(self):
        self.train_result = self.apply_best_model(self.feature.train_X)
        self.val_result = self.apply_best_model(self.feature.val_X)

    def persist_test_result(self, stage):
        if len(self.test_result) == 0:
            raise Exception('Test has not been performed')
        for index in range(3):
            output_file = '../output/' + 'P00' + str(index+1) + '_' + stage + '.lst'
            with open(output_file, 'w') as f:
                for file, score in zip(self.feature.test_name, self.test_result[index]):
                    value = file + ' ' + str(score)
                    f.write("%s\n" % score)

    def apply_best_model(self, input, kaggle=False):
        if len(self.best_models) == 0:
            raise Exception('Model has not been initialized')
        result = []
        X = input.copy()
        for i, model in enumerate(self.best_models):
            if not kaggle:
                if hasattr(model, 'decision_function'):
                    y_pred = model.decision_function(X)
                else:
                    y_pred = model.predict_proba(X)
            else:
                y_pred = model.predict(X)
            result.append(y_pred)
        return result