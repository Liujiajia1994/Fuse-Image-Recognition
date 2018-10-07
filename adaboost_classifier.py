import numpy as np
from numpy.core.umath_tests import inner1d
from sklearn.ensemble import AdaBoostClassifier


class AdaCostClassifier(AdaBoostClassifier):
    def _boost_real(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME.R real algorithm."""
        estimator = self._make_estimator(random_state=random_state)
        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict_proba = estimator.predict_proba(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1),
                                       axis=0)

        incorrect = y_predict != y

        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

        if estimator_error <= 0:
            return sample_weight, 1., 0.

        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1. / (n_classes - 1), 1.])
        y_coding = y_codes.take(classes == y[:, np.newaxis])
        proba = y_predict_proba  # alias for readability
        proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps

        estimator_weight = (-1. * self.learning_rate
                            * (((n_classes - 1.) / n_classes) *
                               inner1d(y_coding, np.log(y_predict_proba))))

        # 样本更新的公式，只需要改写这里
        if not iboost == self.n_estimators - 1:
            sample_weight *= np.exp(estimator_weight *
                                    ((sample_weight > 0) |
                                     (estimator_weight < 0)) *
                                    self._beta(y, y_predict))  # 在原来的基础上乘以self._beta(y, y_predict)，即代价调整函数
        return sample_weight, 1., estimator_error


        #  新定义的代价调整函数

    def _beta(self, y, y_hat):
        res = []
        for i in zip(y, y_hat):
            if i[0] == i[1]:
                res.append(1)  # 正确分类，系数保持不变，按原来的比例减少
            elif i[0] == 1 and i[1] == -1:
                res.append(1.25)  # 在信用卡的情景下，将好人误杀代价应该更大一些，比原来的增加比例要高
            elif i[0] == -1 and i[1] == 1:
                res.append(1)  # 将负例误判为正例，代价不变，按原来的比例增加
            else:
                print(i[0], i[1])

        return np.array(res)