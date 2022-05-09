import numpy as np

class NaiveBayes:
    def __init__(self, df_train, df_test, no_labels):
        self.no_labels = no_labels
        self.N = df_train.shape[0]
        self.df_train = df_train
        self.df_test = df_test
        self.featureSet = ["Fear", "Anger", "Guilt", "Joy", "Shame", "Disgust", "Sadness", "GuiltUnique",
                           "ShameUnique", "JoyUnique", "Negative", "length_binned", "ttr_binned"]
        self.map_data_by_class = {}
        self.class_prob = [0] * self.no_labels
        self.evidence_prob = [[0] * len(self.featureSet) for i in range(self.no_labels)]

        self.pred_prob = [[0] * len(self.no_labels) for i in range(df_test.shape[0])]

    def naive_bayes_fit(self):

        # split data by class
        for i in range(1, self.no_labels + 1):
            self.map_data_by_class[i] = self.df_train[self.df_train['Emotion-Target'] == i]

            # compute prior probability of each class P(y=i)
            self.class_prob[i - 1] = self.map_data_by_class.get(i).shap[0] / self.N

        # compute likelihood of evidence in each class
        for i in range(1, self.no_labels + 1):
            for j, feature in enumerate(self.featureSet):
                cdf = self.map_data_by_class.get(i)  # class i
                self.evidence_prob[i - 1][j] = cdf[cdf[feature] == 1].shape[0] / cdf.shape[
                    0]  # prob. that the feature is 1 in class i

    def naive_bayes_predict(self):
        pred_prob = []
        for i in range(self.no_labels):
            # multiply test data feature values by the evidence probabilities
            evidence_prob = [
                self.df_test[self.featureSet[j]] * self.evidence_prob[i][j] + (1 - self.df_test[self.featureSet[j]]) * (
                1 - self.evidence_prob[i][j]) for j in range(len(self.featureSet))]
            # divide probability by probability of the class
            pred_prob.append(np.prod(evidence_prob) / self.class_prob[i])

        # set class with max prob as predicted class
        self.df_test["Target-Predict"] = np.argmax(pred_prob) + 1