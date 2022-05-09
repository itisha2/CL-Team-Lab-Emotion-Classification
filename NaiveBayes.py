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

    def naive_bayes_fit(self):

        # split data by class
        print("data size: ", self.N, "\n")
        for i in range(1, self.no_labels + 1):
            self.map_data_by_class[i] = self.df_train[self.df_train['Emotion-Target'] == i]

            # compute prior probability of each class P(y=i)
            self.class_prob[i - 1] = self.map_data_by_class.get(i).shape[0] / self.N

            print("size of class ", i, ": ", self.map_data_by_class.get(i).shape[0])

        print("class probabilities:", self.class_prob, "\n")

        # compute likelihood of evidence in each class
        for i in range(1, self.no_labels + 1):
            for j, feature in enumerate(self.featureSet):
                cdf = self.map_data_by_class.get(i)  # class i
                self.evidence_prob[i - 1][j] = cdf[cdf[feature] == 1].shape[0] / cdf.shape[
                    0]  # prob. that the feature is 1 in class i
        print("Evidence Probability: ", self.evidence_prob, "\n")

    def naive_bayes_predict(self):
        self.df_test["pred_prob"] = pd.Series([])
        for i in range(self.no_labels):
            # multiply test data feature values by the evidence probabilities (NOMINATOR)
            self.df_test["prob_nom"] = pd.Series([])
            for j in range(len(self.featureSet)):
                feature = self.featureSet[j]
                self.df_test.loc[self.df_test[feature] == 1]["prob_nom"] *= self.evidence_prob[i][j]
                self.df_test.loc[self.df_test[feature] == 0]["prob_nom"] *= (1 - self.evidence_prob[i][j])
            print(self.df_test["prob_nom"], "\n")
        # divide probability by probability of the class
        self.df_test["pred_prob"].append(self.df_test["prob_nom"] / self.class_prob[i])

        # set class with max prob as predicted class
        self.df_test["Target-Predict"] = np.argmax(self.df_test["pred_prob"]) + 1

