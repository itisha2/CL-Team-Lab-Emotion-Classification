"""

Approach:

Using Language Models
1. Assumption for Naive Bayes : All words are independent
slide no 38.
2. Naive Bayes is a probabilistic classifier.
3. We compute the probability of document being in class C.
4. The goal of the naive bayes class is to find the best class.
5. Naive Bayes = probability of each class + prob of each token belonging to that class.
6. Formula = argmax(log(P(c) + log(P(t/c)))
7. Smoothing in NB
8. P(t/c) = T(no of particular token) + 1 / T(sum of all tokens in a class) + V(size of vocab)
9. Naive bayes is good for predicting the class and not for estimating probabilites.
10. Naive Bayes is robust to concept drift. (change of definition of class over time).
11. For text, the independence assumption does not hold for naive bayes, but for other domains it does hold.
12. Advantages:
    1. Very Fast
    2. Low storage requirement
"""

import numpy as np
import re


class NaiveBayesClassifier:
    def __init__(self):
        self.vocabSize = 0
        self.trainingSize = 0
        self.train_labels = set()
        self.vocab = []
        self.fear = {}
        self.anger = {}
        self.guilt = {}
        self.joy = {}
        self.shame = {}
        self.disgust = {}
        self.sadness = {}
        self.fearSize = 0
        self.angerSize = 0
        self.guiltSize = 0
        self.joySize = 0
        self.shameSize = 0
        self.disgustSize = 0
        self.sadnessSize = 0
        self.priorDict = {}
        """
        set of stopwords
        """
        self.stopwords = {'hers', 'below', "wouldn't", 'nor', 'for', 'over', "hasn't", 'at', 'shouldn', 'only', 'above',
                          'itself', 'yourselves', 'what', "don't", "it's", 'which', 'against', "that'll", 'has', 'i',
                          'his', 'having', 'then', "shan't", 'myself', 'do', 'yours', 'up', 'own', 'the', 'same',
                          'aren', 'few', 'through', 'here', 'whom', 'o', "aren't", 'were', 'are', 'both', "didn't",
                          'll', 'again', 'is', 're', "wasn't", "you'll", 'm', "haven't", 'such', 'off', 'of', 'it',
                          'did', 'into', 'to', 'other', 'was', 'just', 've', "mustn't", 'while', 'about', 'each', 'by',
                          'this', 'isn', 'ourselves', 'in', 'our', 'couldn', 'until', 'where', "couldn't", 'ain',
                          "you'd", 'all', 'when', 'does', 'before', 'weren', 'y', 'doing', 'than', 'being', 'my',
                          'mightn', 'yourself', 'with', 'theirs', 'so', "needn't", 'a', "doesn't", "isn't", 'its',
                          'your', 'if', "should've", 'ma', 'can', 'herself', 'but', 'too', 'more', 'her', "hadn't",
                          'hadn', 'there', "you're", 'from', 'should', 'we', 'how', 'out', 'once', 'mustn', 'won',
                          'their', 'don', 'had', 'he', 'or', 'didn', 'd', 'down', 't', "she's", 'that', 'himself',
                          'wouldn', "you've", "mightn't", 'between', 'them', 'on', 'haven', 'after', 'themselves',
                          'because', 'and', 'you', 'very', 's', 'these', 'no', 'now', 'him', 'been', 'those', 'during',
                          'doesn', 'wasn', 'am', 'under', 'an', 'some', 'have', 'me', 'any', 'who', 'shan', 'why',
                          'will', "shouldn't", 'not', 'they', "won't", 'needn', 'further', 'most', 'be', 'ours', 'she',
                          'as', 'hasn', "weren't", "a", "''"}
        self.paraMapping = {"fear": [self.fear, self.fearSize],
                            "anger": [self.anger, self.angerSize],
                            "guilt": [self.guilt, self.guiltSize],
                            "joy": [self.joy, self.joySize],
                            "shame": [self.shame, self.shameSize],
                            "disgust": [self.disgust, self.disgustSize],
                            "sadness": [self.sadness, self.sadnessSize]
                            }
        self.mapping = {"fear": 1, "anger": 2, "guilt": 3, "joy": 4, "shame": 5, "disgust": 6, "sadness": 7}
        self.reverseMapping = {1: "fear", 2: "anger", 3: "guilt", 4: "joy", 5: "shame", 6: "disgust", 7: "sadness"}

    def getVocabulary(self, text):
        tokens = [re.sub("[^A-Za-z]", "", i).strip().lower() for i in text.split(" ") if
                  i not in self.stopwords and len(i) > 1]
        return list(set(tokens))

    def updateProbabilityDict(self, emodf, resDict):
        """
        :used by fit method to update probability dictionary of words for each emotion.
        :param emodf:
        :param resDict: initial emotion dictionary for update.
        :return: frequency count dictionary for an emotion, total corpus size of the emotion.
        """
        text = " ".join(list(emodf["X"])).split(" ")
        for word in text:
            word = re.sub("[^A-Za-z]", "", word).strip().lower()
            if word not in self.stopwords and len(word) > 1:
                if word in resDict:
                    resDict[word] += 1
                else:
                    resDict[word] = 1
        size = sum([value for key, value in resDict.items()])
        return resDict, size

    def maximum_likelihood_estimation(self, instance, emoDict, emotionCorpusSize, emotion):
        """

        :used by predict method to determine maximum likelihood estimation
        :param instance: one test instance
        :param emoDict: emotion dictionary for a particular emotion(Ex: self.fear)
        :param emotionCorpusSize: total corpus size of the emotion
        :param emotion: emotion name
        :return:  log(P(c) + log(P(t/c))
        """

        p_tc = 0
        tokens = [re.sub("[^A-Za-z]", "", i).strip().lower() for i in instance.split(" ") if
                  i not in self.stopwords and len(i) > 1]
        for word in tokens:
            if word in emoDict:
                p_tc += (emoDict[word] + 1) / (emotionCorpusSize + self.vocabSize)
            else:
                p_tc += 1 / (emotionCorpusSize + self.vocabSize)
        return np.log(self.priorDict[emotion]) + np.log(p_tc)

    # def labelEncoding(self, temp):
    #     """
    #
    #     buffer utility, not currently used.
    #     :param temp:
    #     :return:
    #     """
    #
    #     # print(temp["Y"].value_counts())
    #     mapped_emotions = temp["Y"].map(self.mapping)
    #     temp["Y"] = mapped_emotions
    #     temp = temp.dropna().reset_index(drop=True)
    #     return temp

    def cleanData(self, input_df):
        """

        :param input_df: input dataframe
        :return: verified dataframe
        :functionality: discards all those rows from the train dataframe which have label other than the 7 labels in
        isear data-set.
        """
        temp = input_df[input_df["Y"].isin(["fear", "anger", "guilt", "joy", "shame", "disgust", "sadness"])]
        return temp

    def fit(self, train_df):
        """

        :param train_df: training dataframe
        :functionality:
            1) :updates tokenDict... : holds the probability of occurrence of each word in the class, therefore the
            length of dict = vocab size.
            2) :updates priorDict: holds the probability of occurrence of each class
        """

        train_df = self.cleanData(train_df)
        # train_df = self.labelEncoding(train_df)
        self.train_labels = set(train_df["Y"])
        self.vocab = self.getVocabulary(" ".join(list(train_df["X"])))
        self.vocabSize = len(self.vocab)
        self.trainingSize = train_df.shape[0]

        for label in self.train_labels:
            self.priorDict[label] = train_df[train_df["Y"] == label].shape[0] / self.trainingSize
            emoParam = self.paraMapping[label]
            emoParam[0], emoParam[1] = self.updateProbabilityDict(train_df[train_df["Y"] == label], emoParam[0])

    def predict(self, xtest):
        """

        :param xtest: list of test data
        :return: predictions
        :functionality: This method uses the probabilities learned in the
        fit function and applies the formula to get the correct class for the given instance.
        """

        k = 0
        predictions = []
        for instance in xtest:
            k += 1
            # print(k, instance)
            prob = []
            for label in self.train_labels:
                emoPrm = self.paraMapping[label]
                proba_emo = self.maximum_likelihood_estimation(instance, emoPrm[0], emotionCorpusSize=emoPrm[1],
                                                               emotion=label)
                prob.append((label, proba_emo))
            # print(sorted(prob, key = lambda x: x[1]))
            prediction = sorted(prob, key=lambda x: x[1], reverse=True)[0][0]
            predictions.append(prediction)
            # predictions.append(self.mapping[prediction])
        return predictions

