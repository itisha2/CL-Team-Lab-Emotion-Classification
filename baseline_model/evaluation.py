import numpy as np
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


class evaluation:
    """

    :param y_actual: ground_truth
    :param y_pred: predictions
    :param labels: list of unique emotion labels in the training data
    :returns: (each class --> precision, recall, f1score) + marco average
    main_execution_method: main()
    """

    def __init__(self, y_actual, y_pred, labels):
        self.y_actual = np.array(y_actual)
        self.y_pred = np.array(y_pred)
        self.labels = labels

    def confusion_matrix(self, actual, pred):
        """

        :param actual: ground truth
        :param pred: predictions
        :return: confusion matrix
        """
        tp = fp = tn = fn = 0
        for i, j in zip(actual, pred):
            if i == 1:
                """positive"""
                if i == j:
                    tp += 1
                else:
                    fp += 1
            else:
                """negative"""
                if i == j:
                    tn += 1
                else:
                    fn += 1
        cf = pd.DataFrame([[tp, fp], [fn, tn]], columns=["actual_pos", "actual_neg"], index=["pred_pos", "pred_neg"])
        return cf, tp, fp, tn, fn

    def recall(self, actual, pred):
        """

        :param actual: ground truth
        :param pred: predictions
        :return: recall
        """
        cf, tp, fp, tn, fn = self.confusion_matrix(actual, pred)
        return round(tp / (tp + fn), 2)

    def precision(self, actual, pred):
        """

        :param actual: ground truth
        :param pred: predictions
        :return: precision
        """
        cf, tp, fp, tn, fn = self.confusion_matrix(actual, pred)
        return round(tp / (tp + fp), 2)

    def f1(self, actual, pred):
        """

        :param actual: ground truth
        :param pred: predictions
        :return: F1score, harmonic mean of precision and recall
        """
        pr = self.precision(actual, pred)
        re = self.recall(actual, pred)
        f1 = 2 * ((pr * re) / (pr + re))
        return round(f1, 2)

    def main(self):
        """

        :return: flattened_dataframe, normal_dataframe (including, recall, precision, f1score, macro average)
        """
        # mapping = {1: "fear", 2: "anger", 3: "guilt", 4: "joy", 5: "shame", 6: "disgust", 7: "sadness"}
        res = {}
        for cls in self.labels:
            c = 0
            mod_y_actual = []
            for i in self.y_actual:
                if i == cls:
                    c += 1
                    mod_y_actual.append(1)
                else:
                    mod_y_actual.append(0)

            mod_y_pred = []
            for i in self.y_pred:
                if i == cls:
                    mod_y_pred.append(1)
                else:
                    mod_y_pred.append(0)
            """
            print()
            print()
            print("Confusion Matrix : \n", self.confusion_matrix(mod_y_actual, mod_y_pred)[0])
            print("*******************************************************\n")
            print("Precision : \n", self.precision(mod_y_actual, mod_y_pred))
            print("*******************************************************\n")
            print("Recall : \n", self.recall(mod_y_actual, mod_y_pred))
            print("*******************************************************\n")
            print("F1 Score : \n", self.f1(mod_y_actual, mod_y_pred))
            print()
            print()
        """
            temp = [self.precision(mod_y_actual, mod_y_pred), self.recall(mod_y_actual, mod_y_pred),
                    self.f1(mod_y_actual, mod_y_pred), c]
            res[cls] = temp
        res = pd.DataFrame(res, index=["Precision", "Recall", "F1-Score", "Count"]).transpose()

        """
        Macro average calculation
        """
        avg_pr = np.sum((res["Precision"] * res["Count"])) / np.sum(res["Count"])
        avg_re = np.sum((res["Recall"] * res["Count"])) / np.sum(res["Count"])
        avg_f1 = np.sum((res["F1-Score"] * res["Count"])) / np.sum(res["Count"])
        res.loc["macro_average"] = [round(avg_pr, 2), round(avg_re, 2), round(avg_f1, 2), np.sum(res["Count"])]
        # res = pd.concat([res, pd.DataFrame([avg_pr, avg_re, avg_f1, np.sum(res["Count"])]).transpose()])
        flattened_index = []
        for fl_lab in self.labels:
            flattened_index.append(fl_lab + "-precision")
            flattened_index.append(fl_lab + "-recall")
            flattened_index.append(fl_lab + "-f1score")
            flattened_index.append(fl_lab + "-count")
        flattened_index = flattened_index + ["macro-average-precision", "macro-average-recall", "macro-average-F1score",
                                             "macro-average-count"]
        res_flattened = pd.DataFrame(res.to_numpy().flatten(), index=flattened_index).transpose()
        return res_flattened, res


if __name__ == "__main__":
    """
    code testing
    """
    isear_train_df = pd.read_csv(r"C:\Users\Itisha Yadav\CL-Team-Lab-Emotion-Classification\data\isear\isear-train.csv",
                                 error_bad_lines=False, warn_bad_lines=False, header=None)
    isear_train_df = isear_train_df.rename(columns={0: "Y", 1: "X"})
    trainlabels = ["fear", "anger", "guilt", "joy", "shame", "disgust", "sadness"]
    evalObj = evaluation(y_actual=isear_train_df["Y"], y_pred=isear_train_df["Y"], labels=trainlabels)
    print(evalObj.main()[0])

"""
Sample Output: for the above given input:
**************************************************************************************************************************
 
1) Normal dataframe: print(evalObj.main()[1])
    
                   Precision  Recall  F1-Score   Count
fear                 1.0     1.0       1.0   751.0
anger                1.0     1.0       1.0   758.0
guilt                1.0     1.0       1.0   766.0
joy                  1.0     1.0       1.0   777.0
shame                1.0     1.0       1.0   757.0
disgust              1.0     1.0       1.0   757.0
sadness              1.0     1.0       1.0   760.0
macro_average        1.0     1.0       1.0  5326.0


**************************************************************************************************************************

2) Flattened dataframe (complete result in a single row): print(evalObj.main()[0])   

   fear-precision  fear-recall  ...  macro-average-F1score  macro-average-count
0             1.0          1.0  ...                    1.0               5326.0

"""
