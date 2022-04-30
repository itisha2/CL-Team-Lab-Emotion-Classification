
class Evaluation:
    train_results: []
    test_results: []
    labels: []

    def precision(self):
        num_labels = len(self.labels)
        TP = TN = FP = FN = 0

        for i, test_result in enumerate(self.test_results):
            if test_result == self.train_results[i]:
                TP += 1
                TN += (num_labels - 1)
            else:
                FP += 1
                FN +=1
                TN += (num_labels - 2)

        return len(TP) / (len(TP) + len(FP))

