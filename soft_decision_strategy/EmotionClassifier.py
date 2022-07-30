"""
Place: Stuttgart, Germany
"""
from inference_bert import *
from utilities_ord import *
import torch
import pickle
from transformers import BertTokenizer
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def load_pickle_file(model_path):
    """

    :param model_path:
    :return: loaded model
    """
    filehandler = open(model_path, 'rb')
    model = pickle.load(filehandler)
    filehandler.close()
    return model


class EmotionCLassifier:
    """
    Emotion Classifier class
    1.) Firstly, it calls the trained bert model for confidence scores.
    2.) Secondly, it calls the overlapping region detector and soft decision maker classifiers for detecting,
            if there is more than one emotion present in a given sentence.
    """
    def __init__(self):
        self.pca_ord = load_pickle_file("./classifier_models/pca_crisp.p")
        self.scalar_ord = load_pickle_file("./classifier_models/scalar_crisp.p")
        self.clf_ord = load_pickle_file("./classifier_models/classifier_crisp.p")

        self.pca_sdm_1 = load_pickle_file("./classifier_models/pca_soft_top2.p")
        self.scalar_sdm_1 = load_pickle_file("./classifier_models/scalar_soft_top2.p")
        self.clf_sdm_1 = load_pickle_file("./classifier_models/classifier_soft_top2.p")

        self.pca_sdm_2 = load_pickle_file("./classifier_models/pca_soft_top3.p")
        self.scalar_sdm_2 = load_pickle_file("./classifier_models/scalar_soft_top3.p")
        self.clf_sdm_2 = load_pickle_file("./classifier_models/classifier_soft_top3.p")

    def inference_bert(self, sentences, num_labels):
        """

        :param sentences:  list of input sentences by user
        :param num_labels: number of emotion labels
        :return: 7 confidence scores
        """
        n_sentences = len(sentences)

        model = BertForMultilabelSequenceClassification.from_pretrained("bert-base-uncased",
                                                                        num_labels=num_labels,
                                                                        output_attentions=False,
                                                                        output_hidden_states=False)
        print("Loading the model....")
        model.load_state_dict(torch.load('./finetuned_BERT_epoch_2.model', map_location=torch.device('cpu')))

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                  do_lower_case=True)
        encoded_input = tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=True,
            return_attention_mask=True,
            pad_to_max_length=True,
            max_length=256,
            return_tensors='pt'
        )
        output = model(**encoded_input)
        label = torch.argmax(output[0], dim=1)
        # print(np.asarray(sentences).T.reshape(2,1).shape)
        # print(output[0].detach().numpy().shape)
        return np.concatenate((np.asarray(sentences).T.reshape(n_sentences, 1), label.numpy().reshape(n_sentences, 1),
                               output[0].detach().numpy()), axis=1)

    def inference_ord_sdm(self, bert_output):
        """

        :param bert_output: 7 confidence scores
        :return: final output with crisp or soft decision and a low confidence flag,
              which signifies whether the model is too sure with predictions or not.
        """
        mapping = {1: "fear", 2: "anger", 3: "guilt", 4: "joy", 5: "shame", 6: "disgust", 0: "sadness"}
        bert_all, conf_df_1, conf_df_2, conf_df_3, missclassified = bert_output_formatter(bert_output)

        bert_all["classifier_1"] = bert_all.apply(
            lambda row: get_binary_classifier_predictions(self.pca_ord, self.scalar_ord, self.clf_ord, row[2], row[3], row[4], row[5],
                                                          row[6], row[7], row[8]), axis=1)
        bert_all["classifier_2"] = bert_all.apply(
            lambda row: get_binary_classifier_predictions(self.pca_sdm_1, self.scalar_sdm_1, self.clf_sdm_1, row[2], row[3], row[4], row[5],
                                                          row[6], row[7], row[8]), axis=1)
        bert_all["classifier_3"] = bert_all.apply(
            lambda row: get_binary_classifier_predictions(self.pca_sdm_2, self.scalar_sdm_2, self.clf_sdm_2, row[2], row[3], row[4], row[5],
                                                          row[6],
                                                          row[7], row[8]), axis=1)

        bert_all["top_2nd_recommendation_by_model"] = bert_all["top2"] \
            .map(mapping)
        bert_all["top_3rd_recommendation_by_model"] = bert_all["top3"] \
            .map(mapping)

        bert_all["final"] = bert_all.apply(lambda row: final_output(row), axis=1)
        bert_all["low_confidence"] = bert_all.apply(lambda row: final_output_LC(row), axis=1)
        return bert_all

    def main(self, input):
        """

        :param input: list of input sentences by user
        :return: return the final dataframe, with the following operations performed:
            1. BERT model run.
            2. Overlapping region detector run.
            3. Two classifiers of the soft decision maker run.
        """
        bert_all = pd.DataFrame(self.inference_bert(input, 7))

        bert_all = self.inference_ord_sdm(bert_output=bert_all)
        bert_all = bert_all[["sentence", "final", "low_confidence"]]
        bert_all = bert_all.rename(columns={"sentence": "input_sentences", "final": "predicted_emotions", "low_confidence": "low_confidence_flag"})
        bert_all["low_confidence_flag"] = bert_all["low_confidence_flag"].map({0: False, 1: True})
        return bert_all


if __name__ == "__main__":
    emoClassifier = EmotionCLassifier()
    emoClassifier.main(["Got a big fishing", "I am feeling very nervous", "I am feeling very happy"])
