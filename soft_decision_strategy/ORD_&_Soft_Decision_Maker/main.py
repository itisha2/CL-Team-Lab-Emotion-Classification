from utilities import *
from mlp_classifier import *
import os
import pickle

def load_pickle_file(model_path):
  filehandler = open(model_path, 'rb')
  model = pickle.load(filehandler)
  filehandler.close()
  return model


def saveModel(model, model_name, path):
    with open(os.path.join(path, model_name), 'wb') as f:
        pickle.dump(model, f)
    print("Model " + model_name + ", saved at " + path)


def train(bert_output):
    """

    :param bert_output: formatted bert output dataframe
    :calls the training script, for all 3 classifiers and saves the models
    """
    bert_all, conf_df_1, conf_df_2, conf_df_3, missclassified = bert_output_formatter(bert_output)
    a, pca_1, scalar_1, clf_1 = mlp_classifier(conf_df_1.drop(columns=[0, 1]).reset_index(drop=True), conf_df_1[1])
    print("Saving Models for Crisp Decision : ")
    saveModel(pca_1, "pca_crisp.p", "./classifier_models/")
    saveModel(scalar_1, "scalar_crisp.p", "./classifier_models/")
    saveModel(clf_1, "classifier_crisp.p", "./classifier_models/")
    print()
    print("Saving Models for Soft Decision : ")
    b, pca_2, scalar_2, clf_2 = mlp_classifier(conf_df_2.drop(columns=[0, 1]).reset_index(drop=True), conf_df_2[1])
    saveModel(pca_2, "pca_soft_top2.p", "./classifier_models/")
    saveModel(scalar_2, "scalar_soft_top2.p", "./classifier_models/")
    saveModel(clf_2, "classifier_soft_top2.p", "./classifier_models/")
    print()
    c, pca_3, scalar_3, clf_3 = mlp_classifier(conf_df_3.drop(columns=[0, 1]).reset_index(drop=True), conf_df_3[1])
    saveModel(pca_3, "pca_soft_top3.p", "./classifier_models/")
    saveModel(scalar_3, "scalar_soft_top3.p", "./classifier_models/")
    saveModel(clf_3, "classifier_soft_top3.p", "./classifier_models/")


def inference(bert_output):
    mapping = {1: "fear", 2: "anger", 3: "guilt", 4: "joy", 5: "shame", 6: "disgust", 0: "sadness"}
    bert_all, conf_df_1, conf_df_2, conf_df_3, missclassified = bert_output_formatter(bert_output)
    """
    Loading models
    """
    pca_1 = load_pickle_file("./classifier_models/pca_crisp.p")
    scalar_1 = load_pickle_file("./classifier_models/scalar_crisp.p")
    clf_1 = load_pickle_file("./classifier_models/classifier_crisp.p")

    pca_2 = load_pickle_file("./classifier_models/pca_soft_top2.p")
    scalar_2 = load_pickle_file("./classifier_models/scalar_soft_top2.p")
    clf_2 = load_pickle_file("./classifier_models/classifier_soft_top2.p")

    pca_3 = load_pickle_file("./classifier_models/pca_soft_top3.p")
    scalar_3 = load_pickle_file("./classifier_models/scalar_soft_top3.p")
    clf_3 = load_pickle_file("./classifier_models/classifier_soft_top3.p")

    bert_all["classifier_1"] = bert_all.apply(
        lambda row: get_binary_classifier_predictions(pca_1, scalar_1, clf_1, row[2], row[3], row[4], row[5], row[6], row[7], row[8]), axis=1)
    bert_all["classifier_2"] = bert_all.apply(
        lambda row: get_binary_classifier_predictions(pca_2, scalar_2, clf_2, row[2], row[3], row[4], row[5], row[6], row[7], row[8]), axis=1)
    bert_all["classifier_3"] = bert_all.apply(
        lambda row: get_binary_classifier_predictions(pca_3, scalar_3, clf_3, row[2], row[3], row[4], row[5], row[6],
                                                      row[7], row[8]), axis=1)

    bert_all["top_2nd_recommendation_by_model"] = bert_all["top2"] \
        .map(mapping)
    bert_all["top_3rd_recommendation_by_model"] = bert_all["top3"] \
        .map(mapping)

    bert_all["final"] = bert_all.apply(lambda row: final_output(row), axis=1)
    bert_all["low_confidence"] = bert_all.apply(lambda row: final_output_LC(row), axis=1)
    return bert_all


