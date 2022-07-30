
"""
This is the training script for the following modules:
    1.) Overlapping region detector
    2.) Soft decision maker:
        a) classifier 1: for verification of top 2 emotion recommendation
        b) classifier: for verification of top 3 emotion recommendation
"""

from utilities_ord import *
from mlp_classifier_ord import *
import os
import pickle


def saveModel(model, model_name, path):
    """

    :param model: the trained model
    :param model_name: name of the model
    :param path: path to save the model
    :return:
    """
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




