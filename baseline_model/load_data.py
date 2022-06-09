import os
import pandas as pd
from baseline_classifier import *
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def load_train_data(data_path, train_labels):
    isear_train_df = pd.read_csv(os.path.join(data_path, "isear-train.csv"), error_bad_lines=False,
                                 warn_bad_lines=False, header=None)
    isear_train_df = isear_train_df.rename(columns={0: "Y", 1: "X"})
    isear_train_df = isear_train_df[isear_train_df["Y"].isin(train_labels)]
    return isear_train_df


def load_test_data(data_path, train_labels):
    isear_test1_df = pd.read_csv(os.path.join(data_path, "isear-test.csv"), error_bad_lines=False, warn_bad_lines=False,
                                 header=None)
    isear_val_df = pd.read_csv(os.path.join(data_path, "isear-val.csv"), error_bad_lines=False, warn_bad_lines=False,
                               header=None)
    isear_test_df = pd.concat([isear_test1_df, isear_val_df])
    isear_test_df = isear_test_df.rename(columns={0: "Y", 1: "X"})
    isear_test_df = NaiveBayesClassifier().cleanData(isear_test_df)
    isear_test_df = isear_test_df[isear_test_df["Y"].isin(train_labels)]
    return isear_test_df
