import numpy as np
import hashlib
import heapq
import pandas as pd


def bert_output_formatter(df):
    """

    :param df: bert output
    :return: sorted bert output, with top 3 output labels, and dataframe of missclassified examples
    """
    mapping = {1: "fear", 2: "anger", 3: "guilt", 4: "joy", 5: "shame", 6: "disgust", 0: "sadness"}
    pred = []
    match = []
    match1 = []
    match2 = []
    top2 = []
    top3 = []
    hash = []
    conf = []
    conf1 = []
    conf2 = []
    for ind, row in df.iterrows():
        m = 0
        m1 = 0
        m2 = 0
        hash_object = hashlib.sha256(row[0].encode('utf-8'))
        hex_dig = hash_object.hexdigest()
        hash.append(hex_dig)
        proba = list(row[[2, 3, 4, 5, 6, 7, 8]])
        #largest = np.argmax(proba)
        arr = heapq.nlargest(3, range(len(proba)), key=proba.__getitem__)
        largest = arr[0]
        sec_largest = arr[1]
        third_largest = arr[2]
        if float(largest) == row[1]:
            m = 1
        if row[1] in [float(largest), float(sec_largest)]:
            m1 = 1
        if row[1] in [float(largest), float(sec_largest), float(third_largest)]:
            m2 = 1
        match.append(m)
        match1.append(m1)
        match2.append(m2)
        pred.append(float(largest))
        top2.append(float(sec_largest))
        top3.append(float(third_largest))
        proba.sort(reverse=True)
        conf.append([hex_dig, m] + proba)
        conf1.append([hex_dig, m1] + proba)
        conf2.append([hex_dig, m2] + proba)

    df["predictions"] = pred
    df["match"] = match
    df["_id"] = hash
    df["match_top2"] = match1
    df["match_top3"] = match2
    df["top2"] = top2
    df["top3"] = top3
    df = df.rename(columns={0: "sentence", 1: "ytrue"})
    df["ytrue_emotions"] = df["ytrue"]\
    .map(mapping)
    df["predictions_emotions"] = df["predictions"]\
    .map(mapping)
    missclassified_df = df[df["match"] == 0].reset_index(drop=True)
    missclassified_df = missclassified_df[missclassified_df["sentence"] != "[ No response.]"]
    return df, pd.DataFrame(conf), pd.DataFrame(conf1), pd.DataFrame(conf2), missclassified_df


def get_binary_classifier_predictions(pca, scalar, model, a, b, c, d, e, f, g):
    """

    :param pca: trained pca model
    :param scalar: trained standardiser
    :param model: trained classifier
    :param a-g: 7 bert confidence values in sorted order
    :return: prediction (0 or 1)
    """
    data = [a, b, c, d, e, f, g]
    data.sort(reverse=True)
    data = np.array(data).reshape(1, -1)
    pca_data = pca.transform(data)
    standard_data = scalar.transform(pca_data)
    pred = model.predict(standard_data)
    return pred[0]


def final_output(row):
    flag = 0
    out = []
    if row["classifier_1"] == 1:
        out.append(row["predictions_emotions"])
        return out
    elif row["classifier_1"] == 0:
        if row["classifier_2"] == 1:
            out.append(row["predictions_emotions"])
            out.append(row["top_2nd_recommendation_by_model"])
            return out
        else:
            if row["classifier_3"] == 1:
                out.append(row["predictions_emotions"])
                out.append(row["top_2nd_recommendation_by_model"])
                out.append(row["top_3rd_recommendation_by_model"])
                return out
            else:
                out.append(row["predictions_emotions"])
                out.append(row["top_2nd_recommendation_by_model"])
                out.append(row["top_3rd_recommendation_by_model"])
                return out


def final_output_LC(row):
    flag = 0
    out = []
    if row["classifier_1"] == 1:
        return flag
    elif row["classifier_1"] == 0:
        if row["classifier_2"] == 1:
            return flag
        else:
            if row["classifier_3"] == 1:
                return flag
            else:
                flag = 1
                return flag