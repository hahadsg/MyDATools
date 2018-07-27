# coding: utf8
import numpy as np
import pandas as pd


def features_save(df, feature_columns, file_prefix='./data/output/features/features'):
    df_path = file_prefix + '_df.csv'
    fc_path = file_prefix + '_fc.csv'
    df.to_csv(df_path)
    fc_df = pd.DataFrame(feature_columns).to_csv(fc_path, header=False)

def features_read(file_prefix='./data/output/features/features'):
    df_path = file_prefix + '_df.csv'
    fc_path = file_prefix + '_fc.csv'
    df = pd.read_csv(df_path, index_col=0)
    fc_df = pd.read_csv(fc_path, index_col=0, header=None).values[:, 0].tolist()
    return df, fc_df