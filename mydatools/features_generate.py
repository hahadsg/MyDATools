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

def mean_encoding(df, groupby_cols, target_col, gen_feature_name=None, scheme='kfold',
                  kfold_splits=5, kfold_seed=None):
    """mean encoding
    对训练数据根据scheme进行mean encoding，然后再对测试数据进行mean encoding
    
    Args:
        df: pd.DataFrame
            需要进行mean encoding的df
        groupby_cols: list or str
            根据groupby_cols进行groupby
        target_col: str
            目标特征
        gen_feature_name: str, default None
            生成的特征名，如果不指定就使用'mean_encoding-' + target_col + '-' + '_'.join(groupby_cols)
        scheme: str, default kfold
            进行mean_encoding的方法
            kfold: 对训练数据进行KFold(kfold_splits)，然后每次对除当前组进行fit，再对当前组进行transform
        kfold_splits: int, default 5
            scheme kfold使用
    """
    if not isinstance(groupby_cols, list):
        groupby_cols = [groupby_cols]
    if gen_feature_name is None:
        gen_feature_name = 'mean_encoding-' + target_col + '-' + '_'.join(groupby_cols)

    df[gen_feature_name] = np.nan
    if scheme == 'kfold':
        from sklearn.model_selection import KFold
        globalmean = df[target_col].mean()
        trn_df = df
        kf = KFold(5, shuffle=True, random_state=kfold_seed)
        for trn_idx_i, val_idx_i in kf.split(df[target_col].values):
            trn_df_i, val_df_i = df.iloc[trn_idx_i].copy(), df.iloc[val_idx_i].copy()
            mean_map = trn_df_i.groupby(groupby_cols)[target_col].mean()
            val_df_i[gen_feature_name] = val_df_i[groupby_cols].join(mean_map, how='left', on=groupby_cols)[target_col]
            trn_df.iloc[val_idx_i] = val_df_i
        trn_df.fillna(globalmean, inplace=True)

    return df