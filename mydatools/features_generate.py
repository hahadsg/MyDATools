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

def mean_encoding(df, groupby_cols, target_col, test_index=None, gen_feature_name=None,
                  return_feature_name=False, scheme='kfold',
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
        test_index: array like, default None
            测试集的index，默认没有测试集
        gen_feature_name: str, default None
            生成的特征名，如果不指定就使用'mean_encoding-' + target_col + '-' + '_'.join(groupby_cols)
        scheme: str, default kfold
            进行mean_encoding的方法
            kfold: 对训练数据进行KFold(kfold_splits)，然后每次对除当前组进行fit，再对当前组进行transform
            naive: 根据训练集所有行的mean，赋值给训练集和测试集，这种方法要注意会出现过拟合
        kfold_splits: int, default 5
            scheme=kfold: KFold(n_splits=kfold_splits)
        kfold_seed: int, default None
            scheme=kfold: KFold(random_state=kfold_seed)
    """
    if not isinstance(groupby_cols, list):
        groupby_cols = [groupby_cols]
    if gen_feature_name is None:
        gen_feature_name = 'mean_encoding-' + target_col + '-' + '_'.join(groupby_cols)
    if isinstance(test_index, pd.Series):
        test_index = test_index.values

    # 初始化特征
    df[gen_feature_name] = np.nan

    # 得到trn_df和tst_df
    if test_index is None:
        test_index = np.zeros(df.shape[0], dtype=bool)
    trn_df = df[~test_index][groupby_cols + [target_col, gen_feature_name]].copy()
    tst_df = df[test_index][groupby_cols + [target_col, gen_feature_name]].copy()

    # mean encoding
    if scheme == 'kfold':
        from sklearn.model_selection import KFold
        kf = KFold(5, shuffle=True, random_state=kfold_seed)
        for trn_idx_i, val_idx_i in kf.split(trn_df[target_col].values):
            trn_df_i, val_df_i = trn_df.iloc[trn_idx_i].copy(), trn_df.iloc[val_idx_i].copy()
            mean_map = trn_df_i.groupby(groupby_cols)[target_col].mean()
            val_df_i[gen_feature_name] = val_df_i[groupby_cols].join(mean_map, how='left', on=groupby_cols)[target_col]
            trn_df.iloc[val_idx_i] = val_df_i
    elif scheme == 'naive':
        mean_map = trn_df.groupby(groupby_cols)[target_col].mean()
        trn_df.loc[:, gen_feature_name] = trn_df[groupby_cols].join(mean_map, how='left', on=groupby_cols)[target_col]

    # test df mean coding
    mean_map = trn_df.groupby(groupby_cols)[target_col].mean()
    tst_df.loc[:, gen_feature_name] = tst_df[groupby_cols].join(mean_map, how='left', on=groupby_cols)[target_col]

    # 更新df
    df.iloc[~test_index, [df.columns.get_loc(gen_feature_name)]] = trn_df[gen_feature_name]
    df.iloc[test_index, [df.columns.get_loc(gen_feature_name)]] = tst_df[gen_feature_name]

    # fillna with globalmean
    globalmean = df[target_col].mean()
    df[gen_feature_name] = df[gen_feature_name].fillna(globalmean)

    if return_feature_name:
        return df, gen_feature_name
    return df