# coding: utf-8
import numpy as np
import pandas as pd


def get_top_k_corr(df, k=10):
    c = df.corr().abs()
    c.iloc[:] = np.tril(c, -1) # 取下三角矩阵
    sorted_corr = c.unstack().sort_values(ascending=False)
    return sorted_corr if k is None else sorted_corr[:k]

def remove_high_corr(df, threshold=0.95):
    """去掉高相关性的特征，返回低相关性的特征
    Args:
        df: pd.DataFrame
            特征数据，columns即为特征
        threshold: float, default 0.95
            去掉相关性>threshold的特征
    Returns:
        features: list
            相关性较低的特征
    """
    corr_df = df.corr().abs()
    upper_df = pd.DataFrame(np.triu(corr_df.values, k=1), columns=corr_df.columns, index=corr_df.index)
    features = corr_df.columns[upper_df.max(axis=0) <= threshold].tolist()
    return features