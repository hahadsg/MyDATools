# coding: utf-8
import numpy as np
import pandas as pd


def get_top_k_corr(df, k=10):
    c = df.corr().abs()
    c.iloc[:] = np.tril(c, -1) # 取下三角矩阵
    sorted_corr = c.unstack().sort_values(ascending=False)
    return sorted_corr if k is None else sorted_corr[:k]
