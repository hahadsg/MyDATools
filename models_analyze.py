# coding: utf-8
import numpy as np
import pandas as pd


def most_proba_index(y_true, y_proba, y_idx=None):
    """对预测结果进行最可能的错误分析，现只支持二分类
    Args
        y_true: array like
            真实标签
        y_proba: array like
            预测的可能性
        y_idx: array like
            对应的idx，将作为返回的idx；如果为None，则默认从0~n

    Return
        most_error_0_idx: np.array
            判断最错误的样本中，将0分成1的index
        most_error_1_idx
        most_right_0_idx
        most_right_1_idx
    """
    df = pd.DataFrame({
        'idx': range(len(y_true)) if y_idx is None else y_idx,
        'y_true': y_true,
        'y_proba': y_proba,
        'y_pred': np.where(y_proba >= 0.5, 1, 0),
    })
    df['is_error'] = df['y_true'] != df['y_pred']
    
    most_error_0 = df[(df['is_error'] == True) & (df['y_true'] == 0)].sort_values('y_proba', ascending=False)
    most_error_1 = df[(df['is_error'] == True) & (df['y_true'] == 1)].sort_values('y_proba')
    
    most_right_0 = df[(df['is_error'] == False) & (df['y_true'] == 0)].sort_values('y_proba')
    most_right_1 = df[(df['is_error'] == False) & (df['y_true'] == 1)].sort_values('y_proba', ascending=False)
    
    return (most_error_0.idx.values, most_error_1.idx.values,
        most_right_0.idx.values, most_right_1.idx.values)

    