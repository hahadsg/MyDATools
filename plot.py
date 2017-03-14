# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_grid_search_result(model_cv, x=None, y=None, xfunc=None, yfunc=None):
    """ 将cv的结果展示
    如果是一个参数则显示为曲线图，如果是两个参数则显示为heatmap

    Args:
        model_cv: sklearn.model_selection._search.GridSearchCV
            进行GridSearchCV的返回值
        x, y: str, default None
            图的横纵坐标轴是什么
        xfunc, yfunc: func, default None
            是否对横纵坐标的值做处理（有可能需要对x进行log操作）

    """
    n_paras = len(model_cv.cv_results_['params'])
    paras_type = list(model_cv.cv_results_['params'][0].keys())
    n_paras_type = len(paras_type)

    # x为空就赋值为第一个paras_type
    if x is None: 
        x = paras_type[0]
    # y为空，且有两个paras，就赋值为第二个paras_type
    if y is None and n_paras_type == 2: 
        y = paras_type[1]

    # 对坐标轴的值进行转换
    score_trans = lambda score, func: func(score) if func is not None else score

    # plot过程
    # 一个轴
    if n_paras_type == 1:
        # 获取各参数值score
        cv_score = [
            [
                model_cv.cv_results_['mean_test_score'][i],
                score_trans(model_cv.cv_results_['params'][i][x], xfunc),
            ] for i in range(n_paras)
        ]
        # plot
        cv_score_df = pd.Series(*tuple(zip(*cv_score)))
        cv_score_df.plot()

    # 两个轴
    elif n_paras_type == 2:
        # 获取各参数值score
        cv_score = [
            [
                model_cv.cv_results_['mean_test_score'][i],
                score_trans(model_cv.cv_results_['params'][i][x], xfunc),
                score_trans(model_cv.cv_results_['params'][i][y], yfunc),
            ] for i in range(n_paras)
        ]
        # 将score变成二维的形式
        cv_score_df = pd.DataFrame(cv_score, columns=['score']+paras_type)
        cv_score_df = cv_score_df.pivot(y, x, 'score')
        # plot
        sns.heatmap(cv_score_df, annot=True)

    else:
        raise(Expection('参数数量只能为1个或者2个'))

