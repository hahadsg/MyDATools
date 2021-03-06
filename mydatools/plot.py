# coding: utf-8
import itertools

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# --------------------------------------------------------------------------------
# 雷达图
# --------------------------------------------------------------------------------
# https://matplotlib.org/examples/api/radar_chart.html
def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    # rotate theta such that the first axis is at the top
    theta += np.pi/2

    def draw_poly_patch(self):
        verts = unit_poly_verts(theta)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta

def radar_plot(data, varlabels=None, legends=None, ax=None, class_size=None):
    """ 绘制雷达图

    Args:
        data: array like
            行是类别，列是指标
        varlabels: list
            指标list
        legends: list
            类别list
        ax: matplotlib.axes
    """
    case_data = data
    # 如果data是df类型，那index作为类别，columns作为指标
    if type(data) == pd.DataFrame:
        case_data = data.values
        if varlabels is None:
            varlabels = data.columns.tolist()
        if legends is None:
            legends = data.index.tolist()
            if class_size is not None:
                class_ratio = class_size / np.sum(class_size)
                for i, class_name in enumerate(legends):
                    legends[i] = '{}:{:>6.1%}, {}'.format(legends[i], class_ratio[i], class_size[i])
        
    # 获取theta 并产生projection
    theta = radar_factory(case_data.shape[1])

    # 获取ax
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1, subplot_kw=dict(projection='radar'))
        
    for d in case_data:
        ax.plot(theta, d)
    if varlabels is not None:
        ax.set_varlabels(varlabels)
    if legends is not None:
        ax.legend(legends)


# --------------------------------------------------------------------------------
# 绘制confusion_matrix(copy from sklearn example)
# --------------------------------------------------------------------------------
def plot_confusion_matrix(cm, classes=None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if classes is None:
        classes = list(range(cm.shape[0]))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# --------------------------------------------------------------------------------
# 绘制PR_curve
# --------------------------------------------------------------------------------
def plot_precision_recall_curve(y_true, y_proba):
    """绘制PR_curve

    Args:
        y_true: np.array
            真实label, {0,1}或者{-1,1}
        y_proba: np.array
            预测的概率

    Returns:
        precision: np.array
            查准率
        recall: np.array
            查全率
        thresholds: np.array
            阈值
    """
    p, r, th = precision_recall_curve(y_true, y_proba)
    auc = average_precision_score(y_true, y_proba)
    plt.title('Precision-Recall Curve: AUC={0:0.2f}'.format(auc))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(r, p)
    plt.grid()
    return p, r, th


# --------------------------------------------------------------------------------
# 绘制学习曲线
# --------------------------------------------------------------------------------
def plot_learning_curve(param_range, train_scores, test_scores):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="r")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="g")
    plt.plot(param_range, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(param_range, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()


# --------------------------------------------------------------------------------
# cv结果展示
# --------------------------------------------------------------------------------
def plot_grid_search_result(gs_model, x=None, y=None, xfunc=None, yfunc=None):
    """ 将cv的结果展示
    如果是一个参数则显示为曲线图，如果是两个参数则显示为heatmap

    Args:
        gs_model: sklearn.model_selection._search.GridSearchCV
            进行GridSearchCV的返回值
        x, y: str, default None
            图的横纵坐标轴是什么
        xfunc, yfunc: func, default None
            是否对横纵坐标的值做处理（有可能需要对x进行log操作）

    """
    n_paras = len(gs_model.cv_results_['params'])
    paras_type = list(gs_model.cv_results_['params'][0].keys())
    n_paras_type = len(paras_type)

    # x为空就赋值为第一个paras_type
    if x is None: 
        x = paras_type[0]
    # y为空，且有两个paras，就赋值为第二个paras_type
    if y is None and n_paras_type == 2: 
        y = paras_type[1]

    # 对坐标轴的值进行转换
    score_trans = lambda score, func: str(func(score) if func is not None else score)

    # print最优参数
    print('Best params:', gs_model.best_params_)
    print('Best score:', gs_model.best_score_)

    # plot过程（超过两个轴就不plot）
    # 一个轴
    if n_paras_type == 1:
        # 获取各参数值score
        cv_score = [
            [
                gs_model.cv_results_['mean_test_score'][i],
                score_trans(gs_model.cv_results_['params'][i][x], xfunc),
            ] for i in range(n_paras)
        ]
        # df
        cv_score_df = pd.Series(*tuple(zip(*cv_score)))
        # 改变轴显示值的顺序
        x_values = [score_trans(v, xfunc) for v in gs_model.get_params()['param_grid'][x]]
        cv_score_df = cv_score_df[x_values]
        # plot
        cv_score_df.plot()

    # 两个轴
    elif n_paras_type == 2:
        # 获取各参数值score
        cv_score = [
            [
                gs_model.cv_results_['mean_test_score'][i],
                score_trans(gs_model.cv_results_['params'][i][x], xfunc),
                score_trans(gs_model.cv_results_['params'][i][y], yfunc),
            ] for i in range(n_paras)
        ]
        # 将score变成二维的形式
        cv_score_df = pd.DataFrame(cv_score, columns=['score']+paras_type)
        cv_score_df = cv_score_df.pivot(y, x, 'score')
        # 改变轴显示值的顺序
        x_values = [score_trans(v, xfunc) for v in gs_model.get_params()['param_grid'][x]]
        y_values = [score_trans(v, yfunc) for v in gs_model.get_params()['param_grid'][y]]
        cv_score_df = cv_score_df.loc[y_values, x_values]
        # plot
        sns.heatmap(cv_score_df, annot=True)


# --------------------------------------------------------------------------------
# 数据展示模块
# --------------------------------------------------------------------------------
def plot_data_2D(X, y, method='pca'):
    """将数据降维 以2D的形式展现
    
    Args
        X: array like
            特征数据
        y: array like
            标签数据
        method: str or dimensionality reduction solver
            optional: tsne, pca
            或者直接传降维器
    """
    if method == 'tsne':
        solver = TSNE(n_components=2)
    elif method == 'pca':
        solver = PCA(n_components=2)
    else:
        solver = method
    print(solver)

    X_new = solver.fit_transform(X)
    x1, x2 = X_new[:, 0], X_new[:, 1]
    
    for i, iclass in enumerate(np.unique(y)):
        plt.scatter(x1[y==iclass], x2[y==iclass], alpha=0.5)

def plot_multiclass_feature_dist(x, y, label_dict={}):
    """展示各类的数据分布
    Args
        x: array
            展示数据
        y: array
            label
        label_dict: dict
            legend展示的字典
    """
    classes = y.unique()
    classes.sort()
    for c in classes:
        sns.distplot(x[y == c], label=str(label_dict.get(c, c)))
    plt.legend()


# --------------------------------------------------------------------------------
# 绘制分类器相关
# --------------------------------------------------------------------------------
def plot_decision_boundary(X, y, pred_func, h=0.02):
    """绘制X, y以及预测边界（仅限特征维度为2）

    Args:
        X: array_like
            特征，shape=(sample_num, features_num)，这里features_num只能为2
        y: array_like
            标签，shape(sample_num, )
        pred_func: function
            预测的函数
        h: float
            描点间隔（越小，画的越精细）
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.9)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, s=10)

def plot_classifier_paras(model, X, y, paras_dict=None, plot_input=True, col_num=4):
    """绘制分类器在不同参数下的预测分布图

    Args:
        model: machine_learning model
            模型对象
        X: array_like
            特征，shape=(sample_num, features_num)，这里features_num只能为2
        y: array_like
            标签，shape(sample_num, )
        paras_dict: dict, default None
            需要代入模型的超参数
            如：{'a': [1,2], 'b': [10,20]}
            将会生成[{'a':1, 'b':10}, {'a':2, 'b':10}, {'a':1, 'b':20}, {'a':2, 'b':20}] 即生成笛卡尔积，代入模型
        plot_input: boolean, default True
            是否绘制只有样本点的图（即没有预测边界的图）
        col_num: int, default 4
            每行图的数量
    """
    # plot设置
    figsize_col = 15
    figsize_row = figsize_col / col_num * 0.85

    # 转换paras格式 {'a': [1,2], 'b': [10,20]} -> [{'a':1, 'b':10}, ...] 即生成笛卡尔积
    if paras_dict is None:
        paras_list = [{}]
    else:
        paras_keys = list(paras_dict.keys())
        paras_values = itertools.product(*paras_dict.values())
        paras_list = [dict(zip(paras_keys, values)) for values in paras_values]

    if plot_input: paras_list = np.append([None], paras_list)
    plot_row_num = (len(paras_list) - 1) // col_num + 1
    plt.figure(figsize=(figsize_col, plot_row_num * figsize_row))

    for i, paras in enumerate(paras_list):
        plt.subplot(plot_row_num, col_num, i+1)
        if plot_input and i == 0:
            plt.title('input')
            plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Spectral, s=10)
            continue
        plt.title(', '.join(['%s = %s'%(k,v) for k,v in paras.items()]))
        model_obj = model(**paras)
        model_obj.fit(X, y)
        plot_decision_boundary(X, y, model_obj.predict)