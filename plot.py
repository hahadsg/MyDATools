# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


import numpy as np

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection


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

# --------------------------------------------------------------------------------
# 绘制雷达图
# --------------------------------------------------------------------------------
def radar_plot(data, varlabels=None, legends=None, ax=None):
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