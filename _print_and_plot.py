import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates

import cartopy
import seaborn as sns
plt.rc('font', size=10)
matplotlib.rcParams['figure.figsize'] = (4,3)
matplotlib.rcParams['scatter.marker']='.'

import statsmodels.api as sm
lowess = sm.nonparametric.lowess

import numpy as np

from rich.console import Console
from rich.table import Table

plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 1.),  # red   with alpha = 30%
    "axes.facecolor":    (1.0, 1.0, 1.0, 1.),  # green with alpha = 50%
    "savefig.facecolor": (1.0, 1.0, 1.0, 0.),  # blue  with alpha = 20%
})

runs = {
    '1300' : 'darkmagenta',
    '1400' : 'red',
    '1500' : 'darkorange',
}

def savefig(file_name, **kwargs):
    plt.savefig(f"/climca/people/ppfleiderer/decomposition/plots/{file_name}.pdf", bbox_inches='tight', **kwargs)
    plt.savefig(f"/climca/people/ppfleiderer/decomposition/plots/{file_name}.png", bbox_inches='tight', dpi=300, **kwargs)

def seas_mean_trend(x, alpha=0.05):
    x = x._seasonal.mean('day')
    lr = sm.OLS(x.values, sm.add_constant(x.year.values)).fit()
    return lr.params, lr.conf_int(alpha)


def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes


def print_summary_table(data_objects):
    a = data_objects[0]
    table = Table()
    table.add_column("type", justify="right", style="white", no_wrap=True)
    table.add_column("corr\ndaily", justify="right", style="white")
    table.add_column("corr\nyearly", justify="right", style="white")
    table.add_column("trend", justify="right", style="white")
    table.add_column("trend\n95 conf", justify="right", style="white")
    table.add_column("trend\ndiff", justify="right", style="white")
    table.add_column("trend\nrel. diff", justify="right", style="white")
    table.add_column("trend\nwithin conf.", justify="right", style="white")
    lr_a = seas_mean_trend(a)
    for x in data_objects:
        lr = seas_mean_trend(x)
        table.add_row(
            x._label,
            f"{np.corrcoef(a._x, (x._x))[0,1].round(3)}",
            f"{np.corrcoef(a._seasonal.mean('day'), (x._seasonal.mean('day')))[0,1].round(3)}",
            f'{lr[0][1].round(4)}',
            f'({lr[1][1,0].round(4)} {lr[1][1,1].round(4)})', 
            f"{(lr[0][1] - lr_a[0][1]).round(4)}",
            f"{((lr[0][1] - lr_a[0][1]) / np.abs(lr_a[0][1])).round(4)}",
            f'{lr[1][1,0].round(4) < lr[0][1].round(4) and lr[1][1,1].round(4) > lr[0][1].round(4)}',

        )
    return table


def plot_scatter(x,y,ax=None, identical=True):
    if ax is None:
        f, ax = plt.subplots()
    try:
        x = x.values
    except:
        pass
    try:
        y = y.values
    except:
        pass
    x,y = x[np.isfinite(y) & np.isfinite(x)], y[np.isfinite(y) & np.isfinite(x)]
    lr = sm.OLS(y, sm.add_constant(x)).fit()
    if identical:
        label = f"correct sign {int(np.round(np.sum(np.sign(x) == np.sign(y)) / x.shape[0] * 100))}%"
    else:
        label=''
    ax.scatter(x,y, marker='.', color='b', s=10, label=label)
    lim = np.nanpercentile(np.concatenate((x,y)), [0,100])
    lim = [lim[0] - (lim[1] - lim[0]) /100, lim[1] + (lim[1] - lim[0]) /100]
    ax.plot([x.min(), x.max()], np.array([x.min(),x.max()]) * lr.params[1] + lr.params[0], color='r', label=f'R2 : {lr.rsquared.round(4)}')
    if identical:
        ax.set_ylim(lim)
        ax.set_xlim(lim)
        ax.axhline(0, color='k')
        ax.axvline(0, color='k')
        add_identity(ax, color='k', ls='--')
    ax.legend()
    return ax


class plotter():
    def __init__(self, d, ref, target_variable=''):
        
        self._ref = ref
        self._target_variable = target_variable

        for k,c in zip(d.keys(), sns.color_palette("Set2")):
            d[k]._label = k
            d[k]._color = c
        self._d = d
        self._data_objects = d.values()
        

    def standard(self):
        fig = plt.figure(figsize=(10,5), constrained_layout=True, facecolor="w")
        gs = GridSpec(2,4, figure=fig, width_ratios=[2,1,1,2])
        self.ts(ax=fig.add_subplot(gs[0, 0]))
        self.legend(ax=fig.add_subplot(gs[1, -1]))
        self.corr_seasonal(ax=fig.add_subplot(gs[0, 1]))
        self.corr_daily(ax=fig.add_subplot(gs[0, 2]))
        self.trend_seasonal(ax=fig.add_subplot(gs[1, 1]))
        self.trend_daily(ax=fig.add_subplot(gs[1, 2]))
        return fig,gs

    def scatter(self, x_name, y_name, ax, seasonal=True, identical=True):
        if seasonal:
            x = self._d[x_name]._seasonal.mean('day').values
            y = self._d[y_name]._seasonal.mean('day').values
        else:
            x = self._d[x_name]._x.values
            y = self._d[y_name]._x.values
        plot_scatter(x,y, ax=ax, identical=identical)
        ax.set_ylabel(y_name)
        ax.set_xlabel(x_name)

    def ts(self, ax, anomalies=False):
        for y in self._data_objects:
            y_ = y._seasonal.mean('day')
            if anomalies:
                y_ -= y_.mean()
            ax.plot(y_.year, y_, label=y._label, color=y._color)
        ax.set_ylabel(self._target_variable)
        return ax

    def legend(self, ax):
        ax.axis('off')
        for y in self._data_objects:
            y_ = y._seasonal.mean('day')
            ax.plot([], [], label=y._label, color=y._color)
        _ = ax.legend()

    def add_xlabels(self, ax):
        ax.set_xticks(range(len(self._data_objects)))
        ax.set_xticklabels([y._label for y in self._data_objects], rotation=90, fontsize=6)

    def corr_seasonal(self, ax):
        for x,y in enumerate(self._data_objects):
            z = np.corrcoef(y._seasonal.mean('day'), self._ref._seasonal.mean('day'))[0,1]
            ax.bar(x=x, height=z, color=y._color)
            if z > 0:
                ax.annotate(f'{round(z, 2)} ', xy=(x,z), rotation=90, ha='center', va='top', fontsize=8)
        ax.set_ylabel('corr. seasonal')
        self.add_xlabels(ax)

    def corr_daily(self, ax):
        for x,y in enumerate(self._data_objects):
            z = np.corrcoef(y._x, self._ref._x)[0,1]
            ax.bar(x=x, height=z, color=y._color)
            if z > 0:
                ax.annotate(f'{round(z, 2)} ', xy=(x,z), rotation=90, ha='center', va='top', fontsize=8)
        ax.set_ylabel('corr. daily')
        self.add_xlabels(ax)

    def trend_daily(self, ax, alpha=0.05):
        for x,y in enumerate(self._data_objects):
            z = y._x
            lr = sm.OLS(z.values, sm.add_constant(z.time.dt.year.values)).fit()
            coefs, confs = lr.params, lr.conf_int(alpha)
            ax.fill_between([x-0.3,x+0.3], [confs[1,0]]*2, [confs[1,1]]*2, color=y._color)
            #ax.annotate(f'{round(coefs[1], 3)} ', xy=(x,coefs[1]), rotation=90, ha='center', va='center', fontsize=8)
            #ax.scatter([x], [coefs[1]], color='k', marker='*')
        self.add_xlabels(ax)
        ax.set_ylabel('trend')
        ax.axhline(y=0, color='k')

    def trend_seasonal(self, ax, alpha=0.05):
        for x,y in enumerate(self._data_objects):
            z = y._seasonal.mean('day')
            lr = sm.OLS(z.values, sm.add_constant(z.year.values)).fit()
            coefs, confs = lr.params, lr.conf_int(alpha)
            ax.fill_between([x-0.3,x+0.3], [confs[1,0]]*2, [confs[1,1]]*2, color=y._color)
            ax.annotate(f'{round(coefs[1], 3)} ', xy=(x,coefs[1]), rotation=90, ha='center', va='center', fontsize=8)
        self.add_xlabels(ax)
        ax.set_ylabel('trend')
        ax.axhline(y=0, color='k')


def wp_plot_stats(t, obj, ylabel=''):
    fig,axes = plt.subplots(nrows=obj._m, ncols=obj._n, figsize=(6,6), sharex=True, sharey=True)
    for i,rc in obj._label_dict.items():
        r,c = rc.values()
        ax = axes[r,c]
        ax.axhline(y=0, color='k')
        for x,c in enumerate(t.columns):
            ax.bar(x=x, height=t.loc[i,c])
        ax.annotate(i, xy = (0.05,0.95), xycoords = 'axes fraction', ha='left', va='top')
        
    for ax in axes[-1,:]:
        ax.set_xticks(range(len(t.columns)))
        ax.set_xticklabels(t.columns, rotation=90)
    for ax in axes[:,0]:
        ax.set_ylabel(ylabel)