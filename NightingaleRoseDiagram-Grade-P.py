import pandas as pd
from pyecharts.charts import Pie
from pyecharts import options as opts
from snapshot_phantomjs import snapshot
from pyecharts.render import make_snapshot


def draw_pie(grade, num, n):
    color_series = ['#14ADCF', '#209AC9', '#1E91CA', '#2C6BA0', '#2B55A1',
                    '#2D3D8E', '#44388E', '#6A368B', '#7D3990', '#A63F98']

    df = pd.DataFrame({'grade': grade, 'num': num})
    df.sort_values(by='num', ascending=False, inplace=True)
    v = df['grade'].values.tolist()
    d = df['num'].values.tolist()
    pie1 = Pie(init_opts=opts.InitOpts(width='1000px', height='1000px'))
    pie1.set_colors(color_series)
    pie1.add("", [list(z) for z in zip(v, d)],
             radius=["15%", "70%"],
             center=["50%", "60%"],
             rosetype="area"
             )
    pie1.set_global_opts(legend_opts=opts.LegendOpts(is_show=False), toolbox_opts=opts.ToolboxOpts())
    pie1.set_series_opts(label_opts=opts.LabelOpts(is_show=True, position="inside", font_size=12,
                                                   formatter="{b}分:{c}人", font_style="italic",
                                                   font_weight="bold", font_family="Microsoft YaHei"
                                                   ), )
    make_snapshot(snapshot, pie1.render(), f'figuresNightingaleRoseDiagramP\\{n}.png', is_remove_html=True)


draw_pie([10, 8, 1, 9, 5, 6, 7, 4, 2, 3], [2234, 145, 78, 61, 20, 15, 15, 12, 11, 8], 1)
draw_pie([10, 8, 9, 1, 6, 7, 5, 3, 4, 2], [2012, 168, 155, 79, 60, 46, 30, 22, 19, 8], 2)
draw_pie([10, 9, 8, 1, 7, 6, 4, 5, 2, 3], [2135, 169, 113, 62, 49, 27, 19, 11, 7, 7], 3)
draw_pie([10, 8, 9, 7, 1, 6, 5, 4, 3, 2], [2045, 243, 67, 56, 55, 46, 33, 28, 15, 11], 4)
draw_pie([10, 8, 1, 5, 7, 6], [1099, 319, 128, 43, 15, 6], 5)
draw_pie([10, 8, 5, 1, 7, 6, 3], [927, 368, 137, 92, 71, 10, 5], 6)
draw_pie([10, 8, 1, 7, 5, 3, 6], [948, 404, 93, 80, 57, 19, 9], 7)
draw_pie([10, 8, 1, 5, 6, 7], [954, 365, 165, 65, 33, 28], 8)
