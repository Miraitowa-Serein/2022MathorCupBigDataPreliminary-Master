import pandas as pd
from pyecharts.charts import Pie
from pyecharts import options as opts
from snapshot_phantomjs import snapshot
from pyecharts.render import make_snapshot


def draw_pie(grade, num, n):
    color_series = ['#9ECB3C', '#6DBC49', '#37B44E', '#3DBA78', '#14ADCF',
                    '#209AC9', '#1E91CA', '#2C6BA0', '#2B55A1', '#2D3D8E']

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
    make_snapshot(snapshot, pie1.render(), f'figuresNightingaleRoseDiagramF\\{n}.png', is_remove_html=True)


draw_pie([10, 9, 8, 7, 1, 6, 5, 3, 4, 2], [3157, 764, 567, 214, 209, 182, 172, 71, 55, 42], 1)
draw_pie([10, 9, 8, 7, 6, 1, 5, 3, 4, 2], [2701, 786, 658, 327, 271, 237, 211, 90, 90, 62], 2)
draw_pie([10, 9, 8, 7, 6, 1, 5, 4, 3, 2], [2981, 805, 643, 268, 207, 184, 161, 85, 59, 40], 3)
draw_pie([10, 9, 8, 7, 1, 6, 5, 4, 3, 2], [2800, 786, 654, 309, 230, 228, 201, 93, 83, 49], 4)

draw_pie([10, 8, 9, 7, 1, 6, 5, 3, 4, 2], [2890, 991, 851, 533, 455, 435, 409, 209, 160, 87], 5)
draw_pie([10, 8, 9, 7, 6, 5, 1, 3, 4, 2], [2587, 1056, 824, 638, 508, 435, 429, 228, 198, 117], 6)
draw_pie([10, 8, 9, 7, 6, 5, 1, 3, 4, 2], [2556, 1066, 841, 675, 475, 454, 415, 235, 199, 104], 7)
draw_pie([10, 8, 9, 7, 6, 1, 5, 3, 4, 2], [2528, 1003, 846, 637, 516, 462, 462, 254, 191, 121], 8)
