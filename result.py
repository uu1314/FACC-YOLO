# import pandas as pd
# import matplotlib
# matplotlib.use('TkAgg')  # 或 'Qt5Agg'
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体（SimHei）
# plt.rcParams['axes.unicode_minus'] = False     # 正确显示负号
#
# def get_map50(file_path,yolo,color):
#     df = pd.read_csv(file_path)
#     epochs = df.index
#     map50 = df['metrics/mAP50(B)']
#     plt.plot(epochs, map50, label=yolo, color=color, linewidth=2)
#
# colors=['orange','black','red','deepskyblue','green','gray','orange']
# file_paths=['','','','','','']
# file_paths[0] = r'F:\Desktop\YOLO\ultralytics-main\runs\detect\train13\results.csv'  # 或使用绝对路径
# file_paths[1]=r'F:\Desktop\YOLO\ultralytics-main\runs\detect\train12\results.csv'
# file_paths[2]=r'F:\Desktop\YOLO\ultralytics-main\runs\detect\train14\results.csv'
# file_paths[3]=r'F:\Desktop\YOLO\ultralytics-main\runs\detect\train15\results.csv'
# file_paths[4]=r'F:\Desktop\YOLO\ultralytics-main\runs\detect\train16\results.csv'
# file_paths[5]=r'F:\Desktop\YOLO\ultralytics-main\runs\detect\train17\results.csv'
# yolo=['yolov11','yolov8','yolov5','yolov3','yolov6','ours']
# # 绘图
# plt.figure(figsize=(10, 6))
# for i in range(len(file_paths)):
#     get_map50(file_paths[i],yolo[i],colors[i])
# # 设置图形属性
# plt.xlabel('选代次数/次', fontsize=12)
# plt.ylabel('mAP50', fontsize=12)
# plt.title('a.50%的IoU阈值下计算的mAP值变化', fontsize=14)
# plt.legend()
# plt.tight_layout()
# plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')  # 必须在 plt 前设置
import os

# 全局图形配置
plt.rcParams.update({
    'font.sans-serif': ['SimHei'],  # 中文
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'figure.figsize': (12, 7),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10
})

# ✅ 性能指标别名映射（用于图例）
metric_aliases = {
    'train/box_loss':'train/box_loss',
    'metrics/mAP50(B)': 'mAP@0.5',
    'metrics/mAP50-95(B)': 'mAP@0.5:0.95',
    'metrics/precision(B)': 'Precision',
    'metrics/recall(B)': 'Recall'
}

# ✅ 通用数据处理函数
def process_data(file_paths, models, metric):
    results = []
    for path, model in zip(file_paths, models):
        try:
            df = pd.read_csv(path)
            if metric not in df.columns:
                print(f"⚠️ {model} 缺少列：{metric}")
                continue
            df['model'] = model
            df['metric_smooth'] = df[metric].rolling(3, center=True, min_periods=1).mean()
            results.append(df)
        except Exception as e:
            print(f"❌ 加载 {model} 数据失败: {e}")
    if results:
        return pd.concat(results)
    else:
        raise ValueError("❗ 所有文件读取失败或不含指定指标列")

# ✅ 动态标注最大值
def plot_max_points(ax, data, colors):
    unique_models = data['model'].unique()
    model_color_map = dict(zip(unique_models, colors))
    annotate_config = {
        'YOLOv11': {'xytext': (0, 8), 'ha': 'center', 'va': 'bottom'},
        'YOLOv8': {'xytext': (0, -8), 'ha': 'center', 'va': 'top'},
        'YOLOv5': {'xytext': (-8, 0), 'ha': 'right', 'va': 'center'},
        'YOLOv3': {'xytext': (0, -8), 'ha': 'center', 'va': 'top'},
        'YOLOv6': {'xytext': (0, -8), 'ha': 'center', 'va': 'top'},
        'Ours': {'xytext': (0, 8), 'ha': 'center', 'va': 'bottom'},
    }
    for model in unique_models:
        model_data = data[data['model'] == model]
        max_idx = model_data['metric_smooth'].idxmax()
        max_row = model_data.loc[max_idx]
        color = model_color_map.get(model, 'black')
        config = annotate_config.get(model, {'xytext': (15, 0), 'ha': 'left', 'va': 'center'})
        ax.scatter(max_idx, max_row['metric_smooth'], color=color, s=40, zorder=5,
                   edgecolor='white', linewidth=1)
        ax.annotate(f"{max_row['metric_smooth']:.3f}",
                    (max_idx, max_row['metric_smooth']),
                    xytext=config['xytext'], textcoords='offset points',
                    ha=config['ha'], va=config['va'], fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))

# ✅ 通用绘图函数
def plot_comparison(data, colors, metric):
    metric_label = metric_aliases.get(metric, metric)
    fig, ax = plt.subplots(figsize=(12, 7))

    for i, model in enumerate(data['model'].unique()):
        model_data = data[data['model'] == model]
        ax.plot(model_data.index, model_data['metric_smooth'],
                label=model, color=colors[i], linewidth=1.5, alpha=0.9)

    plot_max_points(ax, data, colors)

    ax.set(xlabel='Epoch', ylabel=metric_label,
           xlim=(0, None), ylim=(0, 1.0))
    ax.legend(title='model', framealpha=0.8)
    ax.grid(True, alpha=0.6)
    sns.despine()
    return fig

# ✅ 参数设置
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#17becf", "#9467bd", "#d62728"]
file_paths = [
    r'F:\Desktop\YOLO\ultralytics-main\runs\detect\train13\results.csv',
    r'F:\Desktop\YOLO\ultralytics-main\runs\detect\train12\results.csv',
    r'F:\Desktop\YOLO\ultralytics-main\runs\detect\train14\results.csv',
    r'F:\Desktop\YOLO\ultralytics-main\runs\detect\train15\results.csv',
    r'F:\Desktop\YOLO\ultralytics-main\runs\detect\train16\results.csv',
    r'F:\Desktop\YOLO\ultralytics-main\runs\detect\train17\results.csv'
]
models = ['YOLOv11', 'YOLOv8', 'YOLOv5', 'YOLOv3', 'YOLOv6', 'Ours']
metric_to_plot = 'metrics/recall(B)'  # 可改成 'metrics/precision(B)'、'metrics/recall(B)' 等

# ✅ 执行流程
data = process_data(file_paths, models, metric_to_plot)
fig = plot_comparison(data, colors, metric_to_plot)

# ✅ 输出保存
filename = f"YOLO_comparison_{metric_aliases.get(metric_to_plot, metric_to_plot).replace('/', '_')}.png"
fig.savefig(filename, bbox_inches='tight', dpi=300)
plt.show()
