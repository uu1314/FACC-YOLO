import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')  # 必须在 plt 前设置

plt.rcParams.update({
    'font.sans-serif': ['SimHei'],
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'figure.figsize': (12, 7),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10
})

def process_loss_data(file_paths, models, metric):
    results = []
    for path, model in zip(file_paths, models):
        try:
            df = pd.read_csv(path)

            # 自动尝试匹配含 metric 的列（允许括号等后缀）
            matched_col = [col for col in df.columns if metric in col]
            if not matched_col:
                print(f"⚠️ 模型 {model} 不包含指标列: {metric}")
                continue
            col = matched_col[0]

            df['model'] = model


            # 添加平滑列
            df['Epoch'] = range(len(df))
            df['metric_smooth'] = df[col].rolling(3, center=True, min_periods=1).mean()

            results.append(df)
        except Exception as e:
            print(f"❌ 读取模型 {model} 数据失败: {e}")
    return pd.concat(results, ignore_index=True)

def plot_loss_comparison(data, metric, colors):
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, model in enumerate(data['model'].unique()):
        model_data = data[data['model'] == model]
        ax.plot(model_data['Epoch'], model_data['metric_smooth'],
                label=model, color=colors[i], linewidth=1, alpha=0.9)

    ax.set(
        xlabel='Epoch',
        ylabel=metric,
        title=f'不同模型的 {metric} 损失对比图',
        xlim=(0, None)
    )
    ax.legend(title='模型', framealpha=0.8)
    ax.grid(True, alpha=0.6)
    sns.despine()
    return fig

file_paths = [
    r'F:\Desktop\YOLO\ultralytics-main\runs\detect\train13\results.csv',
    r'F:\Desktop\YOLO\ultralytics-main\runs\detect\train12\results.csv',
    r'F:\Desktop\YOLO\ultralytics-main\runs\detect\train14\results.csv',
    r'F:\Desktop\YOLO\ultralytics-main\runs\detect\train15\results.csv',
    r'F:\Desktop\YOLO\ultralytics-main\runs\detect\train16\results.csv',
    r'F:\Desktop\YOLO\ultralytics-main\runs\detect\train17\results.csv',
    r'F:\Desktop\YOLO\ultralytics-main\runs\detect\train18\results.csv'
]

models = ['YOLOv11', 'YOLOv8', 'YOLOv5', 'YOLOv3', 'YOLOv6', 'Ours','i']
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#17becf", "#9467bd", "#d62728",'black']

# 指标名：例如 'train/box_loss'
metric = 'train/box_loss'

# 处理并绘图
loss_data = process_loss_data(file_paths, models, metric)
fig = plot_loss_comparison(loss_data, metric, colors)

# 保存 & 显示图像

plt.show()
