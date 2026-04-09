import numpy as np
import matplotlib.pyplot as plt
# 加载数据
preds = np.load('preds.npy') # 形状: (样本数, 预测长度5, 节点数9)
trues = np.load('trues.npy')
# 选取最后一个特征（通常是目标价格 OT）进行可视化
# 我们选取测试集中的一段连续序列进行观察
feature_idx = -1
sample_idx = 0 # 观察第一个测试样本
plt.figure(figsize=(12, 6))
# 绘制真实值
plt.plot(trues[sample_idx, :, feature_idx], label='Actual Price', marker='o', color='blue')
# 绘制预测值
plt.plot(preds[sample_idx, :, feature_idx], label='TimeCMA Prediction', marker='x', linestyle='--', color='red')
plt.title('600519 (Moutai) Price Prediction - Next 5 Days')
plt.xlabel('Days Ahead')
plt.ylabel('Normalized Price')
plt.legend()
plt.grid(True)
plt.show()
# 计算每步的误差分析
for i in range(5):
    step_mse = np.mean((preds[:, i, feature_idx] - trues[:, i, feature_idx])**2)
    print(f"Day {i+1} Prediction MSE: {step_mse:.4f}")
