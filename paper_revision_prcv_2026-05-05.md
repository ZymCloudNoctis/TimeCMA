# PRCV 投稿修改建议

针对当前稿件 [Stock_EMNLP26.pdf](/Users/yiming/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_s9qp01swu4lj22_4753/msg/file/2026-04/Stock_EMNLP26.pdf) 的定向修改建议。

## 1. 总体判断

当前稿件已经具备一个可成立的方法主线：

- 多股票收益预测
- 时间序列分支 + LLM prompt 分支
- 新闻诱导股票共现图
- 静态图与动态图

但以 `PRCV` 为目标时，当前版本还有 4 个明显问题：

1. 题目和摘要没有完全对齐当前方法。
   - 当前标题 `Thinking Like Financial Experts` 较虚，不能直接体现“图增强 + 跨模态 + 多股票预测”。
   - PDF 中摘要部分几乎是空的，至少从导出结果看没有有效摘要正文。

2. 论文结构还不完整。
   - 当前 PDF 只有 `Introduction / Related Works / Method / References`
   - 缺少最关键的 `Experiments`、`Ablation`、`Conclusion`

3. 方法描述有若干公式和排版错误。
   - 多处公式行文断裂
   - 个别引用损坏，例如 `Radford et al., 2019;?`
   - 个别句子存在空格和标点错误

4. 论文定位仍偏 `EMNLP/financial NLP`，没有完全转成 `PRCV` 更容易接受的“多模态表示学习 / 图推理 / 跨模态融合”叙述。

## 2. 建议的 PRCV 定位

相比当前“Thinking Like Financial Experts”的叙述，更适合 PRCV 的说法是：

- 这是一个 **multimodal representation learning** 问题
- 这是一个 **cross-modal alignment + graph-enhanced reasoning** 问题
- 文本不只是语义输入，也是结构关系来源

也就是说，论文重心应从：

- “金融专家如何思考”

转为：

- “如何把时间序列、语言表示和文本诱导关系图统一到一个跨模态框架里”

## 3. 必须先修的硬问题

### 3.1 标题需要直接改

当前标题太泛，不利于审稿人第一眼理解贡献。

建议标题 1：

**Graph-TimeCMA: Text-Induced Graph Enhanced Cross-Modal Learning for Multi-Stock Return Prediction**

建议标题 2：

**Graph-TimeCMA: Multimodal Cross-Modal Alignment with Text-Induced Stock Graphs for Return Prediction**

建议标题 3：

**Text-Induced Graph Enhanced Cross-Modal Learning for Multi-Stock Return Prediction**

如果你希望保留 “Graph-TimeCMA” 这个方法名，建议用标题 1。

### 3.2 摘要必须重写

当前 PDF 中摘要基本不可用。下面这版可以直接作为初稿。

## 4. 可直接替换的摘要

**Abstract**

Stock return prediction is inherently a multimodal learning problem, since investment decisions are jointly influenced by historical market behavior, textual market narratives, and cross-stock relations. Existing forecasting models mainly focus on numerical temporal patterns, while text-based financial models usually treat textual signals as stock-level evidence and graph-based stock predictors often rely on predefined or price-derived relations. In this paper, we propose **Graph-TimeCMA**, a text-induced graph-enhanced cross-modal framework for multi-stock return prediction. Our method extends TimeCMA by introducing financial text in two complementary roles. First, a frozen large language model encodes stock-level prompt embeddings from historical market sequences, providing semantic representations. Second, financial social media is used to construct stock co-occurrence graphs, providing structural relations among stocks. The resulting graph is injected into the time-series branch, the prompt branch, and the cross-modal aligned representation through lightweight multi-stage graph propagation. We evaluate the proposed method on HS300 multi-stock prediction under a rolling-window setting and report both regression metrics and finance-oriented cross-sectional metrics, including MSE, MAE, IC, RankIC, ICIR, and RankICIR. Experimental results show that the dynamic graph variant achieves the best average IC, RankIC, ICIR, and RankICIR, demonstrating that text-induced stock relations can effectively enhance cross-modal stock prediction.

## 5. 引言修改建议

当前引言的总体逻辑是对的，但有两个问题：

- 前半部分背景略长
- 方法切入不够快

更稳的结构应该是：

1. 股票预测为什么是多模态问题
2. 现有方法三条线各自缺什么
3. TimeCMA 和 COGRASP 分别提供了什么
4. 你的工作如何把两者统一

### 建议直接替换的引言后半段

你当前第 1 节后半段可改成下面这种更紧的写法：

> Recent progress on stock prediction mainly follows three directions. Numerical forecasting models learn temporal dependencies from historical prices and indicators, but often neglect market narratives and cross-stock textual relations. Financial NLP methods encode news, reports, or social media discussions, yet they typically use text only as stock-level evidence. Graph-based stock predictors explicitly model inter-stock dependencies, but their graphs are often predefined, price-derived, or slowly updated. As a result, existing approaches rarely integrate numerical signals, language representations, and text-induced stock relations in a unified framework.
>
> Two recent studies are especially relevant to our work. TimeCMA aligns time-series representations with LLM-derived prompt embeddings through cross-modality alignment, showing that language representations can enhance multivariate forecasting. However, it does not explicitly model stock-level relations induced from financial text. In contrast, COGRASP constructs stock co-occurrence graphs from reports, newspapers, and social media by linking stocks co-mentioned in the same content. While such graphs provide interpretable and dynamically updated stock relations, they are not integrated into an LLM-enhanced cross-modal forecasting framework.
>
> To bridge this gap, we propose **Graph-TimeCMA**, a text-induced graph-enhanced cross-modal framework for multi-stock return prediction. The key idea is to treat financial text as both a semantic resource and a structural resource. On the semantic side, a frozen LLM produces stock-level prompt embeddings from historical time-series prompts. On the structural side, financial social media is used to construct stock co-occurrence graphs. These text-induced relations are then injected into the time-series branch, the prompt branch, and the aligned representation through lightweight graph propagation. In this way, Graph-TimeCMA unifies temporal information, language representations, and cross-stock relation priors within a single multimodal prediction framework.

## 6. 贡献点建议重写

你当前 contributions 基本能用，但表述还可以更学术、更聚焦。

建议改成：

- We reformulate multi-stock return prediction as a multimodal learning problem that jointly leverages numerical market signals, language representations, and text-induced stock relations.
- We propose Graph-TimeCMA, which extends cross-modal alignment by injecting financial social-media-induced stock co-occurrence graphs into the time-series branch, the prompt branch, and the aligned representation.
- We construct both static and dynamic stock co-occurrence graphs from financial text and investigate the effect of temporal decay in modeling time-varying stock relations.
- We conduct rolling-window experiments and ablation studies on HS300 multi-stock prediction, showing that text-induced graphs improve cross-sectional correlation and prediction stability.

## 7. 方法部分需要修的具体问题

### 7.1 公式排版问题

以下几处需要重排：

1. `3.6 Text-Induced Stock Co-Occurrence Graph`
   - 现在 `log1p` 和归一化公式挤在一起，阅读很差
   - 建议拆成 3 行：
     - 原始共现矩阵
     - 对数压缩
     - 归一化邻接矩阵

建议写成：

\[
\tilde{C}_{ij} = \log (1 + C_{ij}),
\]
\[
\hat{C} = \text{TopK}(\tilde{C}),
\]
\[
A = D^{-1/2}(\hat{C}+I)D^{-1/2}.
\]

2. `3.7 Multi-Stage Graph Propagation`
   - 当前 `g=\sigma(\gamma)` 和 `GProp(H)=...` 混在一起了
   - 应分开写

建议写成：

\[
H^{(0)} = H,\quad H^{(k+1)} = A H^{(k)} ,
\]
\[
H_{\text{prop}} = H^{(K)},
\]
\[
g = \sigma(\gamma),
\]
\[
\text{GProp}(H) = (1-g)H + gH_{\text{prop}}.
\]

3. `3.8 Graph-Enhanced Cross-Modal Alignment`
   - 现在 `Hts_t = GProp(Hts_t)` 会和原变量重复
   - 建议改成带横线或上标的中间变量，例如：
     - `\bar{H}^{ts}_t`
     - `\bar{H}^{p}_t`
     - `\bar{Z}_t`

### 7.2 引用错误

当前文中存在：

- `(Radford et al., 2019;?)`

这类引用必须清掉，说明 `.bib` 或 `\cite{}` 有坏条目。

### 7.3 行文细节错误

当前至少有这些明显问题：

- `stability.Our contributions are as follows:`  
  应改成 `stability. Our contributions are as follows:`

- `(Jin et al., 2023; Liu et al., 2025),we con-`  
  应改成 `..., we con-`

- `framework(Liu et al., 2025; Li et al., 2025).`  
  应改成 `framework (Liu et al., 2025; Li et al., 2025).`

## 8. 当前稿件缺失的核心章节

如果要投稿 PRCV，当前版本至少还缺：

### 8.1 Experiments

建议结构：

1. Dataset and Setting
2. Baselines
3. Main Results
4. Ablation Studies
5. Discussion

### 8.2 需要明确写进实验部分的内容

- 数据范围：`2024-01-01` 到 `2025-09-30`
- 滚动窗口：
  - 训练 `12` 个月
  - 验证 `3` 个月
  - 测试 `1` 个月
  - 测试月份 `2025-04` 到 `2025-09`
- 静态图与动态图定义
- 标签 winsorization 设置
- `dynamic6m` 半衰期 `60` 天

### 8.3 你现在最值得放的主结果

根据当前实验，主结果建议这样组织：

- Conventional baselines:
  - `MLP`
  - `LSTM`
  - `ALSTM`
  - `Transformer`
  - `XGBoost`
- TimeCMA family:
  - `no_graph`
  - `static`
  - `dynamic6m`

并且强调：

- `dynamic6m` 在 `IC / RankIC / ICIR / RankICIR` 上最好
- 常规模型虽然 `MSE / MAE` 更低，但横截面相关性不稳定甚至为负

## 9. 最适合 PRCV 的结论写法

不要写成：

- “本文方法在所有指标上都优于 baseline”

因为这不成立。

建议写成：

> Experimental results show that although some conventional sequence baselines obtain lower point-wise regression errors, the proposed Graph-TimeCMA consistently achieves better cross-sectional correlation and stability metrics, including IC, RankIC, ICIR, and RankICIR. This indicates that text-induced stock relations are particularly useful for modeling relative return structures across stocks.

## 10. 我建议你下一步怎么做

最务实的顺序是：

1. 先把标题、摘要、引言和方法错误全部修干净
2. 补完整 `Experiments` 章节
3. 把 baseline 对比表和 ablation 表放进去
4. 最后再根据 PRCV 模板统一格式

如果你把 `.tex` 源文件发我，而不是只给 PDF，我可以直接帮你逐段改正文，并给出可粘贴回去的英文版本。

