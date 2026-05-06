# PRCV 章节替换包

当前项目里没有你原始论文的 `.tex` 和 `.bib` 源文件，所以我把现有完整草稿进一步整理成“章节替换包”。  
你可以把这些部分分别粘到你自己的论文模板里，而不需要整篇手动拆分。

对应来源文件：

- 正文草稿：
  [paper_prcv_full_draft_latex_2026-05-05.tex](/Users/yiming/project/TimeCMA/paper_prcv_full_draft_latex_2026-05-05.tex)
- 表格草稿：
  [paper_tables_prcv_2026-05-05.tex](/Users/yiming/project/TimeCMA/paper_tables_prcv_2026-05-05.tex)

---

## 1. 标题

```latex
\title{Graph-TimeCMA: Text-Induced Graph Enhanced Cross-Modal Learning for Multi-Stock Return Prediction}
```

如果你觉得标题太长，可以用这个较短版本：

```latex
\title{Graph-TimeCMA: Text-Induced Graph Enhanced Cross-Modal Learning for Stock Return Prediction}
```

---

## 2. 摘要

从正文文件里复制：

- `\begin{abstract}` 到 `\end{abstract}`

直接放在：

```latex
\maketitle
\begin{abstract}
...
\end{abstract}
```

之后接：

```latex
\section{Introduction}
```

---

## 3. 引言

从正文文件里复制：

- `\section{Introduction}` 整段

建议放完引言之后，立刻接：

```latex
\section{Related Work}
```

如果你篇幅紧张，可以优先保留：

- 第一段：任务背景
- 第二段：现有三类方法的缺口
- 第三段：TimeCMA 与 COGRASP
- 第四段：本文方法
- 最后一段：贡献

---

## 4. 相关工作

从正文文件里复制：

- `\section{Related Work}` 整段

当前结构已经分成：

- `Financial Text for Stock Prediction`
- `Multivariate and LLM-based Time-Series Forecasting`
- `Graph-based Stock Prediction and Text-Induced Relations`
- `Positioning`

如果你需要压缩页数，优先删减：

- 金融文本背景里的长铺垫
- 保留和你方法最相关的 `TimeCMA / COGRASP / graph-based stock prediction`

---

## 5. 方法

从正文文件里复制：

- `\section{Method}` 整段

当前已经拆成：

- `Problem Definition`
- `Framework Overview`
- `Market Features and Return Labels`
- `LLM Prompt Encoding`
- `Time-Series Encoding`
- `Text-Induced Stock Co-Occurrence Graph`
- `Multi-Stage Graph Propagation`
- `Graph-Enhanced Cross-Modal Alignment`
- `Prediction Objective and Evaluation Metrics`

这一部分已经是可直接进稿的版本。

你只需要再检查两件事：

1. 数学符号是否与你模板中的变量风格一致
2. `\cite{}` 是否需要改成你 `.bib` 里的真实 key

---

## 6. 实验

从正文文件里复制：

- `\section{Experiments}` 开始到 `\section{Main Results}` 之前

当前实验设置里最重要的信息已经写进去了：

- 数据时间范围：`2024-01-01` 到 `2025-09-30`
- 滚动窗口：
  - 训练 `12` 个月
  - 验证 `3` 个月
  - 测试 `1` 个月
  - 测试月份 `2025-04` 到 `2025-09`
- prompt embedding 使用 frozen GPT-2
- dynamic graph 使用 6 个月窗口 + 60 天半衰期
- 默认 winsorize 是 `1%~99%`

---

## 7. 主结果

从正文文件里复制：

- `\section{Main Results}` 整段

同时插入表格文件中的：

- `tab:main_results`

建议插入位置：

- 放在 `\subsection{Comparison with Conventional Baselines}` 第一段前或后都可以

正文中已经有：

```latex
Table~\ref{tab:main_results}
```

所以插入后引用会自动对上。

---

## 8. 消融实验

从正文文件里复制：

- `\section{Ablation Study}` 整段

同时插入表格文件中的：

- `tab:ablation_results`

建议插入位置：

- 放在 `\subsection{Ablation Results}` 开头附近

---

## 9. 可选逐窗口表

如果你篇幅允许，可以再加入：

- `tab:per_window_timecma`

这张表的作用是：

- 支撑你在正文中关于“动态图更稳定”的结论
- 回答审稿人“是不是只在某个窗口偶然有效”的疑问

如果篇幅不够，可以删掉这张表，把逐窗口分析只保留在正文里。

---

## 10. 讨论与结论

从正文文件里复制：

- `\section{Discussion}`
- `\section{Conclusion}`

如果页数紧：

- `Discussion` 可以压缩成 2 个小节：
  - Why Graph-TimeCMA Improves Cross-Sectional Metrics
  - Limitations

---

## 11. 建议导言区补的宏包

如果你的模板没有这些包，建议补上：

```latex
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{amsmath}
\usepackage{amssymb}
```

---

## 12. 你现在最应该做的事

1. 打开你自己的论文主 `.tex`
2. 按下面顺序替换：
   - 标题
   - 摘要
   - 引言
   - 方法
   - 实验
   - 主结果
   - 消融
   - 结论
3. 粘贴两张核心表：
   - `tab:main_results`
   - `tab:ablation_results`
4. 编译一次
5. 再检查：
   - 引用 key 是否存在
   - 表格是否过宽
   - 数学公式是否和模板冲突

---

## 13. 现实提醒

现在你手上的材料，已经从“半成稿”变成“完整论文初稿”了。  
但在投稿前，你仍然需要自己做最后两轮检查：

- 一轮是技术一致性检查：
  - 实验数值
  - 图表标题
  - 引用 key
  - 模型名称前后一致
- 一轮是篇幅与语言压缩：
  - 把重复的背景删掉
  - 把最重要的结果写得更集中

