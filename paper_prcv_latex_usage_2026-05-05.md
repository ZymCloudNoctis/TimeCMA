# LaTeX 使用说明

生成的两个核心文件是：

- 正文版：
  [paper_prcv_full_draft_latex_2026-05-05.tex](/Users/yiming/project/TimeCMA/paper_prcv_full_draft_latex_2026-05-05.tex)
- 表格版：
  [paper_tables_prcv_2026-05-05.tex](/Users/yiming/project/TimeCMA/paper_tables_prcv_2026-05-05.tex)

## 1. 如何使用正文版

正文文件已经按会议论文结构写好，包含：

- `\title{}`
- `abstract`
- `Introduction`
- `Related Work`
- `Method`
- `Experiments`
- `Main Results`
- `Ablation Study`
- `Discussion`
- `Conclusion`

你可以：

1. 打开你当前论文的主 `.tex` 文件
2. 保留原来的模板、作者区、`\begin{document}`、`\maketitle`
3. 把正文部分替换成这份文件中的对应内容

注意：

- 如果你模板里已经有 `\title{}`，就不要重复粘贴两个 `\title{}`
- 直接从 `\begin{abstract}` 开始往下放，通常最稳

## 2. 如何使用表格版

表格文件里包含三张表：

- `tab:main_results`
- `tab:ablation_results`
- `tab:per_window_timecma`

通常做法是：

1. 把表格文件里的内容粘到主 `.tex` 的实验部分后面
2. 在正文中保留这些引用：
   - `Table~\ref{tab:main_results}`
   - `Table~\ref{tab:ablation_results}`
   - `Table~\ref{tab:per_window_timecma}`

## 3. 你很可能还要补的宏包

如果模板里没有这些宏包，建议检查导言区是否有：

```latex
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{amsmath}
\usepackage{amssymb}
```

## 4. 你下一步最该做的事

1. 把这两份内容塞回你现有 `.tex`
2. 编译一次，看：
   - 公式是否溢出
   - 表格是否过宽
   - 引用是否缺 BibTeX 条目
3. 再做一轮语言压缩和篇幅控制

## 5. 现实提醒

这两份文件是“可投稿初稿”的骨架，不是最终定稿。

你还需要自己再核对：

- 引用是否全部存在于 `.bib`
- 数学符号是否和你前文统一
- 表格数值是否和最终实验结果完全一致
- 是否满足 PRCV 页数限制和格式要求

