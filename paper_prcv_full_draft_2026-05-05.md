# Graph-TimeCMA: Text-Induced Graph Enhanced Cross-Modal Learning for Multi-Stock Return Prediction

Anonymous PRCV submission

## Abstract

Stock return prediction is inherently a multimodal learning problem, since investment decisions are jointly influenced by historical market behavior, textual market narratives, and cross-stock relations. Existing forecasting models mainly focus on numerical temporal patterns, while text-based financial models usually treat textual signals as stock-level evidence and graph-based stock predictors often rely on predefined or price-derived relations. In this paper, we propose **Graph-TimeCMA**, a text-induced graph-enhanced cross-modal framework for multi-stock return prediction. Our method extends TimeCMA by introducing financial text in two complementary roles. First, a frozen large language model encodes stock-level prompt embeddings from historical market sequences, providing semantic representations. Second, financial social media is used to construct stock co-occurrence graphs, providing structural relations among stocks. The resulting graph is injected into the time-series branch, the prompt branch, and the cross-modal aligned representation through lightweight multi-stage graph propagation. We evaluate the proposed method on HS300 multi-stock prediction under a rolling-window setting and report both regression metrics and finance-oriented cross-sectional metrics, including MSE, MAE, IC, RankIC, ICIR, and RankICIR. Experimental results show that the dynamic graph variant achieves the best average IC, RankIC, ICIR, and RankICIR, demonstrating that text-induced stock relations can effectively enhance cross-modal stock prediction.

## 1. Introduction

Stock return prediction remains a central challenge in financial artificial intelligence. Classical market efficiency theory suggests that asset prices incorporate available information rapidly, making persistent prediction difficult. At the same time, behavioral finance and market microstructure research show that prices are also shaped by investor sentiment, media attention, topic diffusion, and belief propagation. In practice, experienced financial analysts rarely rely on a single signal. They jointly inspect price trends, trading volume, sector rotation, investor attention, company-specific narratives, and related stocks. Stock prediction is therefore not merely a numerical forecasting task, but a multimodal reasoning problem involving temporal signals, textual evidence, and cross-stock relations.

Prior studies have explored these components separately. Numerical forecasting models learn temporal dependencies from historical prices and indicators, but often neglect market narratives and external relation evidence. Financial NLP methods encode news, reports, or social media discussions, yet they typically use text only as stock-level evidence. Graph-based stock prediction methods explicitly model inter-stock dependencies, but their graphs are often predefined, price-derived, or slowly updated. As a result, existing approaches rarely integrate numerical signals, language representations, and text-induced stock relations in a unified framework.

Two recent studies are especially relevant to this work. TimeCMA aligns numerical time-series representations with LLM-derived prompt embeddings through cross-modality alignment, showing that language representations can enhance multivariate forecasting. However, it does not explicitly model stock-level relations induced from financial text. In contrast, COGRASP constructs stock co-occurrence graphs from reports, newspapers, and social media by linking stocks co-mentioned in the same content. Such graphs provide interpretable and dynamically updated stock relations, but they are not integrated into an LLM-enhanced cross-modal forecasting framework.

To bridge this gap, we propose **Graph-TimeCMA**, a text-induced graph-enhanced cross-modal framework for multi-stock return prediction. The key idea is to treat financial text as both a semantic resource and a structural resource. On the semantic side, a frozen LLM produces stock-level prompt embeddings from historical time-series prompts. On the structural side, financial social media is used to construct stock co-occurrence graphs. These text-induced relations are then injected into the time-series branch, the prompt branch, and the aligned representation through lightweight graph propagation. In this way, Graph-TimeCMA unifies temporal information, language representations, and cross-stock relation priors within a single multimodal prediction framework.

We evaluate the proposed framework on HS300 multi-stock prediction, where the task is to predict future 5-day cumulative returns under rolling-window evaluation. Besides MSE and MAE, we report finance-oriented cross-sectional metrics, including IC, RankIC, ICIR, and RankICIR, to assess predictive correlation, ranking quality, and stability. The results show that the dynamic graph variant delivers the best average cross-sectional performance, while ablation studies confirm the contributions of text-induced graphs, temporal decay, and label winsorization.

The main contributions of this work are as follows:

- We formulate multi-stock return prediction as a multimodal learning problem that jointly leverages numerical market signals, language representations, and text-induced stock relations.
- We propose Graph-TimeCMA, which extends cross-modal alignment by injecting financial-social-media-induced stock co-occurrence graphs into the time-series branch, the prompt branch, and the aligned representation.
- We construct both static and dynamic stock co-occurrence graphs from financial text and investigate the effect of temporal decay in modeling time-varying stock relations.
- We conduct rolling-window experiments and ablation studies on HS300 multi-stock prediction, showing that text-induced graphs improve cross-sectional correlation and prediction stability.

## 2. Related Work

### 2.1 Financial Text for Stock Prediction

Financial text has long been used to study market behavior and stock movements. Prior finance research shows that media tone, investor sentiment, and attention contain information related to returns and trading activity. Online investor discussions and social media further reveal market beliefs and attention patterns, although they also contain substantial noise. Textual analysis in finance requires domain-specific treatment because financial language differs substantially from general-domain language.

NLP-based stock prediction directly models textual evidence. StockNet jointly models tweets and historical prices for stock movement prediction. Subsequent work combines social media text, financial time series, and company correlations, showing that textual signals and inter-company dependencies can be complementary. Domain-adapted language models such as FinBERT and BloombergGPT further demonstrate the value of financial-domain pretraining. Recent LLM-based financial prediction methods use financial news to obtain return-predictive representations or explainable factors. However, most prior work uses text as stock-level sentiment, event, or factor evidence. Our work instead uses text in two roles: as semantic input for LLM-prompt representations and as structural evidence for constructing stock relations.

### 2.2 Multivariate and LLM-based Time-Series Forecasting

Multi-stock prediction can be formulated as multivariate time-series forecasting, where each stock is a variable or node and the target is its future return. Recent forecasting models improve numerical temporal modeling through efficient attention, decomposition, patching, variate-token attention, and multiscale mixing. These models provide strong numerical backbones, but they mainly infer dependencies from historical co-movement and do not explicitly use financial text as relation evidence.

LLM-based forecasting introduces language representations into time-series modeling. Time-LLM reprograms time-series inputs for frozen LLMs, while semantic alignment approaches map time-series embeddings into language-model-informed representation spaces. TimeCMA is the closest framework to ours: it aligns a time-series branch with an LLM-prompt branch to retrieve useful temporal components from prompt embeddings. However, TimeCMA focuses on temporal-prompt alignment and does not incorporate stock relation priors induced from financial text.

### 2.3 Graph-based Stock Prediction and Text-Induced Relations

Graph-based stock prediction models explicitly represent cross-stock dependencies. In stock prediction, graph neural networks have been used to encode company relations, concept-oriented shared information, and dynamic multi-relational dependencies. These studies show that cross-stock structure matters, but many relation graphs are predefined, price-derived, or slow to update.

COGRASP addresses graph construction from a text perspective. It builds stock co-occurrence graphs from reports, newspapers, and social media by linking stocks co-mentioned in the same content and weighting edges by co-mention frequency. Such text-induced graphs are interpretable because each relation can be traced back to textual co-mentions. However, COGRASP uses co-occurrence graphs mainly as relation mining signals and does not integrate them into an LLM-based cross-modal alignment framework.

### 2.4 Positioning

Graph-TimeCMA connects two previously separate directions: LLM-enhanced cross-modal time-series forecasting and text-induced stock relation modeling. Compared with numerical forecasters, it incorporates financial text. Compared with text-based stock predictors, it uses text as both semantic evidence and structural relation evidence. Compared with graph-based stock models, it injects text-induced relations into the time-series branch, the prompt branch, and the cross-modal aligned representation. This design provides a unified framework for multimodal stock prediction with numerical, textual, and relational evidence.

## 3. Method

### 3.1 Problem Definition

Let \(S = \{s_1, \dots, s_N\}\) denote the stock universe. At prediction day \(t\), each stock is associated with \(F\)-dimensional market features over a look-back window of length \(L\). The multi-stock input tensor is:

\[
X_{t-L+1:t} \in \mathbb{R}^{L \times N \times F}.
\]

In our setting, the market features include open, high, low, close, volume, amount, daily return, log return, and amplitude. The prediction target is the future \(H\)-day cumulative return:

\[
y^{(H)}_{i,t} = \frac{\text{close}_{i,t+H}}{\text{close}_{i,t}} - 1.
\]

The model outputs stock-level return predictions:

\[
\hat{y}^{(H)}_t \in \mathbb{R}^N.
\]

We use \(H = 5\) by default and formulate the task as regression, while evaluation emphasizes cross-sectional metrics such as IC and RankIC.

### 3.2 Framework Overview

Graph-TimeCMA extends TimeCMA by injecting a text-induced stock co-occurrence graph into the cross-modal forecasting framework. The overall mapping is:

\[
\hat{y}^{(H)}_t = f_\theta(X_{t-L+1:t}, E_t, A_t),
\]

where \(E_t\) denotes cached LLM prompt embeddings and \(A_t\) denotes the normalized stock co-occurrence graph.

Graph-TimeCMA contains four components:

1. a time-series encoder for numerical market features,
2. an LLM-prompt encoder for cached GPT-2 last-token embeddings,
3. a text-induced stock co-occurrence graph constructed from financial social media,
4. a multi-stage graph propagation module applied to the time-series representation, prompt representation, and cross-modal aligned representation.

### 3.3 Market Features and Return Labels

For each stock, we construct nine market features: open, high, low, close, volume, amount, daily return, log return, and amplitude. Features are standardized using statistics fitted only on the training split to avoid leakage.

To reduce the impact of extreme returns, we apply label winsorization based on training-set quantiles:

\[
q_i^{\text{low}} = \text{Quantile}(y^{(H)}_{i,t}, \alpha), \quad
q_i^{\text{high}} = \text{Quantile}(y^{(H)}_{i,t}, 1-\alpha),
\]

\[
\tilde{y}^{(H)}_{i,t} = \min \left( \max \left(y^{(H)}_{i,t}, q_i^{\text{low}}\right), q_i^{\text{high}} \right).
\]

We use \(\alpha = 0.01\) by default.

### 3.4 LLM Prompt Encoding

Following prompt-based LLM time-series modeling, we construct a textual prompt from each stock's historical time-series window and feed it into a frozen GPT-2 model. We take the last-token hidden state as the stock-level prompt embedding:

\[
e_{i,t} = g_\phi(p_{i,t})_{\text{last}} \in \mathbb{R}^{768}.
\]

All stock-level embeddings form:

\[
E_t \in \mathbb{R}^{N \times 768}.
\]

These embeddings are precomputed and cached offline. The prompt encoder maps them into the model hidden space:

\[
H_t^{p} = f_p(E_t) \in \mathbb{R}^{N \times d}.
\]

### 3.5 Time-Series Encoding

The numerical branch maps historical market features into stock-level temporal embeddings:

\[
H_t^{ts} = f_{ts}(X_{t-L+1:t}) \in \mathbb{R}^{N \times d}.
\]

Following the variable-token view in multivariate time-series forecasting, each stock is treated as a variable or node. This is crucial in our setting because the co-occurrence graph and the time-series branch share the same node semantics.

### 3.6 Text-Induced Stock Co-Occurrence Graph

Following text-induced relation modeling, let \(D = \{d_1, \dots, d_M\}\) be the document set, and let \(S_m \subseteq S\) be the set of stocks mentioned in document \(d_m\). The static co-occurrence count between stocks \(s_i\) and \(s_j\) is:

\[
C_{ij} = \sum_{m=1}^{M} \mathbb{I}(s_i \in S_m)\mathbb{I}(s_j \in S_m).
\]

For the dynamic graph, only documents in the six-month window before the test month are used. Each document is assigned an exponential decay weight:

\[
w_m(t) = 0.5^{\Delta_m(t)/T_{\text{half}}},
\]

where \(\Delta_m(t)\) is the number of days from the document timestamp to the end of the graph window, and \(T_{\text{half}} = 60\) days by default. The dynamic co-occurrence matrix is:

\[
C^{(t)}_{ij} = \sum_{m : d_m \in W(t)} w_m(t) \mathbb{I}(s_i \in S_m)\mathbb{I}(s_j \in S_m).
\]

The graph is further processed by log compression, top-\(k\) sparsification, self-loop addition, and symmetric normalization:

\[
\tilde{C}_{ij} = \log(1 + C_{ij}),
\]

\[
\hat{C} = \text{TopK}(\tilde{C}),
\]

\[
A = D^{-1/2}(\hat{C} + I)D^{-1/2}.
\]

### 3.7 Multi-Stage Graph Propagation

For any stock-level representation \(H \in \mathbb{R}^{N \times d}\), graph propagation is defined as:

\[
H^{(0)} = H, \quad H^{(k+1)} = A H^{(k)}.
\]

After \(K\) propagation steps:

\[
H_{\text{prop}} = H^{(K)}.
\]

A learnable graph gate combines the original and propagated representations:

\[
g = \sigma(\gamma),
\]

\[
\text{GProp}(H) = (1-g)H + gH_{\text{prop}}.
\]

We use this lightweight propagation operator rather than a full graph neural network to isolate the effect of text-induced relation priors and reduce additional modeling complexity.

### 3.8 Graph-Enhanced Cross-Modal Alignment

We first graph-enhance the time-series representation:

\[
\bar{H}^{ts}_t = \text{GProp}(H^{ts}_t).
\]

We also graph-enhance the prompt representation:

\[
\bar{H}^{p}_t = \text{GProp}(H^{p}_t).
\]

Cross-modal alignment is implemented with scaled dot-product attention, following the TimeCMA alignment principle. Specifically, the graph-enhanced time-series representation is used as query, and the graph-enhanced prompt representation is used as key and value:

\[
Q = \bar{H}^{ts}_t W_Q, \quad
K = \bar{H}^{p}_t W_K, \quad
V = \bar{H}^{p}_t W_V,
\]

\[
Z_t = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V.
\]

Finally, the aligned representation is graph-enhanced again:

\[
\bar{Z}_t = \text{GProp}(Z_t).
\]

The current implementation does not add the graph directly as an attention bias. Instead, the text-induced graph enhances the time-series branch, the prompt branch, and the aligned representation through multi-stage propagation.

### 3.9 Prediction Objective and Evaluation Metrics

The final prediction is obtained by a decoder:

\[
\hat{y}^{(H)}_t = f_{\text{dec}}(\bar{Z}_t).
\]

The model is optimized using Smooth L1 loss:

\[
\mathcal{L}_{\text{reg}} = \frac{1}{BN} \sum_{b=1}^{B} \sum_{i=1}^{N} \text{SmoothL1}(\hat{y}_{b,i}, \tilde{y}_{b,i}).
\]

Although the task is trained as regression, evaluation emphasizes finance-oriented ranking metrics. We report MSE, MAE, IC, RankIC, ICIR, and RankICIR. IC and RankIC measure cross-sectional predictive quality, while ICIR and RankICIR further characterize stability across rolling windows.

## 4. Experiments

### 4.1 Dataset and Preprocessing

We evaluate Graph-TimeCMA on HS300 multi-stock prediction. Market data are daily-frequency A-share quotes covering the period from January 1, 2024 to September 30, 2025. We use a long-form panel with stock-level daily observations and derive nine features for each stock: open, high, low, close, volume, amount, daily return, log return, and amplitude.

Financial text is collected from Snowball social media. After cleaning, time normalization, stock-code extraction, and duplicate removal, we use the resulting posts to construct stock co-occurrence graphs. The stock universe is defined by the intersection of the HS300 stock pool, the available market data, and the graph node set.

### 4.2 Rolling-Window Evaluation

We adopt a rolling-window setting to better reflect realistic forecasting conditions. Each window contains:

- 12 months for training,
- 3 months for validation,
- 1 month for testing,
- and a step size of 1 month.

The test months are April 2025 through September 2025, resulting in six rolling windows. For the static graph, we construct the co-occurrence graph from the training period of each window. For the dynamic graph, we use the six months immediately preceding the test month and apply exponential time decay with half-life 60 days.

### 4.3 Baselines

We compare Graph-TimeCMA against two groups of baselines.

**Conventional sequence models**

- `MLP`: a flattened multilayer perceptron over stock-level sequence features.
- `LSTM`: a standard recurrent sequence encoder.
- `ALSTM`: an attentive LSTM with attention pooling over hidden states.
- `Transformer`: a stock-level transformer encoder over temporal tokens.
- `XGBoost`: a gradient boosting regressor trained on flattened stock-sequence features.

**TimeCMA family**

- `no_graph`: multi-stock TimeCMA without any stock relation graph.
- `static`: Graph-TimeCMA with a training-window static co-occurrence graph.
- `dynamic6m`: Graph-TimeCMA with a six-month dynamic co-occurrence graph and time decay.

### 4.4 Implementation Details

We use a look-back window \(L = 60\) and target horizon \(H = 5\). Prompt embeddings are generated with a frozen GPT-2 model and cached offline. The hidden dimension is set to 128, the batch size is 16, the learning rate is \(10^{-4}\), and the optimizer is AdamW. The graph propagation depth is 1. For graph preprocessing, the default static and dynamic graph settings apply `log1p` compression and keep top-10 neighbors per node. We use 1%-99% training-label winsorization unless otherwise specified in ablations.

### 4.5 Evaluation Metrics

We report:

- `MSE` and `MAE` for point-wise regression error,
- `IC` and `RankIC` for cross-sectional predictive correlation,
- `ICIR` and `RankICIR` for cross-window stability.

Since the task is multi-stock cross-sectional prediction rather than single-series forecasting, we consider the correlation-based metrics especially important.

## 5. Main Results

### 5.1 Comparison with Conventional Baselines

Table 1 reports the average results over six rolling windows.

| Method | MSE ↓ | MAE ↓ | IC ↑ | RankIC ↑ | ICIR ↑ | RankICIR ↑ |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 0.002514 | 0.033878 | -0.0410 | -0.0410 | -0.4259 | -0.4138 |
| LSTM | 0.002270 | 0.031271 | 0.0282 | 0.0191 | 0.0612 | 0.0743 |
| ALSTM | 0.002259 | 0.030624 | -0.0699 | -0.0330 | -0.6437 | -0.2227 |
| Transformer | 0.002517 | 0.033423 | -0.0383 | -0.0248 | -0.4964 | -0.2021 |
| XGBoost | 0.002305 | 0.031593 | -0.0336 | -0.0238 | -0.5025 | -0.3226 |
| TimeCMA w/o Graph | 0.002888 | 0.037947 | 0.0337 | 0.0294 | 0.3521 | 0.3179 |
| Graph-TimeCMA (Static) | **0.002835** | **0.037659** | 0.0355 | 0.0332 | 0.4206 | 0.3916 |
| Graph-TimeCMA (Dynamic6m) | 0.002845 | 0.038213 | **0.0497** | **0.0506** | **0.5072** | **0.4771** |

Several observations can be made.

First, conventional baselines often achieve lower MSE and MAE than the TimeCMA family. This indicates that they are effective point-wise regressors and can produce predictions numerically close to the realized returns.

Second, however, the ranking-oriented metrics tell a different story. The proposed Graph-TimeCMA variants substantially outperform conventional baselines in IC, RankIC, ICIR, and RankICIR. The dynamic graph variant achieves the best average IC (0.0497), RankIC (0.0506), ICIR (0.5072), and RankICIR (0.4771). Even the graph-free TimeCMA baseline outperforms most conventional baselines in these cross-sectional metrics.

Third, this discrepancy suggests that lower regression error does not necessarily imply better cross-sectional prediction quality. Several conventional models appear to produce conservative predictions with smaller numerical deviations, but they fail to preserve the relative ordering of stocks within a day. In contrast, Graph-TimeCMA better captures the relative return structure across stocks, which is more relevant for cross-sectional stock prediction.

### 5.2 Comparison within the TimeCMA Family

Comparing the three TimeCMA-family models further clarifies the benefit of text-induced graphs.

Relative to `no_graph`, the static graph improves IC from 0.0337 to 0.0355, RankIC from 0.0294 to 0.0332, ICIR from 0.3521 to 0.4206, and RankICIR from 0.3179 to 0.3916. The dynamic graph improves these metrics further to 0.0497, 0.0506, 0.5072, and 0.4771, respectively.

These results indicate that the graph prior is useful, and that dynamically updated text-induced relations are more informative than a single static graph when the goal is to model changing cross-stock dependencies.

### 5.3 Per-Window Behavior

The per-window results reveal that conventional baselines can occasionally perform well in individual months. For example, the transformer baseline achieves the best RankIC in April 2025, ALSTM performs strongly in May 2025, and LSTM is strong in August and September 2025. However, these gains are unstable: the same models also produce strongly negative IC or RankIC in other months.

In contrast, the dynamic graph variant remains positive on all six windows for both IC and RankIC. This consistency is the main reason why it achieves the best average ICIR and RankICIR. Thus, the advantage of Graph-TimeCMA is not simply a single-window spike, but more stable cross-sectional performance across market regimes.

## 6. Ablation Study

We conduct ablation experiments to analyze the role of text-induced graphs, temporal decay, graph preprocessing, and winsorization.

### 6.1 Ablation Settings

We consider the following variants:

- `no_graph`: removes the stock graph entirely.
- `static`: uses the default static graph with `log1p + top-k`.
- `dynamic6m`: uses the six-month dynamic graph with exponential time decay.
- `dynamic6m_no_decay`: removes the temporal decay from the dynamic graph.
- `static_no_graph_preproc`: removes `log1p` compression and top-\(k\) filtering.
- `static_no_winsorize`: removes training-label winsorization.

### 6.2 Ablation Results

| Method | MSE ↓ | MAE ↓ | IC ↑ | RankIC ↑ | ICIR ↑ | RankICIR ↑ |
|---|---:|---:|---:|---:|---:|---:|
| no_graph | 0.002888 | 0.037947 | 0.0337 | 0.0294 | 0.3521 | 0.3179 |
| static | 0.002835 | 0.037659 | 0.0355 | 0.0332 | 0.4206 | **0.3916** |
| dynamic6m | 0.002845 | 0.038213 | **0.0497** | **0.0506** | **0.5072** | **0.4771** |
| dynamic6m_no_decay | 0.002839 | 0.038069 | 0.0481 | 0.0469 | 0.4902 | 0.4536 |
| static_no_graph_preproc | **0.002770** | **0.037281** | 0.0405 | 0.0351 | 0.4524 | 0.3853 |
| static_no_winsorize | 0.002865 | 0.037845 | 0.0315 | 0.0290 | 0.3819 | 0.3403 |

### 6.3 Effect of the Graph

Comparing `static` with `no_graph` confirms that introducing a text-induced stock graph improves all correlation and stability metrics. This validates the central hypothesis that stock relations mined from financial text provide useful structural priors beyond pure temporal forecasting.

### 6.4 Effect of Temporal Decay

Comparing `dynamic6m` with `dynamic6m_no_decay` shows that temporal decay is beneficial. The dynamic graph with decay achieves higher IC, RankIC, ICIR, and RankICIR than the version without decay. This suggests that recent co-mentions are more informative than older ones, and that time-sensitive relation weighting is useful in dynamic market environments.

### 6.5 Effect of Winsorization

Comparing `static` with `static_no_winsorize` shows that winsorization improves overall robustness. Without winsorization, both correlation metrics and stability metrics drop. This indicates that moderate clipping of extreme training returns helps the model avoid overreacting to heavy-tailed labels.

### 6.6 Effect of Graph Preprocessing

An interesting finding is that `static_no_graph_preproc` achieves the best MSE and MAE, and even slightly improves IC and RankIC relative to `static`. This suggests that, under the current sparse Snowball co-occurrence graph, aggressive `log1p + top-k` processing may remove some useful weak edges. In other words, graph preprocessing is not universally beneficial; its effectiveness depends on the sparsity and edge-weight distribution of the graph.

## 7. Discussion

### 7.1 Why Graph-TimeCMA Improves Cross-Sectional Metrics

The key advantage of Graph-TimeCMA is not simply stronger point-wise regression. Instead, it improves the quality of relative return modeling across stocks. The prompt branch provides stock-level semantic cues derived from historical trajectories, while the graph branch injects text-induced relation priors reflecting co-mentioned themes, events, or investor attention. The cross-modal alignment mechanism then fuses temporal and semantic representations under relational regularization. This combination appears particularly useful for predicting which stocks are relatively stronger or weaker on a given day.

### 7.2 Why Conventional Baselines Can Have Lower MSE

The conventional baselines often achieve lower MSE and MAE because they can learn conservative regressors that predict values closer to the cross-sectional mean. Such predictions reduce average numerical error, but may fail to capture the relative ranking among stocks. This is why lower regression error can coexist with poor IC and RankIC. For cross-sectional equity prediction, the ranking-oriented metrics are therefore indispensable.

### 7.3 Limitations

This work still has several limitations. First, the graph is constructed from co-occurrence statistics rather than deeper semantic relations, so it may miss more subtle interactions such as causal dependence or event-role structure. Second, the prompt embeddings are generated from historical market sequences rather than directly from raw financial news, which means that text enters the framework primarily through relation mining rather than full document understanding. Third, the experiments focus on HS300 and one market period; broader validation across markets and longer horizons remains future work.

## 8. Conclusion

This paper presented Graph-TimeCMA, a text-induced graph-enhanced cross-modal framework for multi-stock return prediction. By extending TimeCMA with stock co-occurrence graphs mined from financial social media, the proposed method uses financial text as both semantic evidence and structural relation evidence. Experiments on HS300 rolling-window prediction show that the dynamic graph variant achieves the best average IC, RankIC, ICIR, and RankICIR among all compared methods, demonstrating the value of text-induced stock relations for cross-sectional stock prediction. Ablation results further confirm the benefits of graph priors, temporal decay, and label winsorization. Overall, the results suggest that cross-modal alignment and text-induced relation modeling are a promising direction for multimodal stock prediction.

## References

Use your existing BibTeX file and current reference list from the manuscript. The citations already used in the current PDF are sufficient for this draft and can be retained when you move this text into your `.tex` source.

