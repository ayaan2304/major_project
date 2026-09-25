# Mathematical formulation

Every symbol below is implemented in `src/` with the same meaning. Plain English follows each definition.

## 1. Class imbalance

Let \(n\) be the number of transactions and \(n_1\) the number of frauds (`isFraud = 1`).

\[
\pi = \frac{n_1}{n}, \qquad \rho = \frac{n-n_1}{n_1}.
\]

**English:** \(\pi\) is the fraud rate; \(\rho\) is how many legitimate rows exist per fraud row.

## 2. Temporal split

Let \(s_i\) be the PaySim `step` (hour index). With fractions \(f_{tr}, f_{va}, f_{te}\) that sum to 1 and \(s_{\min}=\min_i s_i\), \(s_{\max}=\max_i s_i\):

\[
\tau_{tr} = s_{\min} + (s_{\max}-s_{\min}) f_{tr}, \qquad
\tau_{va} = s_{\min} + (s_{\max}-s_{\min})(f_{tr}+f_{va}).
\]

Train: \(s_i \le \tau_{tr}\); validation: \(\tau_{tr} < s_i \le \tau_{va}\); test: \(s_i > \tau_{va}\).

**English:** earlier hours train the model, the next hours tune thresholds, the latest hours are the only test.

## 3. Behavioral features (examples)

With \(\varepsilon>0\) from config:

\[
\log(a_i+\varepsilon),\quad
\frac{a_i}{b^{old}_{orig,i}+\varepsilon},\quad
e^{orig}_i = b^{old}_{orig,i} - a_i - b^{new}_{orig,i}.
\]

Causal history for origin \(u\), using only earlier rows in time order:

\[
c_{i}(u)=\#\{j: u_j=u,\ j \prec i\}, \qquad
\Delta t_i(u)= s_i - s_{\mathrm{prev}(i,u)}.
\]

**English:** ratios and accounting errors use the current row only. Counts and “time since last transfer” use the past of that account, never the future.

## 4. Synthetic generation (SMOTE)

For a minority point \(x\) and a random minority neighbor \(x_{nn}\):

\[
x^\* = x + \lambda (x_{nn}-x), \quad \lambda \sim U[0,1].
\]

**English:** the new fraud is a point on the line between two real frauds.

## 5. SHAP prototype

Let \(\phi(x)\in\mathbb{R}^d\) be the TreeSHAP vector of a provisional model for the fraud class. For genuine train-fraud set \(F\):

\[
\bar\phi_{\mathrm{mean}}=\frac{1}{|F|}\sum_{x\in F}\phi(x),\quad
\bar\phi_{\mathrm{med}}=\mathrm{median}\{\phi(x)\},\quad
\bar\phi_{\mathrm{trim}}=\mathrm{trim\_mean}(\{\phi(x)\},\alpha).
\]

The strategy with highest mean cosine similarity on a **train-fraud holdout** is selected, then recomputed on all train fraud.

**English:** we summarise “how the model typically explains fraud” with one vector, and we choose the summary that best matches held-out real fraud.

## 6. Explanation similarity

\[
\mathrm{cos}(u,v)=\frac{u^\top v}{\|u\|\,\|v\|},\qquad
m(\phi,\bar\phi)=1-\frac{\big|\|\phi\|-\|\bar\phi\|\big|}{\|\phi\|+\|\bar\phi\|+\varepsilon}.
\]

\[
\mathrm{sim}(\phi,\bar\phi)= w_c \,[ \mathrm{cos}(\phi,\bar\phi)]_+ + w_m\, m(\phi,\bar\phi).
\]

**English:** cosine checks that the *pattern* of important features matches; the magnitude term checks that the *strength* of the explanation is not tiny or huge compared with real fraud.

## 7. Plausibility

With train-fraud mean \(\mu\) and (shrinked) covariance \(\Sigma\):

\[
d_M(x)=\sqrt{(x-\mu)^\top \Sigma^{+} (x-\mu)}.
\]

**English:** a synthetic is implausible if it sits far from the real-fraud cloud, even if SHAP looks aligned.

## 8. Filtering rule

\[
\mathrm{keep}(x^\*) \iff \mathrm{sim}(\phi(x^\*),\bar\phi)\ge t_{\mathrm{sim}}
\ \wedge\  d(x^\*)\le t_{\mathrm{plaus}}.
\]

Percentiles of genuine train-fraud scores define a grid; validation PR-AUC picks \((t_{\mathrm{sim}}, t_{\mathrm{plaus}})\). Test is unused.

## 9. Cost and threshold

\[
C(t)= C_{FN}\, FN(t)+ C_{FP}\, FP(t),\qquad
t^\*=\arg\min_t C(t)\ \text{on validation}.
\]

**English:** we choose the alert cutoff that minimises a weighted miss/false-alarm bill on recent labelled hours, then freeze it.

## 10. Fidelity

For instance \(i\) and \(k\in\{1,3,5\}\), \(\Delta^{top}_{i,k}\) is the drop in fraud probability after zeroing the \(k\) largest \(|\mathrm{SHAP}|\) features; \(\Delta^{rnd}\) uses \(k\) random features.

**English:** if explanations are faithful, knocking out the features SHAP called important should change the score more than knocking out random ones.

## 11. Stability

For runs \(r=1..R\), let \(I^{(r)}\) be mean \(|\mathrm{SHAP}|\) over a frozen sample. Report mean/std of Spearman and Kendall correlations between ranks, and Jaccard overlap of top-\(k\) feature sets.

## 12. Temporal drift

Partition the test steps into \(W\) quantile windows. Report per-window PR-AUC/F1 and Jensen–Shannon divergence between consecutive windows’ predicted-score histograms.

\[
\mathrm{JS}(P,Q)=\tfrac12 \mathrm{KL}(P\|M)+\tfrac12 \mathrm{KL}(Q\|M),\quad M=\tfrac12(P+Q).
\]
