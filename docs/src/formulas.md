# Formulas

Suppose we are given observations $Y_1, ..., Y_T$, with hidden states $X_1, ..., X_T$.
Following [Rabiner1989](@cite), we use the following notations:

* let $\pi \in \mathbb{R}^N$ be the initial state distribution $\pi_i = \mathbb{P}(X_1 = i)$
* let $A_t \in \mathbb{R}^{N \times N}$ be the transition matrix $a_{i,j,t} = \mathbb{P}(X_{t+1}=j | X_t = i)$
* let $B \in \mathbb{R}^{N \times T}$ be the matrix of statewise observation likelihoods $b_{i,t} = \mathbb{P}(Y_t | X_t = i)$

The conditioning on the known controls $U_{1:T}$ is implicit throughout.

## Vanilla forward-backward

### Recursion

The forward and backward variables are defined by

```math
\begin{align*}
\alpha_{i,t} & = \mathbb{P}(Y_{1:t}, X_t=i) \\
\beta_{i,t} & = \mathbb{P}(Y_{t+1:T} | X_t=i)
\end{align*}
```

They are initialized with

```math
\begin{align*}
\alpha_{i,1} & = \pi_i b_{i,1} \\
\beta_{i,T} & = 1
\end{align*}
```

and satisfy the dynamic programming equations

```math
\begin{align*}
\alpha_{j,t+1} & = \left(\sum_{i=1}^N \alpha_{i,t} a_{i,j,t}\right) b_{j,t+1} \\
\beta_{i,t} & = \sum_{j=1}^N a_{i,j,t} b_{j,t+1} \beta_{j,t+1}
\end{align*}
```

### Likelihood

The likelihood of the whole sequence of observations is given by

```math
\mathcal{L} = \mathbb{P}(Y_{1:T}) = \sum_{i=1}^N \alpha_{i,T}
```

### Marginals

We notice that

```math
\begin{align*}
\alpha_{i,t} \beta_{i,t} & = \mathbb{P}(Y_{1:T}, X_t=i) \\
\alpha_{i,t} a_{i,j,t} b_{j,t+1} \beta_{j,t+1} & = \mathbb{P}(Y_{1:T}, X_t=i, X_{t+1}=j)
\end{align*}
```

Thus we deduce the one-state and two-state marginals

```math
\begin{align*}
\gamma_{i,t} & = \mathbb{P}(X_t=i | Y_{1:T}) = \frac{1}{\mathcal{L}} \alpha_{i,t} \beta_{i,t} \\
\xi_{i,j,t} & = \mathbb{P}(X_t=i, X_{t+1}=j | Y_{1:T}) = \frac{1}{\mathcal{L}} \alpha_{i,t} a_{i,j,t} b_{j,t+1} \beta_{j,t+1}
\end{align*}
```

### Derivatives

According to [Qin2000](@cite), derivatives of the likelihood can be obtained as follows:

```math
\begin{align*}
\frac{\partial \mathcal{L}}{\partial \pi_i} &= \beta_{i,1} b_{i,1} \\
\frac{\partial \mathcal{L}}{\partial a_{i,j}} &= \sum_{t=1}^{T-1} \alpha_{i,t} b_{j,t+1} \beta_{j,t+1} \\
\frac{\partial \mathcal{L}}{\partial b_{j,1}} &= \pi_j \beta_{j,1} \\
\frac{\partial \mathcal{L}}{\partial b_{j,t}} &= \left(\sum_{i=1}^N \alpha_{i,t-1} a_{i,j,t-1}\right) \beta_{j,t} 
\end{align*}
```

## Scaled forward-backward

In this package, we use a slightly different version of the algorithm, including both the traditional scaling of [Rabiner1989](@cite) and a normalization of $B$ using $m_t = \max_i b_{i,t}$.

### Recursion

The variables are initialized with

```math
\begin{align*}
\hat{\alpha}_{i,1} & = \pi_i \frac{b_{i,1}}{m_1} & c_1 & = \frac{1}{\sum_i \hat{\alpha}_{i,1}} & \bar{\alpha}_{i,1} & = c_1 \hat{\alpha}_{i,1} \\
\hat{\beta}_{i,T} & = 1 & && \bar{\beta}_{1,T} &= c_T \hat{\beta}_{1,T}
\end{align*}
```

and satisfy the dynamic programming equations

```math
\begin{align*}
\hat{\alpha}_{j,t+1} & = \left(\sum_{i=1}^N \bar{\alpha}_{i,t} a_{i,j,t}\right) \frac{b_{j,t+1}}{m_{t+1}} & c_{t+1} & = \frac{1}{\sum_j \hat{\alpha}_{j,t+1}} & \bar{\alpha}_{j,t+1} = c_{t+1} \hat{\alpha}_{j,t+1} \\
\hat{\beta}_{i,t} & = \sum_{j=1}^N a_{i,j,t} \frac{b_{j,t+1}}{m_{t+1}} \bar{\beta}_{j,t+1} & && \bar{\beta}_{j,t} = c_t \hat{\beta}_{j,t}
\end{align*}
```

In terms of the original variables, we find

```math
\begin{align*}
\bar{\alpha}_{i,t} &= \alpha_{i,t} \left(\prod_{s=1}^t \frac{c_s}{m_s}\right) \\
\bar{\beta}_{i,t} &= \beta_{i,t} \left(c_t \prod_{s=t+1}^T \frac{c_s}{m_s}\right)
\end{align*}
```

### Likelihood

Since we normalized $\bar{\alpha}$ at each time step,

```math
1 = \sum_{i=1}^N \bar{\alpha}_{i,T} = \left(\sum_{i=1}^N \alpha_{i,T}\right) \left(\prod_{s=1}^T \frac{c_s}{m_s}\right) 
```

which means

```math
\mathcal{L} = \sum_{i=1}^N \alpha_{i,T} = \prod_{s=1}^T \frac{m_s}{c_s}
```

### Marginals

We can now express the marginals using scaled variables:

```math
\begin{align*}
\gamma_{i,t} & = \frac{1}{\mathcal{L}} \alpha_{i,t} \beta_{i,t} = \frac{1}{\mathcal{L}} \left(\bar{\alpha}_{i,t} \prod_{s=1}^t \frac{m_s}{c_s}\right) \left(\bar{\beta}_{i,t} \frac{1}{c_t} \prod_{s=t+1}^T \frac{m_s}{c_s}\right) \\
&= \frac{1}{\mathcal{L}} \frac{\bar{\alpha}_{i,t} \bar{\beta}_{i,t}}{c_t} \left(\prod_{s=1}^T \frac{m_s}{c_s}\right) = \frac{\bar{\alpha}_{i,t} \bar{\beta}_{i,t}}{c_t}
\end{align*}
```

```math
\begin{align*}
\xi_{i,j,t} & = \frac{1}{\mathcal{L}} \alpha_{i,t} a_{i,j} b_{j,t+1} \beta_{j,t+1} \\
&= \frac{1}{\mathcal{L}}  \left(\bar{\alpha}_{i,t} \prod_{s=1}^t \frac{m_s}{c_s}\right) a_{i,j,t} b_{j,t+1} \left(\bar{\beta}_{j,t+1} \frac{1}{c_{t+1}} \prod_{s=t+2}^T \frac{m_s}{c_s}\right) \\
&= \frac{1}{\mathcal{L}}  \bar{\alpha}_{i,t} a_{i,j,t} \frac{b_{j,t+1}}{m_{t+1}} \bar{\beta}_{j,t+1} \left(\prod_{s=1}^T \frac{m_s}{c_s}\right) \\
&= \bar{\alpha}_{i,t} a_{i,j,t} \frac{b_{j,t+1}}{m_{t+1}} \bar{\beta}_{j,t+1}
\end{align*}
```

### Derivatives

And we also need to adapt the derivatives.
For the initial distribution,

```math
\begin{align*}
\frac{\partial \mathcal{L}}{\partial \pi_i} &= \beta_{i,1} b_{i,1} = \left(\bar{\beta}_{i,1} \frac{1}{c_1} \prod_{s=2}^T \frac{m_s}{c_s} \right) b_{i,1} \\
&= \left(\prod_{s=1}^T \frac{m_s}{c_s}\right) \bar{\beta}_{i,1} \frac{b_{i,1}}{m_1}  = \mathcal{L} \bar{\beta}_{i,1} \frac{b_{i,1}}{m_1} 
\end{align*}
```

For the transition matrix,

```math
\begin{align*}
\frac{\partial \mathcal{L}}{\partial a_{i,j}} &= \sum_{t=1}^{T-1} \alpha_{i,t} b_{j,t+1} \beta_{j,t+1} \\
&= \sum_{t=1}^{T-1} \left(\bar{\alpha}_{i,t} \prod_{s=1}^t \frac{m_s}{c_s} \right) b_{j,t+1} \left(\bar{\beta}_{j,t+1} \frac{1}{c_{t+1}} \prod_{s=t+2}^T \frac{m_s}{c_s} \right) \\
&= \sum_{t=1}^{T-1} \bar{\alpha}_{i,t} \frac{b_{j,t+1}}{m_{t+1}} \bar{\beta}_{j,t+1} \left(\prod_{s=1}^T \frac{m_s}{c_s} \right) \\
&= \mathcal{L} \sum_{t=1}^{T-1} \bar{\alpha}_{i,t} \frac{b_{j,t+1}}{m_{t+1}} \bar{\beta}_{j,t+1} \\
\end{align*}
```

And for the statewise observation likelihoods,

```math
\begin{align*}
\frac{\partial \mathcal{L}}{\partial b_{j,1}} &= \pi_j \beta_{j,1} = \pi_j \bar{\beta}_{j,1} \frac{1}{c_1} \prod_{s=2}^T \frac{m_s}{c_s} = \mathcal{L} \pi_j \bar{\beta}_{j,1} \frac{1}{m_1}
\end{align*}
```

```math
\begin{align*}
\frac{\partial \mathcal{L}}{\partial b_{j,t}} &= \left(\sum_{i=1}^N \alpha_{i,t-1} a_{i,j,t-1}\right) \beta_{j,t} \\
&= \sum_{i=1}^N \left(\bar{\alpha}_{i,t-1} \prod_{s=1}^{t-1} \frac{m_s}{c_s}\right) a_{i,j,t-1} \left(\bar{\beta}_{j,t} \frac{1}{c_t} \prod_{s=t+1}^T \frac{m_s}{c_s} \right) \\
&= \sum_{i=1}^N \bar{\alpha}_{i,t-1} a_{i,j,t-1} \bar{\beta}_{j,t} \frac{1}{m_t} \left(\prod_{s=1}^T \frac{m_s}{c_s}\right) \\
&= \mathcal{L} \sum_{i=1}^N \bar{\alpha}_{i,t-1} a_{i,j,t-1} \bar{\beta}_{j,t} \frac{1}{m_t} \\
\end{align*}
```

Finally, we note that

```math
\frac{\partial \log \mathcal{L}}{\partial \log b_{j,t}} = \frac{\partial \log \mathcal{L}}{\partial b_{j,t}} b_{j,t}
```

To sum up,

```math
\begin{align*}
\frac{\partial \log \mathcal{L}}{\partial \pi_i} &= \frac{b_{i,1}}{m_1} \bar{\beta}_{i,1} \\
\frac{\partial \log \mathcal{L}}{\partial a_{i,j}} &= \sum_{t=1}^{T-1} \bar{\alpha}_{i,t} \frac{b_{j,t+1}}{m_{t+1}} \bar{\beta}_{j,t+1} \\
\frac{\partial \log \mathcal{L}}{\partial \log b_{j,1}} &= \pi_j \frac{b_{j,1}}{m_1} \bar{\beta}_{j,1} = \frac{\bar{\alpha}_{j,1} \bar{\beta}_{j,1}}{c_1} = \gamma_{j,1} \\
\frac{\partial \log \mathcal{L}}{\partial \log b_{j,t}} &= \sum_{i=1}^N \bar{\alpha}_{i,t-1} a_{i,j,t-1} \frac{b_{j,t}}{m_t} \bar{\beta}_{j,t} = \frac{\bar{\alpha}_{j,t} \bar{\beta}_{j,t}}{c_t} = \gamma_{j,t}
\end{align*}
```

## Hidden semi-Markov models

A Hidden Semi-Markov Model (HSMM) generalizes an HMM by replacing the implicit
geometric sojourn-time distribution with an explicit per-state distribution over
durations. Concretely, a state $j$ is entered, a duration $d \sim p_d(\cdot \mid j)$
is drawn, the chain emits observations from state $j$ for $d$ consecutive timesteps,
and only then transitions to a different state (self-transitions are forbidden).

We use the same notation as above for $\pi$, $A$, and $B$, plus

* let $p_d(d \mid j)$ be the duration distribution for state $j$ on the positive
  integers $\{1, 2, 3, \ldots\}$.

The transition matrix is constrained to have zero diagonal ($a_{j,j} = 0$).

### Segment-based forward variable

Because the chain is governed by sojourns rather than individual transitions, the
forward variable is defined at *segment boundaries* rather than per-timestep:

```math
\alpha^{\text{HSMM}}_{j,t} = \mathbb{P}(Y_{1:t},\; \text{a segment ends at } t \text{ in state } j)
```

The dynamic-programming recursion sums over the duration $d$ of the segment that
ends at $t$ and the state $i$ that the chain was in just before it:

```math
\alpha^{\text{HSMM}}_{j,t} = \sum_{d=1}^{\min(D, t-1)} \left( \sum_{i \neq j} \alpha^{\text{HSMM}}_{i,t-d} \, a_{i,j} \right) p_d(d \mid j) \prod_{\tau=t-d+1}^{t} b_{j,\tau}
```

with the special-case "initial segment" contribution

```math
\alpha^{\text{HSMM}}_{j,t} \mathrel{+}= \pi_j \, p_d(t \mid j) \prod_{\tau=1}^{t} b_{j,\tau} \qquad (1 \le t \le D)
```

Here $D$ is the maximum-duration cap (the `max_duration` keyword in code) — the
recursion is $O(N^2 T D)$ as a result. The product $\prod_{\tau} b_{j,\tau}$ is
implemented in log-space using cumulative per-state observation log-likelihoods so
that the segment observation factor at $(j, t_\text{start}, t_\text{end})$ is one
subtraction rather than $d$ multiplications.

### Likelihood

By construction every observation sequence must end with a segment boundary at $T$, so

```math
\mathcal{L} = \mathbb{P}(Y_{1:T}) = \sum_{j=1}^{N} \alpha^{\text{HSMM}}_{j,T}
```

### Backward variable, marginals, and duration counts

Define $\beta^{\text{HSMM}}_{j,t} = \mathbb{P}(Y_{t+1:T} \mid \text{a segment ends at } t \text{ in state } j)$,
with boundary $\beta^{\text{HSMM}}_{j,T} = 1$. The recursion is

```math
\beta^{\text{HSMM}}_{i,t} = \sum_{j \neq i} a_{i,j} \sum_{d=1}^{\min(D, T-t)} p_d(d \mid j) \left( \prod_{\tau=t+1}^{t+d} b_{j,\tau} \right) \beta^{\text{HSMM}}_{j,t+d}
```

The HSMM produces three posterior aggregates the EM (Baum–Welch) step needs:

```math
\begin{align*}
\gamma_{j,t} & = \mathbb{P}(X_t = j \mid Y_{1:T}) \\
\xi^{\text{HSMM}}_{i,j,t} & = \mathbb{P}(\text{segment ends at } t \text{ in state } i,\; \text{next segment starts at } t+1 \text{ in state } j \mid Y_{1:T}) \\
\eta_{d,j} & = \mathbb{E}\!\left[\, \#\{\text{segments of state } j \text{ with duration } d\} \,\middle|\, Y_{1:T} \right]
\end{align*}
```

`γ` is the smoothed per-timestep state marginal, `ξ` is the segment-to-segment
transition marginal, and `η` is the expected per-state duration count. They are
computed by accumulating contributions from each candidate segment $(j, t_\text{start}, d)$,
weighted by the corresponding prefix (initial: $\pi_j$; internal: $\sum_{i \neq j} \alpha^{\text{HSMM}}_{i,t_\text{start}-1} a_{i,j}$), the duration log-pmf, the segment
observation factor, and the backward value at the segment's end.

### Baum–Welch updates

The M-step formulas for initial, transition, and observation parameters look familiar:

```math
\pi_j^{\text{new}} = \sum_{k=1}^{K} \gamma^{(k)}_{j,1}, \qquad a_{i,j}^{\text{new}} = \frac{\sum_{k,t} \xi^{\text{HSMM},(k)}_{i,j,t}}{\sum_{k,t,j'} \xi^{\text{HSMM},(k)}_{i,j',t}}
```

with $a_{j,j}^{\text{new}} = 0$ enforced structurally. Observation distributions
are refit using $\gamma$ as per-timestep weights, as in the HMM case. The
duration distributions are refit using `η` as weights on the duration support:

```math
p_d(\cdot \mid j)^{\text{new}} = \arg\max_{p_d} \sum_{d=1}^{D} \eta_{d,j} \log p_d(d \mid j)
```

For [`GeometricDuration`](@ref) and [`PoissonDuration`](@ref) this reduces to a
weighted-mean estimator; for [`NegBinomialDuration`](@ref) we profile out $p$ and
do a 1-D Newton iteration on $r$ driven by the digamma score equation.

### Derivatives of the log-likelihood

Mirroring Qin (2000) but in segment form,

```math
\begin{align*}
\frac{\partial \log \mathcal{L}}{\partial \pi_j} & = \frac{\gamma_{j,1}}{\pi_j} \\
\frac{\partial \log \mathcal{L}}{\partial a_{i,j,t}} & = \frac{\xi^{\text{HSMM}}_{i,j,t}}{a_{i,j,t}} \\
\frac{\partial \log \mathcal{L}}{\partial \log b_{j,t}} & = \gamma_{j,t} \\
\frac{\partial \log \mathcal{L}}{\partial \log p_d(d \mid j)} & = \eta_{d,j}
\end{align*}
```

The reverse-mode `ChainRules` rrule shipped with the package uses the first three
of these to backpropagate through `logdensityof` to `init`, `trans`, and `dists`.
The last is omitted because Zygote currently mishandles pullbacks through vectors
of mutable structs (our duration distributions are mutable so `fit!` can update
them in place during Baum–Welch). To get gradients w.r.t. duration parameters,
use `ForwardDiff` — it has no such limitation.

## Bibliography

```@bibliography
```
