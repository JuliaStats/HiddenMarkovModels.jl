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

## Explicit-duration HSMM forward pass

In a hidden semi-Markov model, the sojourn time in each state is given by an explicit duration distribution instead of an implicit geometric one.
Following [Yu2010](@cite) (see also [Johnson2014](@cite)), we add the following notations:

* the transition matrix has zero diagonal, and $a_{i,j,t}$ now denotes $\mathbb{P}(X_{t+1}=j | X_t=i, \text{the sojourn in } i \text{ ends at } t)$
* let $p_i(d) = \mathbb{P}(D_i = d)$ be the sojourn duration mass function of state $i$, supported on $d \in \{1, 2, \dots\}$
* let $S_i(d) = \mathbb{P}(D_i \geq d)$ be the associated survival function
* let $D$ be a truncation level, encoded by the convention $p_i(d) = S_i(d) = 0$ for $d > D$

A segment entered at time $s$ uses the transition probabilities and sojourn distributions selected by the control $U_s$, i.e., the one in effect at its first timestep.

Observations, on the other hand, use the control at the current timestep; $b_{j,u}$ depends on $U_u$.

### Recursion

The forward pass tracks two variables instead of one:

```math
\begin{align*}
E_{j,t} & = \mathbb{P}(Y_{1:t}, X_t=j, \text{the sojourn in } j \text{ ends at } t) \\
F_{j,t} & = \mathbb{P}(Y_{1:t}, X_t=j, \text{the sojourn in } j \text{ covers } t) = \alpha_{j,t}
\end{align*}
```

Only $E$ can be closed under transitions, since a jump requires a completed sojourn, but $F$ is the quantity of interest because the sojourn covering $t$ is right-censored.
Denoting by

```math
I_{j,t} = \sum_{i \neq j} E_{i,t} a_{i,j,t}
```

the mass entering state $j$ at time $t+1$, and decomposing over the starting time $s$ of the sojourn covering $t$, we get

```math
\begin{align*}
E_{j,t} & = \pi_j \left(\prod_{u=1}^{t} b_{j,u}\right) p_j(t) + \sum_{s=1}^{t-1} I_{j,s} \left(\prod_{u=s+1}^{t} b_{j,u}\right) p_j(t-s) \\
F_{j,t} & = \pi_j \left(\prod_{u=1}^{t} b_{j,u}\right) S_j(t) + \sum_{s=1}^{t-1} I_{j,s} \left(\prod_{u=s+1}^{t} b_{j,u}\right) S_j(t-s)
\end{align*}
```

where the first term accounts for the segment starting the sequence.
Unlike the vanilla forward pass, this recursion is carried out in the log domain instead of being scaled.
Each $F_{j,t}$ mixes segments with different starting times, and hence different products of observation likelihoods, so working with log-probabilities is simpler than tracking scaling factors across all of them.

### Likelihood and marginals

```math
\begin{align*}
\mathcal{L} & = \mathbb{P}(Y_{1:T}) = \sum_{j=1}^N F_{j,T} \\
\bar{\alpha}_{i,t} & = \mathbb{P}(X_t=i | Y_{1:t}) = \frac{F_{i,t}}{\sum_{j=1}^N F_{j,t}}
\end{align*}
```

Beware of the notation clash: the forward pass returns the *normalized* filtered marginals $\bar{\alpha}$, and not the unnormalized $\alpha$ of the vanilla section.

### Segment observation likelihoods

Evaluating $\prod_{u=s+1}^{t} b_{j,u}$ naively costs $O(d)$ per segment, so we precompute prefix sums instead.
Since $b_{j,u} = 0$ would turn their difference into $-\infty - (-\infty)$, zeros are left out of the sum and counted separately:

```math
\ell_{j,t} = \sum_{u \leq t ~:~ b_{j,u} > 0} \log b_{j,u} \qquad z_{j,t} = \#\{u \leq t : b_{j,u} = 0\}
```

which brings each segment down to $O(1)$:

```math
\log \prod_{u=s+1}^{t} b_{j,u} = \begin{cases} \ell_{j,t} - \ell_{j,s} & \text{if } z_{j,t} = z_{j,s} \\ -\infty & \text{otherwise} \end{cases}
```

### Truncation and complexity

By default, $D$ is the length of the longest sequence, the constraint $d \leq D$ is vacuous and the result is exact.
For $D < T$, every path containing a completed sojourn longer than $D$ is dropped, so $\mathcal{L}$ is underestimated and the marginals $\bar{\alpha}$ are biased.
Note that the convention $S_j(d) = 0$ for $d > D$ only stops the enumeration of longer segments: a segment running for $d \leq D$ timesteps still carries the full mass $\mathbb{P}(D_j \geq d)$ of all longer sojourns.

The cost is $O(N^2 T)$ for the transitions plus $O(N T D)$ for the segments, with $O(N T)$ memory.
The alternative is to augment the state with the remaining sojourn time and reuse the vanilla forward pass on $N D$ states, which shares more code but requires $O(N D T)$ memory.

## Bibliography

```@bibliography
```
