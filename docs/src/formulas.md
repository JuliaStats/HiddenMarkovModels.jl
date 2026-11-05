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
In particular, as in the vanilla case, $a_{i,j,t}$ is the transition matrix under $U_{t+1}$, since the jump from $t$ to $t+1$ enters a segment starting at $t+1$.

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

Evaluating $\prod_{u=s+1}^{t} b_{j,u}$ naively costs $O(d)$ per segment.
Since the segments starting at $s+1$ are enumerated by increasing end $t$, each product extends the previous one by a single factor, which brings each segment down to $O(1)$:

```math
\log \prod_{u=s+1}^{t+1} b_{j,u} = \log \prod_{u=s+1}^{t} b_{j,u} + \log b_{j,t+1}
```

Differences of prefix sums would also cost $O(1)$, but a single very small $b_{j,u}$ makes the prefix sums so large that the later terms are rounded away.
Once $b_{j,u} = 0$, every longer segment containing $u$ is impossible, so the enumeration stops there.

### Truncation and complexity

By default, $D$ is the length of the longest sequence, the constraint $d \leq D$ is vacuous and the result is exact.
For $D < T$, every path containing a sojourn longer than $D$ is dropped, including the right-censored final one,, so $\mathcal{L}$ is underestimated and the marginals $\bar{\alpha}$ are biased.
Note that the convention $S_j(d) = 0$ for $d > D$ only stops the enumeration of longer segments: a segment running for $d \leq D$ timesteps still carries the full mass $\mathbb{P}(D_j \geq d)$ of all longer sojourns.

The cost is $O(N^2 T)$ for the transitions plus $O(N T D)$ for the segments, with $O(N T)$ memory.
With the default $D$, this means $O(N T^2)$, so a smaller $D$ is advisable for long sequences.
Segments longer than the support of every sojourn distribution are skipped, so bounded supports reduce $D$ automatically.
The alternative is to augment the state with the remaining sojourn time and reuse the vanilla forward pass on $N D$ states, which shares more code but requires $O(N D T)$ memory.

## Explicit-duration HSMM forward-backward

### Recursion

The backward pass mirrors the forward pass with two variables, both conditioned on a segment boundary:

```math
\begin{align*}
G_{i,t} & = \mathbb{P}(Y_{t+1:T} | X_t=i, \text{the sojourn in } i \text{ ends at } t) \\
H_{j,s} & = \mathbb{P}(Y_{s:T} | X_s=j, \text{a sojourn in } j \text{ starts at } s)
\end{align*}
```

No observations remain after the end of the sequence, so $G_{i,T} = 1$.
Decomposing over the duration $d$ of the sojourn starting at $s$, and over the next state, we get

```math
\begin{align*}
H_{j,s} & = \sum_{d=1}^{T-s} \left(\prod_{u=s}^{s+d-1} b_{j,u}\right) p_j(d) \, G_{j,s+d-1} + \left(\prod_{u=s}^{T} b_{j,u}\right) S_j(T-s+1) \\
G_{i,t} & = \sum_{j \neq i} a_{i,j,t} H_{j,t+1}
\end{align*}
```

where the last term of $H$ accounts for the right-censored final segment.
As in the forward pass, $p_j$, $S_j$ and $a_{i,j,t}$ follow the control at the start of the segment they describe, and the durations are truncated at $D$.

### Marginals

With the convention $I_{j,0} = \pi_j$, the mass entering state $j$ at time $s$ is $I_{j,s-1}$.
The posterior probability of a sojourn in $j$ covering exactly $s, \dots, s+d-1$ is

```math
w_{j,s,d} = \frac{1}{\mathcal{L}} I_{j,s-1} \left(\prod_{u=s}^{s+d-1} b_{j,u}\right) \times \begin{cases} p_j(d) \, G_{j,s+d-1} & \text{if } s+d-1 < T \\ S_j(d) & \text{if } s+d-1 = T \end{cases}
```

The one-state marginals add up the sojourns covering $t$, and the two-state marginals are nonzero only for a segment boundary:

```math
\begin{align*}
\gamma_{j,t} & = \mathbb{P}(X_t=j | Y_{1:T}) = \sum_{s \leq t} \sum_{d \geq t-s+1} w_{j,s,d} \\
\xi_{i,j,t} & = \mathbb{P}(X_t=i, X_{t+1}=j | Y_{1:T}) = \frac{1}{\mathcal{L}} E_{i,t} a_{i,j,t} H_{j,t+1} \qquad (i \neq j)
\end{align*}
```

The inner sum in $\gamma$ is a tail sum over durations, which we accumulate from the longest duration down.
This costs $O(N T D)$ like the recursion itself, and unlike the alternative $\gamma_{j,t+1} = \gamma_{j,t} + \sum_{i \neq j} \xi_{i,j,t} - \sum_{k \neq j} \xi_{j,k,t}$ it never subtracts probabilities.
The identity still holds and serves as a consistency check.

Everything is computed during a single backward sweep over $s$, in the log domain: when the sweep reaches $s$, the variables $G_{\cdot, s:T}$ are already known, so $H_{\cdot, s}$, the weights $w_{\cdot,s,\cdot}$ and then $G_{\cdot, s-1}$ and $\xi_{\cdot,\cdot,s-1}$ follow.

### Expected duration counts

Refitting the sojourn distributions requires the expected number of sojourns of each duration [Yu2010](@cite), which we split according to whether the sojourn is completed or right-censored by the end of the sequence:

```math
\begin{align*}
\eta_{j,d} & = \sum_{s ~:~ s+d-1 < T} w_{j,s,d} \\
\zeta_{j,d} & = w_{j,T-d+1,d}
\end{align*}
```

Since each $w_{j,s,d}$ is computed exactly once during the backward sweep, both are accumulated on the fly with $O(N D)$ memory per sequence, instead of storing all the $O(N T D)$ weights.
With several sequences, the counts of each sequence use its own length $T$ and are summed.
With $D < T$, only the durations $d \leq D$ are counted, like in the recursion.
They satisfy the following identities, since every timestep belongs to exactly one sojourn, every completed sojourn is followed by a transition, and every sequence ends with exactly one censored sojourn:

```math
\sum_{d} d \, (\eta_{j,d} + \zeta_{j,d}) = \sum_{t=1}^T \gamma_{j,t} \qquad \sum_d \eta_{j,d} = \sum_{t=1}^{T-1} \sum_{k \neq j} \xi_{j,k,t} \qquad \sum_j \sum_d \zeta_{j,d} = 1
```

## Explicit-duration HSMM Viterbi

The Viterbi algorithm replaces the sums of the forward pass with maximizations.
Since consecutive sojourns are in different states, a sequence of segments determines the state sequence and vice versa, so maximizing over segmentations maximizes the joint likelihood $\mathbb{P}(X_{1:T}, Y_{1:T})$, in which the final segment is right-censored.

### Recursion

With the convention $I^*_{j,0} = \pi_j$, as in the forward-backward marginals, the best path whose sojourn in $j$ ends at $t$, and the best path entering $j$ at $t+1$, satisfy

```math
\begin{align*}
E^*_{j,t} & = \max_{1 \leq s \leq t} I^*_{j,s-1} \left(\prod_{u=s}^{t} b_{j,u}\right) \times \begin{cases} p_j(t-s+1) & \text{if } t < T \\ S_j(T-s+1) & \text{if } t = T \end{cases} \\
I^*_{j,t} & = \max_{i \neq j} E^*_{i,t} a_{i,j,t}
\end{align*}
```

As in the forward-backward pass, $p_j$, $S_j$ and $a_{i,j,t}$ follow the control at the start of the segment they describe.
The joint loglikelihood of the most likely state sequence is $\log \max_j E^*_{j,T}$.
To backtrack, we store the maximizing duration $\delta^*_{j,t} = t-s+1$ for each $E^*_{j,t}$, and the maximizing state $\psi^*_{j,t+1} = i$ for each $I^*_{j,t}$, indexed by the time $t+1$ at which the sojourn in $j$ starts.
Starting from $j = \arg\max_j E^*_{j,T}$ and $t = T$, the last segment covers $s, \dots, t$ with $s = t - \delta^*_{j,t} + 1$, and the previous one ends at $s-1$ in state $\psi^*_{j,s}$, until $s = 1$.
As in the forward pass, the recursion is carried out in the log domain, with the same segment observation likelihoods and complexity.

### Truncation

With $D < T$, the maximum over $s$ is restricted to $t - D + 1 \leq s \leq t$, so only state sequences whose sojourns all last at most $D$ timesteps are considered.
Unlike the forward pass, nothing is underestimated: the result is the most likely of these state sequences, together with its exact joint loglikelihood, since $S_j(d)$ is never truncated for $d \leq D$.

## Explicit-duration HSMM Baum-Welch

### Durations

Let $\eta_{j,d}$ and $\zeta_{j,d}$ be the expected numbers of sojourns in state $j$ that are completed after exactly $d$ steps, and right-censored by the end of the sequence after $d$ steps.

The sojourn distribution of state $j$, with parameters $\theta_j$, appears through

```math
Q_j(\theta_j) = \sum_d \eta_{j,d} \log p_j(d; \theta_j) + \sum_d \zeta_{j,d} \log S_j(d; \theta_j)
```

where the second term accounts for the sojourns that were still running at the end of their sequence, as in [Guedon2003](@cite) (see also [Yu2010](@cite)).
Leaving it out, or counting the censored sojourns as if they had ended, biases the estimated sojourns downwards, and breaks the monotonicity of the algorithm.
Each update below does not decrease $Q_j$, and is carried out by [`HiddenMarkovModels.fit_duration_in_sequence!`](@ref).

In the geometric case $p_j(d) = q_j (1-q_j)^{d-1}$ and $S_j(d) = (1-q_j)^{d-1}$, so $Q_j$ is maximized by

```math
\hat{q}_j = \frac{\sum_d \eta_{j,d}}{\sum_d d \, \eta_{j,d} + \sum_d (d-1) \, \zeta_{j,d}}
```

which is exactly the vanilla update $1 - \hat{a}_{j,j}$ of the equivalent HMM, whose self-transitions also cover the censored sojourn except for its last step.

If state $j$ has no completed sojourn, i.e. $\sum_d \eta_{j,d} = 0$, it was only seen in censored final sojourns, so the data give no upper bound on its durations.
Then $Q_j$ increases towards an infinite mean ($q_j \to 0$ in the geometric case), which no distribution attains.
In this case the duration distribution is kept unchanged, which keeps $Q_j$, and hence the loglikelihood, non-decreasing.
This also covers a state that is never visited.

In general, $\log S_j(d; \theta_j)$ has no closed form, but the censored sojourns can be completed as in an EM algorithm, treating their unobserved remainder as missing data.
Under the current parameters $\theta_j^{\text{old}}$, a sojourn known to last at least $d$ steps lasts exactly $d' \geq d$ steps with probability $p_j(d'; \theta_j^{\text{old}}) / S_j(d; \theta_j^{\text{old}})$, and Jensen's inequality gives

```math
\log S_j(d; \theta_j) \geq \sum_{d' \geq d} \frac{p_j(d'; \theta_j^{\text{old}})}{S_j(d; \theta_j^{\text{old}})} \log \frac{p_j(d'; \theta_j)}{p_j(d'; \theta_j^{\text{old}})} + \log S_j(d; \theta_j^{\text{old}})
```

with equality at $\theta_j = \theta_j^{\text{old}}$.
Plugging this bound into $Q_j$, maximizing it amounts to a weighted fit of the duration distribution on the values $d - 1$ (recall that the distribution supplied by the user models $D_j - 1$) with the weights

```math
\tilde{\eta}_{j,d'} = \eta_{j,d'} + p_j(d'; \theta_j^{\text{old}}) \sum_{d \leq d'} \frac{\zeta_{j,d}}{S_j(d; \theta_j^{\text{old}})}
```

and since the bound is tight at $\theta_j^{\text{old}}$, this cannot decrease $Q_j$.
Therefore the loglikelihood of Baum-Welch still never decreases, and iterating the update alone would converge to a maximizer of $Q_j$ under the usual regularity conditions of EM.
The inner sum is accumulated in log space as $d'$ grows, so each duration costs one evaluation of $p_j$.
The weights extend beyond the longest sojourn, until the imputed mass left, $S_j(d'+1; \theta_j^{\text{old}}) \sum_{d} \zeta_{j,d} / S_j(d; \theta_j^{\text{old}})$, falls below the machine precision relative to the total weight, or the support of $p_j$ ends.

## Bibliography

```@bibliography
```
