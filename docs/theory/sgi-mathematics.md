# SGI Mathematics

This page provides the complete mathematical derivation of the Semantic Grounding Index, its geometric interpretation as a relative proximity measure, the implications of the triangle inequality, and the rationale for the tanh normalization.

## Formal Definition

Let $\phi: \mathcal{T} \to \mathbb{R}^n$ be a sentence embedding function. Given a question $q$, context $\text{ctx}$, and response $r$, the Semantic Grounding Index is defined as:

$$
\text{SGI}(q, \text{ctx}, r) = \frac{d\bigl(\phi(r),\, \phi(q)\bigr)}{d\bigl(\phi(r),\, \phi(\text{ctx})\bigr)}
$$

where $d(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2$ is the Euclidean distance in $\mathbb{R}^n$.

The domain of SGI is $[0, +\infty)$, with the convention that when $d(\phi(r), \phi(\text{ctx})) < \epsilon$ (response identical to context), SGI is set to a large constant (10.0 in the implementation).

## Geometric Interpretation: Relative Proximity

SGI is a **ratio of distances** from the response embedding to two reference points (question and context). This has a clean geometric interpretation.

### The Apollonius Circle (Generalized)

In two dimensions, the locus of points $P$ such that $d(P, A) / d(P, B) = k$ (for fixed points $A$, $B$ and constant $k > 0$, $k \neq 1$) is a circle --- the **Apollonius circle**. In $\mathbb{R}^n$, this generalizes to an **Apollonius hypersphere**.

For SGI, the "Apollonius surface" of constant SGI score $k$ is the set:

$$
\mathcal{A}_k = \left\{\mathbf{x} \in \mathbb{R}^n : \frac{\|\mathbf{x} - \phi(q)\|}{\|\mathbf{x} - \phi(\text{ctx})\|} = k\right\}
$$

This is a hypersphere (for $k \neq 1$) or a hyperplane (for $k = 1$). Specifically:

- **$k = 1$**: $\mathcal{A}_1$ is the **perpendicular bisector hyperplane** between $\phi(q)$ and $\phi(\text{ctx})$. This is the decision boundary between grounded and ungrounded.
- **$k > 1$**: $\mathcal{A}_k$ is a hypersphere surrounding $\phi(\text{ctx})$ (the context side). Responses inside this sphere are closer to context.
- **$k < 1$**: $\mathcal{A}_k$ is a hypersphere surrounding $\phi(q)$ (the question side). Responses inside this sphere are closer to the question.

### The Perpendicular Bisector Hyperplane

The SGI = 1 surface is geometrically significant. It is the hyperplane:

$$
H = \left\{\mathbf{x} \in \mathbb{R}^n : \|\mathbf{x} - \phi(q)\| = \|\mathbf{x} - \phi(\text{ctx})\|\right\}
$$

This hyperplane passes through the midpoint $\frac{1}{2}(\phi(q) + \phi(\text{ctx}))$ and is perpendicular to the line segment connecting $\phi(q)$ and $\phi(\text{ctx})$. Its equation is:

$$
\bigl(\phi(\text{ctx}) - \phi(q)\bigr)^\top \mathbf{x} = \frac{1}{2}\bigl(\|\phi(\text{ctx})\|^2 - \|\phi(q)\|^2\bigr)
$$

Responses on the context side of this hyperplane have SGI > 1; responses on the question side have SGI < 1.

## Triangle Inequality Implications

The triangle inequality in $\mathbb{R}^n$ constrains the range of SGI values. Given three points $\phi(q)$, $\phi(\text{ctx})$, and $\phi(r)$:

$$
\|\phi(r) - \phi(q)\| \leq \|\phi(r) - \phi(\text{ctx})\| + \|\phi(\text{ctx}) - \phi(q)\|
$$

$$
\|\phi(r) - \phi(\text{ctx})\| \leq \|\phi(r) - \phi(q)\| + \|\phi(q) - \phi(\text{ctx})\|
$$

Let $D_{qc} = \|\phi(q) - \phi(\text{ctx})\|$ be the distance between question and context. Then:

$$
\text{SGI} = \frac{d_q}{d_c} \leq \frac{d_c + D_{qc}}{d_c} = 1 + \frac{D_{qc}}{d_c}
$$

And:

$$
\text{SGI} = \frac{d_q}{d_c} \geq \frac{|d_c - D_{qc}|}{d_c} = \left|1 - \frac{D_{qc}}{d_c}\right|
$$

These bounds show that the maximum achievable SGI score depends on the question-context distance and the response-context distance. When the context is very close to the question ($D_{qc}$ small), SGI has limited discriminative power.

!!! abstract "Geometric insight"
    SGI discrimination is strongest when the question and context are **semantically distant** from each other --- i.e., when the context provides genuinely new information beyond what the question implies. If the context merely restates the question, even a perfect response will have SGI $\approx 1$ because the two reference points are too close together.

## Degenerate Cases

The implementation handles two degenerate cases explicitly:

### Response Identical to Context ($d_c < \epsilon$)

When $\phi(r) \approx \phi(\text{ctx})$, the denominator approaches zero and SGI would diverge. This represents the maximum possible grounding --- the response perfectly matches the context. The implementation returns SGI = 10.0, normalized = 1.0, flagged = False.

### Response Identical to Question ($d_q < \epsilon$)

When $\phi(r) \approx \phi(q)$, the numerator approaches zero and SGI $\to 0$. This represents the LLM simply echoing the question without engaging with the context. The implementation returns SGI = 0.0, normalized = 0.0, flagged = True.

## Tanh Normalization

The raw SGI range $[0, +\infty)$ is inconvenient for dashboards, threshold comparison, and aggregation. The tanh normalization maps it to $[0, 1]$:

$$
\text{SGI}_{\text{norm}} = \tanh\bigl(\max(0,\; \text{SGI}_{\text{raw}} - 0.3)\bigr)
$$

### Why Tanh?

The hyperbolic tangent function has several desirable properties for this mapping:

1. **Bounded output**: $\tanh(x) \in (0, 1)$ for $x > 0$, guaranteeing the normalized score stays in $[0, 1]$.

2. **Sensitivity where it matters**: The derivative of $\tanh$ is:

$$
\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x) = \text{sech}^2(x)
$$

This is largest near $x = 0$ and decays toward 0 as $x \to \infty$. After the 0.3 offset, the region of maximum sensitivity corresponds to SGI $\approx 0.3$--$1.5$, which is exactly the decision-relevant range.

3. **Diminishing returns**: Very high raw SGI scores (e.g., 3.0 vs. 5.0) both map to values near 1.0. This is appropriate because both indicate strong grounding --- the distinction between "very grounded" and "extremely grounded" is not actionable.

### The 0.3 Offset

The offset $\text{SGI}_{\text{raw}} - 0.3$ shifts the sigmoid curve so that SGI values below 0.3 map to exactly 0.0. Empirically, SGI values below 0.3 are extremely strong indicators of ungrounded responses, and the normalized score correctly assigns them the minimum value.

### Mapping Table

| Raw SGI | $\max(0, \text{SGI} - 0.3)$ | $\tanh(\cdot)$ |
|---|---|---|
| 0.00 | 0.00 | 0.000 |
| 0.30 | 0.00 | 0.000 |
| 0.50 | 0.20 | 0.197 |
| 0.80 | 0.50 | 0.462 |
| 0.95 | 0.65 | 0.572 |
| 1.00 | 0.70 | 0.604 |
| 1.20 | 0.90 | 0.716 |
| 1.50 | 1.20 | 0.834 |
| 2.00 | 1.70 | 0.935 |
| 3.00 | 2.70 | 0.990 |

## Threshold Selection

The thresholds `SGI_STRONG_PASS = 1.20` and `SGI_REVIEW = 0.95` were derived empirically from the experiments reported in arXiv:2512.13771.

### SGI_REVIEW = 0.95

The review threshold is set slightly below the equidistance boundary (SGI = 1.0) to account for noise in embedding space. A small amount of measurement noise can push a genuinely grounded response slightly below 1.0. By setting the threshold at 0.95, we reduce false positives (incorrectly flagging grounded responses) while still catching responses that are meaningfully closer to the question than to the context.

### SGI_STRONG_PASS = 1.20

The strong pass threshold at 1.20 marks the point where the response is demonstrably closer to the context --- the response-to-question distance exceeds the response-to-context distance by at least 20%. At this level, the probability of a false negative (a hallucinated response scoring above 1.20) is empirically very low.

### Threshold as Verification Triage

!!! warning "Thresholds are not truth detectors"
    The SGI thresholds define a **triage policy**: which outputs should a human reviewer examine? Setting the review threshold lower catches more hallucinations but increases the review workload. Setting it higher reduces workload but misses more hallucinations. The default values balance these tradeoffs for a typical RAG verification scenario.

## SGI as a Metric Space Property

SGI can be understood as a property of the metric space $(\mathbb{R}^n, d)$ rather than of the specific embedding model. Given any three points $Q$, $C$, $R$ in a metric space:

$$
\text{SGI}(Q, C, R) = \frac{d(R, Q)}{d(R, C)}
$$

This ratio is:

- **Scale-invariant**: Multiplying all distances by a constant does not change SGI. If you scale the entire embedding space, SGI values are preserved.
- **Not translation-invariant**: Shifting all embeddings by a constant vector can change the distances and thus the SGI.
- **Not rotation-invariant in general**: However, SGI is invariant under rigid motions (isometries) of $\mathbb{R}^n$ that preserve the relative positions of $Q$, $C$, $R$.

The scale-invariance is particularly useful: it means SGI scores are comparable across embedding models of different scales, provided the models encode semantic similarity consistently.

## Sensitivity Analysis

How sensitive is SGI to perturbations in the embeddings? Consider a small perturbation $\epsilon$ to the response embedding: $\phi(r) \to \phi(r) + \boldsymbol{\epsilon}$.

Using a first-order Taylor expansion:

$$
d(\phi(r) + \boldsymbol{\epsilon}, \phi(q)) \approx d(\phi(r), \phi(q)) + \frac{(\phi(r) - \phi(q))^\top \boldsymbol{\epsilon}}{d(\phi(r), \phi(q))}
$$

The relative change in SGI is approximately:

$$
\frac{\Delta \text{SGI}}{\text{SGI}} \approx \frac{\hat{d}_q^\top \boldsymbol{\epsilon}}{d_q} - \frac{\hat{d}_c^\top \boldsymbol{\epsilon}}{d_c}
$$

where $\hat{d}_q$ and $\hat{d}_c$ are unit vectors in the question and context directions. This shows that SGI is most sensitive to perturbations along the **question-context axis** and least sensitive to perturbations orthogonal to this axis.

## References

- Marin, J. (2025). *Semantic Grounding Index for LLM Hallucination Detection*. arXiv:2512.13771.
- Beyer, K. et al. (1999). When is "nearest neighbor" meaningful? *ICDT*.
- Apollonius of Perga. *Conics* (c. 200 BCE). The original treatment of circles defined by distance ratios.
