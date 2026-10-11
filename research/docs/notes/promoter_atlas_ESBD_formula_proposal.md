promoter_atlas E / S / B / D Formula Proposal
========================================

Purpose
-------
This document proposes a label system for promoter_atlas that is aligned with the experimental goal:
design promoters that can drive relatively stable expression over a long time span, rather than producing
a short-lived hit-and-run activation pattern.

The current project already has a useful GTEx-grounded label system for expression strength and cross-tissue
behavior. The key design decision in this proposal is that cross-tissue stability and temporal durability
should be treated as different biological axes.

Recommended label axes
----------------------
The recommended label system is:

1. $$E$$: expression strength
2. $$S_{tissue}$$: cross-tissue stability
3. $$B$$: expression breadth
4. $$D_{time}$$: temporal durability

Interpretation:

- $$E$$ measures how strong expression is.
- $$S_{tissue}$$ measures how stable expression is across tissues.
- $$B$$ measures how broadly expression is detected across tissues.
- $$D_{time}$$ measures whether expression remains relatively sustained over time.

Why $$S_{tissue}$$ and $$D_{time}$$ should be separated
-------------------------------------------
The current GTEx-style stability axis is based on tissue snapshots:

$$
S_{tissue}(g) = - MAD \big( \{ \log_2(1 + x_{g,t}) : t \in D(g) \} \big)
$$

where:

- $$g$$ is a gene
- $$t$$ is a tissue
- $$x_{g,t}$$ is expression in tissue t
- $$D(g)$$ is the detectable tissue set for gene g

This quantity measures cross-tissue consistency.

However, the new experimental goal is different. The key question is:

Can a promoter maintain a relatively stable output over a long time span?

This is not the same as cross-tissue stability. A promoter can be broadly active across tissues and still
show fast decay in a time-course assay. Conversely, a promoter can be temporally persistent in one context
but not broadly active across tissues.

Therefore:

$$
S_{tissue} \neq D_{time}
$$

and they should not be merged into one raw label.

Recommended formula for $$E$$
-------------------------
For a gene $$g$$, define expression strength as:

$$
E_{raw}(g) = \frac{1}{T_g} \sum_{t=1}^{T_g} \log_2(1 + x_{g,t})
$$

where:

- $$T_g$$ is the number of tissues observed for gene g
- $$x_{g,t}$$ is the mean TPM in tissue t

Rationale:
- log transformation reduces domination by extreme high-expression genes
- averaging across tissues reflects overall strength rather than tissue-specific peaks

Recommended formula for $$B$$
-------------------------
Define expression breadth as:

$$
B_{raw}(g) = \frac{1}{T_g} \sum_{t=1}^{T_g} \mathbf{1}(x_{g,t} \ge \tau)
$$

where the detectable threshold is:

$$
\tau = 1 \text{ TPM}
$$

Rationale:
- breadth should describe how often a gene is genuinely expressed
- this should remain a separate axis rather than being folded into stability

Recommended formula for S_tissue
--------------------------------
Only define tissue stability when the gene has enough detectable tissues to support a meaningful variability estimate.

Main subset gate:

$$
B_{raw}(g) \ge B_{min}
$$

with:

$$
B_{min} = 0.30
$$

Detectable tissue requirement:

$$
|D(g)| \ge 3
$$

Then define:

$$
z_{g,t} = \log_2(1 + x_{g,t}), \qquad t \in D(g)
$$

and:

$$
S_{tissue,raw}(g) = - MAD \big( \{ z_{g,t} : t \in D(g) \} \big)
$$

Rationale:
- the gate avoids calling tissue-specific genes "stable"
- the minimum detectable tissue count avoids degenerate variance estimates
- using MAD makes the label more robust than variance-based definitions

Recommended formula for $$D_{time}$$
------------------------------
For promoter $$p$$ in a specific temporal context $$c$$ with time points $$\tau \in \mathcal{T}_c$$, define:

$$
D_{c,raw}(p) = - MAD \big( \{ \log_2(1 + y_{p,c,\tau}) : \tau \in \mathcal{T}_c \} \big)
$$

where:

- $$y_{p,c,\tau}$$ is promoter activity at time point \tau
- $$c$$ may represent a differentiation trajectory, stimulus response, or other time-course context

This is the temporal analogue of S_tissue.

Recommended activation gate for $$D_{time}$$
--------------------------------------
$$D_{time}$$ should only be defined if the promoter is genuinely active in the time course. For example:

$$
\max_{\tau \in \mathcal{T}_c} y_{p,c,\tau} \ge \tau_{min}
$$

and optionally:

$$
\sum_{\tau \in \mathcal{T}_c} \mathbf{1}(y_{p,c,\tau} \ge \tau_{min}) \ge m
$$

Rationale:
- very low flat traces should not be treated as "durable"
- $$D_{time}$$ should reflect sustained active transcription, not merely low noise

Recommended multi-context durability aggregation
------------------------------------------------
Because temporal data will usually come from multiple experimental contexts rather than one unified database,
first compute $$D$$ in each context, then aggregate across contexts.

For promoter $$p$$:

$$
D_{time,raw}(p) = median_{c \in C_p} D_{c,raw}(p)
$$

where $$C_p$$ is the set of contexts in which promoter p has valid temporal measurements.

This avoids forcing incompatible time-course experiments into one single matrix.

Recommended hit-and-run penalty
-------------------------------
Temporal durability should be complemented with a retention-style quantity.

For each context c:

$$
R_{c}(p) = \frac{y_{p,c,\tau_{late}} + \epsilon} {y_{p,c,\tau_{early}} + \epsilon}
$$

Interpretation:

- $$R_c(p) \approx 1$$ means late expression remains close to early expression
- $$R_c(p) \ll 1$$ means strong decay after early activation, i.e. hit-and-run-like behavior

A robust aggregate form is:

$$
R_{time}(p) = median_{c \in C_p} R_c(p)
$$

This quantity is especially relevant if the experimental objective explicitly rejects CRISPR-a-like transient activation.

Why simple equal-weight geometric mean is no longer ideal
---------------------------------------------------------
The current Candidate E style ranking can be written as:

$$
Score_{HKG} = \big(Score_E \cdot Score_{S_{tissue}} \cdot Score_B \big)^{1/3}
$$

This is mathematically simple and reasonable for a three-axis system, but once temporal durability becomes
a primary experimental goal, equal weighting is no longer the most appropriate choice.

The project goal is not merely to find genes that are:
- highly expressed
- broad
- cross-tissue stable

It is also to find promoters that are likely to maintain expression over time.

Therefore $$D_{time}$$ should not be a minor afterthought.

Recommended normalization
-------------------------
Each raw label should be converted to a rank-scaled score in [0, 1].

For any raw label $$Q_{raw}$$ over $$n$$ valid observations:

$$
Score_Q = \frac{rank(Q_{raw})}{n - 1}
$$

This is appropriate for:
- $$Score_E$$
- $$Score_{S_{tissue}}$$
- $$Score_B$$
- $$Score_D$$

Rationale:
- avoids assuming raw labels are Gaussian
- stabilizes downstream multi-objective combination
- is robust to heavy-tailed expression distributions

Recommended final HKG-like ranking score
----------------------------------------
The best overall choice for the current experimental purpose is a weighted geometric mean:

$$
Score_{HKG^*} = \exp \Big(w_E \log(Score_E + \epsilon) + w_S \log(Score_{S_tissue} + \epsilon) + w_B \log(Score_B + \epsilon) + w_D \log(Score_D + \epsilon) \Big)
$$

with:

$$
w_E + w_S + w_B + w_D = 1
$$

Recommended weights:

$$
w_E = 0.25
$$

$$
w_S = 0.25
$$

$$
w_B = 0.15
$$

$$
w_D = 0.35
$$

Therefore:

$$
Score_{HKG^*} = \exp \Big(0.25 \log(Score_E + \epsilon) + 0.25 \log(Score_{S_tissue} + \epsilon) + 0.15 \log(Score_B + \epsilon) + 0.35 \log(Score_D + \epsilon) \Big)
$$

Why this is the best choice
---------------------------
This formula is recommended because:

1. It preserves the conjunctive nature of housekeeping-like behavior.
   A promoter should not rank highly if one essential dimension is extremely poor.

2. It gives temporal durability the largest weight.
   This directly reflects the updated experimental goal: sustained output over time is one of the main objectives.

3. It still preserves expression strength and cross-tissue stability as major axes.
   A durable promoter that is too weak or too context-restricted is not a practical target.

4. It prevents breadth from dominating the final score.
   Breadth matters, but it should not overwhelm the more mechanistic stability dimensions.

Why weighted geometric mean is preferable to weighted sum
---------------------------------------------------------
A weighted sum such as

$$
Score = a Score_E + b Score_S + c Score_B + d Score_D
$$

is easier to interpret, but it allows a very high value in one dimension to compensate too easily for
a very poor value in another.

For promoter design, this is undesirable. A promoter with very high expression but very poor durability
should not rank among the best candidates for long-term stable systems.

The geometric mean imposes a stronger penalty on weak dimensions and is therefore better aligned with
the experimental objective.

Recommended current-stage implementation
----------------------------------------
Before temporal data are fully available, the project can proceed in two stages.

Stage 1:
Use the current GTEx-grounded system:

$$
(E,\ S_{tissue},\ B)
$$

with the current three-axis score:

$$
Score_{HKG,3} = \exp \Big(0.40 \log(Score_E + \epsilon) + 0.35 \log(Score_{S_tissue} + \epsilon) + 0.25 \log(Score_B + \epsilon) \Big)
$$

This is preferred over equal weights even now, because breadth should be a little less dominant than
expression strength and cross-tissue stability.

Stage 2:
Once FANTOM5 or other time-course promoter-level data are integrated, move to:

$$
(E,\ S_{tissue},\ B,\ D_{time})
$$

and then use the four-axis final score:

$$
Score_{HKG^*} = \exp \Big(0.25 \log(Score_E + \epsilon) + 0.25 \log(Score_{S_tissue} + \epsilon) + 0.15 \log(Score_B + \epsilon) + 0.35 \log(Score_D + \epsilon) \Big)
$$

Practical interpretation of the four axes
-----------------------------------------
The four-axis system should be read as:

- $$E$$ : how strong the promoter-driven output can be
- $$S_{tissue}$$ : how consistently that output is supported across tissues or biological backgrounds
- $$B$$ : how broadly the output is detected
- $$D_{time}$$ : how well the promoter maintains output across time

This means:

- a high- $$E$$ promoter is strong
- a high- $$S_{tissue}$$ promoter is broadly stable across tissues
- a high- $$B$$ promoter is broadly detectable
- a high- $$D_{time}$$ promoter is persistent rather than transient

Final recommendation
--------------------
For scientific clarity and experimental usefulness, the recommended path is:

1. Keep current GTEx-grounded labels as:

$$E , \ S_{tissue} , \ B$$

2. Do not redefine $$S_{tissue}$$ to mean temporal stability.

3. Introduce temporal durability as a new independent axis:

$$D_{time}$$

4. Use a weighted geometric mean as the final HKG-like ranking formula.

5. Once temporal proxy data are available, make $$D_{time}$$ the highest-weight stability-related axis.

The core conceptual recommendation is:

$$
S_{tissue} \neq D_{time}
$$

and the final long-term promoter design objective should be centered on:

$$
(E,\ S_{tissue},\ B,\ D_{time})
$$

rather than forcing all notions of stability into one single label.
