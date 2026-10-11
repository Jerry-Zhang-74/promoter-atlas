# Temporal Stability Target Note

Current Candidate E uses cross-tissue stability rather than temporal durability.

Current tissue-stability definition:

$$
S_{tissue}(g) = -\mathrm{MAD}\left(\left\{\log_2(1+x_{g,t}) : t \in D(g)\right\}\right)
$$

where $D(g)$ is the detectable tissue set for gene $g$.

This captures cross-tissue snapshot consistency, not long-term persistence over time.

The new design target should be formulated as a separate durability axis:

$$
D_{time}(p) = -\mathrm{MAD}\left(\left\{\log_2(1+y_{p,\tau}) : \tau \in \mathcal{T}\right\}\right)
$$

where $p$ is a promoter and $y_{p,\tau}$ is expression measured at time point $\tau$.

To explicitly penalize hit-and-run behavior, use an early/late retention score:

$$
R_{late/early}(p) = \frac{y_{p,\tau_{late}} + \varepsilon}{y_{p,\tau_{early}} + \varepsilon}
$$

Interpretation:

- high $S_{tissue}$: broad endogenous stability across tissues
- high $D_{time}$: durable expression across time
- high $R_{late/early}$: less hit-and-run behavior

Recommendation before Phase 5:

- keep current Candidate E unchanged as the GTEx-grounded tissue-stability label system
- treat temporal durability as a new downstream target, not a replacement for current S
- when time-course assay data becomes available, model durability as an additional output