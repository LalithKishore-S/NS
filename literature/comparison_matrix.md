# Comparison matrix (plan §7 gate: 5 lines per paper)

Rows marked *(fill on reading)* are drafted from abstracts; confirm/correct after you read the full paper.

### Buldyrev et al., Nature 2010
- **Observed:** — (theory + simulation paper; full graph known)
- **Hidden:** —
- **Task:** derive when two coupled networks keep a mutually connected giant component
- **Target:** critical threshold p_c and size of the surviving component; shows the transition is abrupt (first-order)
- **We differ:** we use their cascade as the *data generator*, then hide the graph and learn from the failure sequence.

### Dilmoney, Gross, Havlin & Shnerb, PRL 2025
- **Observed:** the cascade over time near criticality
- **Hidden:** —
- **Task:** describe the cascade as a branching / birth-death process
- **Target:** collapse probability, avalanche duration, length of the plateau before collapse, vs. system size and trigger strength
- **We differ:** they characterise the plateau analytically; we use the plateau as the window for *data-driven* early warning on hidden, varied topologies.

### Machado et al., AAAI 2023
- **Observed:** time series of a subset of nodes (linear stochastic dynamics)
- **Hidden:** the remaining nodes and the edges
- **Task:** decide, pair by pair, if an edge exists (features + CNN)
- **Target:** individual edges of a single network
- **We differ:** we infer the *generating family* of a two-layer interdependent network from cascade dynamics, not individual edges.

### Salehpour & Al-Anbagi, RESP, IEEE Syst. J. 2024 *(fill on reading)*
- **Observed:** early-stage component failures in a smart grid (coupled power + communication)
- **Hidden:** how the cascade will end
- **Task:** real-time early-stage failure prediction (ML, XGBoost per the plan)
- **Target:** failure outcome for one fixed grid
- **We differ:** we predict collapse across synthetic families and test transfer to a family never seen in training.

### Deb et al., R. Soc. Open Sci. 2022 (EWSNet) *(fill on reading)*
- **Observed:** a single time series approaching a transition
- **Hidden:** the generating model
- **Task:** deep net trained on simple simulated models
- **Target:** whether / what kind of critical transition is coming
- **We differ:** we use their classical signals (variance, lag-1 autocorrelation) as *baseline features* on network cascades.

### Liu et al., PRX 2024 *(fill on reading)*
- **Observed:** node dynamics on a network
- **Hidden:** when the transition starts
- **Task:** predict the onset of a critical transition in networked systems
- **Target:** the transition point
- **We differ:** binary collapse/no-collapse from partial cascades, with leave-one-family-out testing.

### Ma et al., Comm. Phys. 2025 (optional) *(fill on reading)*
- **Observed:** historical time series of the system
- **Hidden:** future transition
- **Task:** train ML on surrogates of the system's own history instead of generic models
- **Target:** upcoming critical transition
- **We differ:** our training set is simulated cascades on several topology families; the held-out family plays the role of an "unseen system".
