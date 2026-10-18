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

### Deb et al., R. Soc. Open Sci. 9, 211475 (2022) (EWSNet) *(abstract checked; confirm details on reading)*
- **Observed:** a single time series approaching a transition (no knowledge of system structure)
- **Hidden:** the generating model / system structure
- **Task:** deep net (EWSNet) trained on simple simulated models, tested on real climate and ecological collapses
- **Target:** whether a critical transition is coming, and whether it is catastrophic or non-catastrophic
- **We differ:** we use their classical signals (variance, lag-1 autocorrelation) as *baseline features* on network cascades.

### Liu et al., Phys. Rev. X 14, 031009 (2024) *(summary checked; confirm on reading)*
- **Observed:** short time series (≈20 points) at every node, plus the network the GNN runs on
- **Hidden:** when the regime shift will happen
- **Task:** GIN + GRU neural network trained on simulated / real networked dynamics
- **Target:** the onset time of a critical transition
- **We differ:** we see only aggregate failure counts and no graph, predict collapse vs. recovery with classical ML, and explicitly test transfer to an unseen topology family (they leave cross-network transfer to future work).

### Ma, Zeng, Zhang & Bury, Comm. Phys. 2025 / arXiv:2410.09707 (optional) *(abstract checked)*
- **Observed:** historical time series of the system's own past transitions
- **Hidden:** whether the current series is approaching a transition
- **Task:** train SVM / LSTM / CNN classifiers on surrogates (permutation, Fourier, AAFT, IAAFT) of that history
- **Target:** probability of being "pre-transition"; compared to variance and lag-1 AC by ROC-AUC
- **We differ:** our training data are simulated cascades on several topology families; the held-out family plays the role of a "new system", and we keep a simple baseline (state-only) in every comparison, as they do.
