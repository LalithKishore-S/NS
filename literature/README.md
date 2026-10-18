# Reading map — paper → code

Read the paper **before** opening the code it backs. ★ = listed in Project_Plan (Tier 1/2). Others are the
foundational papers the code directly implements.

| Step | Paper | Backs this code | Status |
|---|---|---|---|
| 1 | Erdős & Rényi, *On random graphs I*, Publ. Math. Debrecen (1959) | `network_generation.py` → ER | read in step 1 |
| 1 | Barabási & Albert, *Emergence of scaling in random networks*, Science 286 (1999) | `network_generation.py` → BA | read in step 1 |
| 1 | Watts & Strogatz, *Collective dynamics of 'small-world' networks*, Nature 393 (1998) | `network_generation.py` → WS | read in step 1 |
| 1 | Holland, Laskey & Leinhardt, *Stochastic blockmodels: first steps*, Social Networks 5 (1983) | `network_generation.py` → SBM | read in step 1 |
| 2 | ★ Buldyrev, Parshani, Paul, Stanley & Havlin, *Catastrophic cascade of failures in interdependent networks*, Nature 464, 1025 (2010) | `interdependency.py` (q=1), `cascade_simulator.py`, `experiments/01_buldyrev_check.py` | **read now** |
| 2 | Parshani, Buldyrev & Havlin, *Interdependent networks: reducing the coupling strength leads to a change from a first to second order percolation transition*, PRL 105, 048701 (2010) | `interdependency.py` → `dependency_fraction` < 1 | skim now |
| 3 | ★ Dilmoney, Gross, Havlin & Shnerb, *Dynamics of critical cascades in interdependent networks*, PRL 134, 187401 (2025) — [arXiv:2504.06862](https://arxiv.org/abs/2504.06862) | `threshold_analysis.py`, R1 check (plateau = warning window) | step 3 |
| 4 | ★ Deb, Sidheekh, Clements, Krishnan & Dutta, *Machine learning methods trained on simple models can predict critical transitions in complex natural systems* (EWSNet), R. Soc. Open Sci. 9, 211475 (2022) — [PMC8847887](https://pmc.ncbi.nlm.nih.gov/articles/PMC8847887) | `feature_extraction.py` → variance, lag-1 autocorrelation | step 4 |
| 5 | ★ Machado, Sridhar, Gil, Henriques, Moura & Santos, *Recovering the graph underlying networked dynamical systems under partial observability: a deep learning approach*, AAAI 2023 — [arXiv:2208.04405](https://arxiv.org/abs/2208.04405) | `mechanism_classifier.py` (closest competitor — we classify the family, not edges) | step 5 |
| 6 | ★ Salehpour & Al-Anbagi, *RESP: A real-time early-stage failure prediction mechanism for smart grid systems*, IEEE Systems Journal (2024) | `collapse_predictor.py` (features from partial failures + XGBoost baseline) | step 6 |
| 6 | ★ Liu et al., *Early predictor for the onset of critical transitions in networked dynamical systems*, Phys. Rev. X 14, 031009 (2024) | evaluation framing in `evaluation.py` | step 6 |
| 6 | ★ (optional) Ma et al., *Learning from the past: predicting critical transitions with machine learning trained on surrogates of historical data* — [arXiv:2410.09707](https://arxiv.org/abs/2410.09707) (Comm. Phys. 2025) | baseline-comparison design | step 6 |

Per-step notes: `notes/stepN_*.md`.  Five-line summaries: `comparison_matrix.md`.
