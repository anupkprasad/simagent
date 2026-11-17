# Apo vs holo ATP binding (four pseudokinases)

Representative human pseudokinases used in early SimAgent demos and in the
paper’s multi-protein examples:

| PDB | UniProt | Name |
|-----|---------|------|
| `p21860.pdb` | P21860 | ERBB3 |
| `q8iv63.pdb` | Q8IV63 | VRK3 |
| `q8nb16.pdb` | Q8NB16 | MLKL |
| `q8wz42.pdb` | Q8WZ42 | TITIN |

Each input PDB is protein + ATP + Mg (copied from the robustness campaign
starting structures). The locked goal asks SimAgent to run **apo** (protein
only) and **holo** (protein + ATP + Mg) for each — eight simulations total.

## Files

- `goal.txt` — natural-language objective for SimAgent
- `run_simagent_nohup.sh` — nohup launcher (review, then run)
- `*.pdb` — starting structures

## Run

```bash
# Repo root; Ollama serving gpt-oss:20b; conda env SimAgentEnv active
bash tutorial/pseudo_apo_holo/run_simagent_nohup.sh
```

Outputs (preprocess / simsetup / hpc / analysis / reporter) are written under
this directory as the working directory.
