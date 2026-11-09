# Ollama + `gpt-oss:20b` setup (separate from conda)

**Why a separate guide?**  
`environment.yml` installs the **Python client** (`pip install ollama`) that
SimAgent uses to call an LLM API. It does **not** install the **Ollama
server** or download any model weights. Those are system / GPU services you
install once per machine (or on an HPC node), then point SimAgent at them.

Default SimAgent flags (already match this guide):

```bash
--llm-model gpt-oss:20b
--llm-base-url http://127.0.0.1:11434
```

---

## 1. What you need

| Piece | Role | Where it lives |
|-------|------|----------------|
| Conda env `SimAgentEnv` | GROMACS, MDAnalysis, LangChain, `ollama` Python package | `environment.yml` |
| **Ollama server** | Runs models, listens on port **11434** | OS install or Singularity/Docker |
| **Model `gpt-oss:20b`** | Weights pulled into Ollama’s model store | `ollama pull gpt-oss:20b` |

Hardware notes for **`gpt-oss:20b`**:

- Prefer a **CUDA GPU** with enough VRAM (order-of-magnitude: ~12–24+ GB
  depending on quantization / context; check current Ollama model card).
- CPU-only is possible but slow for agentic multi-call workflows.
- For parallel SimAgent workers, set `OLLAMA_NUM_PARALLEL` to match
  `--llm-concurrency` (default often 4).

---

## 2. New laptop / workstation (Linux / macOS)

### 2.1 Install the Ollama server

**Linux (official script):**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS:** download the app from [https://ollama.com/download](https://ollama.com/download)
or use Homebrew if you prefer (`brew install ollama`).

**Windows:** use the installer from the same download page (WSL2 + Linux
install also works well for this project).

Confirm the binary:

```bash
ollama --version
```

### 2.2 Start the server

```bash
# Foreground (good for first test)
ollama serve

# Or rely on the desktop app / systemd user service if the installer enabled it.
```

Leave this running, or enable it as a service so it starts at login.

Health check (another terminal):

```bash
curl -s http://127.0.0.1:11434/api/tags
# Expect JSON with a "models" list (may be empty before pull)
```

### 2.3 Pull `gpt-oss:20b`

```bash
ollama pull gpt-oss:20b
```

List models:

```bash
ollama list
# Should show gpt-oss:20b
```

Quick chat test:

```bash
ollama run gpt-oss:20b "Reply with one word: ready"
```

### 2.4 Optional server tuning

```bash
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_NUM_PARALLEL=4          # concurrent LLM calls
export OLLAMA_MAX_QUEUE=512
export OLLAMA_KEEP_ALIVE=-1           # keep model loaded
export OLLAMA_FLASH_ATTENTION=true    # if supported on your GPU build
# Optional custom model/cache dirs:
# export OLLAMA_MODELS=$HOME/ollama_models
# export OLLAMA_CACHE=$HOME/ollama_cache
```

Restart `ollama serve` after changing these.

### 2.5 Create the SimAgent conda env (if not done)

```bash
cd /path/to/simagent
conda env create -f environment.yml
conda activate SimAgentEnv
pip install -e .
```

### 2.6 Run SimAgent against local Ollama

```bash
conda activate SimAgentEnv

python SimAgent.py \
  --goal "Preprocess and setup MD for my_protein.pdb" \
  --working-dir /work/run1 \
  --llm-base-url http://127.0.0.1:11434 \
  --llm-model gpt-oss:20b
```

Omit `--llm-*` flags to use the same defaults. **A reachable LLM is required**
unless you pass `--no-llm` (heuristic / registry-based plans only).

If Ollama is down and no API key is set, SimAgent exits with setup guidance
instead of silently mocking the LLM.

---

## 3. Remote Ollama (another host on your network)

On the GPU machine: install Ollama, pull the model, and bind the host so
clients can connect (example — adjust firewall / auth for your site):

```bash
export OLLAMA_HOST=0.0.0.0:11434
ollama serve
```

On the analysis machine:

```bash
python SimAgent.py \
  --goal "..." \
  --working-dir /work/run1 \
  --llm-base-url http://GPU_HOST:11434 \
  --llm-model gpt-oss:20b
```

There is an `--llm-api-key` flag (also `LLM_API_KEY` / `OPENAI_API_KEY`) for
OpenAI-compatible APIs. For a local Ollama server, no key is needed — point
`--llm-base-url` at the host.

---

## 4. HPC / Singularity (site-specific)

Write a site SLURM script for the Ollama server. Do not commit account names,
partitions, or container paths. A typical submit is:

```bash
sbatch ollama_server.slurm
# After the job is RUNNING, point SimAgent at the node/port your site exposes
# (often via SSH tunnel):
#   ssh -L 11434:compute-node:11434 login-node
#   --llm-base-url http://127.0.0.1:11434
```

Pull the model **once** on a node that can write to `OLLAMA_MODELS`:

```bash
singularity exec --nv ~/containers/ollama_cuda.sif ollama pull gpt-oss:20b
```

---

## 5. Verify end-to-end with SimAgent’s client

With the conda env active and Ollama serving:

```bash
python - <<'PY'
from ollama import Client
c = Client(host="http://127.0.0.1:11434")
print("models:", [m.model for m in c.list().models])
r = c.chat(model="gpt-oss:20b", messages=[{"role": "user", "content": "Say OK"}])
print(r["message"]["content"][:200])
PY
```

If this works, SimAgent’s `LLMClient` (`agentic/llm.py`) can reach the same
endpoint.

---

## 6. Troubleshooting

| Symptom | Likely fix |
|---------|------------|
| `Connection refused` on `:11434` | Start `ollama serve` (or the desktop app) |
| Model not found | `ollama pull gpt-oss:20b` then `ollama list` |
| Very slow / OOM on GPU | Close other GPU jobs; reduce `OLLAMA_NUM_PARALLEL`; use a larger GPU |
| SimAgent exits: no reachable LLM | Start Ollama, or set `--llm-api-key`, or pass `--no-llm` |
| SimAgent plans are empty / mid-run fallback | Check `agent_conversation.log` for LLM errors; confirm `--llm-model` spelling |
| Conda env created but LLM missing | Normal — install this guide’s server + model; conda only has the client |
| New machine `conda env create` fails on pins | Use the portable `environment.yml` in the repo root (no machine `prefix:`) |

---

## 7. Checklist (new computer)

1. [ ] `conda env create -f environment.yml` && `conda activate SimAgentEnv` && `pip install -e .`
2. [ ] Install **Ollama server** (this guide §2.1)
3. [ ] `ollama serve` && `curl http://127.0.0.1:11434/api/tags`
4. [ ] `ollama pull gpt-oss:20b`
5. [ ] Optional: install CHARMM36 ff if you use phospho / CHARMM (see README)
6. [ ] `python SimAgent.py ... --llm-model gpt-oss:20b --llm-base-url http://127.0.0.1:11434`
7. [ ] Optional smoke-test: [tutorial/TUTORIAL.md](../tutorial/TUTORIAL.md) (`pseudo_apo_holo`)
