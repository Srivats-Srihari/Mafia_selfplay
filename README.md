# Mafia Self-Play (GGUF, Colab, Google Drive)

Run high-volume Mafia simulations with a local GGUF model (0 API cost), save full game logs to Drive, and control diversity and behavior with many knobs.

## Included files

- `mafia_self_play.py`: main runner
- `mafia_self_play_colab.ipynb`: ready-to-run Colab notebook
- `requirements.txt`

## Latest features in this version

- Resume support: `--resume` skips already-saved games.
- Parallel scaling: `--parallel_games` (set `0` for auto).
- Optional compressed storage: `--compress_games` writes `.json.gz`.
- Extra model runtime controls: `--n_batch`, `--no_mmap`, `--mlock`.
- Richer game controls:
  - `--talks_per_day`
  - `--night_mode random|model`
  - `--no_shuffle_speaking_order`
  - `--no_tie_break_random`
- Persona customization from file: `--persona_file`.
- Automatic outputs for analysis:
  - `summary_all.jsonl`
  - `leaderboard.csv`
  - `stats.json`
  - `run_manifest.json`

## Colab quick start

Open and run:

- `mafia_self_play_colab.ipynb`

Notebook flow:

1. Mount Drive.
2. Clone your repo.
3. Install requirements.
4. Set GGUF path/output path + controls.
5. Run self-play.
6. Inspect `stats.json` and `leaderboard.csv`.

## CLI example (direct)

```bash
python mafia_self_play.py \
  --model_path "/content/drive/MyDrive/models/your_model.gguf" \
  --out_dir "/content/drive/MyDrive/mafia_runs/run_001" \
  --n_games 5000 \
  --parallel_games 8 \
  --resume \
  --compress_games \
  --players 9 \
  --mafia_count 2 \
  --max_rounds 10 \
  --talks_per_day 1 \
  --night_mode model \
  --temperature 0.95 \
  --top_p 0.90 \
  --top_k 50 \
  --min_p 0.05 \
  --repeat_penalty 1.1 \
  --base_seed 20260305 \
  --n_ctx 2048 \
  --n_threads 2 \
  --n_gpu_layers -1 \
  --n_batch 512
```

## Output layout

- `games/game_0000001.json` or `.json.gz`
- `summary_worker_*.jsonl` (worker shards)
- `summary_all.jsonl` (merged)
- `leaderboard.csv` (flat table)
- `stats.json` (distribution and average rounds)
- `run_manifest.json` (full run config)

## Notes for free Colab

- Zero rupees for API usage: yes.
- Throughput depends on model size and Colab resources.
- Start safe:
  - `parallel_games=4`
  - small quantized model
  - `n_ctx=1024` or `2048`
- Increase workers gradually until near resource limits.

