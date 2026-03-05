import argparse
import csv
import gzip
import json
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from llama_cpp import Llama


ROLE_MAFIA = "mafia"
ROLE_SILENCER = "silencer"
ROLE_DOCTOR = "doctor"
ROLE_DETECTIVE = "detective"
ROLE_JESTER = "jester"
ROLE_VILLAGER = "villager"
MAFIA_TEAM_ROLES = {ROLE_MAFIA, ROLE_SILENCER}

DEFAULT_PERSONAS = (
    "calm and logical",
    "aggressive accuser",
    "cautious and skeptical",
    "chaotic and playful",
    "empathetic mediator",
    "data-driven analyst",
    "deceptive storyteller",
    "quiet observer",
)
SCHEMA_VERSION = "3.0"


@dataclass
class SamplingConfig:
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 40
    min_p: float = 0.05
    repeat_penalty: float = 1.1
    max_tokens_chat: int = 96
    max_tokens_vote: int = 24


@dataclass
class GameConfig:
    players: int = 8
    mafia_count: int = 2
    silencer_count: int = 1
    doctor_count: int = 1
    detective_count: int = 1
    jester_count: int = 1
    max_rounds: int = 8
    talks_per_day: int = 1
    allow_tie_break_random: bool = True
    shuffle_speaking_order: bool = True
    night_mode: str = "model"
    persona_pool: Tuple[str, ...] = DEFAULT_PERSONAS


@dataclass
class RuntimeConfig:
    model_path: str
    out_dir: str
    n_games: int
    parallel_games: int
    base_seed: int
    n_ctx: int
    n_threads: int
    n_gpu_layers: int
    n_batch: int
    use_mmap: bool
    use_mlock: bool
    resume: bool
    compress_games: bool
    progress_every: int
    verbose: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mafia self-play with GGUF on local/Colab.")
    parser.add_argument("--model_path", required=True, help="Path to .gguf model file.")
    parser.add_argument("--out_dir", required=True, help="Output directory for logs.")
    parser.add_argument("--n_games", type=int, default=1000)
    parser.add_argument("--parallel_games", type=int, default=4, help="Use 0 to auto-select from CPU count.")
    parser.add_argument("--base_seed", type=int, default=12345)
    parser.add_argument("--n_ctx", type=int, default=2048)
    parser.add_argument("--n_threads", type=int, default=max(2, (os.cpu_count() or 4) // 2))
    parser.add_argument("--n_gpu_layers", type=int, default=-1)
    parser.add_argument("--n_batch", type=int, default=512)
    parser.add_argument("--no_mmap", action="store_true")
    parser.add_argument("--mlock", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip existing game files and continue.")
    parser.add_argument("--compress_games", action="store_true", help="Write games as .json.gz instead of .json.")
    parser.add_argument("--progress_every", type=int, default=25, help="Worker progress print interval.")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--players", type=int, default=9)
    parser.add_argument("--mafia_count", type=int, default=2, help="Total mafia team size (includes silencer).")
    parser.add_argument("--silencer_count", type=int, default=1)
    parser.add_argument("--doctor_count", type=int, default=1)
    parser.add_argument("--detective_count", type=int, default=1)
    parser.add_argument("--jester_count", type=int, default=1)
    parser.add_argument("--max_rounds", type=int, default=8)
    parser.add_argument("--talks_per_day", type=int, default=1)
    parser.add_argument("--night_mode", choices=["random", "model"], default="model")
    parser.add_argument("--no_tie_break_random", action="store_true")
    parser.add_argument("--no_shuffle_speaking_order", action="store_true")
    parser.add_argument("--persona_file", type=str, default="", help="Optional text file with one persona per line.")

    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--min_p", type=float, default=0.05)
    parser.add_argument("--repeat_penalty", type=float, default=1.1)
    parser.add_argument("--max_tokens_chat", type=int, default=96)
    parser.add_argument("--max_tokens_vote", type=int, default=24)
    return parser.parse_args()


def chunked(items: List[int], n_chunks: int) -> List[List[int]]:
    buckets = [[] for _ in range(max(1, n_chunks))]
    for idx, item in enumerate(items):
        buckets[idx % len(buckets)].append(item)
    return [b for b in buckets if b]


def choose_parallel_games(value: int) -> int:
    if value > 0:
        return value
    return max(1, min(16, os.cpu_count() or 2))


def load_persona_pool(persona_file: str) -> Tuple[str, ...]:
    if not persona_file:
        return DEFAULT_PERSONAS
    path = Path(persona_file)
    if not path.exists():
        raise FileNotFoundError(f"persona file not found: {persona_file}")
    personas = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text and not text.startswith("#"):
                personas.append(text)
    if not personas:
        raise ValueError(f"persona file has no valid entries: {persona_file}")
    return tuple(personas)


def load_model(runtime: RuntimeConfig, seed: int) -> Llama:
    return Llama(
        model_path=runtime.model_path,
        n_ctx=runtime.n_ctx,
        n_threads=runtime.n_threads,
        n_gpu_layers=runtime.n_gpu_layers,
        n_batch=runtime.n_batch,
        use_mmap=runtime.use_mmap,
        use_mlock=runtime.use_mlock,
        seed=seed,
        verbose=False,
    )


def model_generate(llm: Llama, prompt: str, sampling: SamplingConfig, max_tokens: int) -> str:
    out = llm.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        top_k=sampling.top_k,
        min_p=sampling.min_p,
        repeat_penalty=sampling.repeat_penalty,
        stop=["\n\n###", "\n[END]"],
    )
    return out["choices"][0]["text"].strip()


def assign_roles(game_cfg: GameConfig, rng: random.Random) -> Dict[str, str]:
    players = [f"P{i}" for i in range(1, game_cfg.players + 1)]
    roles = {p: ROLE_VILLAGER for p in players}
    remaining = players[:]

    mafia_team = rng.sample(remaining, game_cfg.mafia_count)
    for p in mafia_team:
        roles[p] = ROLE_MAFIA
    for p in mafia_team:
        remaining.remove(p)

    silencer_members = rng.sample(mafia_team, game_cfg.silencer_count) if game_cfg.silencer_count > 0 else []
    for p in silencer_members:
        roles[p] = ROLE_SILENCER

    def pick_and_assign(count: int, role: str) -> None:
        if count <= 0:
            return
        chosen = rng.sample(remaining, count)
        for p in chosen:
            roles[p] = role
            remaining.remove(p)

    pick_and_assign(game_cfg.doctor_count, ROLE_DOCTOR)
    pick_and_assign(game_cfg.detective_count, ROLE_DETECTIVE)
    pick_and_assign(game_cfg.jester_count, ROLE_JESTER)
    return roles


def win_status(alive: List[str], roles: Dict[str, str]) -> str:
    mafia_alive = sum(1 for p in alive if roles[p] in MAFIA_TEAM_ROLES)
    non_mafia_alive = len(alive) - mafia_alive
    if mafia_alive == 0:
        return "villagers"
    if mafia_alive >= non_mafia_alive:
        return "mafia"
    return ""


def talk_prompt(
    player: str,
    role: str,
    persona: str,
    day_round: int,
    talk_round: int,
    alive: List[str],
    discussion: List[Dict[str, str]],
    detective_memory: List[Dict[str, str]],
) -> str:
    last_talk = "\n".join(f"{x['speaker']}: {x.get('text', '')}" for x in discussion[-14:] if "speaker" in x)
    private_tip = ""
    if role in MAFIA_TEAM_ROLES:
        private_tip = "You are mafia-aligned. Avoid exposing mafia identities.\n"
    if role == ROLE_DETECTIVE and detective_memory:
        memories = ", ".join(f"{m['target']}={m['role']}" for m in detective_memory[-4:])
        private_tip += f"Private detective notes: {memories}\n"
    return (
        "You are in a Mafia game.\n"
        f"Your id: {player}\n"
        f"Your hidden role: {role}\n"
        f"Style: {persona}\n"
        f"Day round: {day_round}\n"
        f"Talk pass: {talk_round}\n"
        f"Alive players: {', '.join(alive)}\n"
        f"{private_tip}"
        "Recent discussion:\n"
        f"{last_talk if last_talk else '(none)'}\n\n"
        "Speak 1-2 short sentences to influence the vote. Do not reveal hidden role.\n"
        "[END]"
    )


def vote_prompt(
    player: str,
    role: str,
    persona: str,
    day_round: int,
    alive: List[str],
    discussion: List[Dict[str, str]],
) -> str:
    last_talk = "\n".join(f"{x['speaker']}: {x.get('text', '')}" for x in discussion[-14:] if "speaker" in x)
    return (
        "You are in a Mafia game.\n"
        f"Your id: {player}\n"
        f"Your hidden role: {role}\n"
        f"Style: {persona}\n"
        f"Day round: {day_round}\n"
        f"Alive players: {', '.join(alive)}\n"
        "Recent discussion:\n"
        f"{last_talk if last_talk else '(none)'}\n\n"
        "Pick one alive player (not yourself) to eliminate.\n"
        "Output strictly:\n"
        "VOTE: P<number>\n"
        "REASON: <short reason>\n"
        "[END]"
    )


def mafia_night_prompt(
    player: str,
    role: str,
    mafia_team: List[str],
    persona: str,
    night_round: int,
    alive: List[str],
    transcript: List[Dict[str, str]],
) -> str:
    last_talk = "\n".join(f"{x['speaker']}: {x.get('text', '')}" for x in transcript[-14:] if "speaker" in x)
    return (
        "You are in Mafia game NIGHT phase.\n"
        f"Your id: {player}\n"
        f"Your role: {role}\n"
        f"Mafia team alive: {', '.join(sorted(mafia_team))}\n"
        f"Style: {persona}\n"
        f"Night round: {night_round}\n"
        f"Alive players: {', '.join(alive)}\n"
        "Recent discussion:\n"
        f"{last_talk if last_talk else '(none)'}\n\n"
        "Pick one NON-mafia alive player to kill.\n"
        "Output strictly:\n"
        "VOTE: P<number>\n"
        "REASON: <short reason>\n"
        "[END]"
    )


def silencer_prompt(
    player: str,
    mafia_team: List[str],
    persona: str,
    night_round: int,
    alive: List[str],
    transcript: List[Dict[str, str]],
) -> str:
    last_talk = "\n".join(f"{x['speaker']}: {x.get('text', '')}" for x in transcript[-14:] if "speaker" in x)
    return (
        "You are Silencer in Mafia game NIGHT phase.\n"
        f"Your id: {player}\n"
        f"Mafia team alive: {', '.join(sorted(mafia_team))}\n"
        f"Style: {persona}\n"
        f"Night round: {night_round}\n"
        f"Alive players: {', '.join(alive)}\n"
        "Recent discussion:\n"
        f"{last_talk if last_talk else '(none)'}\n\n"
        "Pick one alive NON-mafia player to silence for next day round.\n"
        "Output strictly:\n"
        "VOTE: P<number>\n"
        "REASON: <short reason>\n"
        "[END]"
    )


def doctor_prompt(
    player: str,
    persona: str,
    night_round: int,
    alive: List[str],
    last_saved: str,
    transcript: List[Dict[str, str]],
) -> str:
    last_talk = "\n".join(f"{x['speaker']}: {x.get('text', '')}" for x in transcript[-14:] if "speaker" in x)
    restriction = f"You cannot save {last_saved} this night." if last_saved else "No save restriction this night."
    return (
        "You are Doctor in Mafia game NIGHT phase.\n"
        f"Your id: {player}\n"
        f"Style: {persona}\n"
        f"Night round: {night_round}\n"
        f"Alive players: {', '.join(alive)}\n"
        f"{restriction}\n"
        "Recent discussion:\n"
        f"{last_talk if last_talk else '(none)'}\n\n"
        "Pick one alive player to protect from night death.\n"
        "Output strictly:\n"
        "VOTE: P<number>\n"
        "REASON: <short reason>\n"
        "[END]"
    )


def detective_prompt(
    player: str,
    persona: str,
    night_round: int,
    alive: List[str],
    transcript: List[Dict[str, str]],
) -> str:
    last_talk = "\n".join(f"{x['speaker']}: {x.get('text', '')}" for x in transcript[-14:] if "speaker" in x)
    return (
        "You are Detective in Mafia game NIGHT phase.\n"
        f"Your id: {player}\n"
        f"Style: {persona}\n"
        f"Night round: {night_round}\n"
        f"Alive players: {', '.join(alive)}\n"
        "Recent discussion:\n"
        f"{last_talk if last_talk else '(none)'}\n\n"
        "Pick one alive player (not yourself) to investigate.\n"
        "Output strictly:\n"
        "VOTE: P<number>\n"
        "REASON: <short reason>\n"
        "[END]"
    )


def parse_vote(text: str, alive: List[str], self_player: str, rng: random.Random, allowed: List[str] = None) -> str:
    match = re.search(r"\bP\d+\b", text)
    allowed_set = set(allowed) if allowed else set(alive)
    if match:
        vote = match.group(0)
        if vote in alive and vote != self_player and vote in allowed_set:
            return vote
    fallback = [p for p in alive if p != self_player and p in allowed_set]
    return rng.choice(fallback) if fallback else self_player


def resolve_day_vote(votes: Dict[str, str], rng: random.Random, allow_random_tie: bool) -> str:
    counter = Counter(votes.values())
    top = counter.most_common()
    if not top:
        return ""
    high = top[0][1]
    tied = [target for target, n in top if n == high]
    if len(tied) == 1:
        return tied[0]
    return rng.choice(tied) if allow_random_tie else sorted(tied)[0]


def run_single_game(
    llm: Llama,
    game_id: int,
    game_cfg: GameConfig,
    sampling: SamplingConfig,
    seed: int,
) -> Dict:
    rng = random.Random(seed)
    players = [f"P{i}" for i in range(1, game_cfg.players + 1)]
    roles = assign_roles(game_cfg, rng)
    persona = {p: rng.choice(game_cfg.persona_pool) for p in players}
    alive = players[:]
    public_transcript: List[Dict[str, str]] = []
    night_private_events: List[Dict[str, str]] = []
    detective_memory: Dict[str, List[Dict[str, str]]] = {p: [] for p in players if roles[p] == ROLE_DETECTIVE}
    doctor_last_saved: Dict[str, str] = {p: "" for p in players if roles[p] == ROLE_DOCTOR}
    muted_today: set = set()
    winner = ""
    win_reason = ""

    for day_round in range(1, game_cfg.max_rounds + 1):
        discussion: List[Dict[str, str]] = []
        for talk_round in range(1, game_cfg.talks_per_day + 1):
            speaking_order = alive[:]
            if game_cfg.shuffle_speaking_order:
                rng.shuffle(speaking_order)
            for p in speaking_order:
                if p in muted_today:
                    discussion.append(
                        {
                            "phase": "day_talk",
                            "round": day_round,
                            "talk_round": talk_round,
                            "speaker": p,
                            "text": "(silenced)",
                            "silenced": True,
                        }
                    )
                    continue
                prompt = talk_prompt(
                    p,
                    roles[p],
                    persona[p],
                    day_round,
                    talk_round,
                    alive,
                    public_transcript + discussion,
                    detective_memory.get(p, []),
                )
                text = model_generate(llm, prompt, sampling, sampling.max_tokens_chat)
                item = {
                    "phase": "day_talk",
                    "round": day_round,
                    "talk_round": talk_round,
                    "speaker": p,
                    "text": text,
                    "silenced": False,
                }
                discussion.append(item)
                public_transcript.append(item)

        votes: Dict[str, str] = {}
        for p in alive:
            prompt = vote_prompt(p, roles[p], persona[p], day_round, alive, public_transcript)
            raw = model_generate(llm, prompt, sampling, sampling.max_tokens_vote)
            vote = parse_vote(raw, alive, p, rng)
            votes[p] = vote
            public_transcript.append({"phase": "day_vote", "round": day_round, "speaker": p, "raw": raw, "vote": vote})

        eliminated = resolve_day_vote(votes, rng, game_cfg.allow_tie_break_random)
        if eliminated and eliminated in alive:
            eliminated_role = roles[eliminated]
            alive.remove(eliminated)
            public_transcript.append(
                {
                    "phase": "day_elimination",
                    "round": day_round,
                    "target": eliminated,
                    "revealed_role": eliminated_role,
                }
            )
            if eliminated_role == ROLE_JESTER:
                winner = "jester"
                win_reason = f"{eliminated} (jester) was voted out during day."
                break

        winner = win_status(alive, roles)
        if winner:
            win_reason = "Standard parity/elimination win condition after day vote."
            break

        mafia_alive = [p for p in alive if roles[p] in MAFIA_TEAM_ROLES]
        non_mafia_targets = [p for p in alive if roles[p] not in MAFIA_TEAM_ROLES]
        silencer_alive = [p for p in alive if roles[p] == ROLE_SILENCER]
        doctors_alive = [p for p in alive if roles[p] == ROLE_DOCTOR]
        detectives_alive = [p for p in alive if roles[p] == ROLE_DETECTIVE]

        next_muted = set()
        for s in silencer_alive:
            if not non_mafia_targets:
                continue
            if game_cfg.night_mode == "model":
                raw = model_generate(
                    llm,
                    silencer_prompt(s, mafia_alive, persona[s], day_round, alive, public_transcript),
                    sampling,
                    sampling.max_tokens_vote,
                )
                target = parse_vote(raw, alive, s, rng, allowed=non_mafia_targets)
            else:
                target = rng.choice(non_mafia_targets)
                raw = f"VOTE: {target}"
            next_muted.add(target)
            night_private_events.append(
                {"phase": "silence", "round": day_round, "actor": s, "target": target, "raw": raw}
            )

        for d in detectives_alive:
            allowed = [p for p in alive if p != d]
            if not allowed:
                continue
            if game_cfg.night_mode == "model":
                raw = model_generate(
                    llm,
                    detective_prompt(d, persona[d], day_round, alive, public_transcript),
                    sampling,
                    sampling.max_tokens_vote,
                )
                target = parse_vote(raw, alive, d, rng, allowed=allowed)
            else:
                target = rng.choice(allowed)
                raw = f"VOTE: {target}"
            detected_role = roles[target]
            detective_memory[d].append({"round": day_round, "target": target, "role": detected_role})
            night_private_events.append(
                {
                    "phase": "detective_investigation",
                    "round": day_round,
                    "actor": d,
                    "target": target,
                    "result_role": detected_role,
                    "raw": raw,
                }
            )

        if non_mafia_targets and mafia_alive:
            mafia_votes = {}
            for m in mafia_alive:
                if game_cfg.night_mode == "model":
                    raw = model_generate(
                        llm,
                        mafia_night_prompt(m, roles[m], mafia_alive, persona[m], day_round, alive, public_transcript),
                        sampling,
                        sampling.max_tokens_vote,
                    )
                    target = parse_vote(raw, alive, m, rng, allowed=non_mafia_targets)
                else:
                    target = rng.choice(non_mafia_targets)
                    raw = f"VOTE: {target}"
                mafia_votes[m] = target
                night_private_events.append(
                    {"phase": "night_vote", "round": day_round, "speaker": m, "target": target, "raw": raw}
                )
            night_kill_target = resolve_day_vote(mafia_votes, rng, allow_random_tie=True)
        else:
            night_kill_target = ""

        doctor_saved = ""
        for d in doctors_alive:
            allowed = alive[:]
            last_saved = doctor_last_saved.get(d, "")
            if last_saved and len(allowed) > 1:
                allowed = [p for p in allowed if p != last_saved]
            if not allowed:
                allowed = alive[:]
            if game_cfg.night_mode == "model":
                raw = model_generate(
                    llm,
                    doctor_prompt(d, persona[d], day_round, alive, last_saved, public_transcript),
                    sampling,
                    sampling.max_tokens_vote,
                )
                saved = parse_vote(raw, alive, d, rng, allowed=allowed)
            else:
                saved = rng.choice(allowed)
                raw = f"VOTE: {saved}"
            doctor_last_saved[d] = saved
            doctor_saved = saved
            night_private_events.append(
                {"phase": "doctor_save", "round": day_round, "actor": d, "target": saved, "raw": raw}
            )
            break

        if night_kill_target and night_kill_target in alive:
            if night_kill_target == doctor_saved:
                public_transcript.append({"phase": "night_result", "round": day_round, "result": "no_death"})
            else:
                killed_role = roles[night_kill_target]
                alive.remove(night_kill_target)
                public_transcript.append(
                    {
                        "phase": "night_kill",
                        "round": day_round,
                        "target": night_kill_target,
                        "revealed_role": killed_role,
                    }
                )

        muted_today = {p for p in next_muted if p in alive}
        if muted_today:
            public_transcript.append(
                {"phase": "day_mute_applied", "round": day_round + 1, "targets": sorted(muted_today)}
            )

        winner = win_status(alive, roles)
        if winner:
            win_reason = "Standard parity/elimination win condition after night."
            break

    if not winner:
        winner = "draw"
        win_reason = "Reached max rounds without decisive win condition."

    return {
        "schema_version": SCHEMA_VERSION,
        "game_id": game_id,
        "seed": seed,
        "players": players,
        "roles": roles,
        "persona": persona,
        "winner": winner,
        "win_reason": win_reason,
        "rounds_played": max((x["round"] for x in public_transcript if "round" in x), default=0),
        "alive_end": alive,
        "public_transcript": public_transcript,
        "private_night_events": night_private_events,
    }


def game_path(out_dir: Path, game_id: int, compress_games: bool) -> Path:
    suffix = ".json.gz" if compress_games else ".json"
    return out_dir / "games" / f"game_{game_id:07d}{suffix}"


def write_game_file(path: Path, game_obj: Dict, compress_games: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress_games:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(game_obj, f, ensure_ascii=True)
    else:
        with path.open("w", encoding="utf-8") as f:
            json.dump(game_obj, f, ensure_ascii=True, indent=2)


def read_game_file(path: Path) -> Dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def existing_game_ids(out_dir: Path) -> set:
    games_dir = out_dir / "games"
    if not games_dir.exists():
        return set()
    seen = set()
    for p in games_dir.glob("game_*.json"):
        match = re.search(r"game_(\d+)\.json$", p.name)
        if match:
            seen.add(int(match.group(1)))
    for p in games_dir.glob("game_*.json.gz"):
        match = re.search(r"game_(\d+)\.json\.gz$", p.name)
        if match:
            seen.add(int(match.group(1)))
    return seen


def worker_run(worker_id: int, game_ids: List[int], runtime: RuntimeConfig, game_cfg: GameConfig, sampling: SamplingConfig) -> Dict:
    worker_seed = runtime.base_seed + worker_id * 100_000
    llm = load_model(runtime, worker_seed)
    out_dir = Path(runtime.out_dir)
    shard_path = out_dir / f"summary_worker_{worker_id:02d}_{int(time.time())}.jsonl"
    result_counts = Counter()
    done = 0

    with shard_path.open("a", encoding="utf-8") as shard:
        for local_idx, game_id in enumerate(game_ids):
            seed = worker_seed + game_id * 97 + local_idx
            game = run_single_game(llm, game_id, game_cfg, sampling, seed)
            file_path = game_path(out_dir, game_id, runtime.compress_games)
            write_game_file(file_path, game, runtime.compress_games)

            summary = {
                "game_id": game["game_id"],
                "winner": game["winner"],
                "rounds_played": game["rounds_played"],
                "seed": game["seed"],
                "file": str(file_path),
            }
            shard.write(json.dumps(summary, ensure_ascii=True) + "\n")
            result_counts[game["winner"]] += 1
            done += 1
            if runtime.progress_every > 0 and done % runtime.progress_every == 0:
                print(f"worker={worker_id} progress={done}/{len(game_ids)}", flush=True)

    return {"worker_id": worker_id, "games_done": len(game_ids), "counts": dict(result_counts)}


def write_run_manifest(runtime: RuntimeConfig, game_cfg: GameConfig, sampling: SamplingConfig, pending_games: int, skipped_games: int) -> None:
    out_dir = Path(runtime.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_unix": int(time.time()),
        "runtime": asdict(runtime),
        "game": asdict(game_cfg),
        "sampling": asdict(sampling),
        "pending_games_this_run": pending_games,
        "skipped_existing_games": skipped_games,
    }
    with (out_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)


def build_merged_summary(out_dir: Path) -> Tuple[int, Counter, float]:
    games_dir = out_dir / "games"
    files = sorted(list(games_dir.glob("game_*.json")) + list(games_dir.glob("game_*.json.gz")))
    counts = Counter()
    total_rounds = 0
    merged_jsonl = out_dir / "summary_all.jsonl"
    leaderboard_csv = out_dir / "leaderboard.csv"
    rows = []

    with merged_jsonl.open("w", encoding="utf-8") as out:
        for p in files:
            obj = read_game_file(p)
            summary = {
                "game_id": obj.get("game_id"),
                "winner": obj.get("winner", "unknown"),
                "rounds_played": obj.get("rounds_played", 0),
                "seed": obj.get("seed"),
                "file": str(p),
            }
            out.write(json.dumps(summary, ensure_ascii=True) + "\n")
            counts[summary["winner"]] += 1
            total_rounds += int(summary["rounds_played"])
            rows.append(summary)

    with leaderboard_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["game_id", "winner", "rounds_played", "seed", "file"])
        writer.writeheader()
        writer.writerows(rows)

    avg_rounds = (total_rounds / len(rows)) if rows else 0.0
    stats = {
        "games_total": len(rows),
        "winner_distribution": dict(counts),
        "avg_rounds": round(avg_rounds, 4),
    }
    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=True, indent=2)
    return len(rows), counts, avg_rounds


def validate_role_counts(game_cfg: GameConfig) -> None:
    if game_cfg.mafia_count <= 0 or game_cfg.mafia_count >= game_cfg.players:
        raise ValueError("--mafia_count must be > 0 and < --players")
    if game_cfg.silencer_count < 0 or game_cfg.silencer_count > game_cfg.mafia_count:
        raise ValueError("--silencer_count must be between 0 and --mafia_count")
    if game_cfg.doctor_count < 0 or game_cfg.detective_count < 0 or game_cfg.jester_count < 0:
        raise ValueError("role counts cannot be negative")
    special_non_mafia = game_cfg.doctor_count + game_cfg.detective_count + game_cfg.jester_count
    if game_cfg.mafia_count + special_non_mafia > game_cfg.players:
        raise ValueError("sum of mafia/special role counts exceeds --players")
    if game_cfg.talks_per_day <= 0:
        raise ValueError("--talks_per_day must be >= 1")


def main() -> None:
    args = parse_args()
    parallel_games = choose_parallel_games(args.parallel_games)
    runtime = RuntimeConfig(
        model_path=args.model_path,
        out_dir=args.out_dir,
        n_games=args.n_games,
        parallel_games=parallel_games,
        base_seed=args.base_seed,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
        n_batch=args.n_batch,
        use_mmap=not args.no_mmap,
        use_mlock=args.mlock,
        resume=args.resume,
        compress_games=args.compress_games,
        progress_every=args.progress_every,
        verbose=args.verbose,
    )
    personas = load_persona_pool(args.persona_file)
    game_cfg = GameConfig(
        players=args.players,
        mafia_count=args.mafia_count,
        silencer_count=args.silencer_count,
        doctor_count=args.doctor_count,
        detective_count=args.detective_count,
        jester_count=args.jester_count,
        max_rounds=args.max_rounds,
        talks_per_day=args.talks_per_day,
        allow_tie_break_random=not args.no_tie_break_random,
        shuffle_speaking_order=not args.no_shuffle_speaking_order,
        night_mode=args.night_mode,
        persona_pool=personas,
    )
    validate_role_counts(game_cfg)
    sampling = SamplingConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repeat_penalty=args.repeat_penalty,
        max_tokens_chat=args.max_tokens_chat,
        max_tokens_vote=args.max_tokens_vote,
    )

    out_dir = Path(runtime.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_game_ids = list(range(1, runtime.n_games + 1))
    skipped = 0
    if runtime.resume:
        done = existing_game_ids(out_dir)
        target_game_ids = [gid for gid in target_game_ids if gid not in done]
        skipped = runtime.n_games - len(target_game_ids)

    write_run_manifest(runtime, game_cfg, sampling, pending_games=len(target_game_ids), skipped_games=skipped)
    if not target_game_ids:
        print("No pending games to run. Building fresh summary from existing files.", flush=True)
        total, counts, avg_rounds = build_merged_summary(out_dir)
        print(f"games indexed: {total}")
        print(f"winner distribution: {dict(counts)}")
        print(f"avg rounds: {avg_rounds:.3f}")
        print(f"outputs at: {runtime.out_dir}")
        return

    batches = chunked(target_game_ids, runtime.parallel_games)
    start = time.time()
    with ProcessPoolExecutor(max_workers=runtime.parallel_games) as ex:
        futures = [
            ex.submit(worker_run, worker_id, batch, runtime, game_cfg, sampling)
            for worker_id, batch in enumerate(batches)
        ]
        for f in as_completed(futures):
            result = f.result()
            print(f"worker={result['worker_id']} done={result['games_done']} counts={result['counts']}", flush=True)

    elapsed = time.time() - start
    total, counts, avg_rounds = build_merged_summary(out_dir)
    print(f"run complete: new_games={len(target_game_ids)} total_indexed={total} elapsed_sec={elapsed:.2f}")
    print(f"winner distribution: {dict(counts)}")
    print(f"avg rounds: {avg_rounds:.3f}")
    print(f"outputs at: {runtime.out_dir}")


if __name__ == "__main__":
    main()
