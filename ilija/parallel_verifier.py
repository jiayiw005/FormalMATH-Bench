import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil


# ANSI helpers

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
RED    = "\033[31m"
DIM    = "\033[2m"

def col(text, *codes):
    return "".join(codes) + str(text) + RESET


# Process helpers

def _process_uss_kb(proc: psutil.Process) -> int:
    """USS (unique set size) — memory private to this process only.
    Falls back to RSS if USS is unavailable (some platforms/permissions)."""
    try:
        return proc.memory_full_info().uss // 1024
    except (psutil.AccessDenied, AttributeError):
        try:
            return proc.memory_info().rss // 1024
        except psutil.NoSuchProcess:
            return 0


def _group_uss_kb(root_pid: int) -> int:
    """Sum USS across the entire process tree rooted at root_pid."""
    try:
        root  = psutil.Process(root_pid)
        procs = [root] + root.children(recursive=True)
        return sum(_process_uss_kb(p) for p in procs)
    except psutil.NoSuchProcess:
        return 0


def _killpg(proc: subprocess.Popen):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass


# Result store (shared, thread-safe)
# Writes results to disk as they arrive; builds a sorted final file at the end.

class ResultStore:
    def __init__(self, total: int, results_path: str):
        self.total        = total
        self.results_path = results_path
        self.records      : dict[int, dict] = {}
        self._lock        = threading.Lock()
        # Open output file for streaming writes
        self._fh = open(results_path, "w")

    def _add_locked(self, record: dict):
        """Must be called with self._lock held."""
        idx = record.get("index")
        if idx is None or idx in self.records:
            return
        self.records[idx] = record
        # Write immediately so results are on disk as they arrive.
        # We'll re-sort at the end; for now order is arrival order.
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def add(self, record: dict):
        with self._lock:
            self._add_locked(record)

    def add_error(self, idx: int, reason: str):
        self.add({"index": idx, "error": reason, "answers": []})

    def has(self, idx: int) -> bool:
        with self._lock:
            return idx in self.records

    def close_and_sort(self):
        """Rewrite the output file in index order."""
        with self._lock:
            self._fh.close()
            records = dict(self.records)

        with open(self.results_path, "w") as f:
            for idx in range(self.total):
                rec = records.get(idx, {"index": idx, "error": "missing", "answers": []})
                f.write(json.dumps(rec) + "\n")

    @property
    def n_done(self) -> int:
        with self._lock:
            return len(self.records)

    @property
    def n_ok(self) -> int:
        with self._lock:
            return sum(1 for r in self.records.values() if "error" not in r)

    @property
    def n_error(self) -> int:
        with self._lock:
            return sum(1 for r in self.records.values() if "error" in r)


# Per-worker display state

class WorkerState:
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.start     = 0
        self.end       = 0
        self.processed = 0
        self.restarts  = 0
        self.rss_mb    = 0.0
        self.phase     = "idle"
        self.lock      = threading.Lock()

    def assign(self, start: int, end: int):
        with self.lock:
            self.start     = start
            self.end       = end
            self.processed = 0
            self.rss_mb    = 0.0
            self.phase     = "idle"

    def update_from_line(self, line: str):
        line = line.strip()
        with self.lock:
            if "Importing Mathlib" in line:
                self.phase = "importing"
            elif "Starting processing" in line:
                self.phase = "starting"
            elif line.startswith("Processed theorem "):
                self.phase = "running"
                try:
                    frac = line.split("Processed theorem ")[1]
                    self.processed = int(frac.split("/")[0])
                except (IndexError, ValueError):
                    pass
            elif line.startswith("Finished processing"):
                self.phase = "done"

    @property
    def total(self) -> int:
        return max(self.end - self.start, 0)

    @property
    def _frac(self) -> float:
        return self.processed / self.total if self.total else 0.0

    def render_bar(self, width: int = 16) -> str:
        filled = int(self._frac * width)
        bar    = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {self._frac * 100:5.1f}%"

    @property
    def phase_label(self) -> str:
        labels = {
            "idle":      col("idle",       DIM),
            "importing": col("importing…", YELLOW),
            "starting":  col("starting…",  YELLOW),
            "running":   col("running",    CYAN),
            "done":      col("done ✓",     GREEN),
            "crash":     col("crash ✗",    RED),
            "timeout":   col("timeout ✗",  RED),
            "oom":       col("oom ✗",      RED),
        }
        return labels.get(self.phase, self.phase)


# Display

class Display:
    def __init__(self, n_workers: int, total: int, mem_limit_mb: float):
        self.n_workers    = n_workers
        self.total        = total
        self.mem_limit_mb = mem_limit_mb
        self._lines_drawn = 0
        self._lock        = threading.Lock()

    def _up(self, n: int):
        if n > 0:
            sys.stdout.write(f"\033[{n}A")

    def render(self, workers: list, store: ResultStore):
        with self._lock:
            self._up(self._lines_drawn)
            lines = []

            n_done = store.n_done
            pct    = n_done / self.total * 100 if self.total else 0
            filled = int(n_done / self.total * 26) if self.total else 0
            bar    = "█" * filled + "░" * (26 - filled)
            lines.append(
                col(f" Overall  [{bar}] {pct:5.1f}%"
                    f"  {n_done}/{self.total}"
                    f"  ✓ {store.n_ok}  ✗ {store.n_error}", BOLD)
            )
            lines.append(col("─" * 76, DIM))

            for w in workers:
                with w.lock:
                    bar_str  = w.render_bar()
                    detail   = f"{w.processed}/{w.total}"
                    rng      = f"{w.start}–{w.end - 1}" if w.total else "—"
                    phase    = w.phase_label
                    restarts = col(f" ↺{w.restarts}", YELLOW) if w.restarts else ""
                    rss      = w.rss_mb

                if rss <= 0:
                    mem_str = col(f"{'—':>7}", DIM)
                else:
                    mem_pct = rss / self.mem_limit_mb
                    mc      = RED if mem_pct > 0.85 else YELLOW if mem_pct > 0.6 else CYAN
                    mem_str = col(f"{rss:6.0f}MB", mc)

                lines.append(
                    f"  Worker {w.worker_id:>2}  {bar_str}"
                    f"  {detail:>9}  [{rng}]"
                    f"  {mem_str}{restarts}  {phase}"
                )

            lines.append("")

            for line in lines:
                sys.stdout.write("\033[2K\r" + line + "\n")
            sys.stdout.flush()
            self._lines_drawn = len(lines)


# JSONL scratch-file reader

def _harvest_new_lines(path: str, offset: int) -> tuple[list[dict], int]:
    records = []
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            for raw in f:
                line = raw.decode(errors="replace").strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            new_offset = f.tell()
    except FileNotFoundError:
        new_offset = offset
    return records, new_offset


# Spawn

def _spawn(
    input_file  : str,
    start       : int,
    end         : int,
    scratch_out : str,
) -> subprocess.Popen:
    cmd = [
        shutil.which("lake"),
        "exe",
        "repl",
        input_file,
        str(start),
        str(end),
        scratch_out,
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd="repl",
    )


# Worker slot

def run_worker_slot(
    worker_id       : int,
    input_file      : str,
    start           : int,
    end             : int,
    output_dir      : str,
    timeout         : float,
    memory_limit_kb : int,
    state           : WorkerState,
    store           : ResultStore,
    display         : Display,
    all_states      : list,
):
    scratch_out = os.path.join(output_dir, f"scratch_worker_{worker_id}.jsonl")
    current_start = start

    while current_start < end:
        state.assign(current_start, end)
        display.render(all_states, store)

        proc          = _spawn(input_file, current_start, end, scratch_out)
        file_offset   = 0
        last_activity = [time.monotonic()]
        killed_reason = [None]
        absolute_idx  = [current_start]  # stamped onto harvested records

        # watchdog
        def watchdog(proc=proc, last_activity=last_activity,
                     killed_reason=killed_reason, state=state):
            while proc.poll() is None:
                time.sleep(1)

                if time.monotonic() - last_activity[0] > timeout:
                    killed_reason[0] = "timeout"
                    with state.lock:
                        state.phase = "timeout"
                    _killpg(proc)
                    return

                rss_kb = _group_uss_kb(proc.pid)
                with state.lock:
                    state.rss_mb = rss_kb / 1024

                display.render(all_states, store)

                if rss_kb > memory_limit_kb and state.phase not in ("idle", "importing", "starting"):
                    killed_reason[0] = "oom"
                    with state.lock:
                        state.phase = "oom"
                    _killpg(proc)
                    return

        threading.Thread(target=watchdog, daemon=True).start()

        # stdout reader
        for raw in proc.stdout:
            last_activity[0] = time.monotonic()
            line = raw.decode(errors="replace").strip()

            state.update_from_line(line)

            if line.startswith("Processed theorem "):
                new_records, file_offset = _harvest_new_lines(scratch_out, file_offset)
                for rec in new_records:
                    rec["index"] = absolute_idx[0]
                    absolute_idx[0] += 1
                    store.add(rec)
                    with state.lock:
                        state.processed += 1

            display.render(all_states, store)

        proc.wait()

        # final harvest
        new_records, file_offset = _harvest_new_lines(scratch_out, file_offset)
        for rec in new_records:
            rec["index"] = absolute_idx[0]
            absolute_idx[0] += 1
            store.add(rec)
            with state.lock:
                state.processed += 1
        display.render(all_states, store)

        # assess outcome
        if killed_reason[0] is None and proc.returncode != 0:
            killed_reason[0] = "crash"
            with state.lock:
                state.phase = "crash"

        something_went_wrong = killed_reason[0] is not None or proc.returncode != 0

        bad_idx = next(
            (i for i in range(current_start, end) if not store.has(i)),
            None,
        )

        if bad_idx is not None and something_went_wrong:
            reason = killed_reason[0] or "crash"
            store.add_error(bad_idx, reason)
            display.render(all_states, store)
            current_start = bad_idx + 1
            with state.lock:
                state.restarts += 1
                state.rss_mb    = 0.0
        else:
            break

    with state.lock:
        if state.phase not in ("crash", "timeout", "oom"):
            state.phase = "done"
        state.rss_mb = 0.0
    display.render(all_states, store)


# Chunking

def chunk_ranges(total: int, n: int) -> list[tuple[int, int]]:
    base, rem = divmod(total, n)
    ranges, cur = [], 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        ranges.append((cur, cur + size))
        cur += size
    return ranges


# Orchestration

def run(
    input_file      : str,
    n_workers       : int,
    output_dir      : str,
    timeout         : float,
    memory_limit_gb : float,
):
    input_file = str(Path(input_file).resolve())
    output_dir = str(Path(output_dir).resolve())

    with open(input_file) as f:
        data = json.load(f)

    if not isinstance(data, list):
        sys.exit("Error: input JSON must be a top-level array.")
    if not data:
        sys.exit("Error: input JSON array is empty.")

    total           = len(data)
    del data
    n_workers       = min(n_workers, total)
    memory_limit_kb = int(memory_limit_gb * 1024 * 1024)
    memory_limit_mb = memory_limit_gb * 1024

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(col(f"\n  Input    : {input_file}  ({total} items)", BOLD))
    print(col(f"  Workers  : {n_workers}", BOLD))
    print(col(f"  Timeout  : {timeout}s of silence per item", BOLD))
    print(col(f"  Mem limit: {memory_limit_gb:.1f} GB per worker", BOLD))
    print(col(f"  Output   : {output_dir}\n", BOLD))

    results_path = os.path.join(output_dir, "results.jsonl")
    store   = ResultStore(total, results_path)
    states  = [WorkerState(i) for i in range(n_workers)]
    display = Display(n_workers, total, memory_limit_mb)
    ranges  = chunk_ranges(total, n_workers)

    display.render(states, store)

    threads = [
        threading.Thread(
            target=run_worker_slot,
            args=(
                i, input_file, s, e, output_dir,
                timeout, memory_limit_kb, states[i], store, display, states,
            ),
            daemon=True,
        )
        for i, (s, e) in enumerate(ranges)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Rewrite results file in index order.
    store.close_and_sort()

    # Clean up scratch files.
    for i in range(n_workers):
        scratch = os.path.join(output_dir, f"scratch_worker_{i}.jsonl")
        try:
            os.unlink(scratch)
        except FileNotFoundError:
            pass

    # Summary
    n_ok    = store.n_ok
    n_error = store.n_error
    print(col(f"\n  ✓  {n_ok}/{total} theorems verified successfully", GREEN + BOLD))
    if n_error:
        by_reason: dict[str, list[int]] = {}
        for idx, rec in store.records.items():
            if "error" in rec:
                by_reason.setdefault(rec["error"], []).append(idx)
        for reason, idxs in sorted(by_reason.items()):
            preview = idxs[:8]
            suffix  = " …" if len(idxs) > 8 else ""
            print(col(f"  ✗  {len(idxs):>4} × {reason}: indices {preview}{suffix}", RED))
    print(col(f"\n     Results → {results_path}\n", DIM))


# CLI

def main():
    p = argparse.ArgumentParser(
        description="Parallel aggregator with crash/timeout/OOM recovery."
    )
    p.add_argument("--input",        required=True,
                   help="Input JSON file (array of objects).")
    p.add_argument("--workers",      type=int,   default=2,
                   help="Number of parallel worker processes (default: 2).")
    p.add_argument("--output-dir",   default="./output",
                   help="Directory for output files (default: ./output).")
    p.add_argument("--timeout",      type=float, default=120,
                   help="Seconds of stdout silence before killing a worker "
                        "(default: 120).")
    p.add_argument("--memory-limit", type=float, default=7.0,
                   help="Max USS in GB for the worker process tree before "
                        "it is killed (default: 6.0).")
    args = p.parse_args()
    run(
        args.input, args.workers,
        args.output_dir, args.timeout, args.memory_limit,
    )


if __name__ == "__main__":
    main()