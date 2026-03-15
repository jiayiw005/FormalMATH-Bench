import argparse
import json
import os
import subprocess
import sys
import threading
import shutil
from pathlib import Path


# ── ANSI helpers ──────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
RED    = "\033[31m"
DIM    = "\033[2m"

def colored(text, *codes):
    return "".join(codes) + text + RESET


# ── Worker state (one per process) ───────────────────────────────────────────

class WorkerState:
    def __init__(self, worker_id: int, start: int, end: int):
        self.worker_id  = worker_id
        self.start      = start          # inclusive
        self.end        = end            # exclusive
        self.total      = end - start
        self.processed  = 0
        self.phase      = "queued"       # queued | importing | starting | running | done | error
        self.lock       = threading.Lock()

    # Called from the reader thread
    def update(self, line: str):
        line = line.strip()
        with self.lock:
            if "Importing Mathlib..." in line:
                self.phase = "importing"
            elif "Starting processing..." in line:
                self.phase = "starting"
            elif line.startswith("Processed theorem "):
                # "Processed theorem x/total"
                self.phase = "running"
                try:
                    fraction = line.split("Processed theorem ")[1]
                    numerator = int(fraction.split("/")[0])
                    self.processed = numerator
                except (IndexError, ValueError):
                    pass
            elif line.startswith("Finished processing"):
                self.phase = "done"
                self.processed = self.total

    @property
    def progress_fraction(self) -> float:
        return self.processed / self.total if self.total else 0.0

    @property
    def bar(self, width: int = 20) -> str:
        filled = int(self.progress_fraction * width)
        bar    = "█" * filled + "░" * (width - filled)
        pct    = self.progress_fraction * 100
        return f"[{bar}] {pct:5.1f}%"

    @property
    def phase_label(self) -> str:
        labels = {
            "queued":    colored("queued",    DIM),
            "importing": colored("importing…", YELLOW),
            "starting":  colored("starting…",  YELLOW),
            "running":   colored("running",    CYAN),
            "done":      colored("done ✓",     GREEN),
            "error":     colored("error ✗",    RED),
        }
        return labels.get(self.phase, self.phase)


# ── Display ───────────────────────────────────────────────────────────────────

class Display:
    """Redraws a fixed block of lines in-place using ANSI cursor control."""

    def __init__(self, num_workers: int, total_items: int):
        self.num_workers  = num_workers
        self.total_items  = total_items
        self._lines_drawn = 0
        self._lock        = threading.Lock()

    def _move_up(self, n: int):
        if n > 0:
            sys.stdout.write(f"\033[{n}A")

    def _clear_line(self):
        sys.stdout.write("\033[2K\r")

    def render(self, workers: list[WorkerState]):
        with self._lock:
            # Move cursor back to the top of our block
            self._move_up(self._lines_drawn)

            lines = []

            # ── Header ──
            total_processed = sum(w.processed for w in workers)
            overall_pct     = total_processed / self.total_items * 100 if self.total_items else 0
            overall_filled  = int(total_processed / self.total_items * 30) if self.total_items else 0
            overall_bar     = "█" * overall_filled + "░" * (30 - overall_filled)

            lines.append(
                colored(f" Overall  [{overall_bar}] {overall_pct:5.1f}%"
                        f"  {total_processed}/{self.total_items} items", BOLD)
            )
            lines.append(colored("─" * 68, DIM))

            # ── Per-worker rows ──
            for w in workers:
                with w.lock:
                    phase   = w.phase_label
                    bar     = w.bar
                    rng     = f"{w.start}–{w.end - 1}"
                    detail  = f"{w.processed}/{w.total}"
                lines.append(
                    f"  Worker {w.worker_id:>2}  {bar}  {detail:>12}  "
                    f"items [{rng}]  {phase}"
                )

            lines.append("")  # trailing blank line

            for line in lines:
                self._clear_line()
                sys.stdout.write(line + "\n")

            sys.stdout.flush()
            self._lines_drawn = len(lines)


# ── Core logic ────────────────────────────────────────────────────────────────

def chunk_ranges(total: int, n_workers: int) -> list[tuple[int, int]]:
    """Return (start, end) pairs that partition [0, total)."""
    base, remainder = divmod(total, n_workers)
    ranges, cursor = [], 0
    for i in range(n_workers):
        size = base + (1 if i < remainder else 0)
        ranges.append((cursor, cursor + size))
        cursor += size
    return ranges


def stream_worker(proc: subprocess.Popen, state: WorkerState,
                  display: Display, all_workers: list[WorkerState]):
    """Read stdout of a worker line-by-line and update state + display."""
    for raw in proc.stdout:
        line = raw.decode(errors="replace")
        state.update(line)
        display.render(all_workers)

    proc.wait()
    if proc.returncode != 0:
        with state.lock:
            state.phase = "error"
    display.render(all_workers)


def run(input_file: str, n_workers: int, output_dir: str):
    # ── Load input ──
    with open(input_file) as f:
        data = json.load(f)

    if not isinstance(data, list):
        sys.exit("Error: input JSON must be a top-level array.")

    total = len(data)
    if total == 0:
        sys.exit("Error: input JSON array is empty.")

    del data

    n_workers = min(n_workers, total)  # no idle workers
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(colored(f"\n  Input : {input_file}  ({total} items)", BOLD))
    print(colored(f"  Workers : {n_workers}", BOLD))
    print(colored(f"  Output  : {output_dir}\n", BOLD))

    ranges  = chunk_ranges(total, n_workers)
    workers = [WorkerState(i, s, e) for i, (s, e) in enumerate(ranges)]
    display = Display(n_workers, total)

    # Reserve display space before spawning anything
    display.render(workers)

    # ── Spawn workers ──
    procs, threads = [], []
    tmp_inputs = []   # keep references so files aren't deleted early

    for w in workers:
        out_file = os.path.join(output_dir, f"output_worker_{w.worker_id}.jsonl")

        cmd = [
            shutil.which("lake"),
            "exe",
            "repl",
            os.path.abspath(input_file),
            str(w.start),
            str(w.end),
            os.path.abspath(out_file),
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr so we catch everything
            cwd=os.path.abspath("repl")
        )
        procs.append(proc)

        t = threading.Thread(
            target=stream_worker,
            args=(proc, w, display, workers),
            daemon=True,
        )
        t.start()
        threads.append(t)

    # ── Wait ──
    for t in threads:
        t.join()

    # Clean up temp input slices
    for p in tmp_inputs:
        try:
            os.unlink(p)
        except OSError:
            pass

    # ── Summary ──
    errors = [w for w in workers if w.phase == "error"]
    if errors:
        print(colored(f"\n  ⚠  {len(errors)} worker(s) finished with errors.", RED + BOLD))
        sys.exit(1)
    else:
        print(colored(f"\n  ✓  All {n_workers} workers completed successfully.", GREEN + BOLD))
        print(colored(f"     Results written to: {output_dir}\n", DIM))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parallel aggregator for worker-based JSON processing."
    )
    parser.add_argument("--input",       required=True,
                        help="Path to the input JSON file (array of objects).")
    parser.add_argument("--workers",     type=int, default=4,
                        help="Number of parallel worker processes (default: 4).")
    parser.add_argument("--output-dir",  default="./output",
                        help="Directory for per-worker output files (default: ./output).")
    args = parser.parse_args()

    run(
        input_file  = args.input,
        n_workers   = args.workers,
        output_dir  = args.output_dir,
    )


if __name__ == "__main__":
    main()