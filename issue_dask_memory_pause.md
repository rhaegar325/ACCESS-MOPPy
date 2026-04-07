# Bug: Batch CMORisation PBS jobs hang indefinitely due to Dask worker pause

## Summary

All PBS subjobs submitted via `moppy-cmorise` hang permanently and are killed by PBS walltime
exhaustion. CPU time is only ~16 minutes but walltime reaches the 1-hour limit. The jobs produce
no meaningful output and no error message, making the failure silent and hard to diagnose.

## Environment

- Platform: NCI Gadi (`normal` queue)
- PBS resources: `ncpus=14`, `mem=32GB`, `jobfs=100GB`, `walltime=01:00:00`
- Module: `conda/analysis3-26.01`
- Affected script: generated `cmor_<variable>.py` via `moppy-cmorise`

## Observed behaviour

**PBS resource usage (typical stuck job):**
```
Exit Status:        -29 (Job failed due to exceeding walltime)
NCPUs Requested:    14        CPU Time Used: 00:16:34
Memory Requested:   32.0GB    Memory Used:  11.0GB
Walltime Requested: 01:00:00  Walltime Used: 01:00:21
JobFS Requested:    100.0GB   JobFS Used:   6.08KB
```

**Key observations:**
- CPU time ≈ 16 min; walltime ≈ 60 min → job is alive but doing nothing for ~44 minutes
- Memory used = 11 GB (well within 32 GB allocation)
- JobFS used = near zero (no Dask spill to disk occurring)
- Python stdout is empty (buffer never flushed; process killed by SIGKILL)

**Dask log pattern in `.err` file:**
```
15:41:54 - 7 workers started, each Memory: 2.79 GiB
15:45:24 - Worker is at 95% memory usage. Pausing worker.
           Process memory: 2.65 GiB -- Worker memory limit: 2.79 GiB
           [repeated for all 7 workers within 1 second]
15:50 → 16:41 - Unmanaged memory: 2.68 GiB (repeated every 5 min, never changes)
16:42 - PBS: job killed: walltime exceeded
```

## Root cause

### `detect_optimal_dask_config()` produces incorrect configuration for `processes=False`

**File:** `src/access_moppy/templates/cmor_python_script.j2`, lines 17–62

The function calculates `memory_per_worker` by dividing total PBS memory by the number of
workers — a formula designed for `processes=True` (separate OS processes). However, the Dask
client is created with `processes=False` (single-process thread mode), where **all workers share
the same Python process and the same memory pool**.

In `processes=False` mode, Dask's memory monitor reads the **total process RSS** via `psutil`
and compares it directly to each individual worker's `memory_limit`:

```
pause condition:  process_total_RSS  >  memory_limit × 0.95
```

Because `memory_limit = total_memory / n_workers`, the effective pause threshold becomes:

```
pause threshold = (total_memory / n_workers) × 0.95
```

With 14 CPUs and 32 GB, the medium-memory branch produces:

```python
n_workers         = max(2, 14 // 2) = 7
threads_per_worker = 2
memory_per_worker  = int(32 // 7 - 1) = 3 GB   →   2.79 GiB
```

| Metric | Value |
|--------|-------|
| PBS allocation | 32 GB |
| Dask pause threshold | 3 GB × 0.95 = **2.85 GB ≈ 2.65 GiB** |
| conda env + Python baseline RSS | **~2.65 GiB** |
| Memory available for actual work | **≈ 0** |

The process hits the pause threshold as soon as the conda environment and libraries finish
loading — before any data is read. All 7 workers pause simultaneously and permanently. The
`cmoriser.run()` call submits tasks to the scheduler, which cannot dispatch them to paused
workers, and blocks forever.

### All four branches of `detect_optimal_dask_config()` are affected

| Branch | n_workers | memory_per_worker | pause threshold | % of PBS used |
|--------|-----------|-------------------|-----------------|---------------|
| `< 32 GB` (low) | 1 | `total - 1` | ~95% of total | ✅ Correct |
| `32–64 GB` (medium) | 7 | `total / 7 - 1 ≈ 3 GB` | **~9% of total** | ❌ |
| `64–128 GB` (high) | 3 | `total / 3 - 2 ≈ 19 GB` | **~28% of total** | ❌ |
| `≥ 128 GB` (very high) | 2 | `total / 2 - 4 ≈ 60 GB` | **~45% of total** | ❌ |

The low-memory branch (`n_workers=1`) happens to be correct by accident. Every other branch
silently under-provisions memory.

### Additional issues in the same function

**Thread count can exceed PBS CPU allocation (very high memory branch):**
```python
# With n_cpus=14, ≥128 GB branch:
n_workers = max(2, 14 // 8) = 2
threads_per_worker = 8
# Total threads = 2 × 8 = 16  >  PBS_NCPUS=14
```

**`chunk_size` is dead code:**
```python
return {
    ...
    'chunk_size': '256MB'   # returned but never read by main()
}
```

## Proposed fix

Replace the entire body of `detect_optimal_dask_config()` with a single configuration that is
correct for `processes=False`:

```python
def detect_optimal_dask_config():
    """Detect optimal Dask configuration for processes=False single-node mode."""
    n_cpus = int(os.environ.get('PBS_NCPUS', psutil.cpu_count()))
    allocated_memory_gb = int(os.environ.get('PBS_MEM_GB', '16'))
    system_memory_gb = psutil.virtual_memory().total / (1024**3)
    effective_memory = min(allocated_memory_gb, system_memory_gb * 0.9)

    print(f"Detected: {n_cpus} CPUs, {allocated_memory_gb}GB allocated, {system_memory_gb:.1f}GB system")

    # processes=False: all workers share one process.
    # memory_limit must reflect the full process budget, not a per-worker fraction.
    # Use n_workers=1 so the single worker's limit equals the full allocation.
    # All CPUs are exposed as threads for parallel I/O.
    return {
        'n_workers': 1,
        'threads_per_worker': n_cpus,
        'memory_per_worker': f"{int(effective_memory * 0.9)}GB",
    }
```

**Effect with 14 CPU / 32 GB:**

| Metric | Before | After |
|--------|--------|-------|
| n_workers | 7 | 1 |
| threads | 7 × 2 = 14 | 14 (unchanged) |
| memory_limit | 3 GB | 28.8 GB |
| pause threshold | **2.65 GiB (= baseline RSS)** | **26.8 GiB** |
| headroom for data | ~0 | ~24 GiB |

## Related issues found during investigation

### 1. Double `client.close()` — `cmor_python_script.j2`

`base.py write()` may call `client.close()` internally (when `data_size > worker_memory`).
The `finally` block in `main()` then calls `client.close()` again on an already-closed client,
causing a hang while waiting for a TCP response from a dead scheduler.

**Fix (already applied):** guard the `finally` close with a status check:
```python
finally:
    if client.status != 'closed':
        client.close()
```

### 2. `psutil.virtual_memory().available` reads node-wide memory — `base.py`

`write()` uses `psutil.virtual_memory().available` to decide whether to load data into memory.
On shared Gadi nodes (192 GB total), this returns the node's free memory, not the PBS job's
allocation. A job with 16 GB PBS allocation may see 140 GB "available" and attempt to load the
full dataset at once, exceeding its PBS limit.

**Fix:** use `PBS_MEM_GB` (already exported by `cmor_job_script.j2`) as the memory budget.

### 3. `_execute_with_retry` is dead code — `tracking.py`

A retry helper with exponential backoff was written but never called. All public write methods
(`mark_running`, `mark_done`, `mark_failed`) use plain `with self.conn:` with no retry logic.
Concurrent PBS jobs writing to the shared SQLite database on Lustre will fail immediately with
`sqlite3.OperationalError: database is locked`, causing silent job failure.

**Fix:** replace `with self.conn: self.conn.execute(...)` in all write methods with
`self._execute_with_retry(...)`.

## Files to change

| File | Lines | Change |
|------|-------|--------|
| `src/access_moppy/templates/cmor_python_script.j2` | 17–62 | Rewrite `detect_optimal_dask_config()` |
| `src/access_moppy/base.py` | 831, 871 | Replace `psutil.virtual_memory().available` with `PBS_MEM_GB` |
| `src/access_moppy/tracking.py` | 49–84 | Use `_execute_with_retry` in all write methods |
