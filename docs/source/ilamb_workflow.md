# ILAMB Evaluation Workflow: CMORisation with Batch Processing

This guide covers the end-to-end workflow for preparing ACCESS-ESM1-5 model output
for evaluation with [ILAMB (International Land Model Benchmarking)](https://www.ilamb.org/).
The workflow uses ACCESS-MOPPy's batch processing system to CMORise multiple
land, atmosphere, and biogeochemistry variables in parallel on NCI's Gadi HPC.

---

## Overview

ILAMB evaluates land surface model performance against observational benchmarks.
It expects variables in CF-compliant NetCDF format with standard CMIP names and units.
ACCESS-MOPPy handles the conversion (CMORisation) from raw ACCESS-ESM1-5 output to
this format, with each variable submitted as an independent PBS job.

**Experiment:** `historical-02` (ACCESS-ESM1-6 production run)
**Variables covered:** 22 variables across Emon, Lmon, and Amon CMIP tables

> **Note on CMIP compliance:** ILAMB does not require strict CMIP6 publication
> compliance. CMORisation here is used to standardise variable names, units, and
> metadata — not for data submission.

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| NCI project access | `iq82`, `p73`, `tm70`, `xp65` |
| Storage access | `gdata/tm70`, `gdata/xp65`, `gdata/p73`, `scratch/tm70` |
| Software | `conda/analysis3-26.04` via `xp65` modules |
| Scheduler | PBS Pro (Gadi) |

Verify your project memberships before running:

```bash
nci_account -P iq82
nci_account -P p73
```

---

## Variables

The workflow processes 22 variables organised by CMIP table. Variables marked
with `*` require **daily** input files; all others use monthly input.

### Emon — Monthly ecosystem diagnostics

| Variable | Long name | Units |
|----------|-----------|-------|
| `cSoil` | Carbon mass in soil pool | kg m⁻² |
| `fBNF` | Biological nitrogen fixation | kg m⁻² s⁻¹ |

### Lmon — Monthly land surface

| Variable | Long name | Units |
|----------|-----------|-------|
| `cVeg` | Carbon mass in vegetation | kg m⁻² |
| `gpp` | Gross primary production | kg m⁻² s⁻¹ |
| `lai` | Leaf area index | 1 |
| `nbp` | Net biome production | kg m⁻² s⁻¹ |
| `ra` | Plant respiration | kg m⁻² s⁻¹ |
| `rh` | Heterotrophic respiration | kg m⁻² s⁻¹ |
| `tsl` | Temperature of soil layers | K |
| `mrro` | Total runoff | kg m⁻² s⁻¹ |

### Amon — Monthly atmosphere / surface energy balance

| Variable | Long name | Units |
|----------|-----------|-------|
| `evspsbl` | Evaporation | kg m⁻² s⁻¹ |
| `hfls` | Surface upward latent heat flux | W m⁻² |
| `hfss` | Surface upward sensible heat flux | W m⁻² |
| `hurs` | Near-surface relative humidity | % |
| `pr` | Precipitation | kg m⁻² s⁻¹ |
| `rlds` | Surface downwelling longwave radiation | W m⁻² |
| `rlus` | Surface upwelling longwave radiation | W m⁻² |
| `rsds` | Surface downwelling shortwave radiation | W m⁻² |
| `rsus` | Surface upwelling shortwave radiation | W m⁻² |
| `tas` | Near-surface air temperature | K |
| `tasmax` * | Daily maximum near-surface air temperature | K |
| `tasmin` * | Daily minimum near-surface air temperature | K |

> `tasmax` and `tasmin` are Amon-table variables but are derived from **daily**
> ACCESS output (`*dai.nc`). Their file patterns differ from the other Amon variables.

### Omon — Ocean (pending)

| Variable | Status |
|----------|--------|
| `hfds` (Downward heat flux at sea surface) | Pending — ocean file pattern not yet confirmed |

---

## Batch Configuration

Save the following as `batch_config_Feb26_PI_CNP.yml`. Update `output_folder`
before running.

```yaml
# Batch CMORisation configuration — Feb26-PI-CNP-concentrations (ACCESS-ESM1-6)
# Run with: python -m access_moppy.batch_cmoriser batch_config_Feb26_PI_CNP.yml

# Variables to process (one PBS job per variable)
variables:
  # --- Emon ---
  - Emon.cSoil
  - Emon.fBNF
  # --- Lmon ---
  - Lmon.cVeg
  - Lmon.gpp
  - Lmon.lai
  - Lmon.nbp
  - Lmon.ra
  - Lmon.rh
  - Lmon.tsl
  - Lmon.mrro
  # --- Amon ---
  - Amon.evspsbl
  - Amon.hfls
  - Amon.hfss
  - Amon.hurs
  - Amon.pr
  - Amon.rlds
  - Amon.rlus
  - Amon.rsds
  - Amon.rsus
  - Amon.tasmax
  - Amon.tasmin
  - Amon.tas
  # --- Omon ---
  # Omon.hfds  # TODO: add ocean file pattern once known

# CMIP6 metadata
experiment_id: historical
source_id: ACCESS-ESM1-5
variant_label: r1i1p1f1
grid_label: gn
activity_id: CMIP

# Input and output paths
input_folder: "/g/data/p73/archive/CMIP7/ACCESS-ESM1-6/production/historical-02"
output_folder: "YOUR_OUTPUT_PATH"

# File patterns (relative to input_folder)
# All atmosphere/land variables share the same pattern
file_patterns:
  Emon.cSoil:   "/output1*/atmosphere/netCDF/*mon.nc"
  Emon.fBNF:    "/output1*/atmosphere/netCDF/*mon.nc"
  Lmon.cVeg:    "/output1*/atmosphere/netCDF/*mon.nc"
  Lmon.gpp:     "/output1*/atmosphere/netCDF/*mon.nc"
  Lmon.lai:     "/output1*/atmosphere/netCDF/*mon.nc"
  Lmon.nbp:     "/output1*/atmosphere/netCDF/*mon.nc"
  Lmon.ra:      "/output1*/atmosphere/netCDF/*mon.nc"
  Lmon.rh:      "/output1*/atmosphere/netCDF/*mon.nc"
  Lmon.tsl:     "/output1*/atmosphere/netCDF/*mon.nc"
  Lmon.mrro:    "/output1*/atmosphere/netCDF/*mon.nc"
  Amon.evspsbl: "/output1*/atmosphere/netCDF/*mon.nc"
  Amon.hfls:    "/output1*/atmosphere/netCDF/*mon.nc"
  Amon.hfss:    "/output1*/atmosphere/netCDF/*mon.nc"
  Amon.hurs:    "/output1*/atmosphere/netCDF/*mon.nc"
  Amon.pr:      "/output1*/atmosphere/netCDF/*mon.nc"
  Amon.rlds:    "/output1*/atmosphere/netCDF/*mon.nc"
  Amon.rlus:    "/output1*/atmosphere/netCDF/*mon.nc"
  Amon.rsds:    "/output1*/atmosphere/netCDF/*mon.nc"
  Amon.rsus:    "/output1*/atmosphere/netCDF/*mon.nc"
  Amon.tasmax:  "/output1*/atmosphere/netCDF/*dai.nc"
  Amon.tasmin:  "/output1*/atmosphere/netCDF/*dai.nc"
  Amon.tas:     "/output1*/atmosphere/netCDF/*mon.nc"

# PBS job configuration (defaults for all variables)
queue: "normal"
cpus_per_node: 12
mem: "190GB"
jobfs: 100GB
walltime: "02:00:00"
scheduler_options: "#PBS -P iq82"
storage: "gdata/tm70+gdata/xp65+gdata/p73+scratch/tm70"

# Environment setup
worker_init: |
  module use /g/data/xp65/public/modules
  module load conda/analysis3-26.04

wait_for_completion: false
```

### Key configuration notes

**File patterns**

The pattern `/output1*/atmosphere/netCDF/*mon.nc` uses shell globbing:

- `output1*` — matches all restart chunks beginning with `output1`
  (e.g. `output100`, `output101`, …). Adjust the prefix if your archive uses
  a different naming scheme.
- `*mon.nc` — monthly-frequency files. Daily files (`*dai.nc`) are used only
  for `tasmax` and `tasmin`.

**Resource allocation**

Each job requests `12 CPUs` and `190 GB` of memory. This is sized for
land/atmosphere monthly data at N96 resolution. If you add 3-D ocean variables
(e.g. `Omon.hfds`) you may need per-variable overrides:

```yaml
variable_resources:
  Omon.hfds:
    cpus_per_node: 28
    mem: "256GB"
    walltime: "04:00:00"
```

**`wait_for_completion: false`**

The controller submits all 22 jobs and exits immediately. Jobs run
independently on Gadi. Use the tracking database or PBS commands to monitor
progress (see [Monitoring](#monitoring) below).

---

## Running the Workflow

### 1. Set your output path

Replace `YOUR_OUTPUT_PATH` in the config with a path on scratch, e.g.:

```bash
output_folder: "/scratch/tm70/$USER/ilamb_cmorised/historical-02"
```

### 2. Submit the batch

From a Gadi login node, with ACCESS-MOPPy available in your environment:

```bash
module use /g/data/xp65/public/modules
module load conda/analysis3-26.04

moppy-cmorise batch_config_Feb26_PI_CNP.yml
```

This will:
1. Parse and validate the configuration
2. Create a SQLite tracking database at `output_folder/cmor_tasks.db`
3. Generate PBS and Python scripts under `output_folder/cmor_job_scripts/`
4. Submit 22 PBS jobs (one per variable) via `qsub`
5. Print submitted job IDs and exit

### 3. (Optional) Launch the monitoring dashboard

In a separate terminal session (or persistent `tmux`/`screen`):

```bash
moppy-dashboard
```

Open a browser tunnel to `http://localhost:8501` to view real-time job status,
start/end times, and error messages.

---

## Monitoring

### PBS commands

```bash
# List your running jobs
qstat -u $USER

# Check a specific job
qstat -f <job_id>

# Watch overall queue
qstat -q normal
```

### Tracking database

```python
import sqlite3, pandas as pd

conn = sqlite3.connect("/scratch/tm70/$USER/ilamb_cmorised/historical-02/cmor_tasks.db")

df = pd.read_sql_query("""
    SELECT variable, status, start_time, end_time, error_message
    FROM cmor_tasks
    ORDER BY start_time
""", conn)

print(df)
```

Possible `status` values: `pending` → `running` → `completed` / `failed`.

### Log files

Each job writes stdout and stderr under the `cmor_job_scripts/` directory:

```
cmor_job_scripts/
├── cmor_Amon_pr.sh       # Generated PBS script
├── cmor_Amon_pr.py       # Generated Python script
├── cmor_Amon_pr.out      # stdout
└── cmor_Amon_pr.err      # stderr (check here first on failure)
```

---

## Error Recovery

Failed jobs can be resubmitted by re-running the same command:

```bash
moppy-cmorise batch_config_Feb26_PI_CNP.yml
```

Completed variables are skipped automatically; only `failed` or `pending`
variables are resubmitted.

To manually reset a failed variable in the database:

```python
conn.execute("""
    UPDATE cmor_tasks
    SET status = 'pending', start_time = NULL, end_time = NULL, error_message = NULL
    WHERE variable = 'Amon.tas' AND status = 'failed'
""")
conn.commit()
```

### Common failures

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `FileNotFoundError` in `.err` | File pattern matches nothing | Check `output1*` prefix against actual archive directory names |
| `MemoryError` / job killed | Data larger than `mem` allocation | Increase `mem` or reduce `cpus_per_node` |
| Job never starts (stuck `Q`) | Insufficient project allocation | Run `nci_account -P iq82` to check SU balance |
| Module not found in job | `worker_init` not sourcing correctly | Test `module use` command interactively on a compute node |
| `tasmax`/`tasmin` empty output | Wrong file pattern | Confirm daily files exist at `*dai.nc`; adjust glob if needed |

---

## Output Structure

CMORised files are written to `output_folder` following the CMIP6 DRS path
convention (when `drs_root` is set) or a flat structure otherwise:

```
YOUR_OUTPUT_PATH/
├── CMIP/
│   └── ACCESS-ESM1-5/
│       └── historical/
│           └── r1i1p1f1/
│               ├── Amon/
│               │   ├── pr/
│               │   ├── tas/
│               │   └── ...
│               └── Lmon/
│                   ├── gpp/
│                   └── ...
└── cmor_tasks.db
```

These files can be passed directly to ILAMB via its `ILAMB_ROOT` configuration.

---

## Adding Omon.hfds

Once the ocean file path is confirmed, uncomment the variable in the config
and add a file pattern:

```yaml
variables:
  # ... existing variables ...
  - Omon.hfds

file_patterns:
  # ... existing patterns ...
  Omon.hfds: "/output1*/ocean/netCDF/*mon.nc"   # update pattern as needed
```

Ocean variables typically require more memory. Add a resource override:

```yaml
variable_resources:
  Omon.hfds:
    cpus_per_node: 28
    mem: "256GB"
    walltime: "04:00:00"
```

---

## Quick Reference

```bash
# Submit all jobs
moppy-cmorise batch_config_Feb26_PI_CNP.yml

# Check job status
qstat -u $USER

# View a failed job's error
cat cmor_job_scripts/cmor_Lmon_gpp.err

# Resubmit failed jobs (completed jobs are skipped automatically)
moppy-cmorise batch_config_Feb26_PI_CNP.yml

# Get an example config to reference
moppy-example-config
```
