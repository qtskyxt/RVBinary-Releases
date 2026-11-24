# HPC Execution

## MPI Modes

`execution.hdf5_mode: shards` is the portable default. Each rank writes its own
immutable sample-range shards; rank 0 then creates external-link indexes. Rank
count may change between runs because sample random streams are rank independent.

`execution.hdf5_mode: mpio` uses collective HDF5 writes. It requires `mpi4py`
and `h5py.get_config().mpi == True`, built against the same MPI implementation.
Use `environment-mpich.yml` or `environment-openmpi.yml` as a starting point.
MPI-enabled HDF5 is useful for initial mock generation; append-new-star mode
uses new shards so existing data remains immutable and recovery is inexpensive.

Set threaded numerical libraries to one thread per rank:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mpiexec -n 32 rvbinary mock rvbinary.yaml
mpiexec -n 32 rvbinary pdf rvbinary.yaml
rvbinary mcmc rvbinary.yaml
```

The PDF job accumulates local histograms and performs MPI reductions; rank 0
writes the small PDF products. MCMC is currently a rank-0/serial stage.

## PBS and Slurm

Configure `scheduler.walltime`, `memory`, `queue`, `account`, and
`conda_environment`, then render an editable script:

```bash
rvbinary submit rvbinary.yaml --scheduler pbs --output rvbinary.pbs
rvbinary submit rvbinary.yaml --scheduler slurm --output rvbinary.slurm
```

Review site-specific resource directives and the cluster MPI launcher before
submitting with `qsub rvbinary.pbs` or `sbatch rvbinary.slurm`. The command only
renders a file and never submits a job.

Batch shells often do not load the interactive shell configuration that defines
`conda`. In that case set `scheduler.conda_init` to the absolute path of
`etc/profile.d/conda.sh` and use either an environment name or absolute path in
`scheduler.conda_environment`:

```yaml
scheduler:
  conda_init: /opt/anaconda3/etc/profile.d/conda.sh
  conda_environment: /home/user/.conda/envs/rvbinary
```

## Browser generator through SSH

On your cluster, activate the environment where RVBinary is installed and add
the optional interface:

```bash
python -m pip install '.[gui]'
rvbinary gui --port 8501
```

Run the installation command from the repository root. In a separate local
terminal, replace `user@cluster` with your cluster login:

```bash
ssh -N -L 8501:127.0.0.1:8501 user@cluster
```

Open `http://127.0.0.1:8501` locally. Both GUI and tunnel must remain running;
stop each with Ctrl-C. If local port 8501 is occupied, forward
`8502:127.0.0.1:8501` and open local port 8502. The GUI binds to loopback and
performs small serial orbit experiments (at most 10,000), not production mock
generation. Respect the site's rules for interactive computation.

For an established MPI environment, first review the dependency resolution or
use a separate GUI environment; avoid replacing the cluster's MPI-enabled h5py
or MPI libraries merely to run the browser interface.

## Recovery and Storage

Keep indexes and their referenced shard directories together. External links
are relative, so the complete output directory can be moved as one tree.
`rvbinary inspect FILE` reports missing links, malformed payloads, sample IDs,
and resumability. `--merge-only` rebuilds indexes in metadata time; it is the
preferred recovery operation for large stores. Physical compaction scales with
the full data volume and should be deferred until needed for transfer/archive.

## MS17 on MPI workers

Install RVBinary 0.3.1 or newer consistently on all workers. The default MS17 trial
table ships with the package; do not copy or mount a separate Gravity data path.
If `model.period_table` explicitly overrides it, every rank must be able to read
matching contents. Rank preflight compares model/table fingerprints and reports
mismatches before generation. The same trial domains apply in serial, shards and
MPI-HDF5. Keep a new output directory for a different model/package version.
See [the MS17 example](../examples/ms17-trial.yaml).
