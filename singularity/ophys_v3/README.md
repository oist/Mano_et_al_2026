# Building Singularity image
```
sudo singularity build ophys_v3.sif ophys_v3.def
```

## Installing PyNWB on Your Host System

PyNWB does not work well when installed inside the container, because it attempts to write cache files (`core_typemap.pkl`, etc.) into its own installation directory. This directory is read-only in standard Singularity `.sif` images.

Instead, install PyNWB outside the container into a writable directory in your home path:

```bash
pip install --target=/home/yourusername/pynwb_site pynwb
```

> Replace `/home/yourusername/pynwb_site` with a suitable path you own.

You will bind this directory into the container during execution and set the `PYTHONPATH` environment variable so that Python uses your host's writable copy of PyNWB.


# Launching Singularity container
```bash
hostPynwb=/home/t/tomoyuki-mano1/pynwb_site
singularity run \
  --bind "$(pwd)":/workdir \
  --bind "${hostPynwb}":/opt/pynwb_site \
  env PYTHONPATH=/opt/pynwb_site:$PYTHONPATH ophys_v3.sif
```

# Launching Singularity container on HPC with slurm

```bash
cd /path/to/your/workingdir
ml singularity
hostPynwb=/home/t/tomoyuki-mano1/pynwb_site

srun -p compute -c 8 --mem=64G -t 0-3 \
  singularity run \
    --bind "$(pwd)":/workdir \
    --bind "${hostPynwb}":/opt/pynwb_site \
    env PYTHONPATH=/opt/pynwb_site:$PYTHONPATH \
    /bucket/ReiterU/singularity/ophys_v3.sif
```
