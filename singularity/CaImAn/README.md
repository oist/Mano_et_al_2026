# Building Singularity image
```
sudo singularity build caiman_v2.sif caiman_v2.def
```

# Launching Singularity container
```
singularity run --bind `pwd`:/workdir caiman_v2.sif
```

```
singularity run -c --bind `pwd`:/workdir caiman_v2.sif
```

# Launching Singularity container on deigo
```
cd TO_YOUR_WORKINGDIR
ml singularity/3.5.2
srun -p compute -c 8 --mem=64G -t 0-3 --pty --x11 singularity run --bind `pwd`:/workdir /bucket/ReiterU/singularity/caiman_v2.sif
```

Saion (with GPU)
```
srun -p gpu -c 32 --mem=64G -t 0-12 --gres=gpu:1 --pty --x11 singularity run --nv --bind `pwd`:/workdir /bucket/ReiterU/singularity/caiman_v2.sif
```
