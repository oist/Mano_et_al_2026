# Building Singularity image
```
sudo singularity build deepinterp_v1.sif deepinterp_v1.def
```

# Launching Singularity container
```
singularity run --bind `pwd`:/workdir deepinterp_v1.sif
```

# Launching Singularity container on saion

Saion (with GPU)
```
cd TO_YOUR_WORKINGDIR
ml singularity
srun -p gpu -c 12 --mem=64G -t 0-4 --gres=gpu:1 --pty --x11 singularity run --nv --bind `pwd`:/workdir /bucket/ReiterU/singularity/deepinterp_v1.sif
```
