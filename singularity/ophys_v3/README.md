# Building Singularity image
```
sudo singularity build ophys_v3.sif ophys_v3.def
```

# Launching Singularity container
```
singularity run --bind `pwd`:/workdir ophys_v3.sif
```

# Launching Singularity container on deigo

```
cd TO_YOUR_WORKINGDIR
ml singularity
srun -p compute -c 8 --mem=64G -t 0-3 singularity run --bind `pwd`:/workdir /bucket/ReiterU/singularity/ophys_v3.sif
```
