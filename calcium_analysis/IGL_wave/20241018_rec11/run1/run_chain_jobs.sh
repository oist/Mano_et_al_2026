#!/bin/bash

# first, submit a rigid motion correction job
jobstr1=$(sbatch motion_correction_rigid.slurm)
jobid1=${jobstr1##Submitted batch job }
echo $jobid1

if [ -z "$jobid1" ]; then
    echo "Failed to submit the rigid motion correction job."
    exit 1
fi

# next, run deepinterpolation on the rigid-corrected movie
jobstr2=$(sbatch --dependency=afterok:${jobid1} interpolation_pre.slurm)
jobid2=${jobstr2##Submitted batch job }
echo $jobid2

if [ -z "$jobid2" ]; then
    echo "Failed to submit the deepinterpolation (pre) job."
    exit 1
fi

# next, run non-rigid motion correction
jobstr3=$(sbatch --dependency=afterok:${jobid1}:${jobid2} motion_correction_nonrigid.slurm)
jobid3=${jobstr3##Submitted batch job }
echo $jobid3

if [ -z "$jobid3" ]; then
    echo "Failed to submit the non-rigid motion correction job."
    exit 1
fi

# finally, run deepinterpolation on the non-rigid corrected movie
jobstr4=$(sbatch --dependency=afterok:${jobid1}:${jobid2}:${jobid3} interpolation_post.slurm)
jobid4=${jobstr4##Submitted batch job }
echo $jobid4

if [ -z "$jobid4" ]; then
    echo "Failed to submit the deepinterpolation (post) job."
    exit 1
fi

# finally, run deepinterpolation on the non-rigid corrected movie
jobstr5=$(sbatch --dependency=afterok:${jobid1}:${jobid2}:${jobid3}:${jobid4} calc_mc_metrics.slurm)
jobid5=${jobstr5##Submitted batch job }
echo $jobid5

if [ -z "$jobid4" ]; then
    echo "Failed to submit the deepinterpolation (post) job."
    exit 1
fi
