#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 0-12
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=tomoyuki.mano1@oist.jp

#modules
module load singularity

#### Parameters ####
rootDir=/bucket/ReiterU/OLdev/experiments/20250520_squid_cal520
recName=rec5
rawImgPath=${rootDir}/2P/${recName}/raw/ch0.hdf5
corrImgPath=${rootDir}/2P/${recName}/mcorr_nonrigid/corrected.hdf5
denoisedImgPath=${rootDir}/2P/${recName}/mcorr_nonrigid_denoised/denoised.h5
saveDir=${rootDir}/2P/${recName}/mcorr_nonrigid/QC

singularityImage=/bucket/ReiterU/singularity/ophys_v3.sif
scriptDir=/home/t/tomoyuki-mano1/projects/OL_calcium/scripts/NoRMCorre
####################

tempDir=$(mktemp -d /work/ReiterU/tmp/tomo.XXXXXX)
mkdir ${tempDir}/QC_raw
mkdir ${tempDir}/QC_nonrigid
mkdir ${tempDir}/QC_denoised

# raw image
singularity exec --bind `pwd`:/workdir ${singularityImage} \
python ${scriptDir}/calc_mc_metrics_v1.py \
--imgpath ${rawImgPath} \
--dataset_name "mov" \
--outdir ${tempDir}/QC_raw \
--crop 32

# non-rigid corrected image
singularity exec --bind `pwd`:/workdir ${singularityImage} \
python ${scriptDir}/calc_mc_metrics_v1.py \
--imgpath ${corrImgPath} \
--dataset_name "mov" \
--outdir ${tempDir}/QC_nonrigid \
--crop 32

# non-rigid corrected and denoised image
singularity exec --bind `pwd`:/workdir ${singularityImage} \
python ${scriptDir}/calc_mc_metrics_v1.py \
--imgpath ${denoisedImgPath} \
--dataset_name "data" \
--outdir ${tempDir}/QC_denoised \
--crop 32

echo "Copying files to /bucket..."
if [[ ! -e ${saveDir} ]]; then
    ssh saion mkdir -p ${saveDir}
fi
scp -r ${tempDir}/* saion:${saveDir}
echo "Done."

echo "Deleting temporary directory..."
rm -r ${tempDir}
echo "Done."
