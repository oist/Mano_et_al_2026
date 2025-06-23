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
rootDir=/bucket/ReiterU/OLdev/analysis/20250416_IGL_waves/20240614_rec4
# recName=rec2
rawImgPath=${rootDir}/raw/ch0.hdf5
corrImgPath=${rootDir}/mcorr_nonrigid/corrected.hdf5
denoisedImgPath=${rootDir}/mcorr_nonrigid_denoised/denoised.h5
saveDir=${rootDir}/mcorr_nonrigid/QC
####################

tempDir=$(mktemp -d /work/ReiterU/tmp/tomo.XXXXXX)
mkdir ${tempDir}/QC_raw
mkdir ${tempDir}/QC_nonrigid
mkdir ${tempDir}/QC_denoised

# main
singularity exec --bind `pwd`:/workdir /bucket/ReiterU/singularity/caiman_v2.sif \
python /home/t/tomoyuki-mano1/projects/OL_calcium/scripts/motion_corr_QC/calc_mc_metrics_v1.py \
--imgpath ${rawImgPath} \
--dataset_name "mov" \
--outdir ${tempDir}/QC_raw \
--crop 48

singularity exec --bind `pwd`:/workdir /bucket/ReiterU/singularity/caiman_v2.sif \
python /home/t/tomoyuki-mano1/projects/OL_calcium/scripts/motion_corr_QC/calc_mc_metrics_v1.py \
--imgpath ${corrImgPath} \
--dataset_name "mov" \
--outdir ${tempDir}/QC_nonrigid \
--crop 48

singularity exec --bind `pwd`:/workdir /bucket/ReiterU/singularity/caiman_v2.sif \
python /home/t/tomoyuki-mano1/projects/OL_calcium/scripts/motion_corr_QC/calc_mc_metrics_v1.py \
--imgpath ${denoisedImgPath} \
--dataset_name "data" \
--outdir ${tempDir}/QC_denoised \
--crop 48

echo "Copying files to /bucket..."
if [[ ! -e ${saveDir} ]]; then
    ssh saion mkdir -p ${saveDir}
fi
scp -r ${tempDir}/* saion:${saveDir}
echo "Done."

echo "Deleting temporary directory..."
rm -r ${tempDir}
echo "Done."
