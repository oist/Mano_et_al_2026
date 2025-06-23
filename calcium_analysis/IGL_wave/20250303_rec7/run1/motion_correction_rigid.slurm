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
imgDir=/bucket/ReiterU/OLdev/analysis/20250416_IGL_waves/20250303_rec7
inputImg=${imgDir}/raw/ch0.hdf5
saveDir=${imgDir}/mcorr_rigid
####################

tempDir=$(mktemp -d /work/ReiterU/tmp/tomo.XXXXXX)
echo ${tempDir}
# copy input image in the temp directory
cp ${inputImg} ${tempDir}/raw.hdf5

# main
singularity exec --bind `pwd`:/workdir --env CAIMAN_TEMP=${tempDir} /bucket/ReiterU/singularity/caiman_v2.sif \
python /home/t/tomoyuki-mano1/projects/OL_calcium/scripts/NoRMCorre/motion_correction_v4.py \
--inputImg ${tempDir}/raw.hdf5 \
--num_cores 8 \
--savedir ${tempDir} \
--rigid \
--max_shifts 64 \
--strides 256

# remove temporary file
rm ${tempDir}/raw.hdf5

echo "Copying files to /bucket..."
if [[ ! -e ${saveDir} ]]; then
    ssh saion mkdir -p ${saveDir}
fi
scp -r ${tempDir}/* saion:${saveDir}
echo "Done."

echo "Deleting temporary directory..."
rm -r ${tempDir}
echo "Done."
