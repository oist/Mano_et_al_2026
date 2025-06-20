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
imgDir=/bucket/ReiterU/OLdev/analysis/20250416_IGL_waves/20240614_rec4
structImg=${imgDir}/mcorr_rigid_denoised/denoised.h5
calciumImg=${imgDir}/mcorr_rigid/corrected.hdf5
saveDir=${imgDir}/mcorr_nonrigid
####################

tempDir=$(mktemp -d /work/ReiterU/tmp/tomo.XXXXXX)
# copy input image in the temp directory
cp ${structImg} ${tempDir}/ch0.hdf5
cp ${calciumImg} ${tempDir}/ch1.hdf5

# remove 30 frames at the beginning and the end
singularity exec --bind `pwd`:/workdir /bucket/ReiterU/singularity/caiman_v2.sif \
python /home/t/tomoyuki-mano1/projects/OL_calcium/scripts/NoRMCorre/remove_frames.py \
${tempDir}/ch1.hdf5 "mov" 30 30 --chunked_hdf5

## this line is needed
# see https://github.com/flatironinstitute/CaImAn/blob/main/caiman/paths.py
export CAIMAN_TEMP="."

# main
singularity exec --bind `pwd`:/workdir --env CAIMAN_TEMP=${tempDir} /bucket/ReiterU/singularity/caiman_v2.sif \
python /home/t/tomoyuki-mano1/projects/OL_calcium/scripts/NoRMCorre/motion_correction_2chan_v2.py \
--structImg ${tempDir}/ch0.hdf5 \
--calciumImg ${tempDir}/ch1.hdf5 \
--num_cores 8 \
--savedir ${tempDir} \
--max_shifts 32

# remove temporary file
rm ${tempDir}/ch0.hdf5
rm ${tempDir}/ch1.hdf5

echo "Copying files to /bucket..."
if [[ ! -e ${saveDir} ]]; then
    ssh saion mkdir -p ${saveDir}
fi
scp -r ${tempDir}/* saion:${saveDir}
echo "Done."

echo "Deleting temporary directory..."
rm -r ${tempDir}
echo "Done."
