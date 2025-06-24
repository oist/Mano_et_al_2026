#!/bin/bash
#SBATCH -p compute
#SBATCH -t 0-12
#SBATCH -c 64
#SBATCH --mem=196G
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=tomoyuki.mano1@oist.jp

#modules
module load singularity

#### Parameters ####
imgDir=/bucket/ReiterU/OLdev/analysis/20250623_calcium_test_data/20250317_rec6
structImg=${imgDir}/mcorr_rigid_denoised/denoised.h5
calciumImg=${imgDir}/mcorr_rigid/corrected.hdf5
saveDir=${imgDir}/mcorr_nonrigid

singularityImage=/bucket/ReiterU/singularity/caiman_v2.sif
scriptDir=/home/t/tomoyuki-mano1/projects/squid_vision_paper_draft/calcium_scripts/preprocessing
hostPynwb=/home/t/tomoyuki-mano1/pynwb_site
####################

tempDir=$(mktemp -d /flash/ReiterU/tmp/tomo.XXXXXX)
cp ${structImg} ${tempDir}/ch0.hdf5
cp ${calciumImg} ${tempDir}/ch1.hdf5

# remove 30 frames at the beginning and the end
singularity exec --bind `pwd`:/workdir ${singularityImage} \
python ${scriptDir}/remove_frames.py \
${tempDir}/ch1.hdf5 "mov" 30 30 --chunked_hdf5

# main
singularity exec --bind `pwd`:/workdir \
--bind ${hostPynwb}:/opt/pynwb_site \
--env PYTHONPATH=/opt/pynwb_site:$PYTHONPATH \
--env CAIMAN_TEMP=${tempDir} \
${singularityImage} \python ${scriptDir}/motion_correction_2chan_v2.py \
--structImg ${tempDir}/ch0.hdf5 \
--calciumImg ${tempDir}/ch1.hdf5 \
--num_cores 64 \
--savedir ${tempDir} \
--max_shifts 32 \
--strides 96

# remove temporary file
rm ${tempDir}/ch0.hdf5
rm ${tempDir}/ch1.hdf5

echo "Copying files to /bucket..."
if [[ ! -e ${saveDir} ]]; then
    ssh deigo mkdir -p ${saveDir}
fi
scp -r ${tempDir}/* deigo:${saveDir}
echo "Done."

echo "Deleting temporary directory..."
rm -r ${tempDir}
echo "Done."
