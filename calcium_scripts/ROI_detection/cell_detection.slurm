#!/bin/bash
#SBATCH -p compute
#SBATCH -t 0-12
#SBATCH -c 24
#SBATCH --mem=64G
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=tomoyuki.mano1@oist.jp

#modules
module load singularity

#### Parameters ####
imgDir=/bucket/ReiterU/OLdev/analysis/20250623_calcium_test_data/20250317_rec6
input_path=${imgDir}/mcorr_nonrigid_denoised/denoised.h5
frameRate=15.4
saveDir=${imgDir}/suite2p

singularityImage=/bucket/ReiterU/singularity/ophys_v3.sif
scriptDir=/home/t/tomoyuki-mano1/projects/squid_vision_paper_draft/calcium_scripts/ROI_detection
####################

tempDir=$(mktemp -d /flash/ReiterU/tmp/tomo.XXXXXX)
# copy input image in the temp directory
cp ${input_path} ${tempDir}/raw.hdf5

# run suite2p cell detection
singularity exec `pwd`:/workdir ${singularityImage} \
python ${scriptDir}/run_cell_detection.py \
--input_hdf5 ${tempDir}/raw.hdf5 \
--hdf5_key "data" \
--output ${tempDir} \
--threshold_scaling 1.5 \
--fs ${frameRate} \
--tau 1.0

# remove temporary file
rm ${tempDir}/raw.hdf5

echo "Copying files to /bucket..."
if [[ ! -e ${saveDir} ]]; then
    ssh saion mkdir -p ${saveDir}
fi
scp -r ${tempDir}/suite2p/* saion:${saveDir}
echo "Done."

echo "Deleting temporary directory..."
rm -r ${tempDir}
echo "Done."
