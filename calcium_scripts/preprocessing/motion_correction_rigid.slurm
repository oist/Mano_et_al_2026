#!/bin/bash
#SBATCH -p compute
#SBATCH -t 0-12
#SBATCH -c 24
#SBATCH --mem=128G
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=tomoyuki.mano1@oist.jp

#modules
module load singularity

#### Parameters ####
imgDir=/bucket/ReiterU/OLdev/analysis/20250623_calcium_test_data/20250317_rec6
inputImg=${imgDir}/ch0.hdf5
saveDir=${imgDir}/mcorr_rigid

singularityImage=/bucket/ReiterU/singularity/ophys_v3.sif
scriptDir=/home/t/tomoyuki-mano1/projects/squid_vision_paper_draft/calcium_scripts/preprocessing
hostPynwb=/home/t/tomoyuki-mano1/pynwb_site
####################

tempDir=$(mktemp -d /flash/ReiterU/tmp/tomo.XXXXXX)

cp ${inputImg} ${tempDir}/raw.hdf5

# main
singularity exec --bind `pwd`:/workdir \
--bind ${hostPynwb}:/opt/pynwb_site \
--env PYTHONPATH=/opt/pynwb_site:$PYTHONPATH \
--env CAIMAN_TEMP=${tempDir} \
${singularityImage} \
python ${scriptDir}/motion_correction_v5.py \
--inputImg ${tempDir}/raw.hdf5 \
--num_cores 24 \
--savedir ${tempDir} \
--rigid \
--max_shifts 48 \
--strides 256

# remove temporary file
rm ${tempDir}/raw.hdf5

echo "Copying files to /bucket..."
if [[ ! -e ${saveDir} ]]; then
    ssh deigo mkdir -p ${saveDir}
fi
scp -r ${tempDir}/* deigo:${saveDir}
echo "Done."

echo "Deleting temporary directory..."
rm -r ${tempDir}
echo "Done."
