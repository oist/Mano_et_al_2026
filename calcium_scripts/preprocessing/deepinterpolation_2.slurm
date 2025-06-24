#!/bin/bash
#SBATCH -p compute
#SBATCH -t 0-12
#SBATCH -c 64
#SBATCH --mem=128G
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=tomoyuki.mano1@oist.jp

#modules
module load singularity

#### Parameters ####
imgDir=/bucket/ReiterU/OLdev/analysis/20250623_calcium_test_data/20250317_rec6
input_path=${imgDir}/mcorr_nonrigid/corrected.hdf5
model=/bucket/ReiterU/OLdev/analysis/20240729_test_deepinterpolation_1/2019_09_11_23_32_unet_single_1024_mean_absolute_error_Ai93-0450.h5
saveDir=${imgDir}/mcorr_nonrigid_denoised

singularityImage=/bucket/ReiterU/singularity/deepinterp_v1.sif
scriptDir=/home/t/tomoyuki-mano1/projects/squid_vision_paper_draft/calcium_scripts/preprocessing
####################

tempDir=$(mktemp -d /flash/ReiterU/tmp/tomo.XXXXXX)
cp ${input_path} ${tempDir}/raw.hdf5

# first, rename the dataset in HDF5
singularity exec --nv --bind `pwd`:/workdir ${singularityImage} \
python ${scriptDir}/rename_hdf5_dataset.py \
${tempDir}/raw.hdf5 "mov" "data"

# run deepinterpolation inference
singularity exec --nv --bind `pwd`:/workdir ${singularityImage} \
python ${scriptDir}/ophys_inference.py \
--inputImg ${tempDir}/raw.hdf5 \
--model ${model} \
--output ${tempDir}/denoised.h5 \
--start 0 \
--end -1 \
--num_cpus 64

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
