#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:V100:1
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
input_path=${imgDir}/mcorr_nonrigid/corrected.hdf5
model=/bucket/ReiterU/OLdev/analysis/20240729_test_deepinterpolation_1/2019_09_11_23_32_unet_single_1024_mean_absolute_error_Ai93-0450.h5
saveDir=${imgDir}/mcorr_nonrigid_denoised
####################

tempDir=$(mktemp -d /work/ReiterU/tmp/tomo.XXXXXX)
# copy input image in the temp directory
cp ${input_path} ${tempDir}/raw.hdf5

# first, rename the dataset in HDF5
singularity exec --nv --bind `pwd`:/workdir /bucket/ReiterU/singularity/deepinterp_v1.sif \
python /home/t/tomoyuki-mano1/projects/OL_calcium/notebook/20240729_deepinterpolation_test1/rename_hdf5_dataset.py \
${tempDir}/raw.hdf5 "mov" "data"

# run deepinterpolation inference
singularity exec --nv --bind `pwd`:/workdir /bucket/ReiterU/singularity/deepinterp_v1.sif \
python /home/t/tomoyuki-mano1/projects/OL_calcium/notebook/20240729_deepinterpolation_test1/ophys_inference.py \
--inputImg ${tempDir}/raw.hdf5 \
--model ${model} \
--output ${tempDir}/denoised.h5 \
--start 0 \
--end -1

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
