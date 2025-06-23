#!/bin/bash
#SBATCH -p compute
#SBATCH -t 0-1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=tomoyuki.mano1@oist.jp

#modules
module load singularity

# Input and output
tempDir=$(mktemp -d /flash/ReiterU/tmp/tomo.XXXXXX)

rootDir=/bucket/ReiterU/OLdev/analysis/20250416_IGL_waves/20250303_rec7/mcorr_nonrigid_denoised
inputHDF5=${rootDir}/denoised.h5
datasetName=data

# main
singularity exec --bind `pwd`:/workdir /bucket/ReiterU/singularity/ophys_v1.sif \
python /home/t/tomoyuki-mano1/projects/OL_calcium/scripts/utils/hdf5_to_tif.py \
--input_hdf5 ${inputHDF5} \
--dataset_name ${datasetName} \
--output_tif ${tempDir}/denoised_16bit.tif

echo "Copying files to /bucket..."
scp -r ${tempDir}/denoised_16bit.tif deigo:${rootDir}
echo "Done."

echo "Deleting temporary directory..."
rm -r ${tempDir}
echo "Done."
