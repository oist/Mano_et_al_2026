#!/bin/bash
#SBATCH -p compute
#SBATCH -t 0-1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=tomoyuki.mano1@oist.jp

#modules
module load singularity

# Input and output
tempDir=$(mktemp -d /flash/ReiterU/tmp/tomo.XXXXXX)

rootDir=/bucket/ReiterU/OLdev/analysis/20250416_IGL_waves/20250303_rec7
inputROI=${rootDir}/ROI_yutaka.zip
outputROI=${rootDir}/ROI_yutaka_renamed.zip
# inputROI=${rootDir}/suite2p/plane0/ROI_corrected.zip
# outputROI=${rootDir}/suite2p/plane0/ROI_corrected_renamed.zip

# main
singularity exec --bind `pwd`:/workdir /bucket/ReiterU/singularity/ophys_v1.sif \
python /home/t/tomoyuki-mano1/projects/OL_calcium/scripts/manual_calcium/fix_ImageJROI.py \
--input_roi ${inputROI} \
--output_roi ${tempDir}/roi_out.zip

echo "Copying files to /bucket..."
scp ${tempDir}/roi_out.zip deigo:${outputROI}
echo "Done."

echo "Deleting temporary directory..."
rm -r ${tempDir}
echo "Done."
