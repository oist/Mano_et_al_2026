#!/bin/bash
#SBATCH -p compute
#SBATCH -t 0-3
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=tomoyuki.mano1@oist.jp

#modules
module load singularity

# Input and output
tempDir=$(mktemp -d /flash/ReiterU/tmp/tomo.XXXXXX)

tiffDir=/bucket/ReiterU/OLdev/experiments/20241018_squid_cal520/2P/rec11
tiffPath=${tiffDir}/Image_scan_1_region_0_0.tif
frameRate=15.4
saveDir=/bucket/ReiterU/OLdev/analysis/20250416_IGL_waves/20241018_rec11/raw

# main
singularity exec --bind `pwd`:/workdir /bucket/ReiterU/singularity/ophys_v1.sif \
python /home/t/tomoyuki-mano1/projects/OL_calcium/scripts/utils/export_2d_rec.py \
$tiffPath ${tempDir}/ch0.hdf5 $frameRate \
--save_avi \
--avi_name ${tempDir}/movie.avi \
--avi_fps 60 \
--crop_frames 25500 31000

echo "Copying files to /bucket..."
if [[ ! -e ${saveDir} ]]; then
    ssh deigo mkdir -p ${saveDir}
fi
scp -r ${tempDir}/* deigo:${saveDir}
echo "Done."

echo "Deleting temporary directory..."
rm -r ${tempDir}
echo "Done."
