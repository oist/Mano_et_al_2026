import argparse
from datetime import datetime
from roifile import ImagejRoi, ROI_TYPE, roiwrite, roiread
import glob, os
import numpy as np
from hilbert import encode

def main(roi_in, roi_out):

    roi_org = roiread(roi_in)
    roi_new = []
    centers = []
    for r in roi_org:
        roi = ImagejRoi.frompoints(r.subpixel_coordinates)
        roi.roitype = ROI_TYPE.POLYGON
        c = [(roi.top + roi.bottom)/2, (roi.left + roi.right)/2]
        roi_new.append(roi)
        centers.append(c)
    print(f"{datetime.now()} - Done.")

    centers = np.array(centers)
    norm_centers = ((centers - centers.min(axis=0)) / (centers.max(axis=0) - centers.min(axis=0)) * (2**6 - 1)).astype(int)
    hilbert_indices = encode(norm_centers, 2, 6)
    sorted_indices = np.argsort(hilbert_indices)
    roi_new_sorted = [roi_new[i] for i in sorted_indices]

    for (k, roi) in enumerate(roi_new_sorted):
        roi.name = f"{k:04d}"

    roiwrite(roi_out, roi_new_sorted)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_roi", type=str)
    parser.add_argument("--output_roi", type=str)
    args = parser.parse_args()

    main(args.input_roi, args.output_roi)
