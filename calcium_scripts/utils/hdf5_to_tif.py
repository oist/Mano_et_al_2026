import h5py
import numpy as np
from skimage.io import imsave
import argparse

def main(input_hdf5, dataset_name, output_tif):

    print("Loading input HDF5...")
    with h5py.File(input_hdf5, 'r') as hf:
        d = hf[dataset_name][:]
    print("Done.")

    d = d.clip(0, 65535).astype("uint16")

    print("Writing TIFF file...")
    imsave(output_tif, d, check_contrast=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_hdf5", type=str, required=True, help="Path of the input HDF5 file.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset in the HDF5")
    parser.add_argument("--output_tif", type=str, required=True, help="Path of the output TIF file.")
    args = parser.parse_args()

    main(args.input_hdf5, args.dataset_name, args.output_tif)
