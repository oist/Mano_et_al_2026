import h5py
import numpy as np
from scipy import ndimage as ndi
import argparse

def main(input_hdf5, dataset_name, output_hdf5, sigma):

    print("Loading input HDF5...")
    with h5py.File(input_hdf5, 'r') as hf:
        d = hf[dataset_name][:]
    print("Done.")

    d = ndi.gaussian_filter(d, sigma=sigma)
    d = d.clip(0, 65535).astype("uint16")

    print("Writing HDF5...")
    with h5py.File(output_hdf5, 'w') as hf:
        hf.create_dataset("mov", chunks=True, data=d)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_hdf5", type=str, required=True, help="Path of the input HDF5 file.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset in the HDF5")
    parser.add_argument("--output_hdf5", type=str, required=True, help="Path of the output TIF file.")
    parser.add_argument("--sigma", nargs=3, type=float, help="Sigma value for Z, Y and X.")
    args = parser.parse_args()

    main(args.input_hdf5, args.dataset_name, args.output_hdf5, tuple(args.sigma))
