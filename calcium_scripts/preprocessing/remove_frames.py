import h5py
import argparse

def remove_frames(file_path, dataset_name, remove_start, remove_end, chunked_hdf5):
    with h5py.File(file_path, 'a') as f:
        d = f[dataset_name][remove_start:-remove_end]
        del f[dataset_name]
        if chunked_hdf5:
            f.create_dataset(dataset_name, chunks=(128,128,128), data=d)
        else:
            f.create_dataset(dataset_name, data=d)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rename a dataset in an HDF5 file.')
    parser.add_argument('file_path', type=str, help='Path to the HDF5 file.')
    parser.add_argument('dataset_name', type=str, help='dataset name.')
    parser.add_argument('remove_start', type=int)
    parser.add_argument('remove_end', type=int)
    parser.add_argument("--chunked_hdf5", action='store_true')
    args = parser.parse_args()
    
    remove_frames(args.file_path, args.dataset_name, args.remove_start, args.remove_end, args.chunked_hdf5)
