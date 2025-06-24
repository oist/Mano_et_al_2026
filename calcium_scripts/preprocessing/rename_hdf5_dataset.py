import h5py
import argparse

def rename_dataset(file_path, old_name, new_name):
    with h5py.File(file_path, 'a') as f:
        # Check if the old dataset exists
        if old_name in f:
            # Copy the dataset to the new name
            f.copy(old_name, new_name)
            # Delete the old dataset
            del f[old_name]
            print(f"Dataset '{old_name}' has been renamed to '{new_name}'.")
        else:
            print(f"Dataset '{old_name}' not found in the file.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rename a dataset in an HDF5 file.')
    parser.add_argument('file_path', type=str, help='Path to the HDF5 file.')
    parser.add_argument('old_name', type=str, help='Old dataset name.')
    parser.add_argument('new_name', type=str, help='New dataset name.')
    
    args = parser.parse_args()
    
    rename_dataset(args.file_path, args.old_name, args.new_name)
