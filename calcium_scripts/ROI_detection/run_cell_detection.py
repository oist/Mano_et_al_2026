import suite2p
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_hdf5", type=str, required=True)
    parser.add_argument("--hdf5_key", type=str, default="data")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--threshold_scaling", type=float, default=1.5, help="this controls the threshold at which to detect ROIs")
    parser.add_argument("--fs", type=float, default=15.4, help="Set the frame rate (Hz)")
    parser.add_argument("--tau", type=float, default=1.0, help="Approximate decay time constant of calcium indicator in seconds")
    args = parser.parse_args()

    # define inputs and outpus
    db = {
        'h5py': args.input_hdf5,
        'h5py_key': args.hdf5_key,
        'save_path0': args.output_dir
        }
    # define parameters
    ops = suite2p.default_ops()
    ops['threshold_scaling'] = args.threshold_scaling
    ops['fs'] = args.fs
    ops['tau'] = args.tau
    ops['nplanes'] = 1  # Single plane
    ops['nchannels'] = 1  # Single channel
    ops['do_registration'] = False
    ops['delete_bin'] = True

    output_ops = suite2p.run_s2p(ops=ops, db=db)
