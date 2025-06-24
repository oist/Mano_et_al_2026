from skimage import io
import numpy as np
from datetime import datetime
import h5py
import os
import argparse

import caiman as cm
from caiman.motion_correction import MotionCorrect

def correct_motion(img_in, max_shifts, strides, nonrigid_flag, num_cores, tmpdir, chunked_hdf5, template_img_path=None):
    """
    img_in: str
        path to the input image (in HDF5 format with chunks enabled)
    max_shifts: (int, int)
        Maxium value allowed for pixel shifts, e.g. (40, 40)
    strides: (int, int)
        e.g. (128, 128)
    num_cores: int
        number of cpus to allocate to this process
    tmpdir: str
        Temporary directory which will be used by Caiman; e.g. /flash/ReiterU/tmp/caiman_tmp    
    """
    fnames = [img_in]

    # get the number of frames
    with h5py.File(img_in, 'r') as hf:
        dataset_name = list(hf.keys())[0]  # Get the first dataset name
        dataset = hf[dataset_name]  # Access the dataset
        num_frames = dataset.shape[0]
    
    if template_img_path is not None:
        template_image = io.imread(template_img_path)
    else:
        template_image = None

    # start the cluster (if a cluster already exists terminate it)
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=num_cores, single_thread=False)

    # create a motion correction object
    mc = MotionCorrect(fnames,
                       dview=dview,
                       max_shifts=max_shifts,
                       strides=strides,
                       pw_rigid=nonrigid_flag,  # flag for performing rigid or piecewise rigid motion correction
                       overlaps=(32, 32),  # overlap between pathes (size of patch strides+overlaps)
                       max_deviation_rigid=6, # maximum deviation allowed for patch with respect to rigid shifts
                       shifts_opencv=True, # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
                       nonneg_movie=True,
                       border_nan=True,  # flag for allowing NaN in the boundaries. True allows NaN, whereas ‘copy’ copies the value of the nearest data point.
                       splits_els=int(num_frames/200),
                       splits_rig=int(num_frames/200),
                       )
    print("Input image:", fnames)
    print(datetime.now(), "Start motion correction...")
    mc.motion_correct(template=template_image, save_movie=True)
    print(datetime.now(), "Done.\n")

    # save movie after clipping borders
    # border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
    border_to_0 = 0
    fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C', border_to_0=border_to_0, dview=dview) # exclude borders
    print("Corrected movie was saved as", fname_new)

    # save template image
    fname_template = os.path.join(tmpdir, "template.tif")
    if nonrigid_flag:
        io.imsave(fname_template, mc.total_template_els, check_contrast=False)
    else:
        io.imsave(fname_template, mc.total_template_rig, check_contrast=False)
    print("Template image was saved as", fname_template)

    # save shifts
    if nonrigid_flag:
        x_shifts = np.array(mc.x_shifts_els)
        y_shifts = np.array(mc.y_shifts_els)
        np.save(os.path.join(tmpdir, "x_shifts_els.npy"), x_shifts)
        np.save(os.path.join(tmpdir, "y_shifts_els.npy"), y_shifts)
        print("X shift & Y shift was saved in .npy format\n")
    else:
        shifts = np.array(mc.shifts_rig)
        np.save(os.path.join(tmpdir, "shifts_rig.npy"), shifts)
        print("X shift & Y shift was saved in .npy format\n")

    # save output as HDF5
    Yr, dims, T = cm.load_memmap(fname_new)
    Y = np.reshape(Yr.T, [T] + list(dims), order='F')
    Y = np.clip(Y, 0, 65535).astype("uint16")
    print("Writing array as HDF5...")
    with h5py.File(os.path.join(tmpdir, "corrected.hdf5"), 'w', driver='stdio') as hf:
        if chunked_hdf5:
            data = hf.create_dataset("mov", chunks=True, data=Y)
        else:
            data = hf.create_dataset("mov", data=Y)
    print("Done.")

    # delete unnecessary files
    os.remove(mc.mmap_file[0])
    os.remove(fname_new)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputImg", type=str, required=True)
    parser.add_argument("--savedir", type=str, required=True)
    parser.add_argument("--max_shifts", type=int, default=30)
    parser.add_argument("--strides", type=int, default=144)
    parser.add_argument("--num_cores", type=int, required=True)
    parser.add_argument('--nonrigid', action='store_true')
    parser.add_argument('--rigid', dest='nonrigid', action='store_false')
    parser.set_defaults(nonrigid=True)
    parser.add_argument("--chunked_hdf5", action='store_true')
    parser.add_argument("--template_img_path", type=str, default=None)
    args = parser.parse_args()

    # this is needed for CaImAn
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CAIMAN_RELEASE'] = '1'

    correct_motion(args.inputImg, (args.max_shifts, args.max_shifts), (args.strides, args.strides),
                   args.nonrigid, args.num_cores, args.savedir, args.chunked_hdf5, args.template_img_path)
