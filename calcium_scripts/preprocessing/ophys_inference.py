import os
import json
import argparse
from deepinterpolation.generator_collection import OphysGenerator
from deepinterpolation.inference_collection import core_inference

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputImg", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--num_cpus", type=int, default=None)
    args = parser.parse_args()

    if args.num_cpus is not None:
        os.environ["OMP_NUM_THREADS"] = str(args.num_cpus)  # Or however many you allocate per job
        os.environ["TF_NUM_INTRAOP_THREADS"] = str(args.num_cpus)
        os.environ["TF_NUM_INTEROP_THREADS"] = str(args.num_cpus)

    generator_param = {}
    inference_param = {}

    # We are reusing the data generator for training here.
    generator_param["pre_post_frame"] = 30
    generator_param["pre_post_omission"] = 0
    generator_param["steps_per_epoch"] = -1
    generator_param["train_path"] = args.inputImg
    generator_param["batch_size"] = 1
    generator_param["start_frame"] = args.start
    generator_param["end_frame"] = args.end
    generator_param["randomize"] = 0

    inference_param["model_path"] = args.model
    inference_param["output_file"] = args.output

    # Pass the path of the JSON files to SingleTifGenerator and core_inferrence
    data_generator = OphysGenerator(generator_param)
    inference_class = core_inference(inference_param, data_generator)

    inference_class.run()
