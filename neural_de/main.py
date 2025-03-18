import argparse


import sys
import cv2
import os 
from PIL import Image
import matplotlib.pyplot as plt
from os import walk
from pathlib import Path
import glob
from neural_de.transformations import TransformationPipeline

def main():

    parser = argparse.ArgumentParser(description="Main script of Neural DE")

    # parse input arguments
    
    parser.add_argument("--input_source_path", required=True, type=str, help="Path to the input source to process. If path represent a single file path. The single file will be processed. If path represents a directory, all images present in directory will be processed")
    parser.add_argument("--output_target_path",required=True,type=str, help="Output path where results are stored. If ionput path is a file, this path represents the output file, else this is the path to oupput direcotry")   
    parser.add_argument("--output_prefix", required=False, type=str, default="", help="output_prefix to add to output images filenames")
    parser.add_argument("--pipeline_file_path", required=True, type=str, help="Pipeline Configuration file to use")
    args = parser.parse_args()

    print("args", args)

    # Create output directory if it does not exists

    directory= not os.path.isfile(args.input_source_path)
    
    # Create working path lists of images to process 
    
    if directory:       
        image_paths_list=glob.glob(args.input_source_path+"/*")
        output_paths_list=[args.output_target_path+os.sep+args.output_prefix+x.split(os.sep)[-1] for x in image_paths_list] # Adding prefix
    else:
        image_paths_list=[args.input_source_path]
        output_paths_list=[args.output_target_path]
       
    print("input image list", image_paths_list)
    print("output image list", output_paths_list)
 
    # Create output dir if needed
    
    print("output dir",str(args.output_target_path))
    
    if directory:
        Path(args.output_target_path).mkdir(parents=True, exist_ok=True)
    else:
        target_dir=os.sep.join(str(args.output_target_path).split("/")[:-1])
        print("targetdir", target_dir)
        Path(target_dir).mkdir(parents=True, exist_ok=True)

    print("output dir",str(args.output_target_path))

    # Create list of input batch to pass (interpret files as numpy array of tensors) 
    input_images_list = [cv2.imread(x) for x in image_paths_list]
    
    # print("original images sizes", [x.shape for x in input_images_list])
    
    # Resizing for test in debug
    # input_images_list = [cv2.resize(x,(382,416), interpolation= cv2.INTER_LINEAR) for x in input_images_list]
    # input_images_list = [cv2.resize(x,(381,562), interpolation= cv2.INTER_LINEAR) for x in input_images_list]

    # print("resized images sizes", [x.shape for x in input_images_list])
    
    # Apply transformation pipeline  
    input_images_list = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in input_images_list]
    pipeline = TransformationPipeline(args.pipeline_file_path)
    output_images_list = pipeline.transform(input_images_list)

    # Save output images to target output paths
    print("Save images to output paths  . .")

    for idx in range(0,len(output_paths_list)): 
        plt.imsave(output_paths_list[idx],output_images_list[idx])   
    
    print("Work is completed")

if __name__ == "__main__":
    main()
    
    
    
    
     