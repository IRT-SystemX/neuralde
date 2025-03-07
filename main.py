import argparse
# from neural_de.transformations import TransformationPipeline

import sys
import cv2
import os 
from PIL import Image
import matplotlib.pyplot as plt
from os import walk
from pathlib import Path
import glob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Main script of Neural DE"
    )

    # parse input arguments
    parser.add_argument("--input_image_path", required=False, type=str)
    parser.add_argument("--input_dir_path",required=False,type=str)
    # parser.add_argument("--input_list_files_path",required=False,type=str)
    
    parser.add_argument("--output_dir_path", required=False, type=str)
    parser.add_argument("--output_image_path", required=False, type=str)
    parser.add_argument("--output_prefix", required=False, type=str,default="")
    
    parser.add_argument("--config_file_path", required=False, type=str)
    parser.add_argument("--transformation", required=False, type=str)
    
    args = parser.parse_args()


    print("args", args)

    print("input image", args.input_image_path)
    print("input dir", args.input_dir_path)
    print("output_îmage", args.output_image_path)
    print("output_dir", args.output_dir_path)
    print("config_file", args.config_file_path)
    print("transformation", args.transformation)

    # Create output_dir if it doesnot exists
    if args.output_image_path:
        os.makedirs(os.path.dirname(args.output_image_path), exist_ok=True)
    if args.output_dir_path:
        print("dossier a creer",os.path.dirname(args.output_dir_path)) 
        Path(args.output_dir_path).mkdir(parents=True, exist_ok=True)

    # Create working path lists of images to process 
    
    if args.input_image_path:
        image_paths_list=[args.input_image_path]
        output_paths_list=[args.output_image_path]
    if args.input_dir_path:       
        image_paths_list=glob.glob(args.input_dir_path+"/*")
        output_paths_list=[args.output_dir_path+os.sep+args.output_prefix+x.split(os.sep)[-1] for x in image_paths_list] # Adding prefix
    if  args.config_file_path: 
            print("todo")
    
    print("image list", image_paths_list)

    # Create list of input batch to pass 
    input_images_list = [cv2.imread(x) for x in image_paths_list]
    print("image tailles orig", [x.shape for x in input_images_list])
    # input_images_list = [cv2.resize(x,(382,416), interpolation= cv2.INTER_LINEAR) for x in input_images_list]
    # input_images_list = [cv2.resize(x,(381,562), interpolation= cv2.INTER_LINEAR) for x in input_images_list]

    
    print("image tailles", [x.shape for x in input_images_list])
    
    #Apply chosen transformation to list of images
    
    if args.transformation=="DeSnowEnhancer":
        print("preprocess list of input images . .")
        from neural_de.transformations import DeSnowEnhancer   
        
        corrupted_images_list = [x[:, :, ::-1] for x in input_images_list]
        print("Preprocessing ok")
        purifier = DeSnowEnhancer()
        print("Apply tranformation on images . . .")
        output_images_list = purifier.transform(corrupted_images_list)
        print("Tranformations completed")

    if args.transformation=="BrightnessEnhancer":
        from neural_de.transformations import BrightnessEnhancer
        image_enhancer = BrightnessEnhancer()
        input_images_list = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in input_images_list]
        output_images_list = image_enhancer.transform(input_images_list)

    if args.transformation=="Pipeline" :
        from neural_de.transformations import TransformationPipeline
        input_images_list = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in input_images_list]
        pipeline = TransformationPipeline(args.config_file_path)
        output_images_list = pipeline.transform(input_images_list)
    # Save transformed images to output path
    
    # Create list of output _path
    print("Save images to output paths  . .")
    for idx in range(0,len(output_paths_list)): 
        plt.imsave(output_paths_list[idx],output_images_list[idx])   
    print("Work is completed")
    
    
    
    
     