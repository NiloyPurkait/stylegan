# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import argparse


def get_args():
  parser = argparse.ArgumentParser(description="Main Arguments")

  parser.add_argument(
    '--file_path', default='', type=str, required=False,
    help='Path to training data')
    
  parser.add_argument(
    '--url_path', default='', type=str, required=False,
    help='Path to training data')

  args = parser.parse_args()
  return args


def legacy_url_loader(url, cache_dir):
    with dnnlib.util.open_url(url, cache_dir=cache_dir) as f:
        return pickle.load(f)

    
def network_loader():
    args = get_args()
    
    # Load pre-trained network.
    if args.url_path:
        print('Loading from : %s' % args.url_path )
        return legacy_url_loader(args.url_path, config.cache_dir)
    
    elif args.file_path:
        print('Loading from file at : %s' % args.file_path)
        with open(args.file_path, 'rb') as f:
            return pickle.load(f) 
        
    else:
        print('Loading original pre-trained model.')
        return legacy_url_loader(url, config.cache_dir)

def main():
    # Initialize TensorFlow.
    tflib.init_tf()
    
    # Option to load from disk or custom URL
    _G, _D, Gs = network_loader()
    # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
    # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
    # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    # Print network details.
    Gs.print_layers()

    # Pick latent vector.
    rnd = np.random.RandomState(5)
    latents = rnd.randn(1, Gs.input_shape[1])

    # Generate image.
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

    # Save image.
    os.makedirs(config.result_dir, exist_ok=True)
    png_filename = os.path.join(config.result_dir, 'example.png')
    PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

if __name__ == "__main__":
    main()
