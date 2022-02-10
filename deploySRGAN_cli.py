import tensorflow as tf
from PIL import Image
import os
from model.srgan import generator
from model import common, resolve_single
from utils import load_image, plot_sample
import time
import argparse

parser = argparse.ArgumentParser(description='Super-resolves an image using SRGAN')
parser.add_argument('-i','--input', help='Input image, located in /demo', required=True)
parser.add_argument('-o','--output', help='Output image filename, saved in /output', required=True)
args = vars(parser.parse_args())

# Settting up files and directories
input_image_dir = "demo/"
input_image = args['input']
output_image_dir = "output/"
if '.png' in args['output'] or '.jpg' in args['output']:
  output_image = args['output']
else:
  output_image = args['output'] + '.png'
os.makedirs(output_image_dir, exist_ok=True)

# Loading the generator model and its weights
print("Loading the model...")
model = generator()
model.load_weights('weights/srgan/gan_generator.h5')

# Loading the image and super-resolving it
print("Loading the image...")
lr = load_image(os.path.join(input_image_dir, input_image)) 
print("Super-resolving the low-res image...")
startTime = time.time()
sr = resolve_single(model, lr)
executionTime = (time.time() - startTime)
sr = sr.numpy()

print("Low resolution image size:", lr.shape)
print("Super-resolution image size:", sr.shape)
print('Execution time in seconds: ' + str(executionTime))

# Saving the Super-Resolved Image
pil_image=Image.fromarray(sr)
pil_image.save(os.path.join(output_image_dir, output_image))

print("Super Resolution Complete")
print("Output image located at", os.path.join(output_image_dir, output_image))