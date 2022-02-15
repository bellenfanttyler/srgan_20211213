import os
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

from data import DIV2K
from model.srgan import generator, discriminator
from train import SrganTrainer, SrganGeneratorTrainer

# Parsing the command line inputs
parser = argparse.ArgumentParser(description='Trains an SRGAN model')
parser.add_argument('-s','--steps', help='Number of Training Steps (Multiple of 10,000', required=True)
parser.add_argument('-w','--weights', help='Weights file to start with (.h5 file) from the weights/srgan directory', required=True)
args = vars(parser.parse_args())

base_steps = 10000
total_steps = int(args['steps'])
iteration_count = int(total_steps / base_steps)

# Location of model weights (needed for demo)
weights_dir = 'weights/srgan'
weights_file = lambda filename: os.path.join(weights_dir, filename)

os.makedirs(weights_dir, exist_ok=True)

# Loading the dataset
div2k_train = DIV2K(scale=4, subset='train', downgrade='bicubic')
div2k_valid = DIV2K(scale=4, subset='valid', downgrade='bicubic')

# Processing and caching the dataset
train_ds = div2k_train.dataset(batch_size=16, random_transform=True)
valid_ds = div2k_valid.dataset(batch_size=16, random_transform=True, repeat_count=1)

# Setting up the generator and the discriminator
gan_generator = generator()
gan_discriminator = discriminator()
gan_generator.load_weights(weights_file(str(args['weights'])))

# Setting up the Trainer
gan_trainer = SrganTrainer(generator=gan_generator, discriminator=gan_discriminator)

# Running the training
training_start = datetime.now()
print(" ")
print("TRAINING STARTING: ", training_start)

for iterations in range(iteration_count):
  start_time = datetime.now()
  print("Iteration Starting: ", (iterations + 1), "/", iteration_count, ", time: ", start_time)

  gan_trainer.train(train_ds, steps=base_steps)
  step_stamp = str(base_steps * (iterations + 1)) + '_'
  gan_trainer.generator.save_weights(weights_file((step_stamp + 'gan_generator.h5')))
  gan_trainer.discriminator.save_weights(weights_file((step_stamp + 'gan_discriminator.h5')))

  end_time = datetime.now()
  print("Iteration finished, end time: ", end_time)
  print(" ")

training_end = datetime.now()
training_duration = training_end - training_start
print("TRAINING FINISHED: ", training_end)
print("Training duration (minutes): ", (training_duration.total_seconds() / 60))