# Leveraging Relational Information for Learning Weakly Disentangled Representations

This repository provides the Python implementation of the paper "Leveraging Relational Information for Learning Weakly Disentangled Representations", introducing Weak Disentanglement, a new way for learning disentangled representations that relaxes some of the limitations of the previous methods. The project is written in Pytorch. The paper has been accepted at WCCI 2022.

Weakly disentangled representations are able to structure themselves in order to separate the different regions encoding specific combinations of factor of variations. Knowing the location of these regions in the latent space, it is then possible to manipulate the representation in order to obtain controlled changes in the corresponding decoded data samples.

If you want to know more about weak disentanglement, you can read the [paper on ArXiv](https://arxiv.org/pdf/2205.10056.pdf), which also contains a link to some examples of learned realations by the trained models.

## Getting Started
The project is separated into three subfolders, corresponding to the different datasets on which the model is trained. Each subfolder contain similar files, and should allow for a complete training of the model on the specific dataset, without the need of files from other folders.

All the folder are structured as follows:
- File *train.py* is the main training script, implementing the training/eval loop of the model.
- File *config.py* contains the model's configuration settings, structured as a Python dictionary.
- File *prior.py* contains all the prior of the adversarial autoencoder taht have been considered during the developement of the model. Some of the prior are not actually used in the final model.
- File *model.py* contains the implementation of the layers of the AbsAE and ReL models.
- File *loss.py* contains the implementations of all the losses used  during training.
- File *log.py* contains a collections of logging functions used to produce the final results and to check the progress of training.
- File *dataset.py* contains the implementation of the dataset object and all data-related utilities.

## Installation/Dependencies

The model is implemented in Pytorch. 
You should be able to install all the required dependencies using the *ap2.yml* file that will be (soon) added to the repository.

## Usage

First, you need to download the corresponding dataset. Place the raw data into the "./data" subfolder, in the same folder as the one of the scripts.

After that, you are ready to go! You can start a new training by running the *train.py* script.

In *config.py* you can freely change many model parameters, from the number of latent variable to the number of hidden layers of the many architectural components. The parameters name should be quite auto-explicatives.


## Getting Help

For any other additional information, you can email me at andrea.valenti@phd.unipi.it.

## Contributing Guidelines  

You're free (and encouraged) to make your own contributions to this project.

## Code of Conduct

Just be nice, and respecful of each other!

## License

All source code from this project is released under the MIT license.
