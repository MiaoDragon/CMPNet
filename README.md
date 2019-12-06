# Code Explaination:
* cmpnet_train.py, cmpnet_test.py: general training and testing pipeline
* data_loader_*.py: data loader for different env settings
* gem_eval.py: called by cmpnet_test.py to evaluate the model on the testing dataset
* plan_general.py: common plannning functions used in different envs
* plan_*.py: implements env-specific planning functions: Collision-Checker
* utility_*.py: implements normalization and unnormalization of data depending on envs
* plot_util.py: functions for plotting
* cmpnet_compare_sample.py: general pipeline to compare different sampling methods in GEM
* mpnet_train.py: End-2-End training MPNet without Continual Learning functionality
* Model: implements the neural network model for planner network and encoder network
    * GEM_end2end_model.py: implements the MPNet combined with Continual Learning functions (GEM)
    * model_*.py: different planner models
    * gem_utility.py: utility functions used to implement GEM
    * GEM_end2end_model_*.py: network implementing different sampling methods
    * AE: encoder network architecture
* exp: bash codes for training and testing
    * cmpnet_train*.sh: code for training for each envs in Continual Learning mode
    * cmpnet_test*.sh: code for testing for each envs in Continual Learning mode
    * compare_sampling_method*.sh: code to compare different sampling methods
    * mpnet_train*.sh: code for training without Continual Learning mode
    * mpnet_test*.sh: code for testing without Continual Learning mode

# Instructions for reproduction:
* obtain dataset (Coming soon)
* change the data_path in the training/testing script under exp folder to the dataset path
* change the model_path in the training/testing script under exp folder to your desired path for saving the model, and important statistics
* run the bash file in exp folder to conduct experiments (GPU needed)


# Instructions for C++ MPNet:
* make a new directory named deps, and download libtorch into it by following this obtained from pytorch website:
    e.g.: the following works for cuda 10.1
    Download here (Pre-cxx11 ABI):
    https://download.pytorch.org/libtorch/cu101/libtorch-shared-with-deps-1.3.1.zip
    Download here (cxx11 ABI):
    https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.3.1.zip
* make a new directory called build in the main directory.
* in the build directory, run the following:
    `cmake c++`
    `make`
* convert Python trained MPNet model to C++ by running py_model_to_cpp.py inside c++ folder.
