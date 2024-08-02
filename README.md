# eXplainable Artificial Intelligence (XAI) for Time Series Classification (TSC)

### Function and Motivation
In their work (https://doi.org/10.1016/j.patcog.2024.110309), Vielhaben et al. introduced the concept of virtual inspection layers (VILs) to improve the understandability of XAI explanations for TSC.
A VIL can be added in front of the input layer of a neural network. Consisting of two weightless layers, it serves the purpose of transforming the input into a different format, and inverting this transformation before passing the reconstructed input data to the original model.
Time domain audio data, for example, can be transformed to the frequency domain—and back.
Utilizing feature relevancy techniques—such as Layer-wise Relevancy Propagation (LRP)—on this temporarily transformed data, different XAI explanations can be generated without having to retrain the original model.

As understandability is a subjective measure, the literature on XAI emphasizes the current need for user evaluations of XAI explanations. Since Vielhaben et al. did not conduct a user study to evaluate the results of their XAI technique, I decided to implement their approach, add two novel VILs, and evaluate the generated results as part of my bachelor thesis.
The implementation uses Becker et al.'s AudioMNIST data set (https://github.com/soerenab/AudioMNIST), as well as a slightly adjusted re-build of their AudioNet model for spoken digit classification and XAI explanation generation.

### Model Adjustments

- Implemented PyTorch model instead of CAFFE model (due to library dependency conflicts)
- Omitted slicing and silencing logic, as only the digit is considered as the class (for simplification)
- Used Adaptive Moment Estimation (ADAM) optimizer instead Stochastic Gradient Descent (SGD), as the latter did not show any reduction in
train loss
- Added early stopping to avoid overfitting and to recover the model's best parameters achieved throughout training

### Repository Structure

- methods.py: contains all custom implemented methods
- model_training.ipynb: a Jupyter notebook to train the PyTorch AudioNet CNN on the preprocessed audio data
- models.py: contains the AudioNet CNN and the VIL model class
- preprocess_data.ipynb: a Jupyter notebook to preprocess the AudioMNIST data according to Becker et al.'s preprocessing steps for the AudioNet model
- virtual_inspection_layer_implementation.ipynb: a Jupyter notebook to create instance-based explanations for a classification decision, utilizing a virtual inspection layer on the AudioNet model
- virtual_inspection_layers.py: contains all implemented VILs (DFT, STDFT, DCT, and DWT)

### Setup

Please note that this code is a prototype and thus **not** production-ready. There are still open errors and missing functions.

1. Clone repository and initialize with Python version 3.9
2. Install dependencies using the requirements.txt (might happen automatically)
3. Add cloned AudioMNIST repository (https://github.com/soerenab/AudioMNIST) to the project's root folder (or adjust data paths instead)
4. To enable Captum's LRP calculation for complex values (like DFT), please navigate to its Propagation Rule class and change torch.sign(outputs) to torch.sgn(outputs) in the _create_backward_hook_output method. You can quickly access the class by navigating into the captum.attr._utils.lrp_rules import in the virtual_inspection_layer_implementation.ipynb notebook. Please restart your Jupyter notebook for the change to take effect.

### Usage

1. Preprocess data by running preprocess_data.ipynb (optional: skip and use provided test data for model validation)
2. Train model by running model_training.ipynb (optional: skip and use provided model weights)
3. Generate data transformation and model classification decision plots by running virtual_inspection_layer_implementation.ipynb

### Features

- Specify spoken digit, speaker, and recording (or get random sample)
- Specify data transformation and model classification decision representation (either averaged convolution layer output, DFT, STDFT, DCT, or DWT)
- Define custom range for the generated plots (only fully implemented for DFT and DCT)
- Audibly verify correct reconstruction of temporarily transformed audio input
- Verify VIL model(s) by evaluating it using the original model's test set