## What is an Adversarial Image?

Adversarial images are specialized inputs created with the purpose of confusing a neural network, resulting in the misclassification of a given input. These notorious inputs are indistinguishable to the human eye but cause the network to incorrectly identify the contents of the image. There are several types of such attacks; however, here the focus is on the `Fast Gradient Sign Method (FGSM)` attack, a `White-Box Attack` aimed at ensuring misclassification.

In a white-box attack, the attacker has complete access to the model being targeted. One of the most famous examples of an adversarial image, shown below, is taken from the 2014 research paper [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572).

![Image](Summary.png)

Here, starting with the image of a panda, the attacker adds small perturbations (distortions) to the original image, causing the model to label it as a gibbon with high confidence! The process of adding these perturbations is explained below.

## Fast Gradient Sign Method

The Fast Gradient Sign Method works by using the gradients of the neural network to create an adversarial example. For a given input image, the method calculates the gradients of the loss with respect to the input and uses them to create a new image that maximizes the loss. This new image is the adversarial example.

It can be summarized with the following expression:

$$adv\_x = x + \epsilon*\text{sign}(\nabla_xJ(\theta, x, y))$$

where:
* $adv_x \Rightarrow$ Adversarial Image
* $x \Rightarrow$ Original Input Image
* $y \Rightarrow$ Original Input Label
* $\epsilon \Rightarrow$ Perturbation Multiplier (controls how small the changes are)
* $\theta \Rightarrow$ Model Parameters
* $J \Rightarrow$ Loss Function

An important detail here is that the gradients are taken with respect to the input image itself. This is because the goal is to create an image that maximizes the loss. We need to know how much each pixel contributes to the loss value. Then, a small perturbation is added accordingly.

This method works efficiently because the chain rule allows us to quickly compute how each pixel affects the loss. Also, the model parameters ($\theta$) remain constant during this process, since we are not training the model, we are only trying to fool it.

## About this Project

The best way to defend against FGSM white-box attacks is to train the model on adversarial images.
However, most tutorials only explain how to generate a single adversarial image by manually tuning $\epsilon$, which is tedious if you need many images for training.

To solve this, I automated the process by setting a threshold value for $\epsilon$. Here's how it works:
* $\epsilon$ increases gradually from 0 to 1 in 20 steps (0.05 per step).
* If the model misclassifies an image before reaching the threshold, the process stops and saves that perturbed image.
* If the model still classifies the image correctly even at the threshold, the most perturbed version is saved.
* This process is extended to multiple images packed into a zip file.

## How to Run?

I have deployed a Gradio Application in [HuggingFace Spaces](https://huggingface.co/spaces/Navam9530/Adversarial_Images_Generator).

### Requirements:
* The model must be a Keras model (SavedModel format, .keras, or .h5). (PyTorch models are currently unsupported.)
* The images must be uploaded as a zip file named `images.zip`.
* The labels must be provided in a separate text file, with one label per line, matching the order of the images.

### Steps to Run:
* Upload the Keras CNN Model. (PyTorch models are currently unsupported)
* Upload the Images Zip File.
* Upload the Labels Text File.
* Enter the Image Height.
* Enter the Image Width.
* Enter the Number of Classes. (Enter 1 for Binary Classification)
* Click Submit.

If everything is set up correctly, you will see a progress bar at the top right of the window. Once the process finishes, you can download the generated adversarial images as a zip file.

If you face any errors or are unsure about the format, I have provided two examples: one for binary classification and another for multi-class classification. You can either download them and try locally or run them directly by selecting a row in the app.
