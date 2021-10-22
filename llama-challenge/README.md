# Llama Challenge 1

Welcome to the ByteFuse AI Llama challenge. These challenges differ from others in that we want you to focus on what you can do with models rather than on model design and training.



To understand the goal of this challenge, put yourself in the shoes of one of the engineers at ByteFuse. You arrive at work one morning to learn that one of the research scientists has begun work on a system capable of performing multimodal search over both images and speech. However, the scientist only trained the system partially on the [Flickr Audio Caption Corpus](https://groups.csail.mit.edu/sls/downloads/flickraudio/) because the research team is now focusing on finishing their submissions for a conference. This means that the current models aren't ideal or anywhere near news worthy, but if you search for a dog, you'll get images of dogs, but fine-grained detail will be missed.


The research scientist approaches you and asks if you can create a proof of concept (POC) for a product using the system to demonstrate to everyone the importance of the system, while also allowing them to find out what should be changed in the training and design process for the models to work. The only constraint for the product, according to the scientist, is that it performs multimodal search, either speech->image, image->speech, or both directions. Despite the scientist's belief that the provided models will be a good starting point, the product does not have to be built using the provided models. You can either fine-tune the existing models or create entirely new model architectures and train them from scratch.

This is a common scenario at ByteFuse, where we are tasked with finding practical applications for machine learning systems before devoting all of our time and resources to the machine learning system. We want you to feel this rush right now. Take advantage of our models (or don't) and impress us with your work! We will not be judging how well the machine learning component works, but rather how novel the idea is and how well your product works as a concept.

Here are some guidelines and additional rules:

* All of the model architectures and weights can be found in this repository.
* For the models to work, the data must be in a specific format, and utils.py includes functions for converting a given image/audio signal (or file path) to the correct format. Make yourself comfortable with all the code in the file.
* As long as the product performs multimodal search, you are not required to use the models or anything else provided. We are looking for a proof of concept rather than a finished product. **Do not spend money on this; instead, make use of free resources.**
* The product and code **must** be open source.
* If you have finished your product, send an email to careers@bytefuse.ai saying you have finshed. We will then organize a meeting with you where you can demo your product to us. This meeting can then be recorded, if you wish so, and shared to the community. 
* For this challenge, the final deadline to send an email is December 10th.


If your proof of concept really impresses us, you'll be invited to join our Llama program, where we can help you take this project even further while introducing you to our ecosystem and giving you experience working with systems that process millions of requests.

This repo also includes a starter notebook that will download the data and current weights, allowing you to then go crazy with the models and data to get you started and perhaps spur you on if you plan to fine tune the models. You can see where the models are still struggling to generalize or are simply not working by experimenting with the data. 

You can open a getting started notebook with this link [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByteFuse/indabax-2021/blob/main/llama-challenge/llama-challenge-starter-kit.ipynb)

Lastly, just have fun! If anything, this challenge really is all about growing your machine learning and software skills. Any questions can be raised in the issues, or contact us at info@bytefuse.

