# Llama Hackathon

Welcome to the [ByteFuse AI](https://bytefuse.ai/) Llama hackathon. This hackathon differs from other machine learning hackathons in that we want you to focus on what you can do with models, rather than on model design and training.

To understand the goal of this hackathon, put yourself in the shoes of one of the engineers at ByteFuse. You arrive at the office one morning with the news that one of the research scientists has begun developing a system capable of performing multimodal search over both images and speech. The scientist was only able to partially train the system on the [Flickr Audio Caption Corpus](https://groups.csail.mit.edu/sls/downloads/flickraudio/) due to budget constraints. This means that the models are not optimal, but if one searches for a dog, one will get images for dogs back, but fine-grained detail, such as requesting two dogs, will be missed.

The research scientist approaches you, asking if you can create a proof of concept (POC) for a product using the system in order to demonstrate to the people at the top that his system will not be a bad investment. The scientist tells you that the only constraint for the product is that it performs multimodal search, either speech->image, image->speech, or both directions. Despite the scientist's belief that the models provided will be a good starting point, the product does not have to be built using the models provided. You can either fine-tune the models further or design entirely new model architectures.

This is a common scenario at ByteFuse, where we are tasked with finding practical applications for machine learning systems. We want you to experience this rush.  Take our models (or don't ;)) and wow us with your work! The panel will not be judging how well the machine learning component works, but rather how novel the idea is and how well the product as concept works.

Here are some guidelines and additional information:

* This repository contains all of the model architectures (models.py) and weights. The utils.py file contains code for downloading the flickr8 dataset as well as the CSV containing the meta data.
* The data must be in a specific format for the models to work, where  utils.py also includes functions for converting a given image/audio signal (or file path) to the correct format.
* You are not required to use the models or anything else provided, as long as the product performs multimodal search. We want a proof of concept, not a finished product. **Do not spend money on this; instead, use free resources.**
* If you have finished your product, send an email to careers@bytefuse.ai saying you have finshed. We will then organize a meeting with you where you can showcase your product. This meeting can then be recorded, if you wish so, and shared to the community. **The final deadline to send an email is 10 December.**

You can open the getting started notebook with this link [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByteFuse/indabax-2021/blob/main/llama-hackathon/llama-hackathon-starter-kit.ipynb)


The winner will receive **R7500** as well as a guaranteed spot in our LLama program in 2022. 

To get you started, and perhaps spur you on if you are planning to fine tune the models, this repo also contains a starter notebook that will download the required data and weights, allowing you to then go crazy with the models and data. By playing with the data, you will see where the models are still strugling to generalize or simply not working. 

Lastly, just have fun! If anything, this hackathon really is all about growing your machine learning and software skills. Any questions can be raised in the issues, or contact us at info@bytefuse.
