import os

import numpy as np
import librosa

import torch
import torch.nn.functional as F
import torchvision

import torch.nn as nn
import torch.nn.functional as F


def set_parameter_requires_grad_for_finetuning(model):
    '''
    Function to speed up the process of freezing layers in a network you want to 
    fine-tune. 

    args:

    model: the model you want to fine tune. An instance of a torch.nn.Module class.

    returns:

    The input model, where all parameters are set to not require a gradient. In other 
    words a model with frozen weights.
    '''

    for param in model.parameters():
        param.requires_grad = False
            
    return model


class ImageCnn(nn.Module):
    def __init__(self, encode_dim=256):
        super().__init__()
        backbone = torchvision.models.resnet18(pretrained=False)
        backbone = set_parameter_requires_grad_for_finetuning(backbone)
        backbone.fc = nn.Linear(512, encode_dim)
        self.backbone = backbone
        self.gelu = nn.GELU()
        
    def forward(self, x):
        x = self.backbone(x)  
        return x


class AudioCnn(nn.Module):
    def __init__(self, encode_dim=256):
        super().__init__()
        backbone = torchvision.models.resnet18(pretrained=False)
        backbone.fc = nn.Linear(512, encode_dim)
        self.backbone = backbone
        self.gelu = nn.GELU()
        
    def forward(self, x):
        x = x.repeat(1,3,1,1)
        x = self.backbone(x)
        return x


def process_audio(original_sample_rate=None, audio_signal=None, audio_path=None):
    
    '''
    Function to preprocess either a raw audio input, or audio from a given file path. The 
    function will return a melspectrogram, with a shape of (1, 128, 701) that can be used by
    your model.

    Arguments:

    audio:                   Raw audio input signal, either a numpy array or a torch tensor. When you provide this,
                             audio_path must be None.
    audio_path:              Full path to a file to be loaded. File formats supported are wav, mp3 or mp4. When you
                             provide this, audio_signal and sample_rate must be None.
    original_sample_rate:    This is the sample rate at which the audio was sampled.


    Output:

    A melspectrogram version of the input audio provided, as a tensor. The shape of the tensor is (1, 128, 701), 
    corresponding to ONLY the first 7 seconds of your audio data. 

    '''

    assert audio_signal or audio_path, 'Either audio_signal or audio_path must be provided'
    
    if audio_path:
        audio, _ = librosa.load(audio_path, original_sample_rate)


    required_sample_rate = 16e3
    max_samples = int(required_sample_rate*7)

    if original_sample_rate!=required_sample_rate:
        audio = librosa.resample(audio, original_sample_rate, required_sample_rate)
    
    audio = torch.tensor(np.expand_dims(audio, axis=0))

    if audio.shape[0]>1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    if audio.size(-1)<max_samples:
        audio = F.pad(audio, (0, int(max_samples-audio.size(-1))), 'constant', 0)
    elif audio.size(-1)>max_samples:
        audio = audio[:, :max_samples]  
    
    # try converting tensor to numpy otherwise it is already a numpy array.
    try:
        audio = librosa.effects.preemphasis(audio.numpy()[0], coef=0.97)
    except:
        audio = librosa.effects.preemphasis(audio[0], coef=0.97)

    mel_spec = librosa.feature.melspectrogram(
        audio,
        sr=16000,
        hop_length=int(320/2), 
        win_length=320, 
        n_fft=512,
        n_mels=128, 
        fmin=40,
        norm=1,
        power=1,
    )

    logmel = librosa.amplitude_to_db(mel_spec)
    logmel = np.maximum(logmel, -80)
    logmel = torch.tensor(logmel/80).unsqueeze(0)  

    return logmel


def process_image(image=None, image_path=None):
    
    '''
    Function to preprocess either a given image, or audio from a given file path. The 
    function will return the image resized to a shape (3,224,224) and normalized. 

    Arguments:

    image:          Raw image pixel scaled to be between 0 and 1. Expected to be in the shape (3, H, W), where
                    H and W can be any size.
    image_path:     Full path to a file to be loaded.


    Output:

    Image tensor resized to a shape (3,224,224) and normalized.

    '''

    assert image or image_path, 'Either audio_signal or audio_path must be provided'
    
    imgage_process = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.224, 0.225])
        ])

    if image_path:
        image = torchvision.io.read_image(image_path)/255

    image = imgage_process(image)

    return image


def return_pretrained_models(root_path_to_weights):
    '''
    This function will load the pretrained models, with their respective
    weights in the root folder provided.

    Arguments:

    root_path_to_weights:       The root path towards the folder containing the pretrained weights.

    Output:
    The audio and image model, already placed in evaluation state, placed on the CPU

    audio_model(nn.Module), image_model(nn.Module)

    '''

    audio_model = AudioCnn(encode_dim=256)
    image_model = ImageCnn(encode_dim=256)

    audio_model.load_state_dict(torch.load(os.path.join(root_path_to_weights, 'audio_encoder_weights.pt')))
    image_model.load_state_dict(torch.load(os.path.join(root_path_to_weights, 'image_encoder_weights.pt')))

    audio_model.eval()
    image_model.eval()

    return audio_model, image_model