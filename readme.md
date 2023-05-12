# EECS6895 Video Summary Generation

## Overview

This repo contains a modified version of: [pytorch-vsumm-reinforce](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce), a PyTorch model using reinforcement learning to generate summaries of videos. This project has integrated the model into a web application using Django, where users can upload videos(one at a time) and receive a shortened version of the video, which contains the most representative frames of the original video. The details and the process of training the model can be found in the refered github link above. This repo does not contain functions for training the model. Instead, it is just a web application that takes input videos(currently supports only .mp4 files) and outputs summaries to demonstrate the effectiveness of keyshot-based video summarization using deep learning techniques.
  

## Environment Setup

  
`pip install -r requirements.txt`
  

## Start user app running
  

`python3 manage.py runserver`

![screenshot of homepage](imgs/homepage.png)