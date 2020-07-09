import pickle
import collections as col
import numpy as np
import json
import pdb
import os
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import scipy.stats as stats
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
torch.multiprocessing.set_sharing_strategy('file_system')

class FrameDataset(Dataset):
    '''
    TODO: Have to revisit this for the full dataset (when we have multiple videos)

    I have all the relevant frames saved as Tensors. 
    `frame_ids` has start frames: frames that are the start of a 9-consecutive element sequence.

    Labels are found in the `label_events` and `label_ball` files. 
    Both are dicts. 
        label_events matches a start frame with a tuple of its event labels: (bounce, net). Each is a probability between (0, 1). 
        label_ball matches every frame with the coordinates of the ball in that frame.

    '''

    def __init__(self, phase, root_dir):
        self.root_dir = Path(root_dir)
        self.phase = phase
        self.data_dir = self.root_dir / self.phase 
        
        # Load list of frame_ids, where each id denotes the starting frame of the next 8 elements (total of 9-frame sequence)
        with open(self.data_dir / 'frame_ids.pkl', 'rb') as f:
            self.frame_ids = pickle.load(f)

        # Load label events dictionary
        with open(self.data_dir / 'label_events.pkl', 'rb') as f:
            self.label_events = pickle.load(f)

        with open(self.root_dir / 'ball_markup.json', 'rb') as f:
            self.label_ball = json.load(f)


    def __len__(self):
        return len(self.frame_ids)

    def gaussian(self, mu, variance, length):
        assert mu <= length
        sigma = math.sqrt(variance)
        x = np.arange(0, length)
        g = stats.norm.pdf(x, mu, sigma)
        g[g< 0.02] = 0
        return torch.tensor(g)


    '''
    I need both high-res and low-res images. 
    low-res image --> global detection --> global coords -->  loss. 
    global coords --> crop high-res --> local coords --> loss. Label generated dynamically at run-time. 

    So in the local detection module,
        we're going to take the center predicted by global module. 
        draw a 320 by 128 crop from the high-res image around that center. 
        predict (x2, y2) of the precise ball location on that crop and calculate loss on that.
        ground-truth label needs to be generated live, at run-time. 
            - if ball is not in crop, the x and y vectors are all just zero. 
            - Otherwise, if in view, it's a one-dimensional gaussian fitted around true coordinates. 

    I'm given labels for high-res, ball_label = (x, y) on the original (1920, 1080) image. 
    *Example*:
        - Event 44 is a bounce. 
        - ball_label = (500, 606) corresponding to (1920, 1080).
        - What is the ball label for low-res? 
            (500 * 320 / 1920, 606 * 128 / 1080)
            (83.3, 71.8)

    '''

    def __getitem__(self, idx):

        start = self.frame_ids[idx]
        middle = start + 4

        # Load the 9-length tensor
        hi_frames = torch.zeros((9, 3, 1080, 1920))
        lo_frames = torch.zeros((9, 3, 128, 320))
        for i, j  in enumerate(range(start, start + 9)):
            tensor = torch.load(self.data_dir / '{}.pt'.format(j))
            hi_frames[i] = tensor 

            img = transforms.ToPILImage()(tensor)
            img = transforms.Resize((128, 320))(img)
            lo_frames[i] = transforms.ToTensor()(img)

        # 2-tuple of target probabilities: (bounce, net)
        event_label = self.label_events[start]
        ball_label = self.label_ball[str(start+8)]
        ball_label = (ball_label['x'], ball_label['y'])

        global_coords = (ball_label[0] * 320 // 1920, ball_label[1] * 128 // 1080)
        global_gaussian_x, global_gaussian_y = self.gaussian(global_coords[0], 1, 320), self.gaussian(global_coords[1], 1, 128)

        return hi_frames, lo_frames, torch.tensor(event_label), torch.tensor(ball_label), global_gaussian_x, global_gaussian_y, start, middle

        # Normalize? 




