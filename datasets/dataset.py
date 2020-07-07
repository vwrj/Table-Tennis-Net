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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
torch.multiprocessing.set_sharing_strategy('file_system')

class FrameDataset(Dataset):
    '''
    TODO: Have to revisit this for the full dataset (when we have multiple videos)

    I have all the relevant frames saved as Tensors. 
    I have a list of starting frame_ids. 
        frame_ids has ids that are the start of a 9-consecutive element sequence.

    Labels are found in the label_events and label_ball files. 
    They both are dictionaries. 
        label_events matches the starting frame_id with a tuple of correct event labels: (bounce, net), each a probability between (0, 1). 
        label_ball matches every frame_id with the position of the ball in that frame.

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


    def __getitem__(self, idx):

        start = self.frame_ids[idx]
        middle = start + 4

        # Get the next 8 files
        frames = torch.zeros((9, 3, 1080, 1920))
        for i, j  in enumerate(range(start, start + 9)):
            frames[i] = torch.load(self.data_dir / '{}.pt'.format(j))

        # 2-tuple of target probabilities: (bounce, net)
        event_label = self.label_events[start]
        ball_label = self.label_ball[str(start+8)]
        ball_label = (ball_label['x'], ball_label['y'])

        return frames, torch.tensor(event_label), torch.tensor(ball_label), start, middle

        # Normalize? 




