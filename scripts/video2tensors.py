import json
import os
import pdb
from pathlib import Path
import argparse
import pickle
import cv2
import torch

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Converting relevant video frames to PyTorch tensors')
    parser.add_argument('--root-dir', default='/scratch/vr1059/ttnet/')
    parser.add_argument('--game-dir', default='game_1')
    parser.add_argument('--phase', default='train')
    parser.add_argument('--f', default='events_markup.json', help='input must be JSON file.')
    parser.add_argument('--update-until-video', action='store_true')

    args = parser.parse_args()

    args.root_dir = Path(args.root_dir)
    args.game_dir = Path(args.game_dir)

    output_dir = args.root_dir/args.game_dir/args.phase
    print("Operating under {}".format(args.game_dir))
    print("Output dir --> Saving tensors in {}".format(output_dir))

    # If output_dir not created, do so. 
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # load events json file
    fn = args.root_dir/args.game_dir/args.f
    with open(fn) as json_file:
        loaded_json = json.load(json_file)

    '''
    I need a match between starting frame ID and the events_spotting label. especially if it's a bounce. 
    What are the event labels --> bounce, net, empty event. 
    Empty event --> (0, 0)
    Bounce --> (1, 0)
    Net --> (0, 1)

    However, it's more complicated. 
    Suppose '18' is a bounce event. 
    I have event target probabilities from 14 to 22. 
    The event target probability for a starting frame f is always that of f+4. 
    So I can run through the events dict. 
    And create an label_events dict. 

    In a 9-frame sequence, I always predict the event target probability of the 5th frame. 

    So given 18 is a bounce event. 
    I have event target probabilities from 14 to 22. 
    These correspond to starting ids from 10 to 18. 


    I have target_probabilities for a bounce event: [0, 0.38, 0.70, 0.92, 1, 0.92, 0.70, 0.38, 0]
    I go through events dict. 
    Get 18. 
    Get event target probabilities from 14 to 22. 
    for i, x in range(10, 19):
        label_dict[x] = target_probabilities[i] 

    # start_id: 5th frame event target probability
    label_dict = {
        10: 0 (14),
        11: 0.38 (15),
        12: 0.70 (16),
        13: 0.92 (17),
        14: 1 (18),
        15: 0.92 (19),
        16: 0.70 (20),
        17: 0.38 (21),
        18: 0 (22)

    }


    '''

    # Collect event labels. 
    label_events = dict()
    target_probabilities = [0, 0.38, 0.70, 0.92, 1, 0.92, 0.70, 0.38, 0]
    for (fid, y) in loaded_json.items():
        fid = int(fid)
        if y == 'bounce':
            for i, x in enumerate(range(fid-8, fid+1)):
                label_events[x] = (target_probabilities[i], 0)
        elif y == 'net':
            for i, x in enumerate(range(fid-8, fid+1)):
                label_events[x] = (0, target_probabilities[i])
        elif y == 'empty_event':
            label_events[fid-4] = (0, 0) 

    output_fn = args.root_dir/args.game_dir/args.phase/'label_events.pkl'
    with open(output_fn, 'wb') as output:
        pickle.dump(label_events, output)

    # Collect frame_ids. 
    # Frame_ids collected are at the start of their 9-element consecutive sequence. (For training, we deal with a stack of 9 consecutive frames at at ime). 
    '''
    For an event 18:
    I need 10 to 18 in frame_ids. 
    And I need to save 10 - 26. 

    if I have an empty event,
    then what do I want in frame_ids? Hm. 
    I'm assuming that frame_ids has all starting frames with 9-element sequences.
    We're also just adding .... But it has to be 9-element sequences. 
    Even the empty_event has to have 9-element sequences. 
    Okay so what's the 9-element sequence for an empty_event? 
    Okay. For an empty_event:
    construct *one* 9-len sequence
    with the fifth frame being the empty_event id and the last frame being id+4

    So '58' is an empty_event. 
    '62' is in ball_markup but nothing else. 
    So let 54 be the starting frame of the 9-element sequence. 
    Its middle frame will have a (0, 0) event. And its last frame will have ball coordinates. 

    '''
    f_dict = set()
    frame_ids = []
    valid_events = ('bounce', 'net', 'empty_event')
    for (fid, y) in loaded_json.items():
        assert y in valid_events
        fid = int(fid)
        if y in valid_events[:2]:
            for i in range(fid - 8, fid + 1):
                if i not in f_dict:
                    frame_ids.append(i)
                    f_dict.add(i)
        else:
            i = fid-4
            if i not in f_dict:
                frame_ids.append(i)
                f_dict.add(i)
                
    s = set()
    for x in loaded_json:
        fid = int(x)
        for i in range(fid - 8, fid + 11):
            s.add(i)
                
    print(len(frame_ids))
    print(frame_ids[-10:])

    output_fn = args.root_dir/args.game_dir/args.phase/'frame_ids.pkl'
    with open(output_fn, 'wb') as output:
        pickle.dump(frame_ids, output)

    '''
    I'm just going to save all the relevant frames as a PyTorch tensor. 
    And then in the dataset get_item, pick the next 9 and package them as a sequence. 
    That should be okay. Although I think I have too many of the empty_event? Because I'm doing 9 for each one, even though I don't need to. idk I think it's right. 
    so how should i do this
    go through all the video frames 
    if it matches any event, save it as a PyTorch tensor
    my set s should contain all relevant frame_ids

    *Check* : `events` file has Frame 18 as a 'bounce' event. I loaded the tensor 18.pt and it is in fact a bounce event. Frame 33 is 'net' which corresponds. Frame 44 is 'bounce' which corresponds. 
    *Note*: Frame 33 was a net, but the ball then bounced on the opponent's side afterwards. What happens if it bounces back on the player's side after hitting the net? What is the next event? 


    '''
    if not args.update_until_video:
        saved_frames = []
        video_fn = args.root_dir/args.game_dir/'{}.mp4'.format(args.game_dir)
        cap = cv2.VideoCapture(str(video_fn))
        final_frame = frame_ids[-1] + 1
        print("Final frame is ", final_frame)

        for f in range(final_frame):
            ret, frame = cap.read()
            if f % 5000 == 0:
                print('We are at ', f)
                print('\n')
                print(saved_frames[-100:])

            if f in s:
                if ret:
                    frame = torch.from_numpy(frame)
                    # HWC2CHW
                    frame = frame.permute(2, 0, 1)
                    frame_fn = output_dir/'{}.pt'.format(f)
                    torch.save(frame, frame_fn)
                    saved_frames.append(f)
                else:
                    print("Skipped!")
                    failed_clip = True
                    break
    

