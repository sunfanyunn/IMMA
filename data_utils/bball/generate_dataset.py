from Event import Event
from Game import Game
from tqdm import tqdm
import argparse
import glob
import numpy as np
import pandas as pd

all_data = []
desired_sequence_length = 50

for fil in glob.glob('./data/*.json'):
    print(fil)
    df = pd.read_json(fil)
    num_events = len(df) - 1

    for idx in tqdm(range(num_events)):
        event = Event(df['events'][idx])
        frames = len(event.moments)
        if frames > 0:
            if len(event.moments[0].players) == 10:
                default_color = event.moments[0].players[0].color

                for start_idx in range(frames//desired_sequence_length):
                    assert frames >= (start_idx+1)*desired_sequence_length
                    data = []
                    for frame in range(start_idx*desired_sequence_length, (start_idx+1)*desired_sequence_length):
                        moment = event.moments[frame]
                        features = [[player.x, player.y, int(player.color==default_color), 0] for player in moment.players]
                        features.append([moment.ball.x, moment.ball.y, 0, 1])
                        data.append(features)
                    assert len(data) == desired_sequence_length
                    all_data.append(data)
    print(len(all_data))
