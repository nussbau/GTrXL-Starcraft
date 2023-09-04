from star_eyed_trainer import Star_Eyed_Trainer
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2 import maps
from network import starcraft_network
from config import *
from actions import possible_actions, possible_inputs, possible_actions_async, possible_inputs_async
from torch.optim import Adam
import torch
from os import listdir
from os.path import exists
import random
from utils import tracker
from star_eyed_bot import Star_Eyed_Bot

map_list = [map.replace(".SC2Map", "") for map in listdir("C:\Program Files (x86)\StarCraft II\Maps")]
races = [Race.Zerg, Race.Terran, Race.Protoss]
device = 'cuda' if torch.cuda.is_available() else 'cpu'



# Creates the bot
tracking_device = tracker()
agent = starcraft_network(len(possible_inputs)+len(possible_inputs_async)+len(map_list)+len(races)+ 1, len(possible_actions)+len(possible_actions_async)).to(device)
actor_optim = Adam(agent.parameters(), lr=learning_rate_actor, eps=adam_epsilon)
train_bot = Star_Eyed_Trainer(agent, actor_optim, tracking_device, device)



if exists("model.pth"):
        train_bot.load_checkpoint(torch.load("model.pth"))


for _ in range(num_games):

        # Sets race and map
        map_idx = random.randint(0, len(map_list)-1)
        race_idx = 2
        train_bot.set_race_and_map(map_idx, len(map_list), race_idx, len(races))

        # Runs the game
        result = run_game(maps.get(map_list[map_idx]), 
                [Bot(Race.Zerg, train_bot), 
                Computer(races[race_idx], Difficulty.Medium)],
                realtime=False)
        
        # If there is no last memeory, add the last state
        if len(train_bot.memory.memory['states']) == 0:
                train_bot.memory.store_memory(train_bot.memory.last_memory)

        # Set the last staet to done, and reward based on W/L
        train_bot.memory.memory['dones'][-1] = True
        if str(result) == 'Result.Defeat':                
                train_bot.memory.memory['rewards'][-1] -= 1000
                print(f"Unfortunate Loss :( - Average Reward: {tracking_device.reward/tracking_device.iteration}")
        else:
                train_bot.memory.memory['rewards'][-1] += 1000
                print(f"Congrats on the Win! - Average Reward: {tracking_device.reward/tracking_device.iteration}")  

        # Reset score tracker              
        tracking_device.reset()

        # Save the model
        torch.save({'model': agent.state_dict(),
                    'optim': actor_optim.state_dict(),
                    'memory': train_bot.memory,
                    'lr_frac': train_bot.frac}, "model.pth")