from sc2.bot_ai import BotAI
import torch
from actions import possible_inputs, possible_actions, possible_actions_async, possible_inputs_async, get_map
from config import *

class Star_Eyed_Bot(BotAI):

    def __init__(self, actor,  device: str='cuda'):    
        self.actor = actor
        self.device = device

        self.map_idx = None
        self.race_idx = None


    def load_checkpoint(self, checkpoint):
        self.actor.load_state_dict(checkpoint['model'])

    def set_race_and_map(self, map_idx, num_maps, race_idx, num_races):
        self.map_idx = torch.tensor([1 if i == map_idx else 0 for i in range(num_maps)]).to(self.device)
        self.race_idx = torch.tensor([1 if i == race_idx else 0 for i in range(num_races)]).to(self.device)

    async def on_step(self, iteration:int):
        if iteration % 100 == 0:
            await self.distribute_workers()

        current_state = torch.cat((torch.tensor([iteration/25000] + [get_info(self) for get_info in possible_inputs] + [await get_info(self) for get_info in possible_inputs_async]).to(self.device), self.race_idx, self.map_idx))
        current_map = get_map(self)[None, :]

        current_state = current_state.unsqueeze(dim=0).float()
        current_map= torch.tensor(current_map).float().to(self.device)
        
        action_dist = self.actor(current_state, current_map)

        action = action_dist.sample()

        if action < len(possible_actions):
            possible_actions[action](self)
        else:
            try:
                await possible_actions_async[action-len(possible_actions)](self)
            except:
                pass
        
        



    
            
        
