from sc2.bot_ai import BotAI
import torch
from actions import possible_inputs, possible_actions, possible_actions_async, possible_inputs_async, get_map
from config import *
import numpy as np
import torch.nn as nn
from sc2.ids.unit_typeid import UnitTypeId
from GTrXL_Agent import Memory


def reward_fun(self):
    reward = 0
    for zergling in self.units(UnitTypeId.ZERGLING):
        if zergling.is_attacking:
            if len(self.enemy_units.filter(lambda unit: (not unit.is_flying) and (not unit.is_burrowed) and (not unit.is_cloaked)).closer_than(zergling.ground_range + 2, zergling)) > 0 or len(self.enemy_structures.closer_than(zergling.ground_range + 3, zergling)) > 0:
                reward += 0.02
    reward -= (0.1/((len(self.units(UnitTypeId.ZERGLING)) + (2*self.already_pending(UnitTypeId.ZERGLING)))/10 + 1)) + 5e-2
    return reward




class Star_Eyed_Trainer(BotAI):

    def __init__(self, actor, actor_optim, device: str='cuda'):    
        self.actor = actor

        self.actor_optim = actor_optim
        self.memory = Memory(batch_size)
        self.device = device

        self.current_state = None
        self.current_map = None
        self.map_idx = None
        self.race_idx = None
        self.frac = 1
        self.curr_reward = 0

        self.current_segment = torch.zeros((memory_size-1, 1, embedding_dim)).to(device)
        self.old_memories = torch.zeros((memory_size, 4, memory_size, 1, embedding_dim)).to(device)


    def load_checkpoint(self, checkpoint):
        self.actor.load_state_dict(checkpoint['model'])
        self.actor_optim.load_state_dict(checkpoint['optim'])
        self.memory = checkpoint['memory']
        self.frac = checkpoint['lr_frac']

    def set_race_and_map(self, map_idx, num_maps, race_idx, num_races):
        self.map_idx = torch.tensor([1 if i == map_idx else 0 for i in range(num_maps)]).to(self.device)
        self.race_idx = torch.tensor([1 if i == race_idx else 0 for i in range(num_races)]).to(self.device)
        self.current_segment = torch.zeros((memory_size-1, 1, embedding_dim)).to(self.device)
        self.old_memories = torch.zeros((memory_size, 4, memory_size, 1, embedding_dim)).to(self.device)

    async def on_step(self, iteration:int):
        if iteration % 100 == 0:
            await self.distribute_workers()

        if self.current_state is None:
            self.current_state = torch.cat((torch.tensor([iteration/25000] + [get_info(self) for get_info in possible_inputs] + [await get_info(self) for get_info in possible_inputs_async]).to(self.device), self.race_idx, self.map_idx))
            self.current_map = get_map(self)[None, :]

            self.current_state = self.current_state.unsqueeze(dim=0).float()
            self.current_map = torch.tensor(self.current_map).float().to(self.device)

        value = self.actor.forward_critic(self.current_state, self.current_map, self.current_segment, self.old_memories[0])
        action_dist, next_embedding, old_memory = self.actor(self.current_state, self.current_map, self.current_segment, self.old_memories[0])

        self.old_memories[:memory_size-1] = self.old_memories[-(memory_size-1):].clone()
        self.old_memories[-1] = old_memory

        self.current_segment[:(memory_size-2)] = self.current_segment[-(memory_size-2):].clone()
        self.current_segment[-1] = next_embedding.detach()

        action = action_dist.sample()

        prob = action_dist.log_prob(action).squeeze().item()
        value = value.squeeze().item()

        if action < len(possible_actions):
            self.curr_reward = possible_actions[action](self)
        else:
            try:
                self.curr_reward = await possible_actions_async[action-len(possible_actions)](self)
            except:
                self.curr_reward = 0

        self.curr_reward += reward_fun(self)

        curr_probs = [round(np.exp(action_dist.log_prob(torch.tensor(i).to(self.device)).cpu().item()), 3) for i in range(len(possible_actions) + len(possible_actions_async))]
        print(f"{curr_probs} - Reward: {self.curr_reward}", end='                   \r')

        self.current_state = torch.cat((torch.tensor([iteration/25000] + [get_info(self) for get_info in possible_inputs] + [await get_info(self) for get_info in possible_inputs_async]).to(self.device), self.race_idx, self.map_idx))
        self.current_map = get_map(self)[None, :]

        self.current_state = self.current_state.unsqueeze(dim=0).float()
        self.current_map = torch.tensor(self.current_map).float().to(self.device)

        done = False

        self.memory.store_memory({
            'states': self.current_state,
            'maps': self.current_map,
            'memories': self.current_segment,
            'old_memories': self.old_memories[0],
            'probs': prob,
            'values': value,
            'actions': action,
            'rewards': self.curr_reward,
            'dones': done
        })


        if iteration % N == 0 and iteration != 0:
            self.frac *= .99985
            lrnow = self.frac * learning_rate_agent
            self.actor_optim.param_groups[0]["lr"] = lrnow


            for _ in range(n_epochs):
                states, batches = self.memory.generate_batches()

                rewards = states['rewards']
                values = states['values']

                # GAE
                advantage = np.zeros(len(rewards), dtype=np.float32)

                for t in range(len(rewards)-1):
                    discount = 1
                    a_t = 0
                    for k in range(t, len(rewards)-1):
                        a_t += discount*(rewards[k] + gamma*values[k+1]*(1-int(states['dones'][k])) - values[k])
                        discount *= gamma*gae_lambda
                    advantage[t] = a_t

                advantage = (advantage - np.mean(advantage))/(np.std(advantage) + 1e-8)
                advantage = torch.tensor(advantage).to(self.device)

                rewards = torch.tensor(rewards).to(self.device)
                values = torch.tensor(values).to(self.device)
                observations = torch.cat(states['states']).float().to(self.device)
                maps = torch.cat(states['maps']).float().to(self.device)
                memories = torch.cat(states['memories'], axis=1).to(self.device)
                old_memories = torch.cat(states['old_memories'], axis=2).to(self.device).detach()
                old_probs = torch.tensor(states['probs']).to(self.device)
                actions = torch.tensor(states['actions']).to(self.device)

                # MINIBATCH TRAINING
                for batch in batches:                 
                    
                    # Actor Loss
                    dist, _, _ = self.actor(observations[batch], maps[batch], memories[:, batch], old_memories[:, :, batch])
                    critic_value = self.actor.forward_critic(observations[batch], maps[batch], memories[:, batch], old_memories[:, :, batch]).squeeze()

                    new_probs = dist.log_prob(actions[batch])
                    prob_ratio = (new_probs-old_probs[batch]).exp()

                    weighted_probs = advantage[batch] * prob_ratio
                    weighted_clipped_probs = torch.clamp(prob_ratio, 1-policy_clip, 1+policy_clip)*advantage[batch]
                    actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                    # Value Loss
                    v_loss_unclipped = (critic_value - rewards[batch]) ** 2
                    v_clipped = rewards[batch] + torch.clamp(
                        critic_value - values[batch],
                        -policy_clip,
                        policy_clip,
                    )
                    v_loss_clipped = (v_clipped - values[batch]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                    total_loss = actor_loss - (dist.entropy().mean() * entropy_coef) + (v_loss * vf_coef)

                    self.actor_optim.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
                    self.actor_optim.step()


            self.memory.clear_memory()
            
        
