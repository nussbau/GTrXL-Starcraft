from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.unit_command import UnitCommand
from sc2.ids.upgrade_id import UpgradeId
import random
import math
import numpy as np
from sc2.position import Point2



'''
POSSIBLE INPUTS
'''

upgrade_list = [UpgradeId.ZERGLINGMOVEMENTSPEED,
                UpgradeId.ZERGMELEEWEAPONSLEVEL1,
                UpgradeId.ZERGGROUNDARMORSLEVEL1,
                UpgradeId.ZERGMELEEWEAPONSLEVEL2,
                UpgradeId.ZERGGROUNDARMORSLEVEL2,
                UpgradeId.ZERGLINGATTACKSPEED,
                UpgradeId.ZERGMELEEWEAPONSLEVEL3,
                UpgradeId.ZERGGROUNDARMORSLEVEL3]

build_order = [UnitTypeId.SPAWNINGPOOL,
               UnitTypeId.EVOLUTIONCHAMBER,
               (AbilityId.UPGRADETOLAIR_LAIR, UnitTypeId.HYDRALISKDEN),
               UnitTypeId.INFESTATIONPIT,
               (AbilityId.UPGRADETOHIVE_HIVE, UnitTypeId.VIPER)]

def get_map(self):
    map_size = 128
    ratio = map_size/self.game_info.map_size[0]
    map = np.zeros((8, map_size, map_size), dtype=float)

    # draw the minerals:
    for mineral in self.mineral_field:
        pos = mineral.position * ratio
        fraction = mineral.mineral_contents / 1800
        if mineral.is_visible:
            map[0][math.ceil(pos.y)][math.ceil(pos.x)] = fraction
        else:
            map[0][math.ceil(pos.y)][math.ceil(pos.x)] = 0.1

        # draw the enemy units:
        for enemy_unit in self.enemy_units:
            pos = enemy_unit.position * ratio
            # get unit health fraction:
            fraction = enemy_unit.health / enemy_unit.health_max if enemy_unit.health_max > 0 else 0.0001
            if enemy_unit.is_flying:
                map[7][math.ceil(pos.y)][math.ceil(pos.x)] = fraction
            else:
                map[1][math.ceil(pos.y)][math.ceil(pos.x)] = fraction


        # draw the enemy structures:
        for enemy_structure in self.enemy_structures:
            pos = enemy_structure.position * ratio
            # get structure health fraction:
            fraction = enemy_structure.health / enemy_structure.health_max if enemy_structure.health_max > 0 else 0.0001
            map[2][math.ceil(pos.y)][math.ceil(pos.x)] = fraction

        # draw our structures:
        for our_structure in self.structures:
            # if it's a nexus:
            if our_structure.type_id == UnitTypeId.NEXUS:
                pos = our_structure.position * ratio
                fraction = our_structure.health / our_structure.health_max if our_structure.health_max > 0 else 0.0001
                map[3][math.ceil(pos.y)][math.ceil(pos.x)] = fraction
            
            else:
                pos = our_structure.position * ratio
                fraction = our_structure.health / our_structure.health_max if our_structure.health_max > 0 else 0.0001
                map[3][math.ceil(pos.y)][math.ceil(pos.x)] = fraction


        # draw the vespene geysers:
        for vespene in self.vespene_geyser:
            # draw these after buildings, since assimilators go over them. 
            # tried to denote some way that assimilator was on top, couldnt 
            # come up with anything. Tried by positions, but the positions arent identical. ie:
            # vesp position: (50.5, 63.5) 
            # bldg positions: [(64.369873046875, 58.982421875), (52.85693359375, 51.593505859375),...]
            pos = vespene.position * ratio
            fraction = vespene.vespene_contents / 2250

            if vespene.is_visible:
                map[4][math.ceil(pos.y)][math.ceil(pos.x)] = fraction
            else:
                map[4][math.ceil(pos.y)][math.ceil(pos.x)] = 0.1
        # draw our units:
        for our_unit in self.units:
            # if it is a voidray:
            if our_unit.type_id == UnitTypeId.ZERGLING:
                pos = our_unit.position * ratio
                # get health:
                fraction = our_unit.health / our_unit.health_max if our_unit.health_max > 0 else 0.0001
                map[5][math.ceil(pos.y)][math.ceil(pos.x)] = fraction


            else:
                pos = our_unit.position * ratio
                # get health:
                fraction = our_unit.health / our_unit.health_max if our_unit.health_max > 0 else 0.0001
                map[6][math.ceil(pos.y)][math.ceil(pos.x)] = fraction

    return map

def get_minerals(self):
    return self.minerals/1000

def get_vespene(self):
    return self.vespene/1000

def get_larva(self):
    return len(self.larva)/20

def get_workers(self):
    return (len(self.units(UnitTypeId.DRONE)) + self.already_pending(UnitTypeId.DRONE))/30

def get_army(self):
    return self.supply_army/30

def get_supply(self):
    return (self.supply_cap + 8*self.already_pending(UnitTypeId.OVERLORD))/200

def get_num_ground_enemies(self):
    return len(self.enemy_units.filter(lambda unit: (not unit.is_flying) and (not unit.is_burrowed) and (not unit.is_cloaked)))/30

def get_num_air_enemies(self):
    return len(self.enemy_units.filter(lambda unit: (unit.is_flying) and (not unit.is_burrowed) and (not unit.is_cloaked)))/30

def get_currently_upgrading(self):
    for upgrade in upgrade_list:
        upgrade_progress = self.already_pending_upgrade(upgrade)
        if upgrade_progress != 0 and upgrade_progress != 1:
            return True
    return False 

def get_num_upgrades(self):
    total = 0
    for upgrade in upgrade_list:
        if self.already_pending_upgrade(upgrade) == 1:
            total += 1
    return total/len(upgrade_list)

def get_num_halls(self):
    return len(self.townhalls)/6

def get_num_queens(self):
    return (len(self.units(UnitTypeId.QUEEN)) + self.already_pending(UnitTypeId.QUEEN))/10

def get_spawning_pool(self):
    return self.structures(UnitTypeId.SPAWNINGPOOL).exists and (not self.already_pending(UnitTypeId.SPAWNINGPOOL))

def get_evo_chamber(self):
    return self.structures(UnitTypeId.EVOLUTIONCHAMBER).exists and (not self.already_pending(UnitTypeId.EVOLUTIONCHAMBER))

def get_infestation(self):
    return self.structures(UnitTypeId.INFESTATIONPIT).exists and (not self.already_pending(UnitTypeId.INFESTATIONPIT))

def get_lair(self):
    return self.tech_requirement_progress(UnitTypeId.HYDRALISKDEN)

def get_hive(self):
    return self.tech_requirement_progress(UnitTypeId.VIPER)

async def get_can_inject(self):
    for queen in self.units(UnitTypeId.QUEEN):
        if await self.can_cast(queen, AbilityId.EFFECT_INJECTLARVA, only_check_energy_and_cooldown=True):
            return True
    return False

'''
POSSIBLE ACTIONS TO TAKE
'''

def do_nothing(self):
    return 0

async def expand(self):

    total_fields = 0
    total_gas = 0
    for hall in self.townhalls:
        total_fields += len(self.mineral_field.closer_than(10, hall))
        total_gas += len(self.vespene_geyser.closer_than(10, hall))

    num_workers = len(self.units(UnitTypeId.DRONE)) + self.already_pending(UnitTypeId.DRONE)
    
    if num_workers < total_fields *2:
        self.train(UnitTypeId.DRONE)
    else:
        for hall in self.townhalls:
            for geyser in self.vespene_geyser.closer_than(10, hall):
                if (not self.structures(UnitTypeId.EXTRACTOR).closer_than(2.0, geyser).exists) and (self.can_afford(UnitTypeId.EXTRACTOR)) and (not self.already_pending(UnitTypeId.EXTRACTOR)):
                    await self.build(UnitTypeId.EXTRACTOR, geyser)
                    return 0
        if num_workers < total_fields*2 + total_gas*3:
            self.train(UnitTypeId.DRONE)
        elif self.already_pending(UnitTypeId.HATCHERY) == False and self.can_afford(UnitTypeId.HATCHERY):
            await self.expand_now() 
    return 0      



def train_overlord(self):
    if self.can_afford(UnitTypeId.OVERLORD) and ((get_supply(self)*200 - get_army(self)*30 - get_workers(self)*30) < 30) and (get_supply(self)*200 < 200):
        self.train(UnitTypeId.OVERLORD)
    return 0

def train_queen(self):
    if self.can_afford(UnitTypeId.QUEEN) and self.tech_requirement_progress(UnitTypeId.QUEEN) == True:
        for hall in self.townhalls.ready:
            if len(self.units(UnitTypeId.QUEEN).closer_than(7, hall)) + self.already_pending(UnitTypeId.QUEEN) < 2:
                self.train(UnitTypeId.QUEEN, closest_to=hall.position)
                break
    return 0

def train_zergling(self):
    if self.can_afford(UnitTypeId.ZERGLING) and self.tech_requirement_progress(UnitTypeId.ZERGLING) == True:
        self.train(UnitTypeId.ZERGLING)
    return 0

def attack(self):
    for zergling in self.units(UnitTypeId.ZERGLING):
        # if we can attack:
        if self.enemy_units.closer_than(10, zergling).filter(lambda unit: (not unit.is_flying) and (not unit.is_burrowed) and (not unit.is_cloaked)):
            # attack!
            zergling.attack(random.choice(self.enemy_units.closer_than(10, zergling).filter(lambda unit: (not unit.is_flying) and (not unit.is_burrowed) and (not unit.is_cloaked))).position)
        # if we can attack:
        elif self.enemy_structures.closer_than(10, zergling):
            # attack!
            zergling.attack(random.choice(self.enemy_structures.closer_than(10, zergling)).position)
        # any enemy units:
        elif self.enemy_units.filter(lambda unit: (not unit.is_flying) and (not unit.is_burrowed) and (not unit.is_cloaked)):
            # attack!
            zergling.attack(random.choice(self.enemy_units.filter(lambda unit: (not unit.is_flying) and (not unit.is_burrowed) and (not unit.is_cloaked))).position)
        # any enemy structures:
        elif self.enemy_structures:
            # attack!
            zergling.attack(random.choice(self.enemy_structures).position)
        # if we can attack:
        elif self.enemy_start_locations:
            # attack!
            zergling.attack(self.enemy_start_locations[0])
    return 0

def retreat(self):
    if self.units(UnitTypeId.ZERGLING).amount > 0:
        base_center = Point2((0, 0))
        for hall in self.townhalls:
            base_center += hall.position
        base_center /= (len(self.townhalls) + 1e-6)
        retreat_vector = self.enemy_start_locations[0] - base_center
        retreat_vector = 10*(retreat_vector/((retreat_vector[0]**2 + retreat_vector[1]**2)**0.5))
        for zergling in self.units(UnitTypeId.ZERGLING):
            zergling.move(retreat_vector + base_center)
    return 0

def upgrade(self):
    for upgrade_obj in upgrade_list:
        if self.already_pending_upgrade(upgrade_obj) != 1:
            if self.can_afford(upgrade_obj) and self.already_pending_upgrade(upgrade_obj) == 0:
                self.research(upgrade_obj)
            break
    return 0

async def build(self):
    for building in build_order:
        if isinstance(building, UnitTypeId):
            if not self.structures(building).exists:
                if self.can_afford(building) and (self.already_pending(building) == 0):
                    build_vector = self.enemy_start_locations[0] - self.townhalls[-1].position
                    build_vector = 8*(build_vector/((build_vector[0]**2 + build_vector[1]**2)**0.5))
                    rotation = random.uniform(-math.pi/4, math.pi/4)
                    loc = self.townhalls[-1].position + Point2((math.cos(rotation) * build_vector[0] + math.sin(rotation) * build_vector[1], -math.sin(rotation) * build_vector[0] + math.cos(rotation) * build_vector[1]))
                    await self.build(building, near=loc)
                break
        else:
            if not self.tech_requirement_progress(building[1]):
                if await self.can_cast(self.townhalls[-1], building[0]):
                    self.townhalls[-1](building[0], queue=True)
                break
    return 0

async def inject_larva(self):
    for hall in self.townhalls:
        for queen in self.units(UnitTypeId.QUEEN).idle.closer_than(10, hall):
            if await self.can_cast(queen, AbilityId.EFFECT_INJECTLARVA, only_check_energy_and_cooldown=True):
                queen(AbilityId.EFFECT_INJECTLARVA, target=hall, queue=True)
    return 0

async def spread_creep(self):
    for queen in self.units(UnitTypeId.QUEEN).idle:
        if await self.can_cast(queen, AbilityId.BUILD_CREEPTUMOR_QUEEN, only_check_energy_and_cooldown=True):
            loc = await self.find_placement(UnitTypeId.SPINECRAWLER, queen.position, placement_step=3, max_distance=15)
            if loc is not None:
                queen(AbilityId.BUILD_CREEPTUMOR_QUEEN, target=loc, queue=True)
    return 0




'''
LIST OF FUNCTIONS
'''

possible_inputs = [get_minerals, 
                    get_vespene, 
                    get_larva, 
                    get_workers, 
                    get_army, 
                    get_supply,
                    get_num_ground_enemies,
                    get_num_air_enemies,
                    get_num_upgrades,
                    get_currently_upgrading,
                    get_lair,
                    get_hive,
                    get_spawning_pool,
                    get_evo_chamber,
                    get_infestation,
                    get_num_halls,
                    get_num_queens]

possible_inputs_async = [get_can_inject]
        
        
possible_actions = [do_nothing,
                    train_overlord,
                    train_queen,
                    train_zergling,
                    attack,
                    retreat,
                    upgrade]

possible_actions_async = [expand,
                          build,
                          inject_larva]