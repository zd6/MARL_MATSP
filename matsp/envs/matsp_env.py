import __future__

import json
import os
import pkg_resources
import sys

import gym
from gym.utils import seeding
from .env_map import *

import numpy as np
from .const import *
from scipy.special import comb
from itertools import permutations
import concurrent.futures


"""
Base environment
__init__: create a default environment
reset:
    if configuration file path given: reset the configuration
    else: use previous configuration
    if custom_board given: reset the environment by randomly generate specified number of flags
    else: randomly generate a map with random obstacles, given number of flags and agents

state:
    the space-state is stored as 'self.env_dict'
action:
    takes in an array of integers(0-4) indicating [no move, W, N, E, S]
render:
    gives a 2d array of the scene

"""
class GridEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"],
    "video.frames_per_second" : 50
    }
    ACTION = []
    def __init__(self, map_size = 20, custom_board = None):
        self.seed()
        self.viewer = None
        self.custom_board = custom_board
        self.config_path = pkg_resources.resource_filename(__name__, os.path.join(os.path.curdir, 'default.json'))
        self._parse_config(self.config_path)
        self.flags_lookup = {}
        self.env_dict = {FLAG:[], AGENT:[]}
        self.reset()
        self.error_rate = 0.1
        

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _parse_config(self, config_path):
        with open(config_path) as json_file:
            self.config = json.load(json_file)
    
    def reset(self, custom_board = None, config_path = None):
        # print(type(custom_board))
        self.agent_num = self.config['elements']['NUM_AGENT']
        self.flag_num = self.config['elements']['NUM_FLAG']
        if custom_board is not None:
            self.custom_board = custom_board
        if config_path is not None:
            self.config_path = config_path
        self._parse_config(self.config_path)
        if self.custom_board is not None:
            # print('calling check_custom_map')
            self.env_dict[FLAG] = []
            # print('reset', self.config)
            self.map_dim, nd_map, self.env_dict = check_custom_map(self.custom_board, self.config['elements']['NUM_AGENT'])
            # print('reset', self.env_dict)
        else:
            self.env_dict = {}
            self.map_dim, nd_map, self.env_dict = generate_random_map(\
                (self.config['map']['lx'], self.config['map']['ly']),\
                self.agent_num,\
                self.flag_num,\
                np_random = self.np_random\
            )
        
        self.agent_num = self.config['elements']['NUM_AGENT']
        self.flag_num = self.config['elements']['NUM_FLAG']
        self.error_rate = self.config['settings']['ERROR_RATE']
        self.flags_lookup = {}
        for locs in self.env_dict[FLAG]:
            self.flags_lookup[locs] = True
        self.flag_listing = list(self.flags_lookup.keys())
        # if not self.action_list:
        #     action_list = set()
        #     for tmp in (permutations(range(self.config['elements']['NUM_FLAG'] + self.config['elements']['NUM_AGENT']), self.config['elements']['NUM_AGENT'])):
        #         action_list.add(tuple([0 if i >= self.config['elements']['NUM_FLAG'] else i+1 for i in list(tmp)]))
        #     self.action_list = list(action_list)
        self.stochastic = bool(self.config['settings']['STOCH'])
        self.action_space = gym.spaces.Discrete(5)
        self.run_step = 0
        self.is_done = False
        self.static_map = nd_map[0]
        self.agent_channel = nd_map[1]
        return self.env_dict
    
    # [O, W, N, E, S]
    def step(self, entities_action, train = False):
        moves_list = [[0,0], [-1, 0], [0, 1], [1, 0], [0, -1]]
        reward = STEP_REWARD
        info = {'action':[], 'event':[]}
        for i, action in enumerate(entities_action):
            cur_agent_loc = np.array(self.env_dict[AGENT][i])
            if train:
                self.agent_channel[self.env_dict[AGENT][i]] = BACKGROUND
            else:
                self.agent_channel[self.env_dict[AGENT][i]] = AGENT_TRAIL
            cur_agent_loc += moves_list[action]
            cur_agent_loc_tuple = tuple(cur_agent_loc)
            if cur_agent_loc_tuple[0] >= self.map_dim[0] or cur_agent_loc_tuple[0] < 0 \
                or cur_agent_loc_tuple[1] >= self.map_dim[1] or cur_agent_loc_tuple[1] < 0 \
                or self.static_map[cur_agent_loc_tuple] == OBSTACLE:
                cur_agent_loc_tuple = self.env_dict[AGENT][i]
                info['action'].append((self.env_dict[AGENT][i], 'bump', cur_agent_loc_tuple))
            else:
                info['action'].append((self.env_dict[AGENT][i], action, cur_agent_loc_tuple))
            self.env_dict[AGENT][i] = cur_agent_loc_tuple
            self.agent_channel[cur_agent_loc_tuple] = AGENT
            if cur_agent_loc_tuple in self.flags_lookup and self.flags_lookup[cur_agent_loc_tuple]:
                info['event'].append(cur_agent_loc_tuple)
                self.flags_lookup[cur_agent_loc_tuple] = False
                reward += FLAG_REWARD
                self.env_dict[FLAG].remove(cur_agent_loc_tuple)
                self.static_map[cur_agent_loc_tuple] = BACKGROUND
            if sum(self.flags_lookup.values()) == 0:
                reward += FINISH_REWARD
                self.is_done = True
                break
        
        # print(self.env_dict)
        info['done'] = self.is_done
        return self.env_dict, reward, self.is_done, info



    # def step(self, entities_action): 
    #     plan = self.action_list[entities_action]
    #     info = {}
    #     reward = STEP_REWARD*self.config['communication']['COM_FREQUENCY']
    #     with concurrent.futures.ProcessPoolExecutor(max_workers= 1) as executor:
    #         results = executor.map(self.step_agent, range(len(plan)), plan)
    #     for agent_idx, pos, flags in results:
    #         self.nd_map[1][self.env_dict[AGENT][agent_idx][0]][self.env_dict[AGENT][agent_idx][1]] = BACKGROUND
    #         self.nd_map[1][pos[0]][pos[1]] = AGENT
    #         # print(pos)
    #         self.env_dict[AGENT][agent_idx] = pos
    #         # print('flags', flags)
    #         for flag in flags:
    #             assert flag in self.env_dict[FLAG]
    #             if self.flags_lookup[self.env_dict[FLAG].index(flag)]:
    #                 self.flags_lookup[self.env_dict[FLAG].index(flag)] = False
    #                 reward += FLAG_REWARD
    #             self.nd_map[0][flag[0]][flag[1]] = BACKGROUND
    #             if sum(self.flags_lookup.values()) == 0:
    #                 reward += FINISH_REWARD
    #                 self.is_done = True
    #                 return self.nd_map, reward, self.is_done, info
    #     return self.nd_map, reward, self.is_done, info

    # # [O, W, N, E, S]
    # def step_agent(self, agent_idx, flag_target):
        
    #     moves_list = [[0,0], [-1, 0], [0, 1], [1, 0], [0, -1]]
    #     if flag_target == 0:
    #         return agent_idx, self.env_dict[AGENT][agent_idx], []
    #     agent_route = self.route_astar(self.env_dict[AGENT][agent_idx], self.env_dict[FLAG][flag_target-1])
    #     # print(agent_route)
    #     if len(agent_route) <= 1:
    #         return agent_idx, self.env_dict[AGENT][agent_idx], []
    #     flag_collected = []
    #     if self.stochastic:
    #         drift = np.array([0,0])
    #         # counter = 0
    #         for path_idx, _ in enumerate(agent_route[1:min(self.config['communication']['COM_FREQUENCY']+1, len(agent_route))]):
    #             prob = [self.error_rate, self.error_rate, self.error_rate, self.error_rate, self.error_rate]
    #             move = moves_list.index(list(np.array(agent_route[path_idx+1]) - np.array(agent_route[path_idx])))
    #             # print(move)
    #             prob[move] = 1 - 4*self.error_rate
    #             move = np.random.choice(5, p = prob)
    #             tmp = np.copy(drift)
    #             drift += moves_list[move]
    #             cur_pos = np.array(agent_route[0]) + drift
    #             # print(drift)
    #             if cur_pos[0] < 0  or cur_pos[0] >= self.map_dim[0] or cur_pos[1] < 0 or cur_pos[1] >= self.map_dim[1] or self.static_map[cur_pos[0]][cur_pos[1]] == OBSTACLE:
    #                 drift = tmp
    #                 cur_pos = np.array(agent_route[0]) + drift
    #             if self.nd_map[0][cur_pos[0]][cur_pos[1]] == FLAG:
    #                 flag_collected.append(tuple(cur_pos))
    #         # print('counter', counter)
    #         return agent_idx, tuple(cur_pos), flag_collected
    #     else:
    #         for cur_pos in agent_route[0:min(self.config['communication']['COM_FREQUENCY']+1, len(agent_route))]:
    #             # print(self.nd_map[0][cur_pos[0]][cur_pos[1]], cur_pos)
    #             if self.nd_map[0][cur_pos[0]][cur_pos[1]] == FLAG:
    #                 flag_collected.append(tuple(cur_pos))
    #         return agent_idx, tuple(agent_route[\
    #              min(self.config['communication']['COM_FREQUENCY'], len(agent_route)-1)\
    #              ]), flag_collected
    def close(self):
        pass
    def render(self, mode):
        render_map = np.copy(self.static_map)
        render_map += self.agent_channel
        # for loc in self.env_dict[AGENT]:
        #     if render_map[loc] == OBSTACLE:
        #         print('Agent in the redzone')
        #         raise ValueError
        #     render_map[loc] = AGENT
        return render_map

    def distance(self, start, goal, euc=False):
        """
        Distance between two point
        Use L1 norm distance for grid world

        Args:
            start (tuple)
            end (tuple)
            euc (boolean): Set true to make it Euclidean distance

        return:
            int
        adapted from Neale's code: https://github.com/vanstrn/RL
        """
        if euc:
            return ((start[0]-goal[0])**2 + (start[1]-goal[1])**2) ** 0.5
        return abs(start[0]-goal[0]) + abs(start[1]-goal[1])
    
    def route_astar(self, start, goal):
        """
        Finds route from start to goal.
        Implemented A* algorithm

        *The 1-norm distance was used

        Args:
            start (tuple): coordinate of start position
            end (tuple): coordinate of end position

        Return:
            total_path (list):
                List of coordinate in tuple.
                Return None if path does not exist.

        """

        openSet = set([start])
        closedSet = set()
        cameFrom = {}
        fScore = {}
        gScore = {}
        if len(goal) == 0:
            return None
        fScore[start] = self.distance(start, goal)
        gScore[start] = 0

        while openSet:
            min_score = min([fScore[c] for c in openSet])
            for position in openSet:
                if fScore.get(position,np.inf) == min_score:
                    current = position
                    break

            if current == goal:
                total_path = [current]
                while current in cameFrom:
                    current = cameFrom[current]
                    total_path.append(current)
                total_path.reverse()

                return total_path

            openSet.remove(current)
            closedSet.add(current)

            directions = [(1,0),(-1,0),(0,1),(0,-1)]
            neighbours = []
            for dx, dy in directions:
                x2, y2 = current
                x = x2 + dx
                y = y2 + dy
                if (x >= 0 and x < self.static_map.shape[0]) and \
                   (y >= 0 and y < self.static_map.shape[1]) and \
                   self.static_map[x,y] != OBSTACLE:
                    neighbours.append((x, y))

            for neighbour in neighbours:
                if neighbour in closedSet:
                    continue
                tentative_gScore = gScore[current]  # + transition cost
                if neighbour not in openSet:
                    openSet.add(neighbour)
                elif tentative_gScore >= gScore[neighbour]:
                    continue
                cameFrom[neighbour] = current
                gScore[neighbour] = tentative_gScore
                fScore[neighbour] = gScore[neighbour] + self.distance(neighbour, goal)

        return None

    def _size_to_dim(self, map_size):
        if isinstance(map_size, int):
            return (map_size, map_size)
        elif isinstance(map_size, list):
            if len(map_size) == 2:
                return tuple(map_size)
            else:
                print('map_size should be an integer, a list of length 2, or a tuple of length 2.')
                raise ValueError
        elif isinstance(map_size, tuple):
            if len(map_size) == 2:
                return map_size
            else:
                print('map_size should be an integer, a list of length 2, or a tuple of length 2.')
                raise ValueError
        else:
            print('map_size should be an integer, a list of length 2, or a tuple of length 2.')
            raise ValueError          
# State space for capture the flag
class Board(gym.spaces.Space):
    """A Board in R^3 used for CtF """
    def __init__(self, map_dim, agent_num, flag_num, dtype=np.uint8):
        self.dtype = np.dtype(dtype)
        self.shape = tuple(map_dim)
        self.agent_num, self.flag_num = agent_num, flag_num
        super(Board, self).__init__(self.shape, self.dtype)

    def __repr__(self):
        return "Board" + str(self.shape)

    def sample(self):
        _, state, _ = generate_random_map(self.shape, self.agent_num, self.flag_num)
        return state



        



