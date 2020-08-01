import numpy as np
import random
from .const import *


'''
check if the custom map is valid and randomly generate agents
'''
def check_custom_map(flag_channel, agent_num):
    if type(flag_channel) is np.ndarray:
        map_dim = flag_channel.shape
        
    elif type(flag_channel) is str:
        flag_channel = np.loadtxt(flag_channel)
        map_dim = flag_channel.shape
    else:
        raise ValueError("Custom map should be either path(str) or np.ndarray matrix")
    if len(map_dim) != 2 or map_dim[0] == 0 or map_dim[1] == 0:
        print('Custom map not 2d or have 0 length')
        raise ValueError
    flag_channel.astype(int)
    agent_channel = np.zeros(flag_channel.shape)
    env_dict = {FLAG:[],AGENT:[]}
    # print(np.where(flag_channel == FLAG))
    for lx, ly in np.array([*np.where(flag_channel == FLAG)]).T:
        env_dict[FLAG].append((lx, ly))
        flag_channel[lx,ly] = FLAG
    if AGENT in flag_channel:
        agent_count = 0
        for lx, ly in np.array([*np.where(flag_channel == AGENT)]).T:
            env_dict[AGENT].append((lx, ly))
            agent_channel[lx, ly] = AGENT
            flag_channel[lx,ly] = BACKGROUND
            agent_count += 1
        assert agent_count == agent_num
    else:
        cand_x, cand_y = np.where(flag_channel == BACKGROUND)
        if len(cand_x) >= agent_num and len(cand_y) >= agent_num:
            idx_list = [random.randint(0, len(cand_x)-1) for _ in range(agent_num)]
            # print('check_custom',idx_list)
            for i in idx_list:
                agent_channel[cand_x[i], cand_y[i]] = AGENT
                env_dict[AGENT].append((cand_x[i], cand_y[i]))
        else:
            print('Not enough space for all agents')
            raise ValueError
    
    return map_dim, np.array([flag_channel, agent_channel]), env_dict

'''
generate a random map with specified agent and flag numbers
'''
def generate_random_map(map_dim, agent_num, flag_num, np_random = None, in_seed = None, obs = True):
    
    assert len(map_dim) == 2

    if np_random is None:
        np_random = np.random
    if in_seed is not None:
        np.random.seed(in_seed)
    if obs:
        lx = int(map_dim[0]/2)
        ly = int(map_dim[1]/2)
        # print(lx,ly)
        obs_upper_left_x = np.random.randint(0, lx)
        obs_upper_left_y = np.random.randint(0, ly)
        obs_bottom_right_x = np.random.randint(obs_upper_left_x+1, map_dim[0])
        obs_bottom_right_y = np.random.randint(obs_upper_left_y+1, map_dim[1])
        # print(obs_upper_left_x, obs_upper_left_y)
        ox, oy = obs_bottom_right_x - obs_upper_left_x, obs_bottom_right_y - obs_upper_left_y
        while map_dim[0]*map_dim[1] - ox*oy < flag_num + agent_num:
            foo = np.random.randint(2)
            ox = max(1, ox - foo)
            oy = max(1, oy - (1 - foo))
            obs_bottom_right_x = obs_upper_left_x + ox
            obs_bottom_right_y = obs_upper_left_y + oy
    
    rand_map = np.zeros(map_dim)
    candidates = []
    for i in range(map_dim[0]):
        for j in range(map_dim[1]):
            if i >= obs_upper_left_x and i < obs_bottom_right_x and j >= obs_upper_left_y and j < obs_bottom_right_y:
                rand_map[i,j] = OBSTACLE
                continue
            candidates.append(i*map_dim[0]+j)

    env_dict = {FLAG:[],AGENT:[]}
    agent_channel = np.zeros(map_dim)
    candidates = np.random.choice(candidates, agent_num + flag_num, replace=False)
    for i in candidates[:agent_num]:
        lx, ly = int(i/map_dim[0]), i%map_dim[0]
        agent_channel[lx, ly] = AGENT
        env_dict[AGENT].append((lx,ly))
    for i in candidates[agent_num:]:
        lx, ly = int(i/map_dim[0]), i%map_dim[0]
        rand_map[lx, ly] = FLAG
        env_dict[FLAG].append((lx,ly))
    return map_dim, np.array([rand_map, agent_channel]), env_dict


if __name__ == "__main__":
    a = np.full((10,10), 1.0)
    a[2,3] = 0.0
    print(check_custom_map(a))
