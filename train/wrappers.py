import gym
import numpy as np
import matsp
from matsp.envs.const import *
from itertools import permutations

class AssignmentWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(AssignmentWrapper, self).__init__(env)
        # self.action_list = list(self.update_action_space(self.agent_num, self.flag_num))
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if not hasattr(self.env, 'action_list'):
            self.env.action_list = list(self.update_action_space(self.env.agent_num, self.env.flag_num))
        # print(self.env.action_list, 'updated')
        return obs


    def action(self, actions):
        action_plan = []
        # print(actions, self.env.action_list, self.env.flag_num, self.env.agent_num)
        for agent_idx, action in enumerate(self.env.action_list[actions]):
            if action == 0:
                action_plan.append(self.env.env_dict[AGENT][agent_idx])
            else:
                action_plan.append(self.flag_listing[action-1])
        # print(action_plan)
        return action_plan

    def update_action_space(self, agent_num, flag_num):
        action_list = set()
        # Note: 0 refers to not moving, 1 refers to flag at index 0
        for tmp in (permutations(range(self.flag_num + self.agent_num), self.agent_num)):
            action_list.add(tuple([0 if i >= self.flag_num else i+1 for i in list(tmp)]))
        return action_list
        

class ObservationConverter(gym.ObservationWrapper):
    def __init__(self, env):
        super(ObservationConverter, self).__init__(env)
        # self.observation_space = (self.env.map_dim[0]*2-1, self.env.map_dim[1]*2-1, self.env.config['elements']['NUM_AGENT'])
    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.observation_space_from_wrapper = (self.env.map_dim[0]*2-1, self.env.map_dim[1]*2-1, self.env.config['elements']['NUM_AGENT'])
        return self.observation(observation)
    def observation(self, obs):
        map_with_agent = np.copy(self.env.nd_map[0]+self.env.nd_map[1])
        for agent_idx, agent in enumerate(obs[AGENT]):
            map_with_agent[agent] = AGENT
        nd_map = np.full((self.env.map_dim[0]*2-1, self.env.map_dim[1]*2-1, self.env.config['elements']['NUM_AGENT']), OBSTACLE).astype(int)
        for agent_idx, agent in enumerate(obs[AGENT]):
            nd_map[self.env.map_dim[0]-1-agent[0]:self.env.map_dim[0]*2-1-agent[0],\
                   self.env.map_dim[1]-1-agent[1]:self.env.map_dim[1]*2-1-agent[1],\
                   agent_idx] = map_with_agent
        return nd_map.astype(float)

class ObservationConverter_FOAs(gym.ObservationWrapper):
    def __init__(self, env):
        super(ObservationConverter_FOAs, self).__init__(env)
        # self.observation_space = (self.env.map_dim[0]*2-1, self.env.map_dim[1]*2-1, self.env.config['elements']['NUM_AGENT'])
    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.observation_space_from_wrapper = (self.env.map_dim[0]*2-1, self.env.map_dim[1]*2-1, self.env.config['elements']['NUM_AGENT'])
        return self.observation(observation)
    def observation(self, obs):
        map_with_agent = np.zeros((self.env.map_dim))
        for agent_idx, agent in enumerate(obs[AGENT]):
            map_with_agent[agent] = AGENT
        state = np.full((self.env.map_dim[0]*2-1, self.env.map_dim[1]*2-1, 2+self.env.config['elements']['NUM_AGENT']), BACKGROUND).astype(int)
        # print(self.env.nd_map[0])
        state[int(self.env.map_dim[0]*0.5):int(self.env.map_dim[0]*1.5), int(self.env.map_dim[1]*0.5):int(self.env.map_dim[1]*1.5), 0] = \
            self.env.nd_map[0]
        # print(state[1][int(self.env.map_dim[1]*0.5):int(self.env.map_dim[1]*1.5), int(self.env.map_dim[1]*0.5):int(self.env.map_dim[1]*1.5)].shape)
        state[int(self.env.map_dim[1]*0.5):int(self.env.map_dim[1]*1.5), int(self.env.map_dim[1]*0.5):int(self.env.map_dim[1]*1.5), 1] = \
            self.env.nd_map[1]
        for agent_idx, agent in enumerate(obs[AGENT]):
            state[self.env.map_dim[0]-1-agent[0]:self.env.map_dim[0]*2-1-agent[0],\
                   self.env.map_dim[1]-1-agent[1]:self.env.map_dim[1]*2-1-agent[1],\
                   agent_idx+2] = map_with_agent
        return state.astype(float)

class HighLevelPlanner(gym.ActionWrapper):
    def __init__(self, env, steps = 10):
        super(HighLevelPlanner, self).__init__(env)
        self.steps = steps
    def step(self, actions):
        path_plan = []
        for i,target in enumerate(actions):
            path_plan.append(self.route_astar(self.env_dict[AGENT][i], target)[1])
        reward = 0
        done = False
        info = {}
        for step in range(self.steps):
            if done:
                return self.env.env_dict, reward, done, info
            actions = []
            for path in path_plan:
                if path is not None and len(path) > step:
                    actions.append(path[step])
                else:
                    actions.append(0)
            env_dict, s_reward, s_done, s_info = self.env.step(actions)
            reward += s_reward
            done = s_done
            info[step] = s_info
        return env_dict, reward, done, info
            
            
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
            return None, None
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
                total_actions = []
                while current in cameFrom:
                    actions = [(0,0), (-1,0), (0,1), (1,0), (0,-1)]
                    total_actions.append(actions.index((current[0]-cameFrom[current][0], current[1]-cameFrom[current][1])))
                    current = cameFrom[current]
                    total_path.append(current)
                total_path.reverse()
                total_actions.reverse()
                return total_path, total_actions

            openSet.remove(current)
            closedSet.add(current)

            directions = [(1,0),(-1,0),(0,1),(0,-1)]
            neighbours = []
            for dx, dy in directions:
                x2, y2 = current
                x = x2 + dx
                y = y2 + dy
                if (x >= 0 and x < self.map_dim[0]) and \
                   (y >= 0 and y < self.map_dim[1]) and \
                   self.nd_map[0][x,y] != OBSTACLE:
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

        return None, None

class StochasticActionWrapper(gym.ActionWrapper):
    # error [int] the possiblity of not successful action for each of the 4 other actions(including no action)
    # error [list] of length 5 by 5, full mdp table
    def __init__(self, env, error = 0):
        super(StochasticActionWrapper, self).__init__(env)
        if (isinstance(error, float) or error == 0) and error >= 0 and error <= 0.25:
            self.mdp = np.full((5,5), error)
            for i in range(5):
                self.mdp[i,i] = 1 - error*4
        elif isinstance(error, list):
            mdp = np.array(error)
            if mdp.shape != (5,5):
                print('Given mdp table not complete, should be of dimension 5 by 5')
                raise ValueError 
            elif not all(np.equal(np.sum(mdp, 0), [1,1,1,1,1])):
                print('Given mdp rows not adding up to 1')
                raise ValueError
            else:
                self.mdp = mdp
                      
    def step(self, actions):
        for i in range(len(actions)):
            actions[i] = np.random.choice(5,1, p=self.mdp[actions[i]])[0]
        # print(actions)
        return self.env.step(actions)

