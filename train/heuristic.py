from matsp.envs.const import OBSTACLE, FLAG, AGENT
from itertools import combinations
from copy import deepcopy
import mlrose
import numpy as np

class Multi_TSP_Heuristic():
    def __init__(self, env_dict, static_map):
        self.flag_num = len(env_dict[FLAG])
        self.agent_num = len(env_dict[AGENT])
        self.flag_list = deepcopy(env_dict[FLAG])
        self.agent_list = deepcopy(env_dict[AGENT])
        self.static_map = static_map
        self.pathes = {}
        self.dist_list = []
        self.get_pathes()
        self.solved = {}
        self.cost, self.trajectory, self.route = self.multi_tsp()

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
    def get_pathes(self):
        path_dic = {}
        self.agent_idx_list = []
        self.flag_idx_list = []
        for i, agent_loc in enumerate(self.agent_list):
            path_dic[i] = agent_loc
            self.agent_idx_list.append(i)
        for i, flag_loc in enumerate(self.flag_list):
            path_dic[i+self.agent_num] = flag_loc
            self.flag_idx_list.append(i+self.agent_num)
        self.pathes = {}
        self.path_dic = path_dic
        for combo in combinations(range(self.flag_num+ self.agent_num), 2):
            if combo[0] < self.agent_num and combo[1] < self.agent_num:
                continue
            path = self.route_astar(path_dic[combo[0]], path_dic[combo[1]])
            if path:
                self.pathes[combo] = path
                new_path = deepcopy(path)
                new_path.reverse()
                self.pathes[combo[1], combo[0]] = new_path
    def multi_tsp(self):
        partisan_tree = self.partisan_helper(set(self.agent_idx_list), set(self.flag_idx_list))
        generator = self.partisan_traverser(partisan_tree, {})
        self.optimal_cost = np.inf
        optimal_trajectory = None
        optimal_route = None
        for trajectory, route, cost in generator:
            if self.optimal_cost > cost:
                self.optimal_cost = deepcopy(cost)
                optimal_trajectory = deepcopy(trajectory)
                optimal_route = deepcopy(route)
                # print(self.optimal_cost, optimal_route)
        return self.optimal_cost, optimal_trajectory, optimal_route
    def partisan_helper(self, agent_input, target_input):
        if len(agent_input) == 1:
    #         print('recursion base:', dic_key, [{((), tuple(target_input)):None}])
            return {(tuple(agent_input), tuple(target_input)):None}
        if not agent_input:
            print('error: non reachable state')
            return None
        agent_new = deepcopy(agent_input)
        agent_new.pop()
        cur_dic = {}
        if not target_input:
            cur_dic[(tuple(agent_input), ())] = self.partisan_helper(agent_new, set())
            return cur_dic
        for i in range(len(target_input)+1):
    #         print('agent_inputs, target_input: ', agent_input, target_input)
            for cur_target in combinations(target_input, i):
                cur_target_set = set(cur_target)
                target_left = target_input.difference(cur_target_set)
    #             print('agent_new, cur_target, cur_target_set, target_left', agent_new, cur_target, cur_target_set, target_left)
    #             dic_value.append(partisan_helper(agent_new, cur_target_set, target_left))
                cur_dic[(tuple(agent_input), cur_target)] = \
                        self.partisan_helper(agent_new, target_left)
        return cur_dic
    def partisan_traverser(self, partisan_dic, prev_partisan, route = {}, trajectory = {}, cost = 0):
        if partisan_dic is None:
             yield trajectory, route, cost
        else:
            for key in partisan_dic.keys():
                cur_partisan = deepcopy(prev_partisan)
                cur_partisan[key[0][0]] = key[1]
                cur_trajectory, cur_route = self.travelling_salesman(key[0][0], key[1])
                route[key[0][0]] = cur_route
                trajectory[key[0][0]] = cur_trajectory
                if cost > self.optimal_cost:
                    yield None, None, cost
                else:
                    yield from \
                    self.partisan_traverser(\
                                            partisan_dic[key],\
                                            cur_partisan,\
                                            route,\
                                            trajectory, \
                                            max(cost, len(cur_trajectory))\
                                           )
    def travelling_salesman(self, agent, target_list):
        if (agent, frozenset(target_list)) in self.solved:
            return self.solved[(agent, frozenset(target_list))]
        if not target_list:
            return [], []
        if len(target_list) == 1:
            if (agent, target_list[0]) not in self.pathes:
                return None, None
            else:
                return self.pathes[(agent, target_list[0])], [agent, target_list[0]]
        dist_list = []
        mlrose_dict = list(target_list)
        for combo in combinations(target_list, 2):
            if combo in self.pathes:
                dist_list.append(\
                                 (mlrose_dict.index(combo[0]),\
                                  mlrose_dict.index(combo[1]),\
                                  len(self.pathes[combo])-1)\
                                )
        fitness_dists = mlrose.TravellingSales(distances = dist_list)
        problem_fit = mlrose.TSPOpt(length = len(target_list), fitness_fn = fitness_dists, maximize=False)
        route, _ = mlrose.genetic_alg(problem_fit, mutation_prob = 0.2,
                                              max_attempts = 20, random_state = 2)

        route = [mlrose_dict[i] for i in route]

        max_save = 0
        max_save_node_dir = (None, None)
        trajectory = [self.path_dic[agent]]
        for target in target_list:
            if (agent, target) in self.pathes:
                target_index = route.index(target)
                ccw = len(self.pathes[(route[target_index-1], route[target_index])])
                cw = len(self.pathes[(route[target_index], (route[target_index]+1)%(len(route)))])
                no_return_save = max(ccw, cw)
                current_save = len(self.pathes[agent, target])+no_return_save
                if current_save > max_save:
                    max_save = current_save
                    max_save_node_dir = (target, ccw>cw)
        if max_save_node_dir[1]:
            route_first = route[:max_save_node_dir[0]]
            route_first.reverse()
            route_second = route[max_save_node_dir[0]:]
            route_second.reverse()
        else:
            route_first = route[max_save_node_dir[0]:]
            route_second = route[:max_save_node_dir[0]]

        route = [agent] + route_first + route_second
        for i in range(len(route)-1):
            trajectory += (self.pathes[(route[i], route[i+1])])[1:]
        self.solved[(agent, frozenset(target_list))] = (trajectory, route)
        return trajectory, route

from multiprocessing import Process, Manager

class Multi_TSP_Heuristic_MP():
    def __init__(self, env_dict, static_map):
        self.flag_num = len(env_dict[FLAG])
        self.agent_num = len(env_dict[AGENT])
        self.flag_list = env_dict[FLAG]
        self.agent_list = env_dict[AGENT]
        self.static_map = static_map
        self.pathes = {}
        self.dist_list = []
        self.get_pathes()
        self.solved = {}
        self.cost, self.trajectory, self.route = self.multi_tsp()

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
    def get_pathes(self):
        path_dic = {}
        self.agent_idx_list = []
        self.flag_idx_list = []
        for i, agent_loc in enumerate(self.agent_list):
            path_dic[i] = agent_loc
            self.agent_idx_list.append(i)
        for i, flag_loc in enumerate(self.flag_list):
            path_dic[i+self.agent_num] = flag_loc
            self.flag_idx_list.append(i+self.agent_num)
        self.pathes = {}
        self.path_dic = path_dic
        for combo in combinations(range(self.flag_num+ self.agent_num), 2):
            if combo[0] < self.agent_num and combo[1] < self.agent_num:
                continue
            path = self.route_astar(path_dic[combo[0]], path_dic[combo[1]])
            if path:
                self.pathes[combo] = path
                new_path = deepcopy(path)
                new_path.reverse()
                self.pathes[combo[1], combo[0]] = new_path
    def multi_tsp(self):
        partisan_tree = self.partisan_helper(set(self.agent_idx_list), set(self.flag_idx_list))
        generator = self.partisan_traverser(partisan_tree, {})
        self.optimal_cost = np.inf
        optimal_trajectory = None
        optimal_route = None
        for trajectory, route, cost in generator:
            if self.optimal_cost > cost:
                self.optimal_cost = deepcopy(cost)
                optimal_trajectory = deepcopy(trajectory)
                optimal_route = deepcopy(route)
                print(self.optimal_cost, optimal_route)
        return self.optimal_cost, optimal_trajectory, optimal_route
    def partisan_helper(self, agent_input, target_input):
        if len(agent_input) == 1:
            return {(tuple(agent_input), tuple(target_input)):None}
        if not agent_input:
            print('error: non reachable state')
            return None
        agent_new = deepcopy(agent_input)
        agent_new.pop()
        cur_dic = {}
        if not target_input:
            cur_dic[(tuple(agent_input), ())] = self.partisan_helper(agent_new, set())
            return cur_dic
        for i in range(len(target_input)+1):
            for cur_target in combinations(target_input, i):
                cur_target_set = set(cur_target)
                target_left = target_input.difference(cur_target_set)
                cur_dic[(tuple(agent_input), cur_target)] = \
                        self.partisan_helper(agent_new, target_left)
        return cur_dic
    def partisan_traverser(self, partisan_dic, prev_partisan, route = {}, trajectory = {}, cost = 0):
        if partisan_dic is None:
             yield trajectory, route, cost
        else:
            for key in partisan_dic.keys():
                cur_partisan = deepcopy(prev_partisan)
                cur_partisan[key[0][0]] = key[1]
                cur_trajectory, cur_route = self.travelling_salesman(key[0][0], key[1])
                route[key[0][0]] = cur_route
                trajectory[key[0][0]] = cur_trajectory
                if cost > self.optimal_cost:
                    yield None, None, cost
                else:
                    yield from \
                    self.partisan_traverser(\
                                            partisan_dic[key],\
                                            cur_partisan,\
                                            route,\
                                            trajectory, \
                                            max(cost, len(cur_trajectory))\
                                           )
    def travelling_salesman(self, agent, target_list):
        if (agent, frozenset(target_list)) in self.solved:
            return self.solved[(agent, frozenset(target_list))]
        if not target_list:
            return [], []
        if len(target_list) == 1:
            if (agent, target_list[0]) not in self.pathes:
                return None, None
            else:
                return self.pathes[(agent, target_list[0])], [agent, target_list[0]]
        dist_list = []
        mlrose_dict = list(target_list)
        for combo in combinations(target_list, 2):
            if combo in self.pathes:
                dist_list.append(\
                                 (mlrose_dict.index(combo[0]),\
                                  mlrose_dict.index(combo[1]),\
                                  len(self.pathes[combo])-1)\
                                )
        fitness_dists = mlrose.TravellingSales(distances = dist_list)
        problem_fit = mlrose.TSPOpt(length = len(target_list), fitness_fn = fitness_dists, maximize=False)
        route, _ = mlrose.genetic_alg(problem_fit, mutation_prob = 0.2,
                                              max_attempts = 50, random_state = 2)
        route = [mlrose_dict[i] for i in route]
        max_save = 0
        max_save_node_dir = (None, None)
        trajectory = [self.path_dic[agent]]
        for target in target_list:
            if (agent, target) in self.pathes:
                target_index = route.index(target)
                ccw = len(self.pathes[(route[target_index-1], route[target_index])])
                cw = len(self.pathes[(route[target_index], (route[target_index]+1)%(len(route)))])
                no_return_save = max(ccw, cw)
                current_save = len(self.pathes[agent, target])+no_return_save
                if current_save > max_save:
                    max_save = current_save
                    max_save_node_dir = (target, ccw>cw)
        if max_save_node_dir[1]:
            route_first = route[:max_save_node_dir[0]]
            route_first.reverse()
            route_second = route[max_save_node_dir[0]:]
            route_second.reverse()
        else:
            route_first = route[max_save_node_dir[0]:]
            route_second = route[:max_save_node_dir[0]]

        route = [agent] + route_first + route_second
        for i in range(len(route)-1):
            trajectory += (self.pathes[(route[i], route[i+1])])[1:]
        self.solved[(agent, frozenset(target_list))] = (trajectory, route)
        return trajectory, route