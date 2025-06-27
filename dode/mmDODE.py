import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
import hashlib
import time
import shutil
import scipy
from scipy.sparse import coo_matrix, csr_matrix, eye
import scipy.sparse.linalg as spla
from scipy.optimize import lsq_linear
import pickle
import multiprocessing as mp
from typing import Union
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import r2_score
import re
import copy
import gurobipy as gp
from gurobipy import GRB

matches = [s for s in plt.style.available if re.search(r"seaborn.*poster", s)]
assert(len(matches) > 0)
plt_style = matches[0]

import macposts

def tensor_to_numpy(x: torch.Tensor):
    return x.data.cpu().numpy()

class torch_pathflow_solver(nn.Module):
    def __init__(self, num_assign_interval, 
               num_path_driving, num_path_bustransit, num_path_pnr, num_path_busroute,
               car_driving_scale=1, truck_driving_scale=0.1, passenger_bustransit_scale=1, car_pnr_scale=0.5, bus_scale=0.1,
               fix_bus=True, use_file_as_init=None):
        super(torch_pathflow_solver, self).__init__()

        self.num_assign_interval = num_assign_interval

        self.car_driving_scale = car_driving_scale
        self.truck_driving_scale = truck_driving_scale 
        self.passenger_bustransit_scale = passenger_bustransit_scale 
        self.car_pnr_scale = car_pnr_scale
        self.bus_scale = bus_scale

        self.num_path_driving = num_path_driving
        self.num_path_bustransit = num_path_bustransit
        self.num_path_pnr = num_path_pnr
        self.num_path_busroute = num_path_busroute

        self.fix_bus=fix_bus

        self.initialize(use_file_as_init)

        self.params = None

        self.algo_dict = {
            "SGD": torch.optim.SGD,
            "NAdam": torch.optim.NAdam,
            "Adam": torch.optim.Adam,
            "Adamax": torch.optim.Adamax,
            "AdamW": torch.optim.AdamW,
            "RAdam": torch.optim.RAdam,
            "Adagrad": torch.optim.Adagrad,
            "Adadelta": torch.optim.Adadelta
        }
        self.optimizer = None
        self.scheduler = None
    
    def init_tensor(self, x: torch.Tensor):
        # return torch.abs(nn.init.xavier_uniform_(x))
        return nn.init.xavier_uniform_(x)
        # return nn.init.uniform_(x, 0, 1)

    def initialize(self, use_file_as_init=None):
        log_f_car_driving, log_f_truck_driving, log_f_passenger_bustransit, log_f_car_pnr, log_f_bus = None, None, None, None, None

        if use_file_as_init is not None:
            # log f numpy
            _, _, _, _, log_f_car_driving, log_f_truck_driving, log_f_passenger_bustransit, log_f_car_pnr, log_f_bus,\
                _, _, _, _, _, \
                     _, _, _, _, _, _, _, _ = pickle.load(open(use_file_as_init, 'rb'))

        if log_f_car_driving is None:
            self.log_f_car_driving = nn.Parameter(self.init_tensor(torch.Tensor(self.num_assign_interval * self.num_path_driving, 1)).squeeze(), requires_grad=True)
        else:
            assert(np.prod(log_f_car_driving.shape) == self.num_assign_interval * self.num_path_driving)
            self.log_f_car_driving = nn.Parameter(torch.from_numpy(log_f_car_driving).squeeze(), requires_grad=True)
        
        if log_f_truck_driving is None:
            self.log_f_truck_driving = nn.Parameter(self.init_tensor(torch.Tensor(self.num_assign_interval * self.num_path_driving, 1)).squeeze(), requires_grad=True)
        else:
            assert(np.prod(log_f_truck_driving.shape) == self.num_assign_interval * self.num_path_driving)
            self.log_f_truck_driving = nn.Parameter(torch.from_numpy(log_f_truck_driving).squeeze(), requires_grad=True)

        if log_f_passenger_bustransit is None:
            self.log_f_passenger_bustransit = nn.Parameter(self.init_tensor(torch.Tensor(self.num_assign_interval * self.num_path_bustransit, 1)).squeeze(), requires_grad=True)
        else:
            assert(np.prod(log_f_passenger_bustransit.shape) == self.num_assign_interval * self.num_path_bustransit)
            self.log_f_passenger_bustransit = nn.Parameter(torch.from_numpy(log_f_passenger_bustransit).squeeze(), requires_grad=True)

        if log_f_car_pnr is None:
            self.log_f_car_pnr = nn.Parameter(self.init_tensor(torch.Tensor(self.num_assign_interval * self.num_path_pnr, 1)).squeeze(), requires_grad=True)
        else:
            assert(np.prod(log_f_car_pnr.shape) == self.num_assign_interval * self.num_path_pnr)
            self.log_f_car_pnr = nn.Parameter(torch.from_numpy(log_f_car_pnr).squeeze(), requires_grad=True)

        if not self.fix_bus:
            if log_f_bus is None:
                self.log_f_bus = nn.Parameter(self.init_tensor(torch.Tensor(self.num_assign_interval * self.num_path_busroute, 1)).squeeze(), requires_grad=True)
            else:
                assert(np.prod(log_f_bus.shape) == self.num_assign_interval * self.num_path_busroute)
                self.log_f_bus = nn.Parameter(torch.from_numpy(log_f_bus).squeeze(), requires_grad=True)
        else:
            self.log_f_bus = None

    def get_log_f_tensor(self):
        return self.log_f_car_driving, self.log_f_truck_driving, self.log_f_passenger_bustransit, self.log_f_car_pnr, self.log_f_bus

    def get_log_f_numpy(self):
        return tensor_to_numpy(self.log_f_car_driving).flatten(), tensor_to_numpy(self.log_f_truck_driving).flatten(), \
                tensor_to_numpy(self.log_f_passenger_bustransit).flatten(), tensor_to_numpy(self.log_f_car_pnr).flatten(), \
                None if self.fix_bus else tensor_to_numpy(self.log_f_bus).flatten()

    def add_pathflow(self, num_path_driving_add: int, num_path_bustransit_add: int, num_path_pnr_add: int):
        self.num_path_driving += num_path_driving_add
        self.num_path_bustransit += num_path_bustransit_add
        self.num_path_pnr += num_path_pnr_add

        if num_path_driving_add > 0:
            self.log_f_car_driving = nn.Parameter(self.log_f_car_driving.reshape(self.num_assign_interval, -1))
            log_f_car_driving_add = nn.Parameter(self.init_tensor(torch.Tensor(self.num_assign_interval * num_path_driving_add, 1)).squeeze().reshape(self.num_assign_interval, -1), requires_grad=True)
            self.log_f_car_driving = nn.Parameter(torch.cat([self.log_f_car_driving, log_f_car_driving_add], dim=1))
            assert(self.log_f_car_driving.shape[1] == self.num_path_driving)
            self.log_f_car_driving = nn.Parameter(self.log_f_car_driving.reshape(-1))

            self.log_f_truck_driving = nn.Parameter(self.log_f_truck_driving.reshape(self.num_assign_interval, -1))
            log_f_truck_driving_add = nn.Parameter(self.init_tensor(torch.Tensor(self.num_assign_interval * num_path_driving_add, 1)).squeeze().reshape(self.num_assign_interval, -1), requires_grad=True)
            self.log_f_truck_driving = nn.Parameter(torch.cat([self.log_f_truck_driving, log_f_truck_driving_add], dim=1))
            assert(self.log_f_truck_driving.shape[1] == self.num_path_driving)
            self.log_f_truck_driving = nn.Parameter(self.log_f_truck_driving.reshape(-1))

        if num_path_bustransit_add > 0:
            self.log_f_passenger_bustransit = nn.Parameter(self.log_f_passenger_bustransit.reshape(self.num_assign_interval, -1))
            log_f_passenger_bustransit_add = nn.Parameter(self.init_tensor(torch.Tensor(self.num_assign_interval * num_path_bustransit_add, 1)).squeeze().reshape(self.num_assign_interval, -1), requires_grad=True)
            self.log_f_passenger_bustransit = nn.Parameter(torch.cat([self.log_f_passenger_bustransit, log_f_passenger_bustransit_add], dim=1))
            assert(self.log_f_passenger_bustransit.shape[1] == self.num_path_bustransit)
            self.log_f_passenger_bustransit = nn.Parameter(self.log_f_passenger_bustransit.reshape(-1))

        if num_path_pnr_add > 0:
            self.log_f_car_pnr = nn.Parameter(self.log_f_car_pnr.reshape(self.num_assign_interval, -1))
            log_f_car_pnr_add = nn.Parameter(self.init_tensor(torch.Tensor(self.num_assign_interval * num_path_pnr_add, 1)).squeeze().reshape(self.num_assign_interval, -1), requires_grad=True)
            self.log_f_car_pnr = nn.Parameter(torch.cat([self.log_f_car_pnr, log_f_car_pnr_add], dim=1))
            assert(self.log_f_car_pnr.shape[1] == self.num_path_pnr)
            self.log_f_car_pnr = nn.Parameter(self.log_f_car_pnr.reshape(-1))

    def generate_pathflow_tensor(self):
        # softplus
        f_car_driving = torch.nn.functional.softplus(self.log_f_car_driving) * self.car_driving_scale
        f_truck_driving = torch.nn.functional.softplus(self.log_f_truck_driving) * self.truck_driving_scale
        f_passenger_bustransit = torch.nn.functional.softplus(self.log_f_passenger_bustransit) * self.passenger_bustransit_scale
        f_car_pnr = torch.nn.functional.softplus(self.log_f_car_pnr) * self.car_pnr_scale
        if (not self.fix_bus) and (self.log_f_bus is not None):
            f_bus = torch.nn.functional.softplus(self.log_f_bus) * self.bus_scale
        else:
            f_bus = None

        # f_car_driving = torch.clamp(f_car_driving, max=5e3)
        # f_truck_driving = torch.clamp(f_truck_driving, max=5e3)
        # f_passenger_bustransit = torch.clamp(f_passenger_bustransit, max=1e3)
        # f_car_pnr = torch.clamp(f_car_pnr, max=3e3)
        # if (not self.fix_bus) and (f_bus is not None):
        #     f_bus = torch.clamp(f_bus, max=1e1)
        # else:
        #     f_bus = None

        # relu
        # f_car_driving = torch.clamp(self.log_f_car_driving * self.car_driving_scale, min=1e-6)
        # f_truck_driving = torch.clamp(self.log_f_truck_driving * self.truck_driving_scale, min=1e-6)
        # f_passenger_bustransit = torch.clamp(self.log_f_passenger_bustransit * self.passenger_bustransit_scale, min=1e-6)
        # f_car_pnr = torch.clamp(self.log_f_car_pnr * self.car_pnr_scale, min=1e-6)
        # if (not self.fix_bus) and (self.log_f_bus is not None):
        #     f_bus = torch.clamp(self.log_f_bus * self.bus_scale, min=1e-6)
        # else:
        #     f_bus = None

        return f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus

    def generate_pathflow_numpy(self, f_car_driving: Union[torch.Tensor, None] = None, f_truck_driving: Union[torch.Tensor, None] = None, 
                                f_passenger_bustransit: Union[torch.Tensor, None] = None, f_car_pnr: Union[torch.Tensor, None] = None, f_bus: Union[torch.Tensor, None] = None):
        if (f_car_driving is None) and (f_truck_driving is None) and (f_passenger_bustransit is None) and (f_car_pnr is None) and (f_bus is None):
            f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus = self.generate_pathflow_tensor()
       
        return tensor_to_numpy(f_car_driving).flatten(), tensor_to_numpy(f_truck_driving).flatten(), tensor_to_numpy(f_passenger_bustransit).flatten(), \
            tensor_to_numpy(f_car_pnr).flatten(), None if self.fix_bus else tensor_to_numpy(f_bus).flatten()

    def set_params_with_lr(self, car_driving_step_size=1e-2, truck_driving_step_size=1e-3, passenger_bustransit_step_size=1e-2, car_pnr_step_size=1e-3, bus_step_size=None):
        self.params = [
            {'params': self.log_f_car_driving, 'lr': car_driving_step_size},
            {'params': self.log_f_truck_driving, 'lr': truck_driving_step_size},
            {'params': self.log_f_passenger_bustransit, 'lr': passenger_bustransit_step_size},
            {'params': self.log_f_car_pnr, 'lr': car_pnr_step_size}
        ]
        if (not self.fix_bus) and (self.log_f_bus is not None) and (bus_step_size is not None):
            self.params.append({'params': self.log_f_bus, 'lr': bus_step_size})
    
    def set_optimizer(self, algo='NAdam'):
        self.optimizer = self.algo_dict[algo](self.params)

    def compute_gradient(self, f_car_driving: torch.Tensor, f_truck_driving: torch.Tensor, f_passenger_bustransit: torch.Tensor, f_car_pnr: torch.Tensor, f_bus: Union[torch.Tensor, None],
                         f_car_driving_grad: np.ndarray, f_truck_driving_grad: np.ndarray, f_passenger_bustransit_grad: np.ndarray, f_car_pnr_grad: np.ndarray, f_passenger_pnr_grad: np.ndarray, f_bus_grad: Union[np.ndarray, None],
                         l2_coeff: float = 1e-5):

        f_car_driving.backward(torch.from_numpy(f_car_driving_grad) + l2_coeff * f_car_driving.data)
        f_truck_driving.backward(torch.from_numpy(f_truck_driving_grad) + l2_coeff * f_truck_driving.data)
        f_passenger_bustransit.backward(torch.from_numpy(f_passenger_bustransit_grad) + l2_coeff * f_passenger_bustransit.data)
        f_car_pnr.backward(torch.from_numpy((f_car_pnr_grad + f_passenger_pnr_grad)) + l2_coeff * f_car_pnr.data)
        if not self.fix_bus:
            f_bus.backward(torch.from_numpy(f_bus_grad) + l2_coeff * f_bus.data)

        # torch.nn.utils.clip_grad_value_(self.parameters(), 0.4)
    
    def set_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode='min', factor=0.75, patience=5, 
        #     threshold=0.15, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)

class MMDODE:
    def __init__(self, nb, config, num_procs=1):
        self.reinitialize(nb, config, num_procs)

    def reinitialize(self, nb, config, num_procs=1):
        self.config = config
        self.nb = nb

        self.num_assign_interval = nb.config.config_dict['DTA']['max_interval']
        self.ass_freq = nb.config.config_dict['DTA']['assign_frq']

        self.num_link_driving = nb.config.config_dict['DTA']['num_of_link']
        self.num_link_bus = nb.config.config_dict['DTA']['num_of_bus_link']
        self.num_link_walking = nb.config.config_dict['DTA']['num_of_walking_link']

        self.num_path_driving = nb.config.config_dict['FIXED']['num_driving_path']
        self.num_path_bustransit = nb.config.config_dict['FIXED']['num_bustransit_path']
        self.num_path_pnr = nb.config.config_dict['FIXED']['num_pnr_path']
        self.num_path_busroute = nb.config.config_dict['FIXED']['num_bus_routes']

        # if nb.config.config_dict['DTA']['total_interval'] > 0 and nb.config.config_dict['DTA']['total_interval'] > self.num_assign_interval * self.ass_freq:
        #     self.num_loading_interval = nb.config.config_dict['DTA']['total_interval']
        # else:
        #     self.num_loading_interval = self.num_assign_interval * self.ass_freq  # not long enough
        self.num_loading_interval = self.num_assign_interval * self.ass_freq

        self.data_dict = dict()

        # number of observed data
        self.num_data = self.config['num_data']

        # observed link IDs, np.array
        self.observed_links_driving = self.config['observed_links_driving']
        self.observed_links_bus = self.config['observed_links_bus']
        self.observed_links_walking = self.config['observed_links_walking']

        # observed stops_veh tuple list
        self.observed_stops_vehs_list = self.config['observed_stops_vehs_list']

        # observed path IDs, np.array
        self.paths_list = self.config['paths_list']
        self.paths_list_driving = self.config['paths_list_driving']
        self.paths_list_bustransit = self.config['paths_list_bustransit']
        self.paths_list_pnr = self.config['paths_list_pnr']
        self.paths_list_busroute = self.config['paths_list_busroute']
        assert (len(self.paths_list_driving) == self.num_path_driving)
        assert (len(self.paths_list_bustransit) == self.num_path_bustransit)
        assert (len(self.paths_list_pnr) == self.num_path_pnr)
        assert (len(self.paths_list_busroute) == self.num_path_busroute)
        assert (len(self.paths_list) == self.num_path_driving + self.num_path_bustransit + self.num_path_pnr + self.num_path_busroute)

        # observed aggregated link count data, length = self.num_data
        # every element of the list is a np.array
        # num_aggregation x (num_links_driving x num_assign_interval)
        self.car_count_agg_L_list = None
        # num_aggregation x (num_links_driving x num_assign_interval)
        self.truck_count_agg_L_list = None
        # num_aggregation x (num_links_bus x num_assign_interval)
        self.bus_count_agg_L_list = None
        # num_aggregation x ((num_links_bus + num_links_walking) x num_assign_interval)
        self.passenger_count_agg_L_list = None

        # demand
        self.demand_list_total_passenger = self.nb.demand_total_passenger.demand_list
        self.demand_list_truck_driving = self.nb.demand_driving.demand_list

        self.observed_links_bus_driving = np.array([])
        self.bus_driving_link_relation = None
        if self.config['use_bus_link_flow'] or self.config['use_bus_link_tt']:
            self.register_links_overlapped_bus_driving(self.nb.folder_path)

        self.num_procs = num_procs

    def register_links_overlapped_bus_driving(self, folder):
        a = macposts.mmdta_api()
        a.initialize(folder)

        a.register_links_driving(self.observed_links_driving)
        a.register_links_bus(self.observed_links_bus)
        a.register_links_walking(self.observed_links_walking)

        raw_record = a.get_links_overlapped_bus_driving()
        self.observed_links_bus_driving = np.array(list(set(raw_record[:, 1])), dtype=int)
        
        if type(self.observed_links_bus) == np.ndarray:
            ind = np.array(list(map(lambda x: True if len(np.where(self.observed_links_bus == x)[0]) > 0 else False, raw_record[:, 0].astype(int)))).astype(bool)
            assert(np.sum(ind) == len(ind))
            bus_link_seq = (np.array(list(map(lambda x: np.where(self.observed_links_bus == x)[0][0], raw_record[ind, 0].astype(int))))).astype(int)
        elif type(self.observed_links_bus) == list:
            ind = np.array(list(map(lambda x: True if x in self.observed_links_bus else False, raw_record[:, 0].astype(int)))).astype(bool)
            assert(np.sum(ind) == len(ind))
            bus_link_seq = (np.array(list(map(lambda x: self.observed_links_bus.index(x), raw_record[ind, 0].astype(int))))).astype(int)

        if type(self.observed_links_bus_driving) == np.ndarray:
            ind = np.array(list(map(lambda x: True if len(np.where(self.observed_links_bus_driving == x)[0]) > 0 else False, raw_record[:, 1].astype(int)))).astype(bool)
            assert(np.sum(ind) == len(ind))
            bus_driving_link_seq = (np.array(list(map(lambda x: np.where(self.observed_links_bus_driving == x)[0][0], raw_record[ind, 1].astype(int))))).astype(int)
        elif type(self.observed_links_bus_driving) == list:
            ind = np.array(list(map(lambda x: True if x in self.observed_links_bus_driving else False, raw_record[:, 1].astype(int)))).astype(bool)
            assert(np.sum(ind) == len(ind))
            bus_driving_link_seq = (np.array(list(map(lambda x: self.observed_links_bus_driving.index(x), raw_record[ind, 1].astype(int))))).astype(int)

        mat = coo_matrix((raw_record[:, 2], (bus_link_seq, bus_driving_link_seq)), shape=(len(self.observed_links_bus), len(self.observed_links_bus_driving)))
        mat = mat.tocsr()

        self.bus_driving_link_relation = csr_matrix(scipy.sparse.block_diag([mat for _ in range(self.num_assign_interval)]))


    def check_registered_links_covered_by_registered_paths(self, folder, add=False):
        self.save_simulation_input_files(folder, explicit_bus=1, historical_bus_waiting_time=0)

        a = macposts.mmdta_api()
        a.initialize(folder)

        a.register_links_driving(self.observed_links_driving)
        a.register_links_bus(self.observed_links_bus)
        a.register_links_walking(self.observed_links_walking)

        a.register_paths(self.paths_list)
        a.register_paths_driving(self.paths_list_driving)
        a.register_paths_bustransit(self.paths_list_bustransit)
        a.register_paths_pnr(self.paths_list_pnr)
        a.register_paths_bus(self.paths_list_busroute)

        is_driving_link_covered = a.are_registered_links_in_registered_paths_driving()
        is_bus_link_covered = a.are_registered_links_in_registered_paths_bus()
        is_walking_link_covered = a.are_registered_links_in_registered_paths_walking()

        is_bus_walking_link_covered = np.concatenate((is_bus_link_covered, is_walking_link_covered))
        is_updated = 0
        if add:
            if np.any(is_driving_link_covered == False) or np.any(is_bus_link_covered == False) or np.any(is_walking_link_covered == False):
                is_driving_link_covered = a.generate_paths_to_cover_registered_links_driving()
                is_bus_walking_link_covered = a.generate_paths_to_cover_registered_links_bus_walking()
                is_updated = is_driving_link_covered[0] + is_bus_walking_link_covered[0]
                is_driving_link_covered = is_driving_link_covered[1:]
                is_bus_walking_link_covered = is_bus_walking_link_covered[1:]
            else:
                is_updated = 0
                is_bus_walking_link_covered = np.concatenate((is_bus_link_covered, is_walking_link_covered))
        return is_updated, is_driving_link_covered, is_bus_walking_link_covered
    
    def check_stops_covered_by_transit_paths(self):
        walking_link_id2node = dict()
        for link in self.nb.link_walking_list:
            walking_link_id2node[link.ID] = set()
            walking_link_id2node[link.ID].add(link.from_node_ID)
            walking_link_id2node[link.ID].add(link.to_node_ID)

        bus_link_id2stop = dict()
        for link in self.nb.link_bus_list:
            bus_link_id2stop[link.ID] = set()
            bus_link_id2stop[link.ID].add(link.from_busstop_ID)
            bus_link_id2stop[link.ID].add(link.to_busstop_ID)

        stopsORnodes_involved_in_transit_paths = set()
        for path_id in self.nb.path_table_bustransit.ID2path:
            path = self.nb.path_table_bustransit.ID2path[path_id]
            for link_id in path.link_list:
                if link_id in walking_link_id2node:
                    stopsORnodes_involved_in_transit_paths.update(walking_link_id2node[link_id])
                else:
                    stopsORnodes_involved_in_transit_paths.update(bus_link_id2stop[link_id])
        
        is_observed_stops_vehs_covered = np.array([False] * len(self.config['observed_stops_vehs_list']))
        for idx in range(len(self.config['observed_stops_vehs_list'])):
            tup = self.config['observed_stops_vehs_list'][idx]
            if tup[0] in stopsORnodes_involved_in_transit_paths:
                is_observed_stops_vehs_covered[idx] = True

        return is_observed_stops_vehs_covered

        
    def _add_car_link_flow_data(self, link_flow_df_list):
        # assert(self.config['use_car_link_flow'])
        assert (self.num_data == len(link_flow_df_list))
        self.data_dict['car_link_flow'] = link_flow_df_list

    def _add_truck_link_flow_data(self, link_flow_df_list):
        # assert(self.config['use_truck_link_flow'])
        assert (self.num_data == len(link_flow_df_list))
        self.data_dict['truck_link_flow'] = link_flow_df_list

    def _add_bus_link_flow_data(self, link_flow_df_list):
        # assert(self.config['use_bus_link_flow'])
        assert (self.num_data == len(link_flow_df_list))
        self.data_dict['bus_link_flow'] = link_flow_df_list
    
    def _add_passenger_link_flow_data(self, link_flow_df_list):
        # assert(self.config['use_passenger_link_flow'])
        assert (self.num_data == len(link_flow_df_list))
        self.data_dict['passenger_link_flow'] = link_flow_df_list

    def _add_car_link_tt_data(self, link_spd_df_list):
        # assert(self.config['use_car_link_tt'])
        assert (self.num_data == len(link_spd_df_list))
        self.data_dict['car_link_tt'] = link_spd_df_list

    def _add_truck_link_tt_data(self, link_spd_df_list):
        # assert(self.config['use_truck_link_tt'])
        assert (self.num_data == len(link_spd_df_list))
        self.data_dict['truck_link_tt'] = link_spd_df_list

    def _add_bus_link_tt_data(self, link_spd_df_list):
        # assert(self.config['use_bus_link_tt'])
        assert (self.num_data == len(link_spd_df_list))
        self.data_dict['bus_link_tt'] = link_spd_df_list

    def _add_passenger_link_tt_data(self, link_spd_df_list):
        # assert(self.config['use_passenger_link_tt'])
        assert (self.num_data == len(link_spd_df_list))
        self.data_dict['passenger_link_tt'] = link_spd_df_list

    def _add_veh_run_boarding_alighting_data(self, veh_run_boarding_alighting_record_list):
        assert (self.num_data == len(veh_run_boarding_alighting_record_list))
        self.data_dict['veh_run_boarding_alighting_record'] = veh_run_boarding_alighting_record_list

    def _add_stop_arrival_departure_travel_time_data(self, stop_arrival_departure_travel_time_record_list):
        assert (self.num_data == len(stop_arrival_departure_travel_time_record_list))
        self.data_dict['stop_arrival_departure_travel_time'] = stop_arrival_departure_travel_time_record_list

    def add_data(self, data_dict):
        if self.config['car_count_agg']:
            self.car_count_agg_L_list = data_dict['car_count_agg_L_list']
        if self.config['truck_count_agg']:
            self.truck_count_agg_L_list = data_dict['truck_count_agg_L_list']
        if self.config['bus_count_agg']:
            self.bus_count_agg_L_list = data_dict['bus_count_agg_L_list']
        if self.config['passenger_count_agg']:
            self.passenger_count_agg_L_list = data_dict['passenger_count_agg_L_list']
        
        if self.config['use_car_link_flow'] or self.config['compute_car_link_flow_loss']:
            self._add_car_link_flow_data(data_dict['car_link_flow'])
        if self.config['use_truck_link_flow'] or self.config['compute_truck_link_flow_loss'] :
            self._add_truck_link_flow_data(data_dict['truck_link_flow'])
        if self.config['use_bus_link_flow'] or self.config['compute_bus_link_flow_loss']:
            self._add_bus_link_flow_data(data_dict['bus_link_flow'])
        if self.config['use_passenger_link_flow'] or self.config['compute_passenger_link_flow_loss']:
            self._add_passenger_link_flow_data(data_dict['passenger_link_flow'])

        if self.config['use_car_link_tt'] or self.config['compute_car_link_tt_loss']:
            self._add_car_link_tt_data(data_dict['car_link_tt'])
        if self.config['use_truck_link_tt']or self.config['compute_truck_link_tt_loss']:
            self._add_truck_link_tt_data(data_dict['truck_link_tt'])
        if self.config['use_bus_link_tt'] or self.config['compute_bus_link_tt_loss']:
            self._add_bus_link_tt_data(data_dict['bus_link_tt'])
        if self.config['use_passenger_link_tt']or self.config['compute_passenger_link_tt_loss']:
            self._add_passenger_link_tt_data(data_dict['passenger_link_tt'])

        if self.config['use_veh_run_boarding_alighting'] or self.config['compute_veh_run_boarding_alighting_loss'] or self.config['use_ULP_f_transit']:
            self._add_veh_run_boarding_alighting_data(data_dict['veh_run_boarding_alighting_record'])
            if 'stop_arrival_departure_travel_time' in data_dict:
                self._add_stop_arrival_departure_travel_time_data(data_dict['stop_arrival_departure_travel_time'])

        if 'mask_driving_link' in data_dict:
            self.data_dict['mask_driving_link'] = np.tile(data_dict['mask_driving_link'], self.num_assign_interval)
        else:
            self.data_dict['mask_driving_link'] = np.ones(len(self.observed_links_driving) * self.num_assign_interval, dtype=bool)

        if 'mask_bus_link' in data_dict:
            self.data_dict['mask_bus_link'] = np.tile(data_dict['mask_bus_link'], self.num_assign_interval)
        else:
            self.data_dict['mask_bus_link'] = np.ones(len(self.observed_links_bus) * self.num_assign_interval, dtype=bool)

        if 'mask_walking_link' in data_dict:
            self.data_dict['mask_walking_link'] = np.tile(data_dict['mask_walking_link'], self.num_assign_interval)
        else:
            self.data_dict['mask_walking_link'] = np.ones(len(self.observed_links_walking) * self.num_assign_interval, dtype=bool)

        if 'mask_observed_stops_vehs_record' in data_dict:
            self.data_dict['mask_observed_stops_vehs_record'] = data_dict['mask_observed_stops_vehs_record']
        else:
            self.data_dict['mask_observed_stops_vehs_record'] = np.ones(len(self.observed_stops_vehs_list), dtype=bool)

    def save_simulation_input_files(self, folder_path, f_car_driving=None, f_truck_driving=None, 
                                    f_passenger_bustransit=None, f_car_pnr=None, f_bus=None, 
                                    explicit_bus=1, historical_bus_waiting_time=0):

        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        _flg = False
        # update demand for each mode
        if (f_car_driving is not None) and (f_truck_driving is not None):
            self.nb.update_demand_path_driving(f_car_driving, f_truck_driving)
            _flg = True
        if f_passenger_bustransit is not None:
            self.nb.update_demand_path_bustransit(f_passenger_bustransit)
            _flg = True
        if f_car_pnr is not None:
            self.nb.update_demand_path_pnr(f_car_pnr)
            _flg = True
        # update nb.demand_total_passenger
        if _flg:
            self.nb.get_mode_portion_matrix()

        if f_bus is not None:
            self.nb.update_demand_path_busroute(f_bus)

        # self.nb.config.config_dict['DTA']['flow_scalar'] = 3
        if self.config['use_car_link_tt'] or self.config['use_truck_link_tt'] or self.config['use_passenger_link_tt'] or self.config['use_bus_link_tt']:
            self.nb.config.config_dict['DTA']['total_interval'] = self.num_loading_interval # * 2  # hopefully this is sufficient 
        else:
            self.nb.config.config_dict['DTA']['total_interval'] = self.num_loading_interval  # if only count data is used

        self.nb.config.config_dict['DTA']['routing_type'] = 'Multimodal_Hybrid'

        # no output files saved from DNL
        self.nb.config.config_dict['STAT']['rec_volume'] = 1
        self.nb.config.config_dict['STAT']['volume_load_automatic_rec'] = 0
        self.nb.config.config_dict['STAT']['volume_record_automatic_rec'] = 0
        self.nb.config.config_dict['STAT']['rec_tt'] = 1
        self.nb.config.config_dict['STAT']['tt_load_automatic_rec'] = 0
        self.nb.config.config_dict['STAT']['tt_record_automatic_rec'] = 0

        # TODO: to estimate passenger transit flow,
        #       currently use explicit_bus = 0 and historical_bus_waiting_time = 0 for DODE
        # explicit_bus = 1 model bus vehcile explicitly in DNL, 0 otherwise
        # historical_bus_waiting_time = waiting time in number of unit intervals (5s)
        # estimating passenger flow is difficult, this is a temporary solution
        self.nb.config.config_dict['DTA']['explicit_bus'] = explicit_bus
        self.nb.config.config_dict['DTA']['historical_bus_waiting_time'] = historical_bus_waiting_time

        self.nb.dump_to_folder(folder_path)
        

    def _run_simulation(self, f_car_driving = None, f_truck_driving= None, f_passenger_bustransit= None, f_car_pnr= None, f_bus= None, 
                        counter=0, run_mmdta_adaptive=True, show_loading=False,
                        explicit_bus=1, historical_bus_waiting_time=0):
        # print("Start simulation", time.time())
        hash1 = hashlib.sha1()
        # python 2
        # hash1.update(str(time.time()) + str(counter))
        # python 3
        hash1.update((str(time.time()) + str(counter)).encode('utf-8'))
        new_folder = str(hash1.hexdigest())

        self.save_simulation_input_files(folder_path = new_folder, f_car_driving=f_car_driving, f_truck_driving=f_truck_driving, 
                                            f_passenger_bustransit=f_passenger_bustransit, f_car_pnr=f_car_pnr, f_bus=f_bus,
                                         explicit_bus=explicit_bus, historical_bus_waiting_time=historical_bus_waiting_time)

        a = macposts.mmdta_api()
        a.initialize(new_folder)

        a.register_links_driving(self.observed_links_driving)
        a.register_links_bus(self.observed_links_bus)
        a.register_links_walking(self.observed_links_walking)
        if len(self.observed_links_bus_driving) > 0:
            a.register_links_bus_driving(self.observed_links_bus_driving)

        a.register_paths(self.paths_list)
        a.register_paths_driving(self.paths_list_driving)
        a.register_paths_bustransit(self.paths_list_bustransit)
        a.register_paths_pnr(self.paths_list_pnr)
        a.register_paths_bus(self.paths_list_busroute)

        a.install_cc()
        a.install_cc_tree()

        if run_mmdta_adaptive:
            a.run_mmdta_adaptive(new_folder, -1, show_loading)
            # a.run_mmdta_adaptive('/srv/data/qiling/Projects/CentralOhio_Honda_Project/Multimodal/input_files_CentralOhio_multimodal_AM', 180, show_loading)
        else:
            a.run_whole(show_loading)
        # print("Finish simulation", time.time())

        # travel_stats = a.get_travel_stats()
        print(a.print_travel_stats())

        # print_emission_stats() only works if folder is not removed, cannot find reason
        a.print_emission_stats()

        shutil.rmtree(new_folder)

        a.delete_all_agents()

        return a  

    def get_car_dar_matrix_driving(self, dta, f):
        start_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)
        end_intervals = start_intervals + self.ass_freq
        assert(end_intervals[-1] <= self.num_loading_interval)
        raw_dar = dta.get_car_dar_matrix_driving(start_intervals, end_intervals)
        dar = self._massage_raw_dar(raw_dar, self.ass_freq, f, self.num_assign_interval, self.paths_list_driving, self.observed_links_driving, self.num_procs)
        return dar

    def get_truck_dar_matrix_driving(self, dta, f):
        start_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)
        end_intervals = start_intervals + self.ass_freq
        assert(end_intervals[-1] <= self.num_loading_interval)
        raw_dar = dta.get_truck_dar_matrix_driving(start_intervals, end_intervals)
        dar = self._massage_raw_dar(raw_dar, self.ass_freq, f, self.num_assign_interval, self.paths_list_driving, self.observed_links_driving, self.num_procs)
        return dar

    def get_car_dar_matrix_pnr(self, dta, f):
        start_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)
        end_intervals = start_intervals + self.ass_freq
        assert(end_intervals[-1] <= self.num_loading_interval)
        raw_dar = dta.get_car_dar_matrix_pnr(start_intervals, end_intervals)
        dar = self._massage_raw_dar(raw_dar, self.ass_freq, f, self.num_assign_interval, self.paths_list_pnr, self.observed_links_driving, self.num_procs)
        return dar

    def get_bus_dar_matrix_bustransit_link(self, dta, f):
        start_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)
        end_intervals = start_intervals + self.ass_freq
        assert(end_intervals[-1] <= self.num_loading_interval)
        raw_dar = dta.get_bus_dar_matrix_bustransit_link(start_intervals, end_intervals)
        dar = self._massage_raw_dar(raw_dar, self.ass_freq, f, self.num_assign_interval, self.paths_list_busroute, self.observed_links_bus, self.num_procs)
        return dar

    def get_bus_dar_matrix_driving_link(self, dta, f):
        start_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)
        end_intervals = start_intervals + self.ass_freq
        assert(end_intervals[-1] <= self.num_loading_interval)
        raw_dar = dta.get_bus_dar_matrix_driving_link(start_intervals, end_intervals)
        dar = self._massage_raw_dar(raw_dar, self.ass_freq, f, self.num_assign_interval, self.paths_list_busroute, self.observed_links_driving, self.num_procs)
        return dar

    def get_passenger_dar_matrix_bustransit(self, dta, f):
        start_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)
        end_intervals = start_intervals + self.ass_freq
        assert(end_intervals[-1] <= self.num_loading_interval)
        raw_dar = dta.get_passenger_dar_matrix_bustransit(start_intervals, end_intervals)
        dar = self._massage_raw_dar(raw_dar, self.ass_freq, f, self.num_assign_interval, self.paths_list_bustransit, 
                                    np.concatenate((self.observed_links_bus, self.observed_links_walking)), self.num_procs)
        return dar

    def get_passenger_dar_matrix_pnr(self, dta, f):
        start_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)
        end_intervals = start_intervals + self.ass_freq
        assert(end_intervals[-1] <= self.num_loading_interval)
        raw_dar = dta.get_passenger_dar_matrix_pnr(start_intervals, end_intervals)
        dar = self._massage_raw_dar(raw_dar, self.ass_freq, f, self.num_assign_interval, self.paths_list_pnr, 
                                    np.concatenate((self.observed_links_bus, self.observed_links_walking)), self.num_procs)
        return dar

    def get_car_dar_matrix_bus_driving_link(self, dta, f):
        start_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)
        end_intervals = start_intervals + self.ass_freq
        assert(end_intervals[-1] <= self.num_loading_interval)
        raw_dar = dta.get_car_dar_matrix_bus_driving_link(start_intervals, end_intervals)
        dar = self._massage_raw_dar(raw_dar, self.ass_freq, f, self.num_assign_interval, self.paths_list_driving, self.observed_links_bus_driving, self.num_procs)
        return dar

    def get_truck_dar_matrix_bus_driving_link(self, dta, f):
        start_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)
        end_intervals = start_intervals + self.ass_freq
        assert(end_intervals[-1] <= self.num_loading_interval)
        raw_dar = dta.get_truck_dar_matrix_bus_driving_link(start_intervals, end_intervals)
        dar = self._massage_raw_dar(raw_dar, self.ass_freq, f, self.num_assign_interval, self.paths_list_driving, self.observed_links_bus_driving, self.num_procs)
        return dar

    def get_passenger_dar_matrix_bustransit_bus_link(self, dta, f):
        start_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)
        end_intervals = start_intervals + self.ass_freq
        assert(end_intervals[-1] <= self.num_loading_interval)
        raw_dar = dta.get_passenger_dar_matrix_bustransit_bus_link(start_intervals, end_intervals)
        dar = self._massage_raw_dar(raw_dar, self.ass_freq, f, self.num_assign_interval, self.paths_list_bustransit, self.observed_links_bus, self.num_procs)
        return dar

    def get_passenger_dar_matrix_pnr_bus_link(self, dta, f):
        start_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)
        end_intervals = start_intervals + self.ass_freq
        assert(end_intervals[-1] <= self.num_loading_interval)
        raw_dar = dta.get_passenger_dar_matrix_pnr_bus_link(start_intervals, end_intervals)
        dar = self._massage_raw_dar(raw_dar, self.ass_freq, f, self.num_assign_interval, self.paths_list_pnr, self.observed_links_bus, self.num_procs)
        return dar

    def get_passenger_bus_link_flow_relationship(self, dta):
        start_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)
        end_intervals = start_intervals + self.ass_freq
        assert(end_intervals[-1] <= self.num_loading_interval)
        x_e_bus = dta.get_link_bus_inflow(start_intervals, end_intervals).flatten(order='F')
        x_e_bus_passenger = dta.get_link_bus_passenger_inflow(start_intervals, end_intervals).flatten(order='F')

        passenger_bus_link_flow_relationship = self.nb.config.config_dict['DTA']['bus_capacity'] * x_e_bus >= x_e_bus_passenger

        # passenger_bus_link_flow_relationship = x_e_bus / np.maximum(x_e_bus_passenger, 1e-6)

        # # passenger_bus_link_flow_relationship = np.minimum(passenger_bus_link_flow_relationship, 1 / self.nb.config.config_dict['DTA']['bus_capacity'])

        # passenger_bus_link_flow_relationship[passenger_bus_link_flow_relationship > x_e_bus / self.nb.config.config_dict['DTA']['flow_scalar']] = 0

        # passenger_bus_link_flow_relationship = np.minimum(passenger_bus_link_flow_relationship, 1)

        # # passenger_bus_link_flow_relationship[x_e_bus_passenger <= 1e-5] = 1

        # passenger_bus_link_flow_relationship = x_e_bus_passenger / np.maximum(x_e_bus, 1e-6)
        # passenger_bus_link_flow_relationship[x_e_bus < self.nb.config.config_dict['DTA']['flow_scalar']] = 0

        return passenger_bus_link_flow_relationship

    def get_dar(self, dta, f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus, fix_bus=True):
        
        car_dar_matrix_driving = csr_matrix((self.num_assign_interval * len(self.observed_links_driving), 
                                             self.num_assign_interval * len(self.paths_list_driving)))
        car_dar_matrix_pnr = csr_matrix((self.num_assign_interval * len(self.observed_links_driving), 
                                         self.num_assign_interval * len(self.paths_list_pnr)))
        truck_dar_matrix_driving = csr_matrix((self.num_assign_interval * len(self.observed_links_driving), 
                                               self.num_assign_interval * len(self.paths_list_driving)))
        bus_dar_matrix_transit_link = csr_matrix((self.num_assign_interval * len(self.observed_links_bus), 
                                                  self.num_assign_interval * len(self.paths_list_busroute)))
        bus_dar_matrix_driving_link = csr_matrix((self.num_assign_interval * len(self.observed_links_driving), 
                                                  self.num_assign_interval * len(self.paths_list_busroute)))
        passenger_dar_matrix_bustransit = csr_matrix((self.num_assign_interval * (len(self.observed_links_bus) + len(self.observed_links_walking)), 
                                                      self.num_assign_interval * len(self.paths_list_bustransit)))
        passenger_dar_matrix_pnr = csr_matrix((self.num_assign_interval * (len(self.observed_links_bus) + len(self.observed_links_walking)), 
                                               self.num_assign_interval * len(self.paths_list_pnr)))
        car_dar_matrix_bus_driving_link = csr_matrix((self.num_assign_interval * len(self.observed_links_bus_driving), 
                                                      self.num_assign_interval * len(self.paths_list_driving)))
        truck_dar_matrix_bus_driving_link = csr_matrix((self.num_assign_interval * len(self.observed_links_bus_driving), 
                                                        self.num_assign_interval * len(self.paths_list_driving)))
        passenger_dar_matrix_bustransit_bus_link = csr_matrix((self.num_assign_interval * len(self.observed_links_bus), 
                                                               self.num_assign_interval * len(self.paths_list_bustransit)))
        passenger_dar_matrix_pnr_bus_link = csr_matrix((self.num_assign_interval * len(self.observed_links_bus), 
                                                        self.num_assign_interval * len(self.paths_list_pnr)))                     
        passenger_bus_link_flow_relationship = 0

        passenger_BoardingAlighting_dar_transit = csr_matrix((len(self.observed_stops_vehs_list), len(self.paths_list_bustransit)*self.num_assign_interval))
        passenger_BoardingAlighting_dar_pnr = csr_matrix((len(self.observed_stops_vehs_list), len(self.paths_list_pnr)*self.num_assign_interval))

        if self.config['use_car_link_flow'] or self.config['use_car_link_tt']:
            car_dar_matrix_driving = self.get_car_dar_matrix_driving(dta, f_car_driving)
            if car_dar_matrix_driving.max() == 0.:
                print("car_dar_matrix_driving is empty!")
        
            car_dar_matrix_pnr = self.get_car_dar_matrix_pnr(dta, f_car_pnr)
            if car_dar_matrix_pnr.max() == 0.:
                print("car_dar_matrix_pnr is empty!")
            
        if self.config['use_truck_link_flow'] or self.config['use_truck_link_tt']:
            truck_dar_matrix_driving = self.get_truck_dar_matrix_driving(dta, f_truck_driving)
            if truck_dar_matrix_driving.max() == 0.:
                print("truck_dar_matrix_driving is empty!")

        if self.config['use_bus_link_flow'] or self.config['use_bus_link_tt']:
            if not fix_bus:
                bus_dar_matrix_transit_link = self.get_bus_dar_matrix_bustransit_link(dta, f_bus)
                if bus_dar_matrix_transit_link.max() == 0.:
                    print("bus_dar_matrix_transit_link is empty!")

                # bus_dar_matrix_driving_link = self.get_bus_dar_matrix_driving_link(dta, f_bus)
                # if bus_dar_matrix_driving_link.max() == 0.:
                #     print("bus_dar_matrix_driving_link is empty!")

            car_dar_matrix_bus_driving_link = self.get_car_dar_matrix_bus_driving_link(dta, f_car_driving)
            if car_dar_matrix_bus_driving_link.max() == 0.:
                print("car_dar_matrix_bus_driving_link is empty!")
        
            truck_dar_matrix_bus_driving_link = self.get_truck_dar_matrix_bus_driving_link(dta, f_truck_driving)
            if truck_dar_matrix_bus_driving_link.max() == 0.:
                print("truck_dar_matrix_bus_driving_link is empty!")

            passenger_dar_matrix_bustransit_bus_link = self.get_passenger_dar_matrix_bustransit_bus_link(dta, f_passenger_bustransit)
            if passenger_dar_matrix_bustransit_bus_link.max() == 0.:
                print("passenger_dar_matrix_bustransit_bus_link is empty!")

            passenger_dar_matrix_pnr_bus_link = self.get_passenger_dar_matrix_pnr_bus_link(dta, f_car_pnr)
            if passenger_dar_matrix_pnr_bus_link.max() == 0.:
                print("passenger_dar_matrix_pnr_bus_link is empty!")

        if self.config['use_passenger_link_flow'] or self.config['use_passenger_link_tt']:
            passenger_dar_matrix_bustransit = self.get_passenger_dar_matrix_bustransit(dta, f_passenger_bustransit)
            if passenger_dar_matrix_bustransit.max() == 0.:
                print("passenger_dar_matrix_bustransit is empty!")

            passenger_dar_matrix_pnr = self.get_passenger_dar_matrix_pnr(dta, f_car_pnr)
            if passenger_dar_matrix_pnr.max() == 0.:
                print("passenger_dar_matrix_pnr is empty!")

        if (self.config['use_passenger_link_flow'] or self.config['use_passenger_link_tt']) and (self.config['use_bus_link_flow'] or self.config['use_bus_link_tt']):
            passenger_bus_link_flow_relationship = self.get_passenger_bus_link_flow_relationship(dta)

        if self.config['use_veh_run_boarding_alighting'] or self.config['use_ULP_f_transit']:
            raw_bt_dar = dta.get_sparse_dar_matrix_bt_by_round()
            raw_pnr_dar = dta.get_sparse_dar_matrix_pnr_by_round()
            passenger_BoardingAlighting_dar_transit, passenger_BoardingAlighting_dar_pnr = self._massage_raw_boarding_alighting_dar(raw_bt_dar, raw_pnr_dar, f_passenger_bustransit, f_car_pnr)
            if passenger_BoardingAlighting_dar_transit.max() == 0.:
                print("passenger_BoardingAlighting_dar_transit is empty!")
            if passenger_BoardingAlighting_dar_pnr.max() == 0.:
                print("passenger_BoardingAlighting_dar_pnr is empty!")

        return car_dar_matrix_driving, truck_dar_matrix_driving, car_dar_matrix_pnr, bus_dar_matrix_transit_link, bus_dar_matrix_driving_link, \
               passenger_dar_matrix_bustransit, passenger_dar_matrix_pnr, car_dar_matrix_bus_driving_link, truck_dar_matrix_bus_driving_link, passenger_dar_matrix_bustransit_bus_link, passenger_dar_matrix_pnr_bus_link, \
               passenger_bus_link_flow_relationship, passenger_BoardingAlighting_dar_transit, passenger_BoardingAlighting_dar_pnr

    def get_stop_arrival_departure_travel_time(self, raw_boarding_alighting_record):
        df = pd.DataFrame(raw_boarding_alighting_record, columns=['arrival_time', 'bus_id', 'route_id', 'veh_order', 'stop_id', 'boarding_count', 'alighting_count', 'departure_time'])
        df['veh_order'] = df['veh_order'].astype(int)
        df['route_id'] = df['route_id'].astype(int)
        df['stop_id'] = df['stop_id'].astype(int)
        df = df[['route_id', 'veh_order', 'stop_id', 'arrival_time', 'departure_time']]

        rows = []
        for id in self.nb.path_table_bus.ID2path:
            bus_route = self.nb.path_table_bus.ID2path[id]
            route_id = bus_route.route_ID
            for i in range(len(bus_route.virtual_busstop_list)):
                stop_id = bus_route.virtual_busstop_list[i]
                sequence = i+1
                rows.append({'route_id': route_id, 'stop_id': stop_id, 'sequence': sequence})
        bus_stop_sequence_df = pd.DataFrame(rows)

        predicted = pd.merge(df, bus_stop_sequence_df, how='inner', on=['route_id', 'stop_id'])
        predicted.sort_values(by=['route_id', 'veh_order','sequence'], inplace=True)
        predicted.reset_index(inplace=True, drop=True)

        predicted['pure_travel_time'] = np.nan
        predicted['travel_and_dwell_time'] = np.nan
        grouped = predicted.groupby(['route_id', 'veh_order'])
        predicted['next_arrival_time'] = grouped['arrival_time'].shift(-1)  # last recorded stop in that trip has np.nan
        predicted['pure_travel_time'] = predicted['next_arrival_time'] - predicted['departure_time']
        predicted['travel_and_dwell_time'] = predicted['next_arrival_time'] - predicted['arrival_time']
        predicted.drop(columns=['next_arrival_time'], inplace=True)

        predicted['pure_travel_time'] = predicted['pure_travel_time'] * self.nb.config.config_dict['DTA']['unit_time'] / 60 # convert to minutes
        predicted['travel_and_dwell_time'] = predicted['travel_and_dwell_time'] * self.nb.config.config_dict['DTA']['unit_time'] / 60

        return predicted # note: here are all stops and all veh runs as long as it once arrives at a stop, not only observed ones


    def _massage_boarding_alighting_record(self, raw_boarding_alighting_record):
        # with open('raw_boarding_alighting_record.pkl', 'wb') as f:
        #     pickle.dump(raw_boarding_alighting_record, f)

        full_boarding_alighting_record_dict = dict()
        for stop in self.nb.busstop_virtual_list:
            stop_id = stop.get_virtual_busstop_ID()
            route_id = stop.get_route_ID()
            num_veh_runs = int(self.nb.demand_bus.route_demand[route_id])
            for i in range(num_veh_runs):
                key_tuple = (stop_id, i+1, 'board')
                full_boarding_alighting_record_dict[key_tuple] = 0
                key_tuple = (stop_id, i+1, 'alight')
                full_boarding_alighting_record_dict[key_tuple] = 0
        for record in raw_boarding_alighting_record:
            stop_id = int(record[4])
            veh_order = int(record[3])
            boarding_count = record[5]
            alighting_count = record[6]
            key_tuple = (stop_id, veh_order, 'board')      
            full_boarding_alighting_record_dict[key_tuple] += boarding_count
            key_tuple = (stop_id, veh_order, 'alight')
            full_boarding_alighting_record_dict[key_tuple] += alighting_count

        full_boarding_alighting_record = (list(full_boarding_alighting_record_dict.keys()),np.array(list(full_boarding_alighting_record_dict.values())))
        # self.full_boarding_alighting_record = full_boarding_alighting_record

        full_key_index_map = {tup: idx for idx, tup in enumerate(full_boarding_alighting_record[0])}
        indices = [full_key_index_map[tup] for tup in self.observed_stops_vehs_list]
        simulated_boarding_alighting_count_array = full_boarding_alighting_record[1][indices]

        simulated_boarding_alighting_record_for_observed = (self.observed_stops_vehs_list, simulated_boarding_alighting_count_array)

        # with open('simulated_boarding_alighting_record_for_observed.pkl', 'wb') as f:
        #     pickle.dump(simulated_boarding_alighting_record_for_observed, f)

        return simulated_boarding_alighting_record_for_observed


    def _massage_raw_boarding_alighting_dar(self, raw_bt_dar, raw_pnr_dar, f_transit, f_pnr):
        # with open('raw_dar.pkl', 'wb') as f:
        #     pickle.dump((raw_bt_dar, raw_pnr_dar), f)
        # with open('f_transit.pkl', 'wb') as f:
        #     pickle.dump(f_transit, f)

        if raw_bt_dar[0].shape[0] == 0:
            print("No raw_bt_dar. Consider increase the demand values")

        if raw_pnr_dar[0].shape[0] == 0:
            print("No raw_pnr_dar. Consider increase the demand values")

        # transit 
        transit_dar_col_tuple_list = []
        for assign_interval in range(self.num_assign_interval):
            for path_id in self.nb.path_table_bustransit.ID2path:
                transit_dar_col_tuple_list.append((path_id, assign_interval))
        
        # pnr 
        pnr_dar_col_tuple_list = []
        for assign_interval in range(self.num_assign_interval):
            for path_id in self.nb.path_table_pnr.ID2path:
                pnr_dar_col_tuple_list.append((path_id, assign_interval))

        # transit dar
        transit_dar_row_tuple_list_with_values = [raw_bt_dar[3][idx] for idx in raw_bt_dar[0]]
        transit_dar_col_tuple_list_with_values = [raw_bt_dar[4][idx] for idx in raw_bt_dar[1]]
        transit_dar_values_array = raw_bt_dar[2]
        col_index_map = {tup: idx for idx, tup in enumerate(transit_dar_col_tuple_list)}
        row_index_map = {tup: idx for idx, tup in enumerate(self.observed_stops_vehs_list)}
        transit_dar_row_indices = []
        transit_dar_col_indices = []
        transit_dar_values = []
        for i in range(len(transit_dar_row_tuple_list_with_values)):
            if transit_dar_row_tuple_list_with_values[i] in row_index_map:
                transit_dar_row_indices.append(row_index_map[transit_dar_row_tuple_list_with_values[i]])
                transit_dar_col_indices.append(col_index_map[transit_dar_col_tuple_list_with_values[i]])
                transit_dar_values.append(transit_dar_values_array[i])
        transit_dar_row_indices = np.array(transit_dar_row_indices)
        transit_dar_col_indices = np.array(transit_dar_col_indices)
        transit_dar_values = np.array(transit_dar_values) / f_transit[list(transit_dar_col_indices)]
        # mask = transit_dar_values <= 2
        # transit_dar_values = transit_dar_values[mask]
        # transit_dar_row_indices = transit_dar_row_indices[mask]
        # transit_dar_col_indices = transit_dar_col_indices[mask]
        # transit_dar_values = np.minimum(transit_dar_values, 1)  # cap the values to 1
        transit_dar = coo_matrix((transit_dar_values, (transit_dar_row_indices, transit_dar_col_indices)), shape=(len(self.observed_stops_vehs_list), len(transit_dar_col_tuple_list)))
        transit_dar = transit_dar.tocsr()

        # pnr dar
        pnr_dar_row_tuple_list_with_values = [raw_pnr_dar[3][idx] for idx in raw_pnr_dar[0]]
        pnr_dar_col_tuple_list_with_values = [raw_pnr_dar[4][idx] for idx in raw_pnr_dar[1]]
        pnr_dar_values_array = raw_pnr_dar[2]
        col_index_map = {tup: idx for idx, tup in enumerate(pnr_dar_col_tuple_list)}
        pnr_dar_row_indices = []
        pnr_dar_col_indices = []
        pnr_dar_values = []
        for i in range(len(pnr_dar_row_tuple_list_with_values)):
            if pnr_dar_row_tuple_list_with_values[i] in row_index_map:
                pnr_dar_row_indices.append(row_index_map[pnr_dar_row_tuple_list_with_values[i]])
                pnr_dar_col_indices.append(col_index_map[pnr_dar_col_tuple_list_with_values[i]])
                pnr_dar_values.append(pnr_dar_values_array[i])
        pnr_dar_row_indices = np.array(pnr_dar_row_indices)
        pnr_dar_col_indices = np.array(pnr_dar_col_indices)
        pnr_dar_values = np.array(pnr_dar_values) / f_pnr[list(pnr_dar_col_indices)]
        # mask = pnr_dar_values <= 2
        # pnr_dar_values = pnr_dar_values[mask]
        # pnr_dar_row_indices = pnr_dar_row_indices[mask]
        # pnr_dar_col_indices = pnr_dar_col_indices[mask]
        # pnr_dar_values = np.minimum(pnr_dar_values, 1)  # cap the values to 1
        pnr_dar = coo_matrix((pnr_dar_values, (pnr_dar_row_indices, pnr_dar_col_indices)), shape=(len(self.observed_stops_vehs_list), len(pnr_dar_col_tuple_list)))
        pnr_dar = pnr_dar.tocsr()

        # with open('output_dar.pkl', 'wb') as f:
        #     pickle.dump([transit_dar, pnr_dar], f)

        return transit_dar, pnr_dar


    def _massage_raw_dar(self, raw_dar, ass_freq, f, num_assign_interval, paths_list, observed_links, num_procs=5):
        assert(raw_dar.shape[1] == 5)
        if raw_dar.shape[0] == 0:
            print("No dar. Consider increase the demand values")
            return csr_matrix((num_assign_interval * len(observed_links), 
                               num_assign_interval * len(paths_list)))

        num_e_path = len(paths_list)
        num_e_link = len(observed_links)
        # 15 min
        small_assign_freq = ass_freq * self.nb.config.config_dict['DTA']['unit_time'] / 60

        # # if release_one_interval_biclass set the 1-min assign interval for vehicle for the multiclass modeling
        # raw_dar = raw_dar[(raw_dar[:, 1] < num_assign_interval * small_assign_freq) & (raw_dar[:, 3] < self.num_loading_interval), :]
        # # if release_one_interval_biclass already set the correct assign interval for vehicle, then this is 15 min intervals
        raw_dar = raw_dar[(raw_dar[:, 1] < num_assign_interval) & (raw_dar[:, 3] < self.num_loading_interval), :]
        
        # raw_dar[:, 0]: path no.
        # raw_dar[:, 1]: originally the count of 1 min interval, if release_one_interval_biclass already set the correct assign interval for vehicle, then this is 15 min intervals

        # wrong raw_dar may contain paths from different modes (driving, pnr)
        # print(np.min(raw_dar[:, 0].astype(int)))
        # print(np.max(raw_dar[:, 0].astype(int)))
        # print(np.min(paths_list))
        # print(np.max(paths_list))
        # print('raw_dar path unique length: ', len(np.unique(raw_dar[:, 0].astype(int))))
        # print(paths_list)  
        # for x in raw_dar[:, 0].astype(int):
        #     if len(np.where(x == paths_list)[0]) == 0:
        #         print('*************************************')
        #         print(x)
        #         print('*************************************')

        if num_procs > 1:
            raw_dar = pd.DataFrame(data=raw_dar)

            from pandarallel import pandarallel
            pandarallel.initialize(progress_bar=True, nb_workers=num_procs)

            # ind = raw_dar.loc[:, 0].parallel_apply(lambda x: True if x in set(paths_list) else False)
            # assert(np.sum(ind) == len(ind))
            
            if type(paths_list) == list:
                paths_list = np.array(paths_list)
            elif type(paths_list) == np.ndarray:
                pass
            else:
                raise Exception('Wrong data type of paths_list')

            # # if release_one_interval_biclass set the 1-min assign interval for vehicle for the multimodal modeling
            # path_seq = (raw_dar.loc[:, 0].astype(int).parallel_apply(lambda x: np.nonzero(paths_list == x)[0][0]) 
            #             + (raw_dar.loc[:, 1] / small_assign_freq).astype(int) * num_e_path).astype(int)

            # if release_one_interval_biclass already set the correct assign interval for vehicle, then this is 15 min intervals
            path_seq = (raw_dar.loc[:, 0].astype(int).parallel_apply(lambda x: np.nonzero(paths_list == x)[0][0]) 
                        + raw_dar.loc[:, 1].astype(int) * num_e_path).astype(int)
            
            if type(observed_links) == list:
                observed_links = np.array(observed_links)
            elif type(observed_links) == np.ndarray:
                pass
            else:
                raise Exception('Wrong data type of observed_links')

            link_seq = (raw_dar.loc[:, 2].astype(int).parallel_apply(lambda x: np.where(observed_links == x)[0][0])
                        + (raw_dar.loc[:, 3] / ass_freq).astype(int) * num_e_link).astype(int)

            p = raw_dar.loc[:, 4] / f[path_seq]
                
        else:

            if type(paths_list) == np.ndarray:
                # ind = np.array(list(map(lambda x: True if len(np.where(paths_list == x)[0]) > 0 else False, raw_dar[:, 0].astype(int)))).astype(bool)
                # assert(np.sum(ind) == len(ind))
                # # if release_one_interval_biclass set the 1-min assign interval for vehicle for the multimodal modeling
                # path_seq = (np.array(list(map(lambda x: np.where(paths_list == x)[0][0], raw_dar[:, 0].astype(int))))
                #             + (raw_dar[:, 1] / small_assign_freq).astype(int) * num_e_path).astype(int)
                # if release_one_interval_biclass already set the correct assign interval for vehicle, then this is 15 min intervals
                path_seq = (np.array(list(map(lambda x: np.where(paths_list == x)[0][0], raw_dar[:, 0].astype(int))))
                            + raw_dar[:, 1].astype(int) * num_e_path).astype(int)

            elif type(paths_list) == list:
                # ind = np.array(list(map(lambda x: True if x in paths_list else False, raw_dar[:, 0].astype(int)))).astype(bool)
                # assert(np.sum(ind) == len(ind))
                # # if release_one_interval_biclass set the 1-min assign interval for vehicle for the multimodal modeling
                # path_seq = (np.array(list(map(lambda x: paths_list.index(x), raw_dar[:, 0].astype(int))))
                #             + (raw_dar[:, 1] / small_assign_freq).astype(int) * num_e_path).astype(int)
                # if release_one_interval_biclass already set the correct assign interval for vehicle, then this is 15 min intervals
                path_seq = (np.array(list(map(lambda x: paths_list.index(x), raw_dar[:, 0].astype(int))))
                            + raw_dar[:, 1].astype(int) * num_e_path).astype(int)

            else:
                raise Exception('Wrong data type of paths_list')

            # raw_dar[:, 2]: link no.
            # raw_dar[:, 3]: the count of unit time interval (5s)
            if type(observed_links) == np.ndarray:
                # In Python 3, map() returns an iterable while, in Python 2, it returns a list.
                link_seq = (np.array(list(map(lambda x: np.where(observed_links == x)[0][0], raw_dar[:, 2].astype(int))))
                            + (raw_dar[:, 3] / ass_freq).astype(int) * num_e_link).astype(int)
            elif type(observed_links) == list:
                link_seq = (np.array(list(map(lambda x: observed_links.index(x), raw_dar[:, 2].astype(int))))
                            + (raw_dar[:, 3] / ass_freq).astype(int) * num_e_link).astype(int)
            else:
                raise Exception('Wrong data type of observed_links')
                        
            # print(path_seq)
            # raw_dar[:, 4]: flow
            p = raw_dar[:, 4] / f[path_seq]
        
        # print("Creating the coo matrix", time.time()), coo_matrix permits duplicate entries
        mat = coo_matrix((p, (link_seq, path_seq)), shape=(num_assign_interval * num_e_link, num_assign_interval * num_e_path))
        # pickle.dump((p, link_seq, path_seq), open('test.pickle', 'wb'))
        # print('converting the csr', time.time())
        
        # sum duplicate entries in coo_matrix
        mat = mat.tocsr()
        # print('finish converting', time.time())
        return mat

    def init_demand_vector(self, num_assign_interval, num_col, scale=1):
        # uniform
        d = np.random.rand(num_assign_interval * num_col) * scale

        # Kaiming initialization (not working)
        # d = np.random.normal(0, 1, num_assign_interval * num_col) * scale
        # d *= np.sqrt(2 / len(d))
        # d = np.abs(d)

        # Xavier initialization
        # x = torch.Tensor(num_assign_interval * num_col, 1)
        # d = torch.abs(nn.init.xavier_uniform_(x)).squeeze().data.numpy() * scale
        # d = d.astype(float)
        return d

    def init_demand_flow(self, num_OD, init_scale=0.1):
        # demand matrix (num_OD x num_assign_interval) flattened in F order
        return self.init_demand_vector(self.num_assign_interval, num_OD, init_scale)

    def init_path_flow(self, car_driving_scale=1, truck_driving_scale=0.1, passenger_bustransit_scale=1, car_pnr_scale=0.5, bus_scale=0.1):

        f_car_driving = self.init_demand_vector(self.num_assign_interval, self.num_path_driving, car_driving_scale)
        f_truck_driving = self.init_demand_vector(self.num_assign_interval, self.num_path_driving, truck_driving_scale)
        f_passenger_bustransit = self.init_demand_vector(self.num_assign_interval, self.num_path_bustransit, passenger_bustransit_scale)
        f_car_pnr = self.init_demand_vector(self.num_assign_interval, self.num_path_pnr, car_pnr_scale)

        f_bus = None
        if bus_scale is not None:
            f_bus = self.init_demand_vector(self.num_assign_interval, self.num_path_busroute, bus_scale)
        return f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus

    def compute_path_cost(self, dta):
        # dta.build_link_cost_map() should be called before this method
        start_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)

        # for existing paths
        path_cost = dta.get_registered_path_cost_driving(start_intervals)
        assert(path_cost.shape[0] == self.num_path_driving)
        path_tt = dta.get_registered_path_tt_truck(start_intervals)
        assert(path_tt.shape[0] == self.num_path_driving)
        for i, path_ID in enumerate(self.nb.path_table_driving.ID2path.keys()):
            self.nb.path_table_driving.ID2path[path_ID].path_cost_car = path_cost[i, :]
            self.nb.path_table_driving.ID2path[path_ID].path_cost_truck = path_tt[i, :]
        
        path_cost = dta.get_registered_path_cost_bustransit(start_intervals)
        assert(path_cost.shape[0] == self.num_path_bustransit)
        for i, path_ID in enumerate(self.nb.path_table_bustransit.ID2path.keys()):
            self.nb.path_table_bustransit.ID2path[path_ID].path_cost = path_cost[i, :]
        
        path_cost = dta.get_registered_path_cost_pnr(start_intervals)
        assert(path_cost.shape[0] == self.num_path_pnr)
        for i, path_ID in enumerate(self.nb.path_table_pnr.ID2path.keys()):
            self.nb.path_table_pnr.ID2path[path_ID].path_cost = path_cost[i, :]

    def compute_path_flow_grad_and_loss(self, one_data_dict, f_car_driving, f_truck_driving, 
                                        f_passenger_bustransit, f_car_pnr, f_bus, fix_bus=True, counter=0, run_mmdta_adaptive=False, init_loss = None,
                                        explicit_bus = 1, isUsingbymode = False):
        # print("Running simulation", time.time())
        if isUsingbymode:
            dta = self._run_simulation(counter = counter, run_mmdta_adaptive =run_mmdta_adaptive, show_loading=False,
                                    explicit_bus=explicit_bus, historical_bus_waiting_time=0)
        else:
            dta = self._run_simulation(f_car_driving = f_car_driving, f_truck_driving= f_truck_driving, f_passenger_bustransit= f_passenger_bustransit, 
                                        f_car_pnr= f_car_pnr, f_bus= f_bus,
                                        counter = counter, run_mmdta_adaptive =run_mmdta_adaptive, show_loading=False,
                                        explicit_bus=explicit_bus, historical_bus_waiting_time=0)

        if self.config['use_car_link_tt'] or self.config['use_truck_link_tt'] or self.config['use_passenger_link_tt'] or self.config['use_bus_link_tt']:
            dta.build_link_cost_map(True)
            # TODO: C++
            # if self.config['use_car_link_tt'] or self.config['use_truck_link_tt'] or self.config['use_passenger_link_tt']:
            #     dta.get_link_queue_dissipated_time()
            #     # TODO: unfinished
            #     car_ltg_matrix_driving, truck_ltg_matrix_driving, car_ltg_matrix_pnr, bus_ltg_matrix_transit_link, bus_ltg_matrix_driving_link, \
            #     passenger_ltg_matrix_bustransit, passenger_ltg_matrix_pnr = \
            #         self.get_ltg(dta)

        # print("Getting DAR", time.time())
        car_dar_matrix_driving, truck_dar_matrix_driving, car_dar_matrix_pnr, bus_dar_matrix_transit_link, bus_dar_matrix_driving_link, \
               passenger_dar_matrix_bustransit, passenger_dar_matrix_pnr, car_dar_matrix_bus_driving_link, truck_dar_matrix_bus_driving_link, passenger_dar_matrix_bustransit_bus_link, passenger_dar_matrix_pnr_bus_link, \
               passenger_bus_link_flow_relationship, passenger_BoardingAlighting_dar_transit, passenger_BoardingAlighting_dar_pnr = \
                   self.get_dar(dta, f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus, fix_bus=fix_bus)
        # print("Evaluating grad", time.time())

        # test DAR
        # print('+++++++++++++++++++++++++++++++++++++ test DAR +++++++++++++++++++++++++++++++++++++')
        # x_e = dta.get_link_car_inflow(np.arange(0, self.num_loading_interval, self.ass_freq), 
        #                               np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq).flatten(order='F')
        # print(np.linalg.norm(x_e - car_dar_matrix_driving.dot(f_car_driving) - car_dar_matrix_pnr.dot(f_car_pnr)))

        # x_e = dta.get_link_truck_inflow(np.arange(0, self.num_loading_interval, self.ass_freq), 
        #                                 np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq).flatten(order='F')
        # print(np.linalg.norm(x_e - truck_dar_matrix_driving.dot(f_truck_driving) - bus_dar_matrix_driving_link.dot(f_bus)))

        # x_e_bus_passenger = dta.get_link_bus_passenger_inflow(
        #     np.arange(0, self.num_loading_interval, self.ass_freq), 
        #     np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq)
        # assert(x_e_bus_passenger.shape[0] == len(self.observed_links_bus))
        # x_e_walking_passenger = dta.get_link_walking_passenger_inflow(
        #     np.arange(0, self.num_loading_interval, self.ass_freq), 
        #     np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq)
        # assert(x_e_walking_passenger.shape[0] == len(self.observed_links_walking))
        # x_e = np.concatenate((x_e_bus_passenger, x_e_walking_passenger), axis=0).flatten(order='F')
        # print(np.linalg.norm(x_e - passenger_dar_matrix_bustransit.dot(f_passenger_bustransit) - passenger_dar_matrix_pnr.dot(f_car_pnr)))
        # print('+++++++++++++++++++++++++++++++++++++ test DAR +++++++++++++++++++++++++++++++++++++')

        # if use scaled loss, i.e., modify the weights
        if 'use_scaled_loss' in self.config:
            if self.config['use_scaled_loss']:
                if not init_loss:
                    _, init_loss = self._get_loss(one_data_dict, dta)

                if self.config['use_car_link_flow']:
                    self.config['link_car_flow_weight'] = self.config['link_car_flow_weight'] / init_loss['car_count_loss']
                if self.config['use_truck_link_flow']:
                    self.config['link_truck_flow_weight'] = self.config['link_truck_flow_weight'] / init_loss['truck_count_loss']
                if self.config['use_passenger_link_flow']:
                    self.config['link_passenger_flow_weight'] = self.config['link_passenger_flow_weight'] / init_loss['passenger_count_loss']
                if self.config['use_bus_link_flow']:
                    self.config['link_bus_flow_weight'] = self.config['link_bus_flow_weight'] / init_loss['bus_count_loss']
                if self.config['use_car_link_tt']:
                    self.config['link_car_tt_weight'] = self.config['link_car_tt_weight'] / init_loss['car_tt_loss']
                if self.config['use_truck_link_tt']:
                    self.config['link_truck_tt_weight'] = self.config['link_truck_tt_weight'] / init_loss['truck_tt_loss']
                if self.config['use_passenger_link_tt']:
                    self.config['link_passenger_tt_weight'] = self.config['link_passenger_tt_weight'] / init_loss['passenger_tt_loss']
                if self.config['use_bus_link_tt']:
                    self.config['link_bus_tt_weight'] = self.config['link_bus_tt_weight'] / init_loss['bus_tt_loss']
                if self.config['use_veh_run_boarding_alighting']:
                    self.config['veh_run_boarding_alighting_weight'] = self.config['veh_run_boarding_alighting_weight'] / init_loss['veh_run_boarding_alighting_loss']


        # derivative of count loss with respect to link flow
        car_grad = np.zeros(len(self.observed_links_driving) * self.num_assign_interval)
        truck_grad = np.zeros(len(self.observed_links_driving) * self.num_assign_interval)
        bus_grad = np.zeros(len(self.observed_links_bus) * self.num_assign_interval)
        passenger_grad = np.zeros((len(self.observed_links_bus) + len(self.observed_links_walking)) * self.num_assign_interval)
        car_grad_for_bus = np.zeros(len(self.observed_links_bus_driving) * self.num_assign_interval)
        truck_grad_for_bus = np.zeros(len(self.observed_links_bus_driving) * self.num_assign_interval)

        # derivative of passenger boarding and alighting loss with respect to boarding_alighting record
        passenger_BoardingAlighting_grad = np.zeros(len(self.observed_stops_vehs_list))

        x_e_car, x_e_truck, x_e_passenger, x_e_bus, x_e_BoardingAlighting_count, stop_arrival_departure_travel_time_df = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        if self.config['use_car_link_flow']:
            grad, x_e_car = self._compute_count_loss_grad_on_car_link_flow(dta, one_data_dict)
            car_grad += self.config['link_car_flow_weight'] * grad
        if self.config['use_truck_link_flow']:
            grad, x_e_truck = self._compute_count_loss_grad_on_truck_link_flow(dta, one_data_dict)
            truck_grad += self.config['link_truck_flow_weight'] * grad
        if self.config['use_passenger_link_flow']:
            grad, x_e_passenger = self._compute_count_loss_grad_on_passenger_link_flow(dta, one_data_dict)
            passenger_grad += self.config['link_passenger_flow_weight'] * grad
        if self.config['use_bus_link_flow']:
            grad, x_e_bus = self._compute_count_loss_grad_on_bus_link_flow(dta, one_data_dict)
            bus_grad += self.config['link_bus_flow_weight'] * grad
        if self.config['use_veh_run_boarding_alighting'] or self.config['use_ULP_f_transit']:
            grad, x_e_BoardingAlighting_count, stop_arrival_departure_travel_time_df = self._compute_BoardingAlighting_loss_grad_on_BoardingAlighting_record(dta, one_data_dict)
            if self.config['use_veh_run_boarding_alighting']:
                passenger_BoardingAlighting_grad += self.config['veh_run_boarding_alighting_weight'] * grad

            # car_grad_for_bus += self._compute_link_flow_grad_on_car_link_flow_for_bus(bus_grad)
            # truck_grad_for_bus += self._compute_link_flow_grad_on_truck_link_flow_for_bus(bus_grad)

            # TODO: assume no walking links
            if not fix_bus:
                bus_grad += passenger_grad  # passenger_bus_link_flow_relationship * passenger_grad
            # passenger_grad += passenger_bus_link_flow_relationship * bus_grad # * np.nan_to_num(one_data_dict['bus_link_flow'] > 0)

        # derivative of count loss with respect to path flow
        f_car_driving_grad = car_dar_matrix_driving.T.dot(car_grad) # + car_dar_matrix_bus_driving_link.T.dot(car_grad_for_bus)
        f_truck_driving_grad = truck_dar_matrix_driving.T.dot(truck_grad) # + truck_dar_matrix_bus_driving_link.T.dot(truck_grad_for_bus)
        f_passenger_bustransit_grad = passenger_dar_matrix_bustransit.T.dot(passenger_grad)
        f_car_pnr_grad = car_dar_matrix_pnr.T.dot(car_grad)
        f_passenger_pnr_grad = passenger_dar_matrix_pnr.T.dot(passenger_grad)
        f_bus_grad = bus_dar_matrix_transit_link.T.dot(bus_grad) if not fix_bus else None # + (f_bus - self.nb.demand_bus.path_flow_matrix.flatten(order='F')) * 2

        # derivative of boarding_alighting record with respect to path flow
        if self.config['use_veh_run_boarding_alighting']:
            f_passenger_bustransit_grad += passenger_BoardingAlighting_dar_transit.T.dot(passenger_BoardingAlighting_grad)
            f_passenger_pnr_grad += passenger_BoardingAlighting_dar_pnr.T.dot(passenger_BoardingAlighting_grad)

        if self.config['use_ULP_f_transit']:
            # f_transit_ULP = spla.lsqr(passenger_BoardingAlighting_dar_transit, one_data_dict['veh_run_boarding_alighting_record'][1])[0] 
            # f_transit_ULP = lsq_linear(passenger_BoardingAlighting_dar_transit, one_data_dict['veh_run_boarding_alighting_record'][1], bounds=(0, np.inf), max_iter=2).x
            # f_transit_ULP = compute_ULP_f(passenger_BoardingAlighting_dar_transit, one_data_dict['veh_run_boarding_alighting_record'][1], 5e-1, 3000)
            mask_record = one_data_dict['mask_observed_stops_vehs_record']
            row_indices = np.where(mask_record)[0]
            f_transit_ULP = compute_ULP_f_Gurobi(passenger_BoardingAlighting_dar_transit[row_indices], one_data_dict['veh_run_boarding_alighting_record'][1][mask_record])
            if f_transit_ULP is None:
                print("Warning: f_transit_ULP is None, update based on gradient")
            else:
                f_transit_ULP = np.maximum(f_transit_ULP, 1e-6)  
        else:
            f_transit_ULP = None

        # derivative of travel time loss with respect to link travel time
        car_grad = np.zeros(len(self.observed_links_driving) * self.num_assign_interval)
        truck_grad = np.zeros(len(self.observed_links_driving) * self.num_assign_interval)
        bus_grad = np.zeros(len(self.observed_links_bus) * self.num_assign_interval)
        passenger_grad = np.zeros((len(self.observed_links_bus) + len(self.observed_links_walking)) * self.num_assign_interval)

        tt_e_car, tt_e_truck, tt_e_passenger, tt_e_bus = np.nan, np.nan, np.nan, np.nan

        if self.config['use_car_link_tt']:
            grad, tt_e_car = self._compute_tt_loss_grad_on_car_link_tt(dta, one_data_dict)
            car_grad += self.config['link_car_tt_weight'] * grad
        if self.config['use_truck_link_tt']:
            grad, tt_e_truck = self._compute_tt_loss_grad_on_truck_link_tt(dta, one_data_dict)
            truck_grad += self.config['link_truck_tt_weight'] * grad
        if self.config['use_passenger_link_tt']:
            grad, tt_e_passenger = self._compute_tt_loss_grad_on_passenger_link_tt(dta, one_data_dict)
            passenger_grad += self.config['link_passenger_tt_weight'] * grad
        if self.config['use_bus_link_tt']:
            grad, tt_e_bus = self._compute_tt_loss_grad_on_bus_link_tt(dta, one_data_dict)
            bus_grad += self.config['link_bus_tt_weight'] * grad

        # TODO: derivative of travel time loss with respect to path flow, car finished
        if self.config['use_car_link_tt']:
            _tt_loss_grad_on_car_link_flow = self._compute_tt_loss_grad_on_car_link_flow(dta, car_grad)
            f_car_driving_grad += car_dar_matrix_driving.T.dot(_tt_loss_grad_on_car_link_flow)
            # f_car_pnr_grad += car_dar_matrix_pnr.T.dot(_tt_loss_grad_on_car_link_flow)

            # f_car_driving_grad += car_ltg_matrix_driving.T.dot(car_grad)
            # f_car_pnr_grad += car_ltg_matrix_pnr.T.dot(car_grad)
               
        if self.config['use_truck_link_tt']:
            f_truck_driving_grad += truck_dar_matrix_driving.T.dot(self._compute_tt_loss_grad_on_truck_link_flow(dta, truck_grad))

            # f_truck_driving_grad += truck_ltg_matrix_driving.T.dot(truck_grad)

        if self.config['use_passenger_link_tt']:
            _tt_loss_grad_on_passenger_link_flow = self._compute_tt_loss_grad_on_passenger_link_flow(dta, passenger_grad)
            f_passenger_bustransit_grad += passenger_dar_matrix_bustransit.T.dot(_tt_loss_grad_on_passenger_link_flow)
            f_passenger_pnr_grad += passenger_dar_matrix_pnr.T.dot(_tt_loss_grad_on_passenger_link_flow)

            # f_passenger_bustransit_grad += passenger_ltg_matrix_bustransit.T.dot(passenger_grad)
            # f_passenger_pnr_grad += passenger_ltg_matrix_pnr.T.dot(passenger_grad)

        if self.config['use_bus_link_tt']:
            if not fix_bus:
                f_bus_grad += bus_dar_matrix_transit_link.T.dot(bus_grad)

            # f_car_driving_grad += car_dar_matrix_bus_driving_link.T.dot(self._compute_tt_loss_grad_on_car_link_flow_for_bus(dta, bus_grad))
            # f_truck_driving_grad += truck_dar_matrix_bus_driving_link.T.dot(self._compute_tt_loss_grad_on_truck_link_flow_for_bus(dta, bus_grad))

            _tt_loss_grad_on_passenger_link_flow_for_bus = self._compute_tt_loss_grad_on_passenger_link_flow_for_bus(dta, bus_grad)
            f_passenger_bustransit_grad += passenger_dar_matrix_bustransit_bus_link.T.dot(_tt_loss_grad_on_passenger_link_flow_for_bus)
            # f_passenger_pnr_grad += passenger_dar_matrix_pnr_bus_link.T.dot(_tt_loss_grad_on_passenger_link_flow_for_bus)
        
        # print("Getting Loss", time.time())
        total_loss, loss_dict = self._get_loss(one_data_dict, dta)

        # also save the weighted loss for computation in hypothesis testing
        loss_dict['car_count_loss_weighted'] = self.config['link_car_flow_weight'] * loss_dict['car_count_loss'] if self.config['use_car_link_flow'] else 0
        loss_dict['truck_count_loss_weighted'] = self.config['link_truck_flow_weight'] * loss_dict['truck_count_loss'] if self.config['use_truck_link_flow'] else 0
        loss_dict['bus_count_loss_weighted'] = self.config['link_bus_flow_weight'] * loss_dict['bus_count_loss'] if self.config['use_bus_link_flow'] else 0
        loss_dict['passenger_count_loss_weighted'] = self.config['link_passenger_flow_weight'] * loss_dict['passenger_count_loss'] if self.config['use_passenger_link_flow'] else 0
        loss_dict['car_tt_loss_weighted'] = self.config['link_car_tt_weight'] * loss_dict['car_tt_loss'] if self.config['use_car_link_tt'] else 0
        loss_dict['truck_tt_loss_weighted'] = self.config['link_truck_tt_weight'] * loss_dict['truck_tt_loss'] if self.config['use_truck_link_tt'] else 0
        loss_dict['bus_tt_loss_weighted'] = self.config['link_bus_tt_weight'] * loss_dict['bus_tt_loss'] if self.config['use_bus_link_tt'] else 0
        loss_dict['passenger_tt_loss_weighted'] = self.config['link_passenger_tt_weight'] * loss_dict['passenger_tt_loss'] if self.config['use_passenger_link_tt'] else 0
        loss_dict['veh_run_boarding_alighting_loss_weighted'] = self.config['veh_run_boarding_alighting_weight'] * loss_dict['veh_run_boarding_alighting_loss'] if self.config['use_veh_run_boarding_alighting'] else 0
        loss_dict['total_loss_weighted'] = loss_dict['car_count_loss_weighted'] + loss_dict['truck_count_loss_weighted'] + \
            loss_dict['bus_count_loss_weighted'] + loss_dict['passenger_count_loss_weighted'] + \
            loss_dict['veh_run_boarding_alighting_loss_weighted'] + \
            loss_dict['car_tt_loss_weighted'] + loss_dict['truck_tt_loss_weighted'] + loss_dict['bus_tt_loss_weighted'] + loss_dict['passenger_tt_loss_weighted']

        # _bound = 0.1
        # f_car_driving_grad, f_truck_driving_grad, f_passenger_bustransit_grad, f_car_pnr_grad, f_passenger_pnr_grad, f_bus_grad = \
        #     np.clip(f_car_driving_grad, -_bound, _bound), \
        #     np.clip(f_truck_driving_grad, -_bound, _bound), \
        #     np.clip(f_passenger_bustransit_grad, -_bound, _bound), \
        #     np.clip(f_car_pnr_grad, -_bound, _bound), \
        #     np.clip(f_passenger_pnr_grad, -_bound, _bound), \
        #     np.clip(f_bus_grad, -_bound, _bound)   
        # 

        # below are for Jacobian matrices used in hypothesis testing
        car_count_J_f_driving = car_dar_matrix_driving
        # car_count_J_f_transit = np.zeros((car_dar_matrix_driving.shape[0], self.num_path_bustransit * self.num_assign_interval))
        car_count_J_f_pnr = car_dar_matrix_pnr
        temp_link_tt_grad_on_link_flow_car = self._compute_link_tt_grad_on_link_flow_car(dta)
        car_time_J_f_driving = temp_link_tt_grad_on_link_flow_car.dot(car_dar_matrix_driving)
        # car_time_J_f_transit = np.zeros((temp_link_tt_grad_on_link_flow_car.shape[0], self.num_path_bustransit * self.num_assign_interval))
        car_time_J_f_pnr = temp_link_tt_grad_on_link_flow_car.dot(car_dar_matrix_pnr)
        # BoardingAlightingCount_J_f_driving = np.zeros((len(self.observed_stops_vehs_list), self.num_path_driving * self.num_assign_interval))
        BoardingAlightingCount_J_f_transit = passenger_BoardingAlighting_dar_transit
        BoardingAlightingCount_J_f_pnr = passenger_BoardingAlighting_dar_pnr
        Jacobian_dict = {
            'car_count_J_f_driving': car_count_J_f_driving,
            'car_count_J_f_pnr': car_count_J_f_pnr,
            'car_time_J_f_driving': car_time_J_f_driving,
            'car_time_J_f_pnr': car_time_J_f_pnr,
            'BoardingAlightingCount_J_f_transit': BoardingAlightingCount_J_f_transit,
            'BoardingAlightingCount_J_f_pnr': BoardingAlightingCount_J_f_pnr
        }


        return f_car_driving_grad, f_truck_driving_grad, f_passenger_bustransit_grad, f_car_pnr_grad, f_passenger_pnr_grad, f_bus_grad, \
               total_loss, loss_dict, dta, x_e_car, x_e_truck, x_e_passenger, x_e_bus, x_e_BoardingAlighting_count, tt_e_car, tt_e_truck, tt_e_passenger, \
               tt_e_bus, f_transit_ULP, stop_arrival_departure_travel_time_df, Jacobian_dict

    def _compute_BoardingAlighting_loss_grad_on_BoardingAlighting_record(self, dta, one_data_dict):
        observed_x_e_BoardingAlighting_count = one_data_dict['veh_run_boarding_alighting_record'][1]
        raw_boarding_alighting_record = dta.get_bus_boarding_alighting_record()
        simulated_boarding_alighting_record_for_observed = self._massage_boarding_alighting_record(raw_boarding_alighting_record)
        x_e_BoardingAlighting_count = simulated_boarding_alighting_record_for_observed[1]
        assert(len(x_e_BoardingAlighting_count) == len(observed_x_e_BoardingAlighting_count))
        discrepancy = np.nan_to_num(observed_x_e_BoardingAlighting_count - x_e_BoardingAlighting_count)
        grad = - discrepancy

        stop_arrival_departure_travel_time_df = self.get_stop_arrival_departure_travel_time(raw_boarding_alighting_record)

        return grad, x_e_BoardingAlighting_count, stop_arrival_departure_travel_time_df


    def _compute_count_loss_grad_on_car_link_flow(self, dta, one_data_dict):
        link_flow_array = one_data_dict['car_link_flow']
        # num_links_driving x num_assign_interval flatened in F order
        x_e = dta.get_link_car_inflow(np.arange(0, self.num_loading_interval, self.ass_freq), 
                                      np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq).flatten(order='F')
        assert(len(x_e) == len(self.observed_links_driving) * self.num_assign_interval)
        # print("x_e", x_e, link_flow_array)
        if self.config['car_count_agg']:
            x_e = one_data_dict['car_count_agg_L'].dot(x_e)
        discrepancy = np.nan_to_num(link_flow_array - x_e)
        grad = - discrepancy
        # grad = - np.nan_to_num(discrepancy / np.maximum(np.max(link_flow_array), 1))
        # grad = - np.nan_to_num(discrepancy / np.maximum(link_flow_array, 1))
        # grad = - discrepancy / (np.linalg.norm(discrepancy) * np.linalg.norm(np.nan_to_num(link_flow_array)) + 1e-6)
        if self.config['car_count_agg']:
            grad = one_data_dict['car_count_agg_L'].T.dot(grad)
        # print("final link grad", grad)

        # grad = grad / (np.linalg.norm(grad) + 1e-7)
        return grad, x_e

    def _compute_count_loss_grad_on_truck_link_flow(self, dta, one_data_dict):
        link_flow_array = one_data_dict['truck_link_flow']
        # num_links_driving x num_assign_interval flatened in F order
        x_e = dta.get_link_truck_inflow(np.arange(0, self.num_loading_interval, self.ass_freq), 
                                        np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq).flatten(order='F')
        assert(len(x_e) == len(self.observed_links_driving) * self.num_assign_interval)
        # print("x_e", x_e, link_flow_array)
        if self.config['truck_count_agg']:
            x_e = one_data_dict['truck_count_agg_L'].dot(x_e)
        discrepancy = np.nan_to_num(link_flow_array - x_e)
        grad = - discrepancy
        # grad = - np.nan_to_num(discrepancy / np.maximum(np.max(link_flow_array), 1))
        # grad = - np.nan_to_num(discrepancy / np.maximum(link_flow_array, 1))
        # grad = -discrepancy / (np.linalg.norm(discrepancy) * np.linalg.norm(np.nan_to_num(link_flow_array)) + 1e-6)
        if self.config['truck_count_agg']:
            grad = one_data_dict['truck_count_agg_L'].T.dot(grad)
        # print("final link grad", grad)

        # grad = grad / (np.linalg.norm(grad) + 1e-7)
        return grad, x_e

    def _compute_count_loss_grad_on_passenger_link_flow(self, dta, one_data_dict):
        link_flow_array = one_data_dict['passenger_link_flow']
        # (num_links_bus + num_links_walking)  x num_assign_interval flatenned in F order
        x_e_bus_passenger = dta.get_link_bus_passenger_inflow(
            np.arange(0, self.num_loading_interval, self.ass_freq), 
            np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq)
        assert(x_e_bus_passenger.shape[0] == len(self.observed_links_bus))
        x_e_walking_passenger = dta.get_link_walking_passenger_inflow(
            np.arange(0, self.num_loading_interval, self.ass_freq), 
            np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq)
        assert(x_e_walking_passenger.shape[0] == len(self.observed_links_walking))
        x_e = np.concatenate((x_e_bus_passenger, x_e_walking_passenger), axis=0).flatten(order='F')
        # print("x_e", x_e, link_flow_array)
        if self.config['passenger_count_agg']:
            x_e = one_data_dict['passenger_count_agg_L'].dot(x_e)
        discrepancy = np.nan_to_num(link_flow_array - x_e)
        grad = - discrepancy
        # grad = - np.nan_to_num(discrepancy / np.maximum(np.max(link_flow_array), 1))
        # grad = - np.nan_to_num(discrepancy / np.maximum(link_flow_array, 1))
        # grad = - discrepancy / (np.linalg.norm(discrepancy) * np.linalg.norm(np.nan_to_num(link_flow_array)) + 1e-6)
        if self.config['passenger_count_agg']:
            grad = one_data_dict['passenger_count_agg_L'].T.dot(grad)
        # print("final link grad", grad)

        # grad = grad / (np.linalg.norm(grad) + 1e-7)
        return grad, x_e

    def _compute_count_loss_grad_on_bus_link_flow(self, dta, one_data_dict):
        link_flow_array = one_data_dict['bus_link_flow']
        # num_links_bus x num_assign_interval flatened in F order
        x_e = dta.get_link_bus_inflow(
            np.arange(0, self.num_loading_interval, self.ass_freq), 
            np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq).flatten(order='F')
        # print("x_e", x_e, link_flow_array)
        if self.config['bus_count_agg']:
            x_e = one_data_dict['bus_count_agg_L'].dot(x_e)
        discrepancy = np.nan_to_num(link_flow_array - x_e)
        grad = - discrepancy
        # grad = - np.nan_to_num(discrepancy / np.maximum(np.max(link_flow_array), 1))
        # grad = - np.nan_to_num(discrepancy / np.maximum(link_flow_array, 1))
        # grad = -discrepancy / (np.linalg.norm(discrepancy) * np.linalg.norm(np.nan_to_num(link_flow_array)) + 1e-6)
        if self.config['bus_count_agg']:
            grad = one_data_dict['bus_count_agg_L'].T.dot(grad)
        # print("final link grad", grad)

        # grad = grad / (np.linalg.norm(grad) + 1e-7)
        return grad, x_e

    def _compute_link_flow_grad_on_car_link_flow_for_bus(self, _count_loss_grad_on_bus_link_flow):
        # heuristic: when est_bus > obs_bus, _count_loss_grad_on_bus_link_flow > 0, too few trucks, we increase truck flow;
        #            when est_bus < obs_bus, _count_loss_grad_on_bus_link_flow < 0, too many trucks, we decrease truck flow;
        _link_count_bus_grad_on_link_count_car = self.bus_driving_link_relation.T > 0
        grad = _link_count_bus_grad_on_link_count_car @ (- _count_loss_grad_on_bus_link_flow)
        return grad

    def _compute_link_flow_grad_on_truck_link_flow_for_bus(self, _count_loss_grad_on_bus_link_flow):
        # heuristic: when est_bus > obs_bus, _count_loss_grad_on_bus_link_flow > 0, too few trucks, we increase truck flow;
        #            when est_bus < obs_bus, _count_loss_grad_on_bus_link_flow < 0, too many trucks, we decrease truck flow;
        _link_count_bus_grad_on_link_count_truck = self.bus_driving_link_relation.T > 0
        grad = _link_count_bus_grad_on_link_count_truck @ (- _count_loss_grad_on_bus_link_flow)
        return grad

    def _compute_tt_loss_grad_on_car_link_tt(self, dta, one_data_dict):
        tt_e = dta.get_car_link_tt_robust(np.arange(0, self.num_loading_interval, self.ass_freq),
                                          np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq, self.ass_freq, True).flatten(order='F')
        # tt_e = dta.get_car_link_tt(np.arange(0, self.num_loading_interval, self.ass_freq), True).flatten(order='F')
        assert(len(tt_e) == len(self.observed_links_driving) * self.num_assign_interval)
        # tt_free = np.tile(list(map(lambda x: self.nb.get_link_driving(x).get_car_fft(), self.observed_links_driving)), (self.num_assign_interval))
        tt_free = np.tile(dta.get_car_link_fftt(self.observed_links_driving), (self.num_assign_interval))
        tt_e = np.maximum(tt_e, tt_free)
        tt_o = np.maximum(one_data_dict['car_link_tt'], tt_free) 

        # don't use inf value
        ind = (np.isinf(tt_e) + np.isinf(tt_o))
        tt_e[ind] = 0
        tt_o[ind] = 0

        discrepancy = np.nan_to_num(tt_o - tt_e)
        grad = - discrepancy
        # grad = - np.nan_to_num(discrepancy / tt_o)
        # grad = -discrepancy / (np.linalg.norm(discrepancy) * np.linalg.norm(np.nan_to_num(tt_o)) + 1e-6)
        
        # if self.config['car_count_agg']:
        #   grad = one_data_dict['car_count_agg_L'].T.dot(grad)
        # print(tt_e, tt_o)
        # print("car_grad", grad)

        # grad = grad / (np.linalg.norm(grad) + 1e-7)
        # grad = np.clip(grad, -1, 1)

        tt_e[ind] = np.inf
        return grad, tt_e

    def _compute_tt_loss_grad_on_car_link_flow(self, dta, tt_loss_grad_on_car_link_tt):
        _link_tt_grad_on_link_flow_car = self._compute_link_tt_grad_on_link_flow_car(dta)
        grad = _link_tt_grad_on_link_flow_car.dot(tt_loss_grad_on_car_link_tt)
        grad = grad / (np.linalg.norm(grad) + 1e-7)
        return grad

    def _compute_tt_loss_grad_on_truck_link_tt(self, dta, one_data_dict):
        tt_e = dta.get_truck_link_tt_robust(np.arange(0, self.num_loading_interval, self.ass_freq),
                                            np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq, self.ass_freq, True).flatten(order='F')
        # tt_e = dta.get_truck_link_tt(np.arange(0, self.num_loading_interval, self.ass_freq), True).flatten(order='F')
        assert(len(tt_e) == len(self.observed_links_driving) * self.num_assign_interval)
        # tt_free = np.tile(list(map(lambda x: self.nb.get_link_driving(x).get_truck_fft(), self.observed_links_driving)), (self.num_assign_interval))
        tt_free = np.tile(dta.get_truck_link_fftt(self.observed_links_driving), (self.num_assign_interval))
        tt_e = np.maximum(tt_e, tt_free)
        tt_o = np.maximum(one_data_dict['truck_link_tt'], tt_free)

        # don't use inf value
        ind = (np.isinf(tt_e) + np.isinf(tt_o))
        tt_e[ind] = 0
        tt_o[ind] = 0

        discrepancy = np.nan_to_num(tt_o - tt_e)
        grad = - discrepancy
        # grad = - np.nan_to_num(discrepancy / tt_o)
        # grad = -discrepancy / (np.linalg.norm(discrepancy) * np.linalg.norm(np.nan_to_num(tt_o)) + 1e-6)

        # if self.config['truck_count_agg']:
        #   grad = one_data_dict['truck_count_agg_L'].T.dot(grad)
        # print("truck_grad", grad)

        # grad = grad / (np.linalg.norm(grad) + 1e-7)
        # grad = np.clip(grad, -1, 1)

        tt_e[ind] = np.inf
        return grad, tt_e

    def _compute_tt_loss_grad_on_truck_link_flow(self, dta, tt_loss_grad_on_truck_link_tt):
        _link_tt_grad_on_link_flow_truck = self._compute_link_tt_grad_on_link_flow_truck(dta)
        grad = _link_tt_grad_on_link_flow_truck.dot(tt_loss_grad_on_truck_link_tt)
        grad = grad / (np.linalg.norm(grad) + 1e-7)
        return grad

    def _compute_tt_loss_grad_on_passenger_link_tt(self, dta, one_data_dict):
        tt_e_bus = dta.get_bus_link_tt_robust(np.arange(0, self.num_loading_interval, self.ass_freq),
                                              np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq, self.ass_freq, True, True)
        # tt_e_bus = dta.get_bus_link_tt(np.arange(0, self.num_loading_interval, self.ass_freq), True, True)
        assert(tt_e_bus.shape[0] == len(self.observed_links_bus))
        tt_e_walking = dta.get_passenger_walking_link_tt_robust(np.arange(0, self.num_loading_interval, self.ass_freq),
                                                                np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq, self.ass_freq)
        # tt_e_walking = dta.get_passenger_walking_link_tt(np.arange(0, self.num_loading_interval, self.ass_freq))
        assert(tt_e_walking.shape[0] == len(self.observed_links_walking))
        tt_e = np.concatenate((tt_e_bus, tt_e_walking), axis=0).flatten(order='F')
        # tt_free = np.tile(list(map(lambda x: self.nb.get_link_bus(x).get_bus_fft(), self.observed_links_bus)) + 
        #                   list(map(lambda x: self.nb.get_link_walking(x).get_walking_fft(), self.observed_links_walking)), 
        #                   (self.num_assign_interval))
        # tt_free = np.tile(np.concatenate((dta.get_bus_link_fftt(self.observed_links_bus), dta.get_walking_link_fftt(self.observed_links_walking))), 
        #                   (self.num_assign_interval))
        # tt_e = np.maximum(tt_e, tt_free)
        # tt_o = np.maximum(one_data_dict['passenger_link_tt'], tt_free)

        # fill in the inf value
        # tt_e[np.isinf(tt_e)] = 5 * self.num_loading_interval
        # tt_o = np.nan_to_num(one_data_dict['passenger_link_tt'], posinf = 5 * self.num_loading_interval)

        # don't use inf value
        tt_o = one_data_dict['passenger_link_tt'].copy()
        ind = (np.isinf(tt_e) + np.isinf(tt_o))
        tt_e[ind] = 0
        tt_o[ind] = 0

        discrepancy = np.nan_to_num(tt_o - tt_e)
        grad = - discrepancy
        # grad = - np.nan_to_num(discrepancy / tt_o)
        # grad = -discrepancy / (np.linalg.norm(discrepancy) * np.linalg.norm(np.nan_to_num(tt_o)) + 1e-6)

        # if self.config['passenger_count_agg']:
        #   grad = one_data_dict['passenger_count_agg_L'].T.dot(grad)
        # print("passenger_grad", grad)

        # grad = grad / (np.linalg.norm(grad) + 1e-7)
        # grad = np.clip(grad, -1, 1)

        tt_e[ind] = np.inf
        return grad, tt_e

    def _compute_tt_loss_grad_on_passenger_link_flow(self, dta, tt_loss_grad_on_passenger_link_tt):
        _link_tt_grad_on_link_flow_passenger = self._compute_link_tt_grad_on_link_flow_passenger(dta)
        grad = _link_tt_grad_on_link_flow_passenger.dot(tt_loss_grad_on_passenger_link_tt)
        grad = grad / (np.linalg.norm(grad) + 1e-7)
        return grad

    def _compute_tt_loss_grad_on_bus_link_tt(self, dta, one_data_dict):
        tt_e = dta.get_bus_link_tt_robust(np.arange(0, self.num_loading_interval, self.ass_freq),
                                          np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq, self.ass_freq, True, True).flatten(order='F')
        # tt_e = dta.get_bus_link_tt(np.arange(0, self.num_loading_interval, self.ass_freq), True, True).flatten(order='F')
        # tt_free = np.tile(list(map(lambda x: self.nb.get_link_bus(x).get_bus_fft(), self.observed_links_bus)), (self.num_assign_interval))
        # tt_free = np.tile(dta.get_bus_link_fftt(self.observed_links_bus), (self.num_assign_interval))
        # tt_e = np.maximum(tt_e, tt_free)
        # tt_o = np.maximum(one_data_dict['bus_link_tt'], tt_free)
        # tt_o = one_data_dict['bus_link_tt']

        # fill in the inf value
        # tt_e[np.isinf(tt_e)] = 5 * self.num_loading_interval
        # tt_o = np.nan_to_num(one_data_dict['bus_link_tt'], posinf = 5 * self.num_loading_interval)

        # don't use inf value
        tt_o = one_data_dict['bus_link_tt'].copy()
        ind = (np.isinf(tt_e) + np.isinf(tt_o))
        tt_e[ind] = 0
        tt_o[ind] = 0

        discrepancy = np.nan_to_num(tt_o - tt_e)
        grad = - discrepancy
        # grad = - np.nan_to_num(discrepancy / tt_o)
        # grad = -discrepancy / (np.linalg.norm(discrepancy) * np.linalg.norm(np.nan_to_num(tt_o)) + 1e-6)

        # if self.config['bus_count_agg']:
        #   grad = one_data_dict['bus_count_agg_L'].T.dot(grad)
        # print("bus_grad", grad)

        # grad = grad / (np.linalg.norm(grad) + 1e-7)
        # grad = np.clip(grad, -1, 1)

        tt_e[ind] = np.inf
        return grad, tt_e

    def _compute_tt_loss_grad_on_car_link_flow_for_bus(self, dta, tt_loss_grad_on_bus_link_tt):
        _link_tt_bus_grad_on_link_tt_car = self.bus_driving_link_relation.T
        _link_tt_car_grad_on_link_flow_car = self._compute_link_tt_grad_on_link_flow_car_for_bus(dta)
        grad = _link_tt_car_grad_on_link_flow_car @ (_link_tt_bus_grad_on_link_tt_car @ tt_loss_grad_on_bus_link_tt)
        grad = grad / (np.linalg.norm(grad) + 1e-7)
        return grad

    def _compute_tt_loss_grad_on_truck_link_flow_for_bus(self, dta, tt_loss_grad_on_bus_link_tt):
        _link_tt_bus_grad_on_link_tt_truck = self.bus_driving_link_relation.T
        _link_tt_truck_grad_on_link_flow_truck = self._compute_link_tt_grad_on_link_flow_truck_for_bus(dta)
        grad = _link_tt_truck_grad_on_link_flow_truck @ (_link_tt_bus_grad_on_link_tt_truck @ tt_loss_grad_on_bus_link_tt)
        grad = grad / (np.linalg.norm(grad) + 1e-7)
        return grad

    def _compute_tt_loss_grad_on_passenger_link_flow_for_bus(self, dta, tt_loss_grad_on_bus_link_tt):
        _link_tt_grad_on_link_flow_passenger = self._compute_link_tt_grad_on_link_flow_passenger_for_bus(dta)
        grad = _link_tt_grad_on_link_flow_passenger.dot(tt_loss_grad_on_bus_link_tt)
        grad = grad / (np.linalg.norm(grad) + 1e-7)
        return grad

    # def _compute_grad_on_walking_link_tt(self, dta, one_data_dict):
    #     tt_e = dta.get_passenger_walking_link_tt(np.arange(0, self.num_loading_interval, self.ass_freq)).flatten(order='F')
    #     tt_free = np.tile(list(map(lambda x: self.nb.get_link_walking(x).get_walking_fft(), self.observed_links_walking)), (self.num_assign_interval))
    #     tt_e = np.maximum(tt_e, tt_free)
    #     tt_o = np.maximum(one_data_dict['walking_link_tt'], tt_free)
    #     grad = -np.nan_to_num(tt_o - tt_e)/tt_o
    #     # if self.config['bus_count_agg']:
    #     #   grad = one_data_dict['bus_count_agg_L'].T.dot(grad)
    #     # print("bus_grad", grad)
    #     return grad

    def _compute_link_tt_grad_on_link_flow_car(self, dta):
        assign_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)
        num_assign_intervals = len(assign_intervals)
        num_links = len(self.observed_links_driving)

        car_link_out_cc = dict()
        for link_ID in self.observed_links_driving:
            car_link_out_cc[link_ID] = dta.get_car_link_out_cc(link_ID)

        # tt_e = dta.get_car_link_tt(np.arange(assign_intervals[-1] + self.ass_freq))
        # assert(tt_e.shape[0] == num_links and tt_e.shape[1] == assign_intervals[-1] + self.ass_freq)
        # # average link travel time
        # tt_e = np.stack(list(map(lambda i : np.mean(tt_e[:, i : i+self.ass_freq], axis=1), assign_intervals)), axis=1)
        # assert(tt_e.shape[0] == num_links and tt_e.shape[1] == num_assign_intervals)

        # tt_e = dta.get_car_link_tt(np.arange(0, self.num_loading_interval))

        # tt_e = dta.get_car_link_tt(assign_intervals, False)
        tt_e = dta.get_car_link_tt_robust(assign_intervals, assign_intervals + self.ass_freq, self.ass_freq, False)

        # tt_free = np.array(list(map(lambda x: self.nb.get_link_driving(x).get_car_fft(), self.observed_links_driving)))
        tt_free = dta.get_car_link_fftt(self.observed_links_driving)
        mask = tt_e > tt_free[:, np.newaxis]

        # no congestions
        if np.sum(mask) == 0:
            return csr_matrix((num_assign_intervals * num_links, 
                               num_assign_intervals * num_links))
            # return eye(num_assign_intervals * num_links)

        # cc = np.zeros((num_links, num_assign_intervals + 1), dtype=float)
        # for j, link_ID in enumerate(self.observed_links_driving):
        #     cc[j, :] = np.array(list(map(lambda timestamp : self._get_flow_from_cc(timestamp, car_link_out_cc[link_ID]), 
        #                                  np.concatenate((assign_intervals, np.array([assign_intervals[-1] + self.ass_freq]))))))

        # outflow_rate = np.diff(cc, axis=1) / self.ass_freq / self.nb.config.config_dict['DTA']['unit_time']

        cc = np.zeros((num_links, assign_intervals[-1] + self.ass_freq + 1), dtype=float)
        for j, link_ID in enumerate(self.observed_links_driving):
            cc[j, :] = np.array(list(map(lambda timestamp : self._get_flow_from_cc(timestamp, car_link_out_cc[link_ID]), 
                                         np.arange(assign_intervals[-1] + self.ass_freq + 1))))

        outflow_rate = np.diff(cc, axis=1) / self.nb.config.config_dict['DTA']['unit_time']
        # outflow_rate = outflow_rate[:, assign_intervals]

        if mask.shape[1] == outflow_rate.shape[1]:
            outflow_rate *= mask
        
        if outflow_rate.shape[1] == num_assign_intervals:
            outflow_avg_rate = outflow_rate
        else:
            outflow_avg_rate = np.stack(list(map(lambda i : np.mean(outflow_rate[:, i : i+self.ass_freq], axis=1), assign_intervals)), axis=1)
            if mask.shape[1] == outflow_avg_rate.shape[1]:
                outflow_avg_rate *= mask
        assert(outflow_avg_rate.shape[0] == num_links and outflow_avg_rate.shape[1] == num_assign_intervals)

        val = list()
        row = list()
        col = list()

        for i, assign_interval in enumerate(assign_intervals):
            for j, link_ID in enumerate(self.observed_links_driving):
                _tmp = outflow_avg_rate[j, i]

                if _tmp > 0:
                    _tmp = 1 / _tmp
                else:
                    _tmp = 1

                # if mask[j, i]:
                #     if _tmp > 0:
                #         # in case 1 / c is very big
                #         # _tmp = np.minimum(1 / _tmp, 1)  # seconds
                #         _tmp = 1 / _tmp
                #     else:
                #         _tmp = 1
                # else:
                #     # help retain the sign of - (tt_o - tt_e)
                #     _tmp = 1
                
                val.append(_tmp)
                row.append(j + num_links * i)
                col.append(j + num_links * i)
                        

        grad = coo_matrix((val, (row, col)), 
                           shape=(num_links * num_assign_intervals, num_links * num_assign_intervals)).tocsr()
        
        # grad = grad / (scipy.sparse.linalg.norm(grad) + 1e-7)
        return grad

    def _compute_link_tt_grad_on_link_flow_truck(self, dta):
        assign_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)
        num_assign_intervals = len(assign_intervals)
        num_links = len(self.observed_links_driving)

        truck_link_out_cc = dict()
        for link_ID in self.observed_links_driving:
            truck_link_out_cc[link_ID] = dta.get_truck_link_out_cc(link_ID)

        # tt_e = dta.get_truck_link_tt(np.arange(assign_intervals[-1] + self.ass_freq))
        # assert(tt_e.shape[0] == num_links and tt_e.shape[1] == assign_intervals[-1] + self.ass_freq)
        # # average link travel time
        # tt_e = np.stack(list(map(lambda i : np.mean(tt_e[:, i : i+self.ass_freq], axis=1), assign_intervals)), axis=1)
        # assert(tt_e.shape[0] == num_links and tt_e.shape[1] == num_assign_intervals)

        # tt_e = dta.get_truck_link_tt(assign_intervals, False)
        tt_e = dta.get_truck_link_tt_robust(assign_intervals, assign_intervals + self.ass_freq, self.ass_freq, False)

        # tt_free = np.array(list(map(lambda x: self.nb.get_link_driving(x).get_truck_fft(), self.observed_links_driving)))
        tt_free = dta.get_truck_link_fftt(self.observed_links_driving)
        mask = tt_e > tt_free[:, np.newaxis]

        # no congestions
        if np.sum(mask) == 0:
            return csr_matrix((num_assign_intervals * num_links, 
                               num_assign_intervals * num_links))
            # return eye(num_assign_intervals * num_links)

        # cc = np.zeros((num_links, num_assign_intervals + 1), dtype=float)
        # for j, link_ID in enumerate(self.observed_links_driving):
        #     cc[j, :] = np.array(list(map(lambda timestamp : self._get_flow_from_cc(timestamp, truck_link_out_cc[link_ID]), 
        #                                  np.concatenate((assign_intervals, np.array([assign_intervals[-1] + self.ass_freq]))))))

        # outflow_rate = np.diff(cc, axis=1) / self.ass_freq / self.nb.config.config_dict['DTA']['unit_time']

        cc = np.zeros((num_links, assign_intervals[-1] + self.ass_freq + 1), dtype=float)
        for j, link_ID in enumerate(self.observed_links_driving):
            cc[j, :] = np.array(list(map(lambda timestamp : self._get_flow_from_cc(timestamp, truck_link_out_cc[link_ID]), 
                                         np.arange(assign_intervals[-1] + self.ass_freq + 1))))

        outflow_rate = np.diff(cc, axis=1) / self.nb.config.config_dict['DTA']['unit_time']
        # outflow_rate = outflow_rate[:, assign_intervals]
        
        if mask.shape[1] == outflow_rate.shape[1]:
            outflow_rate *= mask
        
        if outflow_rate.shape[1] == num_assign_intervals:
            outflow_avg_rate = outflow_rate
        else:
            outflow_avg_rate = np.stack(list(map(lambda i : np.mean(outflow_rate[:, i : i+self.ass_freq], axis=1), assign_intervals)), axis=1)
            if mask.shape[1] == outflow_avg_rate.shape[1]:
                outflow_avg_rate *= mask
        assert(outflow_avg_rate.shape[0] == num_links and outflow_avg_rate.shape[1] == num_assign_intervals)

        val = list()
        row = list()
        col = list()

        for i, assign_interval in enumerate(assign_intervals):
            for j, link_ID in enumerate(self.observed_links_driving):
                _tmp = outflow_avg_rate[j, i]

                if _tmp > 0:
                    _tmp = 1 / _tmp
                else:
                    _tmp = 1

                # if mask[j, i]:
                #     if _tmp > 0:
                #         # in case 1 / c is very big
                #         # _tmp = np.minimum(1 / _tmp, 1)  # seconds
                #         _tmp = 1 / _tmp
                #     else:
                #         _tmp = 1
                # else:
                #     # help retain the sign of - (tt_o - tt_e)
                #     _tmp = 1
                
                val.append(_tmp)
                row.append(j + num_links * i)
                col.append(j + num_links * i)

        grad = coo_matrix((val, (row, col)), 
                           shape=(num_links * num_assign_intervals, num_links * num_assign_intervals)).tocsr()
        
        # grad = grad / (scipy.sparse.linalg.norm(grad) + 1e-7)
        return grad

    def _compute_link_tt_grad_on_link_flow_passenger(self, dta):
        assign_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)
        num_assign_intervals = len(assign_intervals)
        num_links = len(self.observed_links_bus) + len(self.observed_links_walking)

        passenger_count = dta.get_link_bus_passenger_inflow(assign_intervals, assign_intervals + self.ass_freq)
        bus_count = dta.get_link_bus_inflow(assign_intervals, assign_intervals + self.ass_freq)
        assert(num_links == passenger_count.shape[0] == bus_count.shape[0])
        mask = bus_count * self.nb.config.config_dict['DTA']['bus_capacity'] > passenger_count

        mask = np.concatenate((mask, np.zeros((len(self.observed_links_walking), num_assign_intervals), dtype=bool)), axis=0)
        assert(mask.shape[0] == num_links)

        # no congestions
        if np.sum(mask) == 0:
            return csr_matrix((num_assign_intervals * num_links, 
                               num_assign_intervals * num_links))
            # return eye(num_assign_intervals * num_links)
        
        val = list()
        row = list()
        col = list()

        for i, assign_interval in enumerate(assign_intervals):
            for j, link_ID in enumerate(np.concatenate((self.observed_links_bus, self.observed_links_walking))):
                _tmp = mask[j, i]
                if _tmp > 0:
                    _tmp = self.nb.config.config_dict['DTA']['boarding_time_per_passenger']  # seconds
                else:
                    # help retain the sign of - (tt_o - tt_e)
                    _tmp = 1
                val.append(_tmp)
                row.append(j + num_links * i)
                col.append(j + num_links * i)


        grad = coo_matrix((val, (row, col)), 
                           shape=(num_links * num_assign_intervals, num_links * num_assign_intervals)).tocsr()
        
        # grad = grad / (scipy.sparse.linalg.norm(grad) + 1e-7)
        return grad

    def _compute_link_tt_grad_on_link_flow_car_for_bus(self, dta):
        assign_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)
        num_assign_intervals = len(assign_intervals)
        num_links = len(self.observed_links_bus_driving)

        car_link_out_cc = dict()
        for link_ID in self.observed_links_bus_driving:
            car_link_out_cc[link_ID] = dta.get_car_link_out_cc(link_ID)

        # tt_e = dta.get_bus_driving_link_tt_car(assign_intervals, False)
        tt_e = dta.get_bus_driving_link_tt_car_robust(assign_intervals, assign_intervals + self.ass_freq, self.ass_freq, False)

        # tt_free = np.array(list(map(lambda x: self.nb.get_link_driving(x).get_car_fft(), self.observed_links_bus_driving)))
        tt_free = dta.get_car_link_fftt(self.observed_links_bus_driving)
        mask = tt_e > tt_free[:, np.newaxis]

        # no congestions
        if np.sum(mask) == 0:
            return csr_matrix((num_assign_intervals * num_links, 
                               num_assign_intervals * num_links))
            # return eye(num_assign_intervals * num_links)

        # cc = np.zeros((num_links, num_assign_intervals + 1), dtype=float)
        # for j, link_ID in enumerate(self.observed_links_bus_driving):
        #     cc[j, :] = np.array(list(map(lambda timestamp : self._get_flow_from_cc(timestamp, car_link_out_cc[link_ID]), 
        #                                  np.concatenate((assign_intervals, np.array([assign_intervals[-1] + self.ass_freq]))))))

        # outflow_rate = np.diff(cc, axis=1) / self.ass_freq / self.nb.config.config_dict['DTA']['unit_time']

        cc = np.zeros((num_links, assign_intervals[-1] + self.ass_freq + 1), dtype=float)
        for j, link_ID in enumerate(self.observed_links_bus_driving):
            cc[j, :] = np.array(list(map(lambda timestamp : self._get_flow_from_cc(timestamp, car_link_out_cc[link_ID]), 
                                         np.arange(assign_intervals[-1] + self.ass_freq + 1))))

        outflow_rate = np.diff(cc, axis=1) / self.nb.config.config_dict['DTA']['unit_time']
        # outflow_rate = outflow_rate[:, assign_intervals]

        if mask.shape[1] == outflow_rate.shape[1]:
            outflow_rate *= mask
        
        if outflow_rate.shape[1] == num_assign_intervals:
            outflow_avg_rate = outflow_rate
        else:
            outflow_avg_rate = np.stack(list(map(lambda i : np.mean(outflow_rate[:, i : i+self.ass_freq], axis=1), assign_intervals)), axis=1)
            if mask.shape[1] == outflow_avg_rate.shape[1]:
                outflow_avg_rate *= mask
        assert(outflow_avg_rate.shape[0] == num_links and outflow_avg_rate.shape[1] == num_assign_intervals)

        val = list()
        row = list()
        col = list()

        for i, assign_interval in enumerate(assign_intervals):
            for j, link_ID in enumerate(self.observed_links_driving):
                _tmp = outflow_avg_rate[j, i]

                if _tmp > 0:
                    _tmp = 1 / _tmp
                else:
                    _tmp = 1

                # if mask[j, i]:
                #     if _tmp > 0:
                #         # in case 1 / c is very big
                #         # _tmp = np.minimum(1 / _tmp, 1)  # seconds
                #         _tmp = 1 / _tmp
                #     else:
                #         _tmp = 1
                # else:
                #     # help retain the sign of - (tt_o - tt_e)
                #     _tmp = 1
                
                val.append(_tmp)
                row.append(j + num_links * i)
                col.append(j + num_links * i)


        grad = coo_matrix((val, (row, col)), 
                           shape=(num_links * num_assign_intervals, num_links * num_assign_intervals)).tocsr()
        
        # grad = grad / (scipy.sparse.linalg.norm(grad) + 1e-7)
        return grad

    def _compute_link_tt_grad_on_link_flow_truck_for_bus(self, dta):
        assign_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)
        num_assign_intervals = len(assign_intervals)
        num_links = len(self.observed_links_bus_driving)

        truck_link_out_cc = dict()
        for link_ID in self.observed_links_bus_driving:
            truck_link_out_cc[link_ID] = dta.get_truck_link_out_cc(link_ID)

        # tt_e = dta.get_bus_driving_link_tt_truck(assign_intervals, False)
        tt_e = dta.get_bus_driving_link_tt_truck_robust(assign_intervals, assign_intervals + self.ass_freq, self.ass_freq, False)

        # tt_free = np.array(list(map(lambda x: self.nb.get_link_driving(x).get_truck_fft(), self.observed_links_bus_driving)))
        tt_free = dta.get_truck_link_fftt(self.observed_links_bus_driving)
        mask = tt_e > tt_free[:, np.newaxis]

        # no congestions
        if np.sum(mask) == 0:
            return csr_matrix((num_assign_intervals * num_links, 
                               num_assign_intervals * num_links))
            # return eye(num_assign_intervals * num_links)

        # cc = np.zeros((num_links, num_assign_intervals + 1), dtype=float)
        # for j, link_ID in enumerate(self.observed_links_bus_driving):
        #     cc[j, :] = np.array(list(map(lambda timestamp : self._get_flow_from_cc(timestamp, truck_link_out_cc[link_ID]), 
        #                                  np.concatenate((assign_intervals, np.array([assign_intervals[-1] + self.ass_freq]))))))

        # outflow_rate = np.diff(cc, axis=1) / self.ass_freq / self.nb.config.config_dict['DTA']['unit_time']

        cc = np.zeros((num_links, assign_intervals[-1] + self.ass_freq + 1), dtype=float)
        for j, link_ID in enumerate(self.observed_links_bus_driving):
            cc[j, :] = np.array(list(map(lambda timestamp : self._get_flow_from_cc(timestamp, truck_link_out_cc[link_ID]), 
                                         np.arange(assign_intervals[-1] + self.ass_freq + 1))))

        outflow_rate = np.diff(cc, axis=1) / self.nb.config.config_dict['DTA']['unit_time']
        # outflow_rate = outflow_rate[:, assign_intervals]
        
        if mask.shape[1] == outflow_rate.shape[1]:
            outflow_rate *= mask
        
        if outflow_rate.shape[1] == num_assign_intervals:
            outflow_avg_rate = outflow_rate
        else:
            outflow_avg_rate = np.stack(list(map(lambda i : np.mean(outflow_rate[:, i : i+self.ass_freq], axis=1), assign_intervals)), axis=1)
            if mask.shape[1] == outflow_avg_rate.shape[1]:
                outflow_avg_rate *= mask
        assert(outflow_avg_rate.shape[0] == num_links and outflow_avg_rate.shape[1] == num_assign_intervals)

        val = list()
        row = list()
        col = list()

        for i, assign_interval in enumerate(assign_intervals):
            for j, link_ID in enumerate(self.observed_links_driving):
                _tmp = outflow_avg_rate[j, i]

                if _tmp > 0:
                    _tmp = 1 / _tmp
                else:
                    _tmp = 1

                # if mask[j, i]:
                #     if _tmp > 0:
                #         # in case 1 / c is very big
                #         # _tmp = np.minimum(1 / _tmp, 1)  # seconds
                #         _tmp = 1 / _tmp
                #     else:
                #         _tmp = 1
                # else:
                #     # help retain the sign of - (tt_o - tt_e)
                #     _tmp = 1
                
                val.append(_tmp)
                row.append(j + num_links * i)
                col.append(j + num_links * i)


        grad = coo_matrix((val, (row, col)), 
                           shape=(num_links * num_assign_intervals, num_links * num_assign_intervals)).tocsr()
        
        # grad = grad / (scipy.sparse.linalg.norm(grad) + 1e-7)
        return grad

    def _compute_link_tt_grad_on_link_flow_passenger_for_bus(self, dta):
        assign_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)
        num_assign_intervals = len(assign_intervals)
        num_links = len(self.observed_links_bus)

        passenger_count = dta.get_link_bus_passenger_inflow(assign_intervals, assign_intervals + self.ass_freq)
        bus_count = dta.get_link_bus_inflow(assign_intervals, assign_intervals + self.ass_freq)
        assert(num_links == passenger_count.shape[0] == bus_count.shape[0])
        mask = bus_count * self.nb.config.config_dict['DTA']['bus_capacity'] > passenger_count

        # no congestions
        if np.sum(mask) == 0:
            return csr_matrix((num_assign_intervals * num_links, 
                               num_assign_intervals * num_links))
            # return eye(num_assign_intervals * num_links)

        val = list()
        row = list()
        col = list()

        for i, assign_interval in enumerate(assign_intervals):
            for j, link_ID in enumerate(self.observed_links_bus):
                _tmp = mask[j, i]
                if _tmp:
                    _tmp = self.nb.config.config_dict['DTA']['boarding_time_per_passenger']
                else:
                    # help retain the sign of - (tt_o - tt_e)
                    _tmp = 1

                val.append(_tmp)
                row.append(j + num_links * i)
                col.append(j + num_links * i)


        grad = coo_matrix((val, (row, col)), 
                           shape=(num_links * num_assign_intervals, num_links * num_assign_intervals)).tocsr()
        
        # grad = grad / (scipy.sparse.linalg.norm(grad) + 1e-7)
        return grad

    def _compute_link_tt_grad_on_path_flow_car_driving(self, dta):
        # dta.build_link_cost_map() is invoked already
        assign_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)
        num_assign_intervals = len(assign_intervals)

        release_freq = 60
        # this is in terms of 5-s intervals
        release_intervals = np.arange(0, self.num_loading_interval, release_freq // self.nb.config.config_dict['DTA']['unit_time'])

        raw_ltg = dta.get_car_ltg_matrix_driving(release_intervals, self.num_loading_interval)

        ltg = self._massage_raw_ltg(raw_ltg, self.ass_freq, num_assign_intervals, self.paths_list_driving, self.observed_links_driving, self.num_procs)
        return ltg

    def _compute_link_tt_grad_on_path_flow_car_pnr(self, dta):
        # dta.build_link_cost_map() is invoked already
        assign_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)
        num_assign_intervals = len(assign_intervals)

        release_freq = 60
        release_intervals = np.arange(0, self.num_loading_interval, release_freq // self.nb.config.config_dict['DTA']['unit_time'])

        raw_ltg = dta.get_car_ltg_matrix_pnr(release_intervals, self.num_loading_interval)

        ltg = self._massage_raw_ltg(raw_ltg, self.ass_freq, num_assign_intervals, self.paths_list_pnr, self.observed_links_driving, self.num_procs)
        return ltg

    def _compute_link_tt_grad_on_path_flow_truck_driving(self, dta):
        # dta.build_link_cost_map() is invoked already
        assign_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)
        num_assign_intervals = len(assign_intervals)

        release_freq = 60
        release_intervals = np.arange(0, self.num_loading_interval, release_freq // self.nb.config.config_dict['DTA']['unit_time'])

        raw_ltg = dta.get_truck_ltg_matrix_driving(release_intervals, self.num_loading_interval)

        ltg = self._massage_raw_ltg(raw_ltg, self.ass_freq, num_assign_intervals, self.paths_list_driving, self.observed_links_driving, self.num_procs)
        return ltg

    def _compute_link_tt_grad_on_path_flow_passenger_bustransit(self, dta):
        # dta.build_link_cost_map() is invoked already
        assign_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)
        num_assign_intervals = len(assign_intervals)

        release_freq = 60
        release_intervals = np.arange(0, self.num_loading_interval, release_freq // self.nb.config.config_dict['DTA']['unit_time'])

        raw_ltg = dta.get_passenger_ltg_matrix_bustransit(release_intervals, self.num_loading_interval)

        ltg = self._massage_raw_ltg(raw_ltg, self.ass_freq, num_assign_intervals, self.paths_list_bustransit, 
                                    np.concatenate((self.observed_links_bus, self.observed_links_walking)), self.num_procs)
        return ltg
    
    def _compute_link_tt_grad_on_path_flow_passenger_pnr(self, dta):
        # dta.build_link_cost_map() is invoked already
        assign_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)
        num_assign_intervals = len(assign_intervals)

        release_freq = 60
        release_intervals = np.arange(0, self.num_loading_interval, release_freq // self.nb.config.config_dict['DTA']['unit_time'])

        raw_ltg = dta.get_passenger_ltg_matrix_pnr(release_intervals, self.num_loading_interval)

        ltg = self._massage_raw_ltg(raw_ltg, self.ass_freq, num_assign_intervals, self.paths_list_pnr,
                                    np.concatenate((self.observed_links_bus, self.observed_links_walking)), self.num_procs)
        return ltg

    def _massage_raw_ltg(self, raw_ltg, ass_freq, num_assign_interval, paths_list, observed_links, num_procs=5):
        assert(raw_ltg.shape[1] == 5)
        if raw_ltg.shape[0] == 0:
            print("No ltg. No congestion.")
            return csr_matrix((num_assign_interval * len(observed_links), 
                               num_assign_interval * len(paths_list)))

        num_e_path = len(paths_list)
        num_e_link = len(observed_links)
        # 15 min
        small_assign_freq = ass_freq * self.nb.config.config_dict['DTA']['unit_time'] / 60

        raw_ltg = raw_ltg[(raw_ltg[:, 1] < self.num_loading_interval) & (raw_ltg[:, 3] < self.num_loading_interval), :]

        if num_procs > 1:
            raw_ltg = pd.DataFrame(data=raw_ltg)

            from pandarallel import pandarallel
            pandarallel.initialize(progress_bar=True, nb_workers=num_procs)

            ind = raw_ltg.loc[:, 0].parallel_apply(lambda x: True if x in set(paths_list) else False)
            assert(np.sum(ind) == len(ind))
            
            if type(paths_list) == list:
                paths_list = np.array(paths_list)
            elif type(paths_list) == np.ndarray:
                pass
            else:
                raise Exception('Wrong data type of paths_list')

            path_seq = (raw_ltg.loc[ind, 0].astype(int).parallel_apply(lambda x: np.where(paths_list == x)[0][0]) 
                        + (raw_ltg.loc[ind, 1] / ass_freq).astype(int) * num_e_path).astype(int)
            
            if type(observed_links) == list:
                observed_links = np.array(observed_links)
            elif type(observed_links) == np.ndarray:
                pass
            else:
                raise Exception('Wrong data type of observed_links')

            link_seq = (raw_ltg.loc[ind, 2].astype(int).parallel_apply(lambda x: np.where(observed_links == x)[0][0])
                        + (raw_ltg.loc[ind, 3] / ass_freq).astype(int) * num_e_link).astype(int)

            p = raw_ltg.loc[ind, 4] / (ass_freq * small_assign_freq)

        else:

            # raw_ltg[:, 0]: path no.
            # raw_ltg[:, 1]: the count of 1 min interval in terms of 5s intervals
                    
            if type(paths_list) == np.ndarray:
                ind = np.array(list(map(lambda x: True if len(np.where(paths_list == x)[0]) > 0 else False, raw_ltg[:, 0].astype(int)))).astype(bool)
                assert(np.sum(ind) == len(ind))
                path_seq = (np.array(list(map(lambda x: np.where(paths_list == x)[0][0], raw_ltg[ind, 0].astype(int))))
                            + (raw_ltg[ind, 1] / ass_freq).astype(int) * num_e_path).astype(int)
            elif type(paths_list) == list:
                ind = np.array(list(map(lambda x: True if x in paths_list else False, raw_ltg[:, 0].astype(int)))).astype(bool)
                assert(np.sum(ind) == len(ind))
                path_seq = (np.array(list(map(lambda x: paths_list.index(x), raw_ltg[ind, 0].astype(int))))
                            + (raw_ltg[ind, 1] / ass_freq).astype(int) * num_e_path).astype(int)
            else:
                raise Exception('Wrong data type of paths_list')

            # raw_ltg[:, 2]: link no.
            # raw_ltg[:, 3]: the count of unit time interval (5s)
            if type(observed_links) == np.ndarray:
                # In Python 3, map() returns an iterable while, in Python 2, it returns a list.
                link_seq = (np.array(list(map(lambda x: np.where(observed_links == x)[0][0], raw_ltg[ind, 2].astype(int))))
                            + (raw_ltg[ind, 3] / ass_freq).astype(int) * num_e_link).astype(int)
            elif type(observed_links) == list:
                link_seq = (np.array(list(map(lambda x: observed_links.index(x), raw_ltg[ind, 2].astype(int))))
                            + (raw_ltg[ind, 3] / ass_freq).astype(int) * num_e_link).astype(int)
            else:
                raise Exception('Wrong data type of observed_links')
                        
            # print(path_seq)
            # raw_ltg[:, 4]: gradient, to be averaged for each large assign interval 
            p = raw_ltg[ind, 4] / (ass_freq * small_assign_freq)
        
        # print("Creating the coo matrix", time.time()), coo_matrix permits duplicate entries
        mat = coo_matrix((p, (link_seq, path_seq)), shape=(num_assign_interval * num_e_link, num_assign_interval * num_e_path))
        # pickle.dump((p, link_seq, path_seq), open('test.pickle', 'wb'))
        # print('converting the csr', time.time())
        
        # sum duplicate entries in coo_matrix
        mat = mat.tocsr()
        # print('finish converting', time.time())
        return mat

    def get_ltg(self, dta):
        
        car_ltg_matrix_driving = csr_matrix((self.num_assign_interval * len(self.observed_links_driving), 
                                             self.num_assign_interval * len(self.paths_list_driving)))
        car_ltg_matrix_pnr = csr_matrix((self.num_assign_interval * len(self.observed_links_driving), 
                                         self.num_assign_interval * len(self.paths_list_pnr)))
        truck_ltg_matrix_driving = csr_matrix((self.num_assign_interval * len(self.observed_links_driving), 
                                               self.num_assign_interval * len(self.paths_list_driving)))
        bus_ltg_matrix_transit_link = csr_matrix((self.num_assign_interval * len(self.observed_links_bus), 
                                                  self.num_assign_interval * len(self.paths_list_busroute)))
        bus_ltg_matrix_driving_link = csr_matrix((self.num_assign_interval * len(self.observed_links_driving), 
                                                  self.num_assign_interval * len(self.paths_list_busroute)))
        passenger_ltg_matrix_bustransit = csr_matrix((self.num_assign_interval * (len(self.observed_links_bus) + len(self.observed_links_walking)), 
                                                      self.num_assign_interval * len(self.paths_list_bustransit)))
        passenger_ltg_matrix_pnr = csr_matrix((self.num_assign_interval * (len(self.observed_links_bus) + len(self.observed_links_walking)), 
                                               self.num_assign_interval * len(self.paths_list_pnr)))
        
        if self.config['use_car_link_tt']:
            car_ltg_matrix_driving = self._compute_link_tt_grad_on_path_flow_car_driving(dta)
            if car_ltg_matrix_driving.max() == 0.:
                print("car_ltg_matrix_driving is empty!")

            # TODO:
            # car_ltg_matrix_pnr = self._compute_link_tt_grad_on_path_flow_car_pnr(dta)
            # if car_ltg_matrix_pnr.max() == 0.:
            #     print("car_ltg_matrix_pnr is empty!")
            
        if self.config['use_truck_link_tt']:
            truck_ltg_matrix_driving = self._compute_link_tt_grad_on_path_flow_truck_driving(dta)
            if truck_ltg_matrix_driving.max() == 0.:
                print("truck_ltg_matrix_driving is empty!")

        # TODO:
        # if self.config['use_passenger_link_tt']:
        #     passenger_ltg_matrix_bustransit = self._compute_link_tt_grad_on_path_flow_passenger_bustransit(dta)
        #     if passenger_ltg_matrix_bustransit.max() == 0.:
        #         print("passenger_ltg_matrix_bustransit is empty!")

        #     passenger_ltg_matrix_pnr = self._compute_link_tt_grad_on_path_flow_passenger_pnr(dta)
        #     if passenger_ltg_matrix_pnr.max() == 0.:
        #         print("passenger_ltg_matrix_pnr is empty!")
            
        return car_ltg_matrix_driving, truck_ltg_matrix_driving, car_ltg_matrix_pnr, bus_ltg_matrix_transit_link, bus_ltg_matrix_driving_link, \
               passenger_ltg_matrix_bustransit, passenger_ltg_matrix_pnr

    def _get_flow_from_cc(self, timestamp, cc):
        # precision issue, consistent with C++
        cc = np.around(cc, decimals=4)
        if any(timestamp >= cc[:, 0]):
            ind = np.nonzero(timestamp >= cc[:, 0])[0][-1]
        else:
            ind = 0
        return cc[ind, 1]

    def _get_timestamp_from_cc(self, flow, cc):
        # precision issue, consistent with C++
        cc = np.around(cc, decimals=4)
        if any(flow == cc[:, 1]):
            ind = np.nonzero(flow == cc[:, 1])[0][0]
        elif any(flow > cc[:, 1]):
            ind = np.nonzero(flow > cc[:, 1])[0][-1]
        else:
            ind = 0
        return cc[ind, 0]

    def _get_link_tt_from_cc(self, timestamp, cc_in, cc_out, fft):
        _flow = self._get_flow_from_cc(timestamp, cc_in)
        _timestamp_1 = self._get_timestamp_from_cc(_flow, cc_in)
        _timestamp_2 = self._get_timestamp_from_cc(_flow, cc_out)
        tt = (_timestamp_2 - _timestamp_1) * self.nb.config.config_dict['DTA']['unit_time']
        assert(tt >= 0)
        if tt < fft:
            tt = fft
        return tt

    def _get_one_data(self, j):
        assert (self.num_data > j)
        one_data_dict = dict()
        if self.config['use_car_link_flow'] or self.config['compute_car_link_flow_loss']:
            one_data_dict['car_link_flow'] = self.data_dict['car_link_flow'][j]
        if self.config['use_truck_link_flow'] or self.config['compute_truck_link_flow_loss']:
            one_data_dict['truck_link_flow'] = self.data_dict['truck_link_flow'][j]
        if self.config['use_bus_link_flow']or self.config['compute_bus_link_flow_loss']:
            one_data_dict['bus_link_flow'] = self.data_dict['bus_link_flow'][j]
        if self.config['use_passenger_link_flow']or self.config['compute_passenger_link_flow_loss']:
            one_data_dict['passenger_link_flow'] = self.data_dict['passenger_link_flow'][j]

        if self.config['use_car_link_tt'] or self.config['compute_car_link_tt_loss']:
            one_data_dict['car_link_tt'] = self.data_dict['car_link_tt'][j]
        if self.config['use_truck_link_tt'] or self.config['compute_truck_link_tt_loss']:
            one_data_dict['truck_link_tt'] = self.data_dict['truck_link_tt'][j]
        if self.config['use_bus_link_tt'] or self.config['compute_bus_link_tt_loss']:
            one_data_dict['bus_link_tt'] = self.data_dict['bus_link_tt'][j]
        if self.config['use_passenger_link_tt'] or self.config['compute_passenger_link_tt_loss']:
            one_data_dict['passenger_link_tt'] = self.data_dict['passenger_link_tt'][j]

        if self.config['car_count_agg']:
            one_data_dict['car_count_agg_L'] = self.car_count_agg_L_list[j]
        if self.config['truck_count_agg']:
            one_data_dict['truck_count_agg_L'] = self.truck_count_agg_L_list[j]
        if self.config['bus_count_agg']:
            one_data_dict['bus_count_agg_L'] = self.bus_count_agg_L_list[j]
        if self.config['passenger_count_agg']:
            one_data_dict['passenger_count_agg_L'] = self.passenger_count_agg_L_list[j]

        if self.config['use_veh_run_boarding_alighting'] or self.config['use_ULP_f_transit']:
            one_data_dict['veh_run_boarding_alighting_record'] = self.data_dict['veh_run_boarding_alighting_record'][j]
            if 'stop_arrival_departure_travel_time' in self.data_dict:
                one_data_dict['stop_arrival_departure_travel_time'] = self.data_dict['stop_arrival_departure_travel_time'][j]

        if 'mask_driving_link' in self.data_dict:
            one_data_dict['mask_driving_link'] = self.data_dict['mask_driving_link']
        else:
            one_data_dict['mask_driving_link'] = np.ones(len(self.observed_links_driving) * self.num_assign_interval, dtype=bool)

        if 'mask_bus_link' in self.data_dict:
            one_data_dict['mask_bus_link'] = self.data_dict['mask_bus_link']
        else:
            one_data_dict['mask_bus_link'] = np.ones(len(self.observed_links_bus) * self.num_assign_interval, dtype=bool)

        if 'mask_walking_link' in self.data_dict:
            one_data_dict['mask_walking_link'] = self.data_dict['mask_walking_link']
        else:
            one_data_dict['mask_walking_link'] = np.ones(len(self.observed_links_walking) * self.num_assign_interval, dtype=bool)

        if 'mask_observed_stops_vehs_record' in self.data_dict:
            one_data_dict['mask_observed_stops_vehs_record'] = self.data_dict['mask_observed_stops_vehs_record']
        else:
            one_data_dict['mask_observed_stops_vehs_record'] = np.ones(len(self.observed_stops_vehs_list), dtype=bool)

        return one_data_dict

    def _get_loss(self, one_data_dict, dta):
        start_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)
        end_intervals = start_intervals + self.ass_freq

        loss_dict = dict()

        # flow loss

        if self.config['use_car_link_flow'] or self.config['compute_car_link_flow_loss']:
            # num_links_driving x num_assign_intervals
            x_e = dta.get_link_car_inflow(start_intervals, end_intervals).flatten(order='F')
            assert(len(x_e) == len(self.observed_links_driving) * self.num_assign_interval)
            if self.config['car_count_agg']:
                x_e = one_data_dict['car_count_agg_L'].dot(x_e)

            # loss = self.config['link_car_flow_weight'] * np.linalg.norm(np.nan_to_num(x_e[one_data_dict['mask_driving_link']] - one_data_dict['car_link_flow'][one_data_dict['mask_driving_link']]))
            loss = np.linalg.norm(np.nan_to_num(x_e[one_data_dict['mask_driving_link']] - one_data_dict['car_link_flow'][one_data_dict['mask_driving_link']]))
            loss_dict['car_count_loss'] = loss

        if self.config['use_truck_link_flow'] or self.config['compute_truck_link_flow_loss']:
            # num_links_driving x num_assign_intervals
            x_e = dta.get_link_truck_inflow(start_intervals, end_intervals).flatten(order='F')
            assert(len(x_e) == len(self.observed_links_driving) * self.num_assign_interval)
            if self.config['truck_count_agg']:
                x_e = one_data_dict['truck_count_agg_L'].dot(x_e)
            # loss = self.config['link_truck_flow_weight'] * np.linalg.norm(np.nan_to_num(x_e[one_data_dict['mask_driving_link']] - one_data_dict['truck_link_flow'][one_data_dict['mask_driving_link']]))
            loss = np.linalg.norm(np.nan_to_num(x_e[one_data_dict['mask_driving_link']] - one_data_dict['truck_link_flow'][one_data_dict['mask_driving_link']]))
            loss_dict['truck_count_loss'] = loss

        if self.config['use_bus_link_flow'] or self.config['compute_bus_link_flow_loss']:
            # num_links_bus x num_assign_intervals
            x_e = dta.get_link_bus_inflow(start_intervals, end_intervals).flatten(order='F')
            assert(len(x_e) == len(self.observed_links_bus) * self.num_assign_interval)
            if self.config['bus_count_agg']:
                x_e = one_data_dict['bus_count_agg_L'].dot(x_e)
            # loss = self.config['link_bus_flow_weight'] * np.linalg.norm(np.nan_to_num(x_e[one_data_dict['mask_bus_link']] - one_data_dict['bus_link_flow'][one_data_dict['mask_bus_link']]))
            loss = np.linalg.norm(np.nan_to_num(x_e[one_data_dict['mask_bus_link']] - one_data_dict['bus_link_flow'][one_data_dict['mask_bus_link']]))
            loss_dict['bus_count_loss'] = loss

        if self.config['use_passenger_link_flow'] or self.config['compute_passenger_link_flow_loss']:
            # (num_links_bus + num_links_walking) x num_assign_intervals
            x_e_bus_passenger = dta.get_link_bus_passenger_inflow(start_intervals, end_intervals)
            assert(x_e_bus_passenger.shape[0] == len(self.observed_links_bus))
            x_e_walking_passenger = dta.get_link_walking_passenger_inflow(start_intervals, end_intervals)
            assert(x_e_walking_passenger.shape[0] == len(self.observed_links_walking))
            x_e = np.concatenate((x_e_bus_passenger, x_e_walking_passenger), axis=0).flatten(order='F')
            if self.config['passenger_count_agg']:
                x_e = one_data_dict['passenger_count_agg_L'].dot(x_e)

            if len(one_data_dict['mask_walking_link']) > 0:
                mask_passenger = np.concatenate(
                    (one_data_dict['mask_bus_link'].reshape(-1, len(self.observed_links_bus)), 
                    one_data_dict['mask_walking_link'].reshape(-1, len(self.observed_links_walking))), 
                    axis=1
                )
                mask_passenger = mask_passenger.flatten()
            else:
                mask_passenger = one_data_dict['mask_bus_link']

            # loss = self.config['link_passenger_flow_weight'] * np.linalg.norm(np.nan_to_num(x_e[mask_passenger] - one_data_dict['passenger_link_flow'][mask_passenger]))
            loss = np.linalg.norm(np.nan_to_num(x_e[mask_passenger] - one_data_dict['passenger_link_flow'][mask_passenger]))
            loss_dict['passenger_count_loss'] = loss

        if self.config['use_veh_run_boarding_alighting'] or self.config['compute_veh_run_boarding_alighting_loss']: 
            observed_x_e_BoardingAlighting_count = one_data_dict['veh_run_boarding_alighting_record'][1]
            raw_boarding_alighting_record = dta.get_bus_boarding_alighting_record()
            simulated_boarding_alighting_record_for_observed = self._massage_boarding_alighting_record(raw_boarding_alighting_record)
            x_e_BoardingAlighting_count = simulated_boarding_alighting_record_for_observed[1]
            assert(len(x_e_BoardingAlighting_count) == len(observed_x_e_BoardingAlighting_count))
            mask_record = one_data_dict['mask_observed_stops_vehs_record']
            loss = np.linalg.norm(np.nan_to_num(x_e_BoardingAlighting_count[mask_record] - observed_x_e_BoardingAlighting_count[mask_record]))
            loss_dict['veh_run_boarding_alighting_loss'] = loss



        # if self.config['use_bus_link_passenger_flow'] or self.config['compute_bus_link_passenger_flow_loss']:
        #     # num_links_bus x num_assign_intervals
        #     x_e = dta.get_link_bus_passenger_inflow(start_intervals, end_intervals).flatten(order='F')
        #     if self.config['bus_passenger_count_agg']:
        #         x_e = one_data_dict['bus_passenger_count_agg_L'].dot(x_e)
        # #     loss = self.config['link_bus_passenger_flow_weight'] * np.linalg.norm(np.nan_to_num(x_e - one_data_dict['bus_link_passenger_flow']))
        #     loss = np.linalg.norm(np.nan_to_num(x_e - one_data_dict['bus_link_passenger_flow']))
        #     loss_dict['bus_passenger_count_loss'] = loss

        # if self.config['use_walking_link_passenger_flow'] or self.config['compute_walking_link_passenger_flow_loss']:
        #     # num_links_walking x num_assign_intervals
        #     x_e = dta.get_link_walking_passenger_inflow(start_intervals, end_intervals).flatten(order='F')
        #     if self.config['walking_passenger_count_agg']:
        #         x_e = one_data_dict['walking_passenger_count_agg_L'].dot(x_e)
        # #     loss = self.config['link_walking_passenger_flow_weight'] * np.linalg.norm(np.nan_to_num(x_e - one_data_dict['walking_link_passenger_flow']))
        #     loss = np.linalg.norm(np.nan_to_num(x_e - one_data_dict['walking_link_passenger_flow']))
        #     loss_dict['walking_passenger_count_loss'] = loss

        # travel time loss

        if self.config['use_car_link_tt'] or self.config['compute_car_link_tt_loss']:
            # num_links_driving x num_assign_intervals
            x_tt_e = dta.get_car_link_tt_robust(np.arange(0, self.num_loading_interval, self.ass_freq),
                                                np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq,
                                                self.ass_freq, True).flatten(order='F')
            # x_tt_e = dta.get_car_link_tt(start_intervals, True).flatten(order='F')
            assert(len(x_tt_e) == len(self.observed_links_driving) * self.num_assign_interval)

            # don't use inf value
            x_tt_o = one_data_dict['car_link_tt'].copy()
            ind = np.isinf(x_tt_e) + np.isinf(x_tt_o)
            x_tt_e[ind] = 0
            x_tt_o[ind] = 0

            # loss = self.config['link_car_tt_weight'] * np.linalg.norm(np.nan_to_num(x_tt_e[one_data_dict['mask_driving_link']] - x_tt_o[one_data_dict['mask_driving_link']]))
            loss = np.linalg.norm(np.nan_to_num(x_tt_e[one_data_dict['mask_driving_link']] - x_tt_o[one_data_dict['mask_driving_link']]))
            loss_dict['car_tt_loss'] = loss
            
        if self.config['use_truck_link_tt'] or self.config['compute_truck_link_tt_loss']:
            # num_links_driving x num_assign_intervals
            x_tt_e = dta.get_truck_link_tt_robust(np.arange(0, self.num_loading_interval, self.ass_freq),
                                                  np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq,
                                                  self.ass_freq, True).flatten(order='F')
            # x_tt_e = dta.get_truck_link_tt(start_intervals, True).flatten(order='F')
            assert(len(x_tt_e) == len(self.observed_links_driving) * self.num_assign_interval)

            # don't use inf value
            x_tt_o = one_data_dict['truck_link_tt'].copy()
            ind = np.isinf(x_tt_e) + np.isinf(x_tt_o)
            x_tt_e[ind] = 0
            x_tt_o[ind] = 0

            # loss = self.config['link_truck_tt_weight'] * np.linalg.norm(np.nan_to_num(x_tt_e[one_data_dict['mask_driving_link']] - x_tt_o[one_data_dict['mask_driving_link']]))
            loss = np.linalg.norm(np.nan_to_num(x_tt_e[one_data_dict['mask_driving_link']] - x_tt_o[one_data_dict['mask_driving_link']]))
            loss_dict['truck_tt_loss'] = loss

        if self.config['use_bus_link_tt'] or self.config['compute_bus_link_tt_loss']:
            # num_links_bus x num_assign_intervals
            x_tt_e = dta.get_bus_link_tt_robust(np.arange(0, self.num_loading_interval, self.ass_freq),
                                                np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq,
                                                self.ass_freq, True, True).flatten(order='F')
            # x_tt_e = dta.get_bus_link_tt(start_intervals, True, True).flatten(order='F')
            assert(len(x_tt_e) == len(self.observed_links_bus) * self.num_assign_interval)

            # fill in the inf value
            # x_tt_e[np.isinf(x_tt_e)] = 5 * self.num_loading_interval
            # x_tt_o = np.nan_to_num(one_data_dict['bus_link_tt'], posinf = 5 * self.num_loading_interval)

            # don't use inf value
            x_tt_o = one_data_dict['bus_link_tt'].copy()
            ind = np.isinf(x_tt_e) + np.isinf(x_tt_o)
            x_tt_e[ind] = 0
            x_tt_o[ind] = 0

            # loss = self.config['link_bus_tt_weight'] * np.linalg.norm(np.nan_to_num(x_tt_e[one_data_dict['mask_bus_link']] - x_tt_o[one_data_dict['mask_bus_link']]))
            loss = np.linalg.norm(np.nan_to_num(x_tt_e[one_data_dict['mask_bus_link']] - x_tt_o[one_data_dict['mask_bus_link']]))
            loss_dict['bus_tt_loss'] = loss
            
        if self.config['use_passenger_link_tt'] or self.config['compute_passenger_link_tt_loss']:
            # (num_links_bus + num_links_walking) x num_assign_intervals
            x_tt_e_bus = dta.get_bus_link_tt_robust(np.arange(0, self.num_loading_interval, self.ass_freq),
                                                    np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq,
                                                    self.ass_freq, True, True)
            # x_tt_e_bus = dta.get_bus_link_tt(start_intervals, True, True)
            assert(x_tt_e_bus.shape[0] == len(self.observed_links_bus))
            x_tt_e_walking = dta.get_passenger_walking_link_tt_robust(np.arange(0, self.num_loading_interval, self.ass_freq),
                                                                      np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq,
                                                                      self.ass_freq)
            # x_tt_e_walking = dta.get_passenger_walking_link_tt(start_intervals)
            assert(x_tt_e_walking.shape[0] == len(self.observed_links_walking))
            x_tt_e = np.concatenate((x_tt_e_bus, x_tt_e_walking), axis=0).flatten(order='F')

            # fill in the inf value
            # x_tt_e[np.isinf(x_tt_e)] = 5 * self.num_loading_interval
            # x_tt_o = np.nan_to_num(one_data_dict['passenger_link_tt'], posinf = 5 * self.num_loading_interval)

            # don't use inf value
            x_tt_o = one_data_dict['passenger_link_tt'].copy()
            ind = np.isinf(x_tt_e) + np.isinf(x_tt_o)
            x_tt_e[ind] = 0
            x_tt_o[ind] = 0

            if len(one_data_dict['mask_walking_link']) > 0:
                mask_passenger = np.concatenate(
                    (one_data_dict['mask_bus_link'].reshape(-1, len(self.observed_links_bus)), 
                    one_data_dict['mask_walking_link'].reshape(-1, len(self.observed_links_walking))), 
                    axis=1
                )
                mask_passenger = mask_passenger.flatten()
            else:
                mask_passenger = one_data_dict['mask_bus_link']

            # loss = self.config['link_passenger_tt_weight'] * np.linalg.norm(np.nan_to_num(x_tt_e[mask_passenger] - x_tt_o[mask_passenger]))
            loss = np.linalg.norm(np.nan_to_num(x_tt_e[mask_passenger] - x_tt_o[mask_passenger]))
            loss_dict['passenger_tt_loss'] = loss

        total_loss = 0.0
        for loss_type, loss_value in loss_dict.items():
            total_loss += loss_value
        return total_loss, loss_dict

    def perturbate_path_flow(self, path_flow, epoch, prob=0.5):
        prob = prob * pow(epoch + 1.0, -0.5)
        ind = path_flow < 1 / self.nb.config.config_dict['DTA']['flow_scalar']
        path_flow[ind] = np.random.choice([1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'], 1], size=sum(ind), replace=True, p=[1 - prob, prob])

        return path_flow

    def perturbate_bus_flow(self, bus_flow, one_data_dict, x_e):
        x_o = one_data_dict['bus_link_flow'].reshape(-1, len(self.observed_links_bus)).T
        x_e = x_e.reshape(-1, len(self.observed_links_bus)).T
        bus_flow = bus_flow.reshape(-1, self.num_path_busroute).T
        route_IDs_for_links = [self.nb.get_link_bus(link_ID).route_ID for link_ID in self.observed_links_bus]
        route_IDs_for_paths = [self.nb.path_table_bus.ID2path[path_ID].route_ID for path_ID in self.paths_list_busroute]
        ind = [np.where(np.array(route_IDs_for_paths, dtype=int) == u)[0][0] for u in route_IDs_for_links]
        for i in range(self.num_assign_interval):
            for j in range(len(self.observed_links_bus)):
                if x_o[j, i] > 0 and x_e[j, i] == 0:           
                    bus_flow[ind[j], (bus_flow[ind[j], :] < 1 / self.nb.config.config_dict['DTA']['flow_scalar']) & (i > np.arange(self.num_assign_interval))] = 1
        return bus_flow.flatten(order='F')

    def estimate_path_flow_pytorch(self, car_driving_scale=10, truck_driving_scale=1, passenger_bustransit_scale=1, car_pnr_scale=5, bus_scale=1,
                                   car_driving_step_size=0.1, truck_driving_step_size=0.01, passenger_bustransit_step_size=0.01, car_pnr_step_size=0.05, bus_step_size=0.01,
                                   link_car_flow_weight=1, link_truck_flow_weight=1, link_passenger_flow_weight=1, link_bus_flow_weight=1,
                                   link_car_tt_weight=1, link_truck_tt_weight=1, link_passenger_tt_weight=1, link_bus_tt_weight=1,
                                   max_epoch=100, algo="NAdam", l2_coeff=1e-4, run_mmdta_adaptive=False, fix_bus=True, column_generation=False, use_tdsp=False, use_file_as_init=None, save_folder=None, starting_epoch=0):
        if save_folder is not None and not os.path.exists(save_folder):
            os.mkdir(save_folder)

        if fix_bus:
            bus_scale = None
            bus_step_size = None
        else:
            assert(bus_scale is not None)
            assert(bus_step_size is not None)

        if np.isscalar(column_generation):
            column_generation = np.ones(max_epoch, dtype=int) * column_generation
        assert(len(column_generation) == max_epoch)

        if np.isscalar(link_car_flow_weight):
            link_car_flow_weight = np.ones(max_epoch, dtype=int) * link_car_flow_weight
        assert(len(link_car_flow_weight) == max_epoch)

        if np.isscalar(link_truck_flow_weight):
            link_truck_flow_weight = np.ones(max_epoch, dtype=int) * link_truck_flow_weight
        assert(len(link_truck_flow_weight) == max_epoch)

        if np.isscalar(link_passenger_flow_weight):
            link_passenger_flow_weight = np.ones(max_epoch, dtype=int) * link_passenger_flow_weight
        assert(len(link_passenger_flow_weight) == max_epoch)

        if np.isscalar(link_bus_flow_weight):
            link_bus_flow_weight = np.ones(max_epoch, dtype=int) * link_bus_flow_weight
        assert(len(link_bus_flow_weight) == max_epoch)

        if np.isscalar(link_car_tt_weight):
            link_car_tt_weight = np.ones(max_epoch, dtype=int) * link_car_tt_weight
        assert(len(link_car_tt_weight) == max_epoch)

        if np.isscalar(link_truck_tt_weight):
            link_truck_tt_weight = np.ones(max_epoch, dtype=int) * link_truck_tt_weight
        assert(len(link_truck_tt_weight) == max_epoch)

        if np.isscalar(link_passenger_tt_weight):
            link_passenger_tt_weight = np.ones(max_epoch, dtype=int) * link_passenger_tt_weight
        assert(len(link_passenger_tt_weight) == max_epoch)

        if np.isscalar(link_bus_tt_weight):
            link_bus_tt_weight = np.ones(max_epoch, dtype=int) * link_bus_tt_weight
        assert(len(link_bus_tt_weight) == max_epoch)
        
        loss_list = list()
        best_epoch = starting_epoch
        best_log_f_car_driving, best_log_f_truck_driving, best_log_f_passenger_bustransit, best_log_f_car_pnr, best_log_f_bus = 0, 0, 0, 0, 0
        best_f_car_driving, best_f_truck_driving, best_f_passenger_bustransit, best_f_car_pnr, best_f_bus = 0, 0, 0, 0, 0
        best_x_e_car, best_x_e_truck, best_x_e_passenger, best_x_e_bus, best_tt_e_car, best_tt_e_truck, best_tt_e_passenger, best_tt_e_bus = 0, 0, 0, 0, 0, 0, 0, 0
        if use_file_as_init is not None:
            # most recent
            _, _, loss_list, best_epoch, _, _, _, _, _, \
                _, _, _, _, _, \
                _, _, _, _, _, _, _, _ = pickle.load(open(use_file_as_init, 'rb'))
            # best
            use_file_as_init = os.path.join(save_folder, '{}_iteration_estimate_path_flow.pickle'.format(best_epoch))
            _, _, _, _, best_log_f_car_driving, best_log_f_truck_driving, best_log_f_passenger_bustransit, best_log_f_car_pnr, best_log_f_bus, \
                best_f_car_driving, best_f_truck_driving, best_f_passenger_bustransit, best_f_car_pnr, best_f_bus, \
                best_x_e_car, best_x_e_truck, best_x_e_passenger, best_x_e_bus, best_tt_e_car, best_tt_e_truck, best_tt_e_passenger, best_tt_e_bus \
                    = pickle.load(open(use_file_as_init, 'rb'))
        
        assert((len(loss_list) >= best_epoch) and (starting_epoch >= best_epoch))

        pathflow_solver = torch_pathflow_solver(self.num_assign_interval, self.num_path_driving, self.num_path_bustransit, self.num_path_pnr, self.num_path_busroute,
                                                car_driving_scale, truck_driving_scale, passenger_bustransit_scale, car_pnr_scale, bus_scale,
                                                fix_bus=fix_bus, use_file_as_init=use_file_as_init)

        # f, tensor
        f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus = pathflow_solver.generate_pathflow_tensor()
        assert(f_bus is None or isinstance(f_bus, torch.Tensor))
        # fixed bus path flow
        if fix_bus:
            f_bus = self.nb.demand_bus.path_flow_matrix.flatten(order='F')
            f_bus = np.maximum(f_bus, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])
        assert(isinstance(f_bus, np.ndarray) or isinstance(f_bus, torch.Tensor))

        pathflow_solver.set_params_with_lr(car_driving_step_size, truck_driving_step_size, passenger_bustransit_step_size, car_pnr_step_size, bus_step_size)
        pathflow_solver.set_optimizer(algo)
        # pathflow_solver.set_scheduler()
        # print(pathflow_solver.state_dict())
        
        for i in range(max_epoch):
            seq = np.random.permutation(self.num_data)
            loss = float(0)
            loss_dict = {
                "car_count_loss": 0.,
                "truck_count_loss": 0.,
                "bus_count_loss": 0.,
                "passenger_count_loss": 0.,
                "car_tt_loss": 0.,
                "truck_tt_loss": 0.,
                "bus_tt_loss": 0.,
                "passenger_tt_loss": 0.
            }

            self.config['link_car_flow_weight'] = link_car_flow_weight[i] * (self.config['use_car_link_flow'] or self.config['compute_car_link_flow_loss'])
            self.config['link_truck_flow_weight'] = link_truck_flow_weight[i] * (self.config['use_truck_link_flow'] or self.config['compute_truck_link_flow_loss'])
            self.config['link_passenger_flow_weight'] = link_passenger_flow_weight[i] * (self.config['use_passenger_link_flow'] or self.config['compute_passenger_link_flow_loss'])
            self.config['link_bus_flow_weight'] = link_bus_flow_weight[i] * (self.config['use_bus_link_flow'] or self.config['compute_bus_link_flow_loss'])

            self.config['link_car_tt_weight'] = link_car_tt_weight[i] * (self.config['use_car_link_tt'] or self.config['compute_car_link_tt_loss'])
            self.config['link_truck_tt_weight'] = link_truck_tt_weight[i] * (self.config['use_truck_link_tt'] or self.config['compute_truck_link_tt_loss'])
            self.config['link_passenger_tt_weight'] = link_passenger_tt_weight[i] * (self.config['use_passenger_link_tt'] or self.config['compute_passenger_link_tt_loss'])
            self.config['link_bus_tt_weight'] = link_bus_tt_weight[i] * (self.config['use_bus_link_tt'] or self.config['compute_bus_link_tt_loss'])

            for j in seq:
                # retrieve one record of observed data
                one_data_dict = self._get_one_data(j)

                # f tensor -> f numpy
                f_car_driving_numpy, f_truck_driving_numpy, f_passenger_bustransit_numpy, f_car_pnr_numpy, f_bus_numpy = \
                    pathflow_solver.generate_pathflow_numpy(f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus=None if fix_bus else f_bus)
                if fix_bus:
                    f_bus_numpy = f_bus

                # f_grad numpy: num_path * num_assign_interval
                f_car_driving_grad, f_truck_driving_grad, f_passenger_bustransit_grad, f_car_pnr_grad, f_passenger_pnr_grad, f_bus_grad, \
                    tmp_loss, tmp_loss_dict, dta, x_e_car, x_e_truck, x_e_passenger, x_e_bus, tt_e_car, tt_e_truck, tt_e_passenger, tt_e_bus = \
                        self.compute_path_flow_grad_and_loss(one_data_dict, f_car_driving_numpy, f_truck_driving_numpy, 
                                                             f_passenger_bustransit_numpy, f_car_pnr_numpy, f_bus_numpy, fix_bus=fix_bus, counter=0, run_mmdta_adaptive=run_mmdta_adaptive)

                if column_generation[i] == 0:
                    dta = 0
                
                pathflow_solver.optimizer.zero_grad()

                # grad of loss wrt log_f = grad of f wrt log_f (pytorch autograd) * grad of loss wrt f (manually)
                pathflow_solver.compute_gradient(f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus=None if fix_bus else f_bus,
                                             f_car_driving_grad=f_car_driving_grad, f_truck_driving_grad=f_truck_driving_grad, f_passenger_bustransit_grad=f_passenger_bustransit_grad, 
                                             f_car_pnr_grad=f_car_pnr_grad, f_passenger_pnr_grad=f_passenger_pnr_grad, f_bus_grad=None if fix_bus else f_bus_grad,
                                             l2_coeff=l2_coeff)
                
                # update log_f
                pathflow_solver.optimizer.step()

                # release memory
                f_car_driving_grad, f_truck_driving_grad, f_passenger_bustransit_grad, f_car_pnr_grad, f_passenger_pnr_grad, f_bus_grad = 0, 0, 0, 0, 0, 0
                pathflow_solver.optimizer.zero_grad()

                # f tensor
                f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus = pathflow_solver.generate_pathflow_tensor()
                assert(f_bus is None or isinstance(f_bus, torch.Tensor))
                if fix_bus:
                    f_bus = self.nb.demand_bus.path_flow_matrix.flatten(order='F')
                    f_bus = np.maximum(f_bus, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])
                assert(isinstance(f_bus, np.ndarray) or isinstance(f_bus, torch.Tensor))

                if column_generation[i]:
                    print("***************** generate new paths *****************")
                    if not self.config['use_car_link_tt'] and not self.config['use_truck_link_tt'] and not self.config['use_passenger_link_tt'] and not self.config['use_bus_link_tt']:
                        # if any of these is true, dta.build_link_cost_map(True) is already invoked in compute_path_flow_grad_and_loss()
                        dta.build_link_cost_map(False)

                    self.update_path_table(dta, use_tdsp)

                    dta = 0

                    # update log_f tensor
                    pathflow_solver.add_pathflow(
                        len(self.nb.path_table_driving.ID2path) - len(f_car_driving) // self.num_assign_interval, 
                        len(self.nb.path_table_bustransit.ID2path) - len(f_passenger_bustransit) // self.num_assign_interval,
                        len(self.nb.path_table_pnr.ID2path) - len(f_car_pnr) // self.num_assign_interval,
                    )
                    # print(pathflow_solver.state_dict())

                    # update f tensor from updated log_f tensor
                    f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, _ = pathflow_solver.generate_pathflow_tensor()

                    # update params and optimizer
                    pathflow_solver.set_params_with_lr(car_driving_step_size, truck_driving_step_size, passenger_bustransit_step_size, car_pnr_step_size, bus_step_size)
                    pathflow_solver.set_optimizer(algo)
                    # pathflow_solver.set_scheduler()

                loss += tmp_loss / float(self.num_data)
                for loss_type, loss_value in tmp_loss_dict.items():
                    loss_dict[loss_type] += loss_value / float(self.num_data)
            
            print("Epoch:", starting_epoch + i, "Loss:", loss, self.print_separate_accuracy(loss_dict))
            loss_list.append([loss, loss_dict])
            
            if (best_epoch == 0) or (loss_list[best_epoch][0] > loss_list[-1][0]):
                best_epoch = starting_epoch + i

                # log_f numpy
                best_log_f_car_driving, best_log_f_truck_driving, best_log_f_passenger_bustransit, best_log_f_car_pnr, best_log_f_bus = pathflow_solver.get_log_f_numpy()

                # f numpy
                best_f_car_driving, best_f_truck_driving, best_f_passenger_bustransit, best_f_car_pnr, best_f_bus = \
                    pathflow_solver.generate_pathflow_numpy(f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus=None if fix_bus else f_bus)
                if fix_bus:
                    best_f_bus = self.nb.demand_bus.path_flow_matrix.flatten(order='F')
                    best_f_bus = np.maximum(f_bus, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])

                best_x_e_car, best_x_e_truck, best_x_e_passenger, best_x_e_bus, best_tt_e_car, best_tt_e_truck, best_tt_e_passenger, best_tt_e_bus = \
                    x_e_car, x_e_truck, x_e_passenger, x_e_bus, tt_e_car, tt_e_truck, tt_e_passenger, tt_e_bus

                if save_folder is not None:
                    self.save_simulation_input_files(os.path.join(save_folder, 'input_files_estimate_path_flow'), 
                                                     best_f_car_driving, best_f_truck_driving, best_f_passenger_bustransit, best_f_car_pnr, f_bus = None if fix_bus else best_f_bus,
                                                     explicit_bus=1, historical_bus_waiting_time=0)
            
            # if 'passenger_count_loss' in loss_list[-1][-1]:
            #     pathflow_solver.scheduler.step(loss_list[-1][-1]['passenger_count_loss'])
            # else:
            #     pathflow_solver.scheduler.step(loss_list[-1][0])
            # pathflow_solver.scheduler.step(loss_list[-1][0])
            # pathflow_solver.scheduler.step()

            if save_folder is not None:
                # log_f numpy
                log_f_car_driving_numpy, log_f_truck_driving_numpy, log_f_passenger_bustransit_numpy, log_f_car_pnr_numpy, log_f_bus_numpy = pathflow_solver.get_log_f_numpy()
                # f numpy
                f_car_driving_numpy, f_truck_driving_numpy, f_passenger_bustransit_numpy, f_car_pnr_numpy, f_bus_numpy = \
                    pathflow_solver.generate_pathflow_numpy(f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus=None if fix_bus else f_bus)
                if fix_bus:
                    f_bus_numpy = f_bus
                pickle.dump([loss, loss_dict, loss_list, best_epoch,
                             log_f_car_driving_numpy, log_f_truck_driving_numpy, log_f_passenger_bustransit_numpy, log_f_car_pnr_numpy, log_f_bus_numpy,
                             f_car_driving_numpy, f_truck_driving_numpy, f_passenger_bustransit_numpy, f_car_pnr_numpy, f_bus_numpy,
                             x_e_car, x_e_truck, x_e_passenger, x_e_bus, tt_e_car, tt_e_truck, tt_e_passenger, tt_e_bus],
                            open(os.path.join(save_folder, str(starting_epoch + i) + '_iteration_estimate_path_flow.pickle'), 'wb'))

        print("Best loss at Epoch:", best_epoch, "Loss:", loss_list[best_epoch][0], self.print_separate_accuracy(loss_list[best_epoch][1]))
        return best_log_f_car_driving, best_log_f_truck_driving, best_log_f_passenger_bustransit, best_log_f_car_pnr, best_log_f_bus, \
               best_f_car_driving, best_f_truck_driving, best_f_passenger_bustransit, best_f_car_pnr, best_f_bus, \
               best_x_e_car, best_x_e_truck, best_x_e_passenger, best_x_e_bus, best_tt_e_car, best_tt_e_truck, best_tt_e_passenger, best_tt_e_bus, loss_list
        
        
    def estimate_path_flow_pytorch2(self, car_driving_scale=10, truck_driving_scale=1, passenger_bustransit_scale=1, car_pnr_scale=5, bus_scale=1,
                                   car_driving_step_size=0.1, truck_driving_step_size=0.01, passenger_bustransit_step_size=0.01, car_pnr_step_size=0.05, bus_step_size=0.01,
                                   link_car_flow_weight=1, link_truck_flow_weight=1, link_passenger_flow_weight=1, link_bus_flow_weight=1,
                                   link_car_tt_weight=1, link_truck_tt_weight=1, link_passenger_tt_weight=1, link_bus_tt_weight=1,
                                   max_epoch=100, algo="NAdam", run_mmdta_adaptive=False, fix_bus=True, column_generation=False, use_tdsp=False, use_file_as_init=None, save_folder=None, starting_epoch=0):
        if save_folder is not None and not os.path.exists(save_folder):
            os.mkdir(save_folder)

        if fix_bus:
            bus_scale = None
            bus_step_size = None
        else:
            assert(bus_scale is not None)
            assert(bus_step_size is not None)

        if np.isscalar(column_generation):
            column_generation = np.ones(max_epoch, dtype=int) * column_generation
        assert(len(column_generation) == max_epoch)

        if np.isscalar(link_car_flow_weight):
            link_car_flow_weight = np.ones(max_epoch, dtype=int) * link_car_flow_weight
        assert(len(link_car_flow_weight) == max_epoch)

        if np.isscalar(link_truck_flow_weight):
            link_truck_flow_weight = np.ones(max_epoch, dtype=int) * link_truck_flow_weight
        assert(len(link_truck_flow_weight) == max_epoch)

        if np.isscalar(link_passenger_flow_weight):
            link_passenger_flow_weight = np.ones(max_epoch, dtype=int) * link_passenger_flow_weight
        assert(len(link_passenger_flow_weight) == max_epoch)

        if np.isscalar(link_bus_flow_weight):
            link_bus_flow_weight = np.ones(max_epoch, dtype=int) * link_bus_flow_weight
        assert(len(link_bus_flow_weight) == max_epoch)

        if np.isscalar(link_car_tt_weight):
            link_car_tt_weight = np.ones(max_epoch, dtype=int) * link_car_tt_weight
        assert(len(link_car_tt_weight) == max_epoch)

        if np.isscalar(link_truck_tt_weight):
            link_truck_tt_weight = np.ones(max_epoch, dtype=int) * link_truck_tt_weight
        assert(len(link_truck_tt_weight) == max_epoch)

        if np.isscalar(link_passenger_tt_weight):
            link_passenger_tt_weight = np.ones(max_epoch, dtype=int) * link_passenger_tt_weight
        assert(len(link_passenger_tt_weight) == max_epoch)

        if np.isscalar(link_bus_tt_weight):
            link_bus_tt_weight = np.ones(max_epoch, dtype=int) * link_bus_tt_weight
        assert(len(link_bus_tt_weight) == max_epoch)
        
        loss_list = list()
        best_epoch = starting_epoch
        best_f_car_driving, best_f_truck_driving, best_f_passenger_bustransit, best_f_car_pnr, best_f_bus = 0, 0, 0, 0, 0
        best_x_e_car, best_x_e_truck, best_x_e_passenger, best_x_e_bus, best_tt_e_car, best_tt_e_truck, best_tt_e_passenger, best_tt_e_bus = 0, 0, 0, 0, 0, 0, 0, 0
        # read from files as init values
        if use_file_as_init is not None:
            # most recent
            _, _, loss_list, best_epoch, _, _, _, _, _, \
                _, _, _, _, _, _, _, _ = pickle.load(open(use_file_as_init, 'rb'))
            # best 
            use_file_as_init = os.path.join(save_folder, '{}_iteration_estimate_path_flow.pickle'.format(best_epoch))
            _, _, _, _, best_f_car_driving, best_f_truck_driving, best_f_passenger_bustransit, best_f_car_pnr, best_f_bus, \
                best_x_e_car, best_x_e_truck, best_x_e_passenger, best_x_e_bus, best_tt_e_car, best_tt_e_truck, best_tt_e_passenger, best_tt_e_bus = \
                    pickle.load(open(use_file_as_init, 'rb'))
            
            f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus = \
                best_f_car_driving, best_f_truck_driving, best_f_passenger_bustransit, best_f_car_pnr, best_f_bus

            # f_bus = self.nb.demand_bus.path_flow_matrix.flatten(order='F')
            # f_passenger_bustransit = np.ones(self.num_assign_interval * self.num_path_bustransit) * passenger_bustransit_scale
        else:
            f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus = \
                self.init_path_flow(car_driving_scale, truck_driving_scale, passenger_bustransit_scale, car_pnr_scale, bus_scale)
            
        assert((len(loss_list) >= best_epoch) and (starting_epoch >= best_epoch))
           
        # fixed bus path flow
        if fix_bus:
            f_bus = self.nb.demand_bus.path_flow_matrix.flatten(order='F')

        # relu function
        f_car_driving = np.maximum(f_car_driving, 1e-6 / self.nb.config.config_dict['DTA']['flow_scalar'])
        f_truck_driving = np.maximum(f_truck_driving, 1e-6 / self.nb.config.config_dict['DTA']['flow_scalar'])
        # this alleviate trapping in local minima, when f = 0 -> grad = 0 is not helping
        f_passenger_bustransit = np.maximum(f_passenger_bustransit, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])
        f_car_pnr = np.maximum(f_car_pnr, 1e-6 / self.nb.config.config_dict['DTA']['flow_scalar'])
        f_bus = np.maximum(f_bus, 1e-6 / self.nb.config.config_dict['DTA']['flow_scalar'])

        f_car_driving_tensor = torch.from_numpy(f_car_driving / np.maximum(car_driving_scale, 1e-6))
        f_truck_driving_tensor = torch.from_numpy(f_truck_driving / np.maximum(truck_driving_scale, 1e-6))
        f_passenger_bustransit_tensor = torch.from_numpy(f_passenger_bustransit / np.maximum(passenger_bustransit_scale, 1e-6))
        f_car_pnr_tensor = torch.from_numpy(f_car_pnr / np.maximum(car_pnr_scale, 1e-6))
        if not fix_bus:
            f_bus_tensor = torch.from_numpy(f_bus / np.maximum(bus_scale, 1e-6))

        # f_car_driving_tensor = torch.from_numpy(f_car_driving)
        # f_truck_driving_tensor = torch.from_numpy(f_truck_driving)
        # f_passenger_bustransit_tensor = torch.from_numpy(f_passenger_bustransit)
        # f_car_pnr_tensor = torch.from_numpy(f_car_pnr)
        # if not fix_bus:
        #     f_bus_tensor = torch.from_numpy(f_bus)

        f_car_driving_tensor.requires_grad = True
        f_truck_driving_tensor.requires_grad = True
        f_passenger_bustransit_tensor.requires_grad = True
        f_car_pnr_tensor.requires_grad = True
        if not fix_bus:
            f_bus_tensor.requires_grad = True

        params = [
            {'params': f_car_driving_tensor, 'lr': car_driving_step_size},
            {'params': f_truck_driving_tensor, 'lr': truck_driving_step_size},
            {'params': f_passenger_bustransit_tensor, 'lr': passenger_bustransit_step_size},
            {'params': f_car_pnr_tensor, 'lr': car_pnr_step_size}
        ]

        if not fix_bus:
            params.append({'params': f_bus_tensor, 'lr': bus_step_size})

        algo_dict = {
            "SGD": torch.optim.SGD,
            "NAdam": torch.optim.NAdam,
            "Adam": torch.optim.Adam,
            "Adamax": torch.optim.Adamax,
            "AdamW": torch.optim.AdamW,
            "RAdam": torch.optim.RAdam,
            "Adagrad": torch.optim.Adagrad,
            "Adadelta": torch.optim.Adadelta
        }
        optimizer = algo_dict[algo](params)

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode='min', factor=0.75, patience=5, 
        #     threshold=0.15, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        
        for i in range(max_epoch):
            seq = np.random.permutation(self.num_data)
            loss = float(0)
            loss_dict = {
                "car_count_loss": 0.,
                "truck_count_loss": 0.,
                "bus_count_loss": 0.,
                "passenger_count_loss": 0.,
                "car_tt_loss": 0.,
                "truck_tt_loss": 0.,
                "bus_tt_loss": 0.,
                "passenger_tt_loss": 0.
            }

            self.config['link_car_flow_weight'] = link_car_flow_weight[i] * (self.config['use_car_link_flow'] or self.config['compute_car_link_flow_loss'])
            self.config['link_truck_flow_weight'] = link_truck_flow_weight[i] * (self.config['use_truck_link_flow'] or self.config['compute_truck_link_flow_loss'])
            self.config['link_passenger_flow_weight'] = link_passenger_flow_weight[i] * (self.config['use_passenger_link_flow'] or self.config['compute_passenger_link_flow_loss'])
            self.config['link_bus_flow_weight'] = link_bus_flow_weight[i] * (self.config['use_bus_link_flow'] or self.config['compute_bus_link_flow_loss'])

            self.config['link_car_tt_weight'] = link_car_tt_weight[i] * (self.config['use_car_link_tt'] or self.config['compute_car_link_tt_loss'])
            self.config['link_truck_tt_weight'] = link_truck_tt_weight[i] * (self.config['use_truck_link_tt'] or self.config['compute_truck_link_tt_loss'])
            self.config['link_passenger_tt_weight'] = link_passenger_tt_weight[i] * (self.config['use_passenger_link_tt'] or self.config['compute_passenger_link_tt_loss'])
            self.config['link_bus_tt_weight'] = link_bus_tt_weight[i] * (self.config['use_bus_link_tt'] or self.config['compute_bus_link_tt_loss'])

            for j in seq:
                # retrieve one record of observed data
                one_data_dict = self._get_one_data(j)

                # f_grad: num_path * num_assign_interval
                f_car_driving_grad, f_truck_driving_grad, f_passenger_bustransit_grad, f_car_pnr_grad, f_passenger_pnr_grad, f_bus_grad, \
                    tmp_loss, tmp_loss_dict, dta, x_e_car, x_e_truck, x_e_passenger, x_e_bus, tt_e_car, tt_e_truck, tt_e_passenger, tt_e_bus = \
                        self.compute_path_flow_grad_and_loss(one_data_dict, f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus, fix_bus=fix_bus, counter=0, run_mmdta_adaptive=run_mmdta_adaptive)

                if column_generation[i] == 0:
                    dta = 0
                
                optimizer.zero_grad()

                f_car_driving_tensor.grad = torch.from_numpy(f_car_driving_grad * car_driving_scale)
                f_truck_driving_tensor.grad = torch.from_numpy(f_truck_driving_grad * truck_driving_scale)
                f_passenger_bustransit_tensor.grad = torch.from_numpy(f_passenger_bustransit_grad * passenger_bustransit_scale)
                f_car_pnr_tensor.grad = torch.from_numpy((f_car_pnr_grad + f_passenger_pnr_grad) * car_pnr_scale)
                if not fix_bus:
                    f_bus_tensor.grad = torch.from_numpy(f_bus_grad * bus_scale)

                # f_car_driving_tensor.grad = torch.from_numpy(f_car_driving_grad)
                # f_truck_driving_tensor.grad = torch.from_numpy(f_truck_driving_grad)
                # f_passenger_bustransit_tensor.grad = torch.from_numpy(f_passenger_bustransit_grad)
                # f_car_pnr_tensor.grad = torch.from_numpy((f_car_pnr_grad + f_passenger_pnr_grad))
                # if not fix_bus:
                #     f_bus_tensor.grad = torch.from_numpy(f_bus_grad)

                optimizer.step()

                # release memory
                f_car_driving_grad, f_truck_driving_grad, f_passenger_bustransit_grad, f_car_pnr_grad, f_passenger_pnr_grad, f_bus_grad = 0, 0, 0, 0, 0, 0
                optimizer.zero_grad()

                f_car_driving = f_car_driving_tensor.data.cpu().numpy() * car_driving_scale
                f_truck_driving = f_truck_driving_tensor.data.cpu().numpy() * truck_driving_scale
                f_passenger_bustransit = f_passenger_bustransit_tensor.data.cpu().numpy() * passenger_bustransit_scale
                f_car_pnr = f_car_pnr_tensor.data.cpu().numpy() * car_pnr_scale
                if not fix_bus:
                    f_bus = f_bus_tensor.data.cpu().numpy() * bus_scale

                # f_car_driving = f_car_driving_tensor.data.cpu().numpy()
                # f_truck_driving = f_truck_driving_tensor.data.cpu().numpy()
                # f_passenger_bustransit = f_passenger_bustransit_tensor.data.cpu().numpy()
                # f_car_pnr = f_car_pnr_tensor.data.cpu().numpy()
                # if not fix_bus:
                #     f_bus = f_bus_tensor.data.cpu().numpy() 

                if column_generation[i]:
                    print("***************** generate new paths *****************")
                    if not self.config['use_car_link_tt'] and not self.config['use_truck_link_tt'] and not self.config['use_passenger_link_tt'] and not self.config['use_bus_link_tt']:
                        # if any of these is true, dta.build_link_cost_map(True) is already invoked in compute_path_flow_grad_and_loss()
                        dta.build_link_cost_map(False)
                    self.update_path_table(dta, use_tdsp)
                    f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr = \
                        self.update_path_flow(f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr,
                                              car_driving_scale, truck_driving_scale, passenger_bustransit_scale, car_pnr_scale)
                    dta = 0
                
                f_car_driving = np.maximum(f_car_driving, 1e-6 / self.nb.config.config_dict['DTA']['flow_scalar'])
                f_truck_driving = np.maximum(f_truck_driving, 1e-6 / self.nb.config.config_dict['DTA']['flow_scalar'])
                # this helps jump out of local minima, when f = 0 -> grad = 0 is not helping
                # f_passenger_bustransit = np.maximum(f_passenger_bustransit, np.random.choice([1e-3, 1]))
                f_passenger_bustransit = np.maximum(f_passenger_bustransit, 1 / self.nb.config.config_dict['DTA']['flow_scalar'])
                # f_passenger_bustransit = self.perturbate_path_flow(f_passenger_bustransit, i, prob=0.5)
                f_car_pnr = np.maximum(f_car_pnr, 1e-6 / self.nb.config.config_dict['DTA']['flow_scalar'])
                if not fix_bus:
                    f_bus = np.maximum(f_bus, 1e-6 / self.nb.config.config_dict['DTA']['flow_scalar'])
                

                if column_generation[i]:
                    f_car_driving_tensor = torch.from_numpy(f_car_driving / np.maximum(car_driving_scale, 1e-6))
                    f_truck_driving_tensor = torch.from_numpy(f_truck_driving / np.maximum(truck_driving_scale, 1e-6))
                    f_passenger_bustransit_tensor = torch.from_numpy(f_passenger_bustransit / np.maximum(passenger_bustransit_scale, 1e-6))
                    f_car_pnr_tensor = torch.from_numpy(f_car_pnr / np.maximum(car_pnr_scale, 1e-6))
                    if not fix_bus:
                        f_bus_tensor = torch.from_numpy(f_bus / np.maximum(bus_scale, 1e-6))

                    # f_car_driving_tensor = torch.from_numpy(f_car_driving)
                    # f_truck_driving_tensor = torch.from_numpy(f_truck_driving)
                    # f_passenger_bustransit_tensor = torch.from_numpy(f_passenger_bustransit)
                    # f_car_pnr_tensor = torch.from_numpy(f_car_pnr)
                    # if not fix_bus:
                    #     f_bus_tensor = torch.from_numpy(f_bus)

                    f_car_driving_tensor.requires_grad = True
                    f_truck_driving_tensor.requires_grad = True
                    f_passenger_bustransit_tensor.requires_grad = True
                    f_car_pnr_tensor.requires_grad = True
                    if not fix_bus:
                        f_bus_tensor.requires_grad = True

                    params = [
                        {'params': f_car_driving_tensor, 'lr': car_driving_step_size},
                        {'params': f_truck_driving_tensor, 'lr': truck_driving_step_size},
                        {'params': f_passenger_bustransit_tensor, 'lr': passenger_bustransit_step_size},
                        {'params': f_car_pnr_tensor, 'lr': car_pnr_step_size}
                    ]

                    if not fix_bus:
                        params.append({'params': f_bus_tensor, 'lr': bus_step_size})

                    optimizer = algo_dict[algo](params)

                    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    #     optimizer, mode='min', factor=0.75, patience=5, 
                    #     threshold=0.15, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)

                loss += tmp_loss / float(self.num_data)
                for loss_type, loss_value in tmp_loss_dict.items():
                    loss_dict[loss_type] += loss_value / float(self.num_data)
            
            print("Epoch:", starting_epoch + i, "Loss:", loss, self.print_separate_accuracy(loss_dict))
            loss_list.append([loss, loss_dict])
            
            if (best_epoch == 0) or (loss_list[best_epoch][0] > loss_list[-1][0]):
                best_epoch = starting_epoch + i
                best_f_car_driving, best_f_truck_driving, best_f_passenger_bustransit, best_f_car_pnr, best_f_bus = \
                     f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus

                best_x_e_car, best_x_e_truck, best_x_e_passenger, best_x_e_bus, best_tt_e_car, best_tt_e_truck, best_tt_e_passenger, best_tt_e_bus = \
                    x_e_car, x_e_truck, x_e_passenger, x_e_bus, tt_e_car, tt_e_truck, tt_e_passenger, tt_e_bus

                if save_folder is not None:
                    self.save_simulation_input_files(os.path.join(save_folder, 'input_files_estimate_path_flow'), 
                                                     f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus = None if fix_bus else f_bus,
                                                     explicit_bus=1, historical_bus_waiting_time=0)
            
            # if 'passenger_count_loss' in loss_list[-1][-1]:
            #     scheduler.step(loss_list[-1][-1]['passenger_count_loss'])
            # else:
            #     scheduler.step(loss_list[-1][0])
            
            # scheduler.step(loss_list[-1][0])

            if save_folder is not None:
                pickle.dump([loss, loss_dict, loss_list, best_epoch,
                             f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus,
                             x_e_car, x_e_truck, x_e_passenger, x_e_bus, tt_e_car, tt_e_truck, tt_e_passenger, tt_e_bus], 
                            open(os.path.join(save_folder, str(starting_epoch + i) + '_iteration_estimate_path_flow.pickle'), 'wb'))

        print("Best loss at Epoch:", best_epoch, "Loss:", loss_list[best_epoch][0], self.print_separate_accuracy(loss_list[best_epoch][1]))
        return best_f_car_driving, best_f_truck_driving, best_f_passenger_bustransit, best_f_car_pnr, best_f_bus, \
               best_x_e_car, best_x_e_truck, best_x_e_passenger, best_x_e_bus, best_tt_e_car, best_tt_e_truck, best_tt_e_passenger, best_tt_e_bus, \
               loss_list
        

    def estimate_path_flow(self, car_driving_scale=10, truck_driving_scale=1, passenger_bustransit_scale=1, car_pnr_scale=5, bus_scale=1,
                           car_driving_step_size=0.1, truck_driving_step_size=0.01, passenger_bustransit_step_size=0.01, car_pnr_step_size=0.05, bus_step_size=0.01,
                           link_car_flow_weight=1, link_truck_flow_weight=1, link_passenger_flow_weight=1, link_bus_flow_weight=1,
                           link_car_tt_weight=1, link_truck_tt_weight=1, link_passenger_tt_weight=1, link_bus_tt_weight=1,
                           max_epoch=100, adagrad=False, run_mmdta_adaptive=False, fix_bus=True, column_generation=False, use_tdsp=False, use_file_as_init=None, save_folder=None, starting_epoch=0):
        if save_folder is not None and not os.path.exists(save_folder):
            os.mkdir(save_folder)

        if fix_bus:
            bus_scale = None
            bus_step_size = None
        else:
            assert(bus_scale is not None)
            assert(bus_step_size is not None)

        if np.isscalar(column_generation):
            column_generation = np.ones(max_epoch, dtype=int) * column_generation
        assert(len(column_generation) == max_epoch)

        if np.isscalar(car_driving_step_size):
            car_driving_step_size = np.ones(max_epoch, dtype=float) * car_driving_step_size
        assert(len(car_driving_step_size) == max_epoch)

        if np.isscalar(truck_driving_step_size):
            truck_driving_step_size = np.ones(max_epoch, dtype=float) * truck_driving_step_size
        assert(len(truck_driving_step_size) == max_epoch)

        if np.isscalar(passenger_bustransit_step_size):
            passenger_bustransit_step_size = np.ones(max_epoch, dtype=float) * passenger_bustransit_step_size
        assert(len(passenger_bustransit_step_size) == max_epoch)

        if np.isscalar(car_pnr_step_size):
            car_pnr_step_size = np.ones(max_epoch, dtype=float) * car_pnr_step_size
        assert(len(car_pnr_step_size) == max_epoch)

        if np.isscalar(bus_step_size):
            bus_step_size = np.ones(max_epoch, dtype=float) * bus_step_size
        assert(len(bus_step_size) == max_epoch)

        if np.isscalar(link_car_flow_weight):
            link_car_flow_weight = np.ones(max_epoch, dtype=bool) * link_car_flow_weight
        assert(len(link_car_flow_weight) == max_epoch)

        if np.isscalar(link_truck_flow_weight):
            link_truck_flow_weight = np.ones(max_epoch, dtype=bool) * link_truck_flow_weight
        assert(len(link_truck_flow_weight) == max_epoch)

        if np.isscalar(link_passenger_flow_weight):
            link_passenger_flow_weight = np.ones(max_epoch, dtype=bool) * link_passenger_flow_weight
        assert(len(link_passenger_flow_weight) == max_epoch)

        if np.isscalar(link_bus_flow_weight):
            link_bus_flow_weight = np.ones(max_epoch, dtype=bool) * link_bus_flow_weight
        assert(len(link_bus_flow_weight) == max_epoch)

        if np.isscalar(link_car_tt_weight):
            link_car_tt_weight = np.ones(max_epoch, dtype=bool) * link_car_tt_weight
        assert(len(link_car_tt_weight) == max_epoch)

        if np.isscalar(link_truck_tt_weight):
            link_truck_tt_weight = np.ones(max_epoch, dtype=bool) * link_truck_tt_weight
        assert(len(link_truck_tt_weight) == max_epoch)

        if np.isscalar(link_passenger_tt_weight):
            link_passenger_tt_weight = np.ones(max_epoch, dtype=bool) * link_passenger_tt_weight
        assert(len(link_passenger_tt_weight) == max_epoch)

        if np.isscalar(link_bus_tt_weight):
            link_bus_tt_weight = np.ones(max_epoch, dtype=bool) * link_bus_tt_weight
        assert(len(link_bus_tt_weight) == max_epoch)
        
        loss_list = list()
        best_epoch = starting_epoch
        best_f_car_driving, best_f_truck_driving, best_f_passenger_bustransit, best_f_car_pnr, best_f_bus = 0, 0, 0, 0, 0
        best_x_e_car, best_x_e_truck, best_x_e_passenger, best_x_e_bus, best_tt_e_car, best_tt_e_truck, best_tt_e_passenger, best_tt_e_bus = 0, 0, 0, 0, 0, 0, 0, 0
        # read from files as init values
        if use_file_as_init is not None:
            # most recent
            _, _, loss_list, best_epoch, _, _, _, _, _, \
                _, _, _, _, _, _, _, _ = pickle.load(open(use_file_as_init, 'rb'))
            # best 
            use_file_as_init = os.path.join(save_folder, '{}_iteration_estimate_path_flow.pickle'.format(best_epoch))
            _, _, _, _, best_f_car_driving, best_f_truck_driving, best_f_passenger_bustransit, best_f_car_pnr, best_f_bus, \
                best_x_e_car, best_x_e_truck, best_x_e_passenger, best_x_e_bus, best_tt_e_car, best_tt_e_truck, best_tt_e_passenger, best_tt_e_bus = \
                pickle.load(open(use_file_as_init, 'rb'))
            
            f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus = \
                best_f_car_driving, best_f_truck_driving, best_f_passenger_bustransit, best_f_car_pnr, best_f_bus
        else:
            f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus = \
                self.init_path_flow(car_driving_scale, truck_driving_scale, passenger_bustransit_scale, car_pnr_scale, bus_scale)

        # fixed bus path flow
        if fix_bus:
            f_bus = self.nb.demand_bus.path_flow_matrix.flatten(order='F')

        f_car_driving = np.maximum(f_car_driving, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])
        f_truck_driving = np.maximum(f_truck_driving, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])
        # this alleviate trapping in local minima, when f = 0 -> grad = 0 is not helping
        f_passenger_bustransit = np.maximum(f_passenger_bustransit, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])
        f_car_pnr = np.maximum(f_car_pnr, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])
        f_bus = np.maximum(f_bus, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])

        for i in range(max_epoch):
            seq = np.random.permutation(self.num_data)
            loss = float(0)
            loss_dict = {
                "car_count_loss": 0.,
                "truck_count_loss": 0.,
                "bus_count_loss": 0.,
                "passenger_count_loss": 0.,
                "car_tt_loss": 0.,
                "truck_tt_loss": 0.,
                "bus_tt_loss": 0.,
                "passenger_tt_loss": 0.
            }

            self.config['link_car_flow_weight'] = link_car_flow_weight[i] * (self.config['use_car_link_flow'] or self.config['compute_car_link_flow_loss'])
            self.config['link_truck_flow_weight'] = link_truck_flow_weight[i] * (self.config['use_truck_link_flow'] or self.config['compute_truck_link_flow_loss'])
            self.config['link_passenger_flow_weight'] = link_passenger_flow_weight[i] * (self.config['use_passenger_link_flow'] or self.config['compute_passenger_link_flow_loss'])
            self.config['link_bus_flow_weight'] = link_bus_flow_weight[i] * (self.config['use_bus_link_flow'] or self.config['compute_bus_link_flow_loss'])

            self.config['link_car_tt_weight'] = link_car_tt_weight[i] * (self.config['use_car_link_tt'] or self.config['compute_car_link_tt_loss'])
            self.config['link_truck_tt_weight'] = link_truck_tt_weight[i] * (self.config['use_truck_link_tt'] or self.config['compute_truck_link_tt_loss'])
            self.config['link_passenger_tt_weight'] = link_passenger_tt_weight[i] * (self.config['use_passenger_link_tt'] or self.config['compute_passenger_link_tt_loss'])
            self.config['link_bus_tt_weight'] = link_bus_tt_weight[i] * (self.config['use_bus_link_tt'] or self.config['compute_bus_link_tt_loss'])

            if adagrad:
                sum_g_square_car_driving = 1e-6
                sum_g_square_truck_driving = 1e-6
                sum_g_square_passenger_bustransit = 1e-6
                sum_g_square_car_pnr = 1e-6
                if not fix_bus:
                    sum_g_square_bus = 1e-6
            for j in seq:
                # retrieve one record of observed data
                one_data_dict = self._get_one_data(j)

                # f_grad: num_path * num_assign_interval
                f_car_driving_grad, f_truck_driving_grad, f_passenger_bustransit_grad, f_car_pnr_grad, f_passenger_pnr_grad, f_bus_grad, \
                    tmp_loss, tmp_loss_dict, dta, x_e_car, x_e_truck, x_e_passenger, x_e_bus, tt_e_car, tt_e_truck, tt_e_passenger, tt_e_bus = \
                        self.compute_path_flow_grad_and_loss(one_data_dict, f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus, fix_bus=fix_bus, counter=0, run_mmdta_adaptive=run_mmdta_adaptive)

                if column_generation[i] == 0:
                    dta = 0
                
                if adagrad:
                    sum_g_square_car_driving = sum_g_square_car_driving + np.power(f_car_driving_grad, 2)
                    f_car_driving -= f_car_driving_grad * car_driving_step_size[i] / np.sqrt(sum_g_square_car_driving)

                    sum_g_square_truck_driving = sum_g_square_truck_driving + np.power(f_truck_driving_grad, 2)
                    f_truck_driving -= f_truck_driving_grad * truck_driving_step_size[i] / np.sqrt(sum_g_square_truck_driving)

                    sum_g_square_passenger_bustransit = sum_g_square_passenger_bustransit + np.power(f_passenger_bustransit_grad, 2)
                    f_passenger_bustransit -= f_passenger_bustransit_grad * passenger_bustransit_step_size[i] / np.sqrt(sum_g_square_passenger_bustransit)

                    sum_g_square_car_pnr = sum_g_square_car_pnr + np.power((f_car_pnr_grad + f_passenger_pnr_grad), 2)
                    f_car_pnr -= (f_car_pnr_grad + f_passenger_pnr_grad) * car_pnr_step_size[i] / np.sqrt(sum_g_square_car_pnr)

                    if not fix_bus:
                        sum_g_square_bus = sum_g_square_bus + np.power(f_bus_grad, 2)
                        f_bus -= f_bus_grad * bus_step_size[i] / np.sqrt(sum_g_square_bus)
                else:
                    f_car_driving -= f_car_driving_grad * car_driving_step_size[i] / np.sqrt(i+1)
                    f_truck_driving -= f_truck_driving_grad * truck_driving_step_size[i] / np.sqrt(i+1)
                    f_passenger_bustransit -= f_passenger_bustransit_grad * passenger_bustransit_step_size[i] / np.sqrt(i+1)
                    f_car_pnr -= (f_car_pnr_grad + f_passenger_pnr_grad) * car_pnr_step_size[i] / np.sqrt(i+1)
                    if not fix_bus:
                        f_bus -= f_bus_grad * bus_step_size[i] / np.sqrt(i+1)

                # release memory
                f_car_driving_grad, f_truck_driving_grad, f_passenger_bustransit_grad, f_car_pnr_grad, f_passenger_pnr_grad, f_bus_grad = 0, 0, 0, 0, 0, 0

                if column_generation[i]:
                    print("***************** generate new paths *****************")
                    if not self.config['use_car_link_tt'] and not self.config['use_truck_link_tt'] and not self.config['use_passenger_link_tt']:
                        # if any of these is true, dta.build_link_cost_map(True) is already invoked in compute_path_flow_grad_and_loss()
                        dta.build_link_cost_map(False)
                    self.update_path_table(dta, use_tdsp)
                    f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr = \
                        self.update_path_flow(f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr,
                                              car_driving_scale, truck_driving_scale, passenger_bustransit_scale, car_pnr_scale)
                    dta = 0
                
                f_car_driving = np.maximum(f_car_driving, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])
                f_truck_driving = np.maximum(f_truck_driving, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])
                # this alleviate trapping in local minima, when f = 0 -> grad = 0 is not helping
                f_passenger_bustransit = np.maximum(f_passenger_bustransit, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])
                f_car_pnr = np.maximum(f_car_pnr, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])
                if not fix_bus:
                    f_bus = np.maximum(f_bus, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])

                loss += tmp_loss / float(self.num_data)
                for loss_type, loss_value in tmp_loss_dict.items():
                    loss_dict[loss_type] += loss_value / float(self.num_data)
            
            print("Epoch:", starting_epoch + i, "Loss:", loss, self.print_separate_accuracy(loss_dict))
            loss_list.append([loss, loss_dict])

            if (best_epoch == 0) or (loss_list[best_epoch][0] > loss_list[-1][0]):
                best_epoch = starting_epoch + i
                best_f_car_driving, best_f_truck_driving, best_f_passenger_bustransit, best_f_car_pnr, best_f_bus = \
                    f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus

                best_x_e_car, best_x_e_truck, best_x_e_passenger, best_x_e_bus, best_tt_e_car, best_tt_e_truck, best_tt_e_passenger, best_tt_e_bus = \
                    x_e_car, x_e_truck, x_e_passenger, x_e_bus, tt_e_car, tt_e_truck, tt_e_passenger, tt_e_bus

                if save_folder is not None:
                    self.save_simulation_input_files(os.path.join(save_folder, 'input_files_estimate_path_flow'), 
                                                     f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus=None if fix_bus else f_bus,
                                                     explicit_bus=1, historical_bus_waiting_time=0)

            if save_folder is not None:
                pickle.dump([loss, loss_dict, loss_list, best_epoch,
                             f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus,
                             x_e_car, x_e_truck, x_e_passenger, x_e_bus, tt_e_car, tt_e_truck, tt_e_passenger, tt_e_bus], 
                            open(os.path.join(save_folder, str(starting_epoch + i) + '_iteration_estimate_path_flow.pickle'), 'wb'))

                # if column_generation[i]:
                #     self.save_simulation_input_files(os.path.join(save_folder, 'input_files_estimate_path_flow'), 
                #                                      f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus,
                #                                      explicit_bus=0, historical_bus_waiting_time=0)

            # if column_generation[i]:
            #     dta.build_link_cost_map()
            #     self.update_path_table(dta, use_tdsp)
            #     f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr = \
            #         self.update_path_flow(f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus,
            #                               car_driving_scale, truck_driving_scale, passenger_bustransit_scale, car_pnr_scale)

        print("Best loss at Epoch:", best_epoch, "Loss:", loss_list[best_epoch][0], self.print_separate_accuracy(loss_list[best_epoch][1]))
        return best_f_car_driving, best_f_truck_driving, best_f_passenger_bustransit, best_f_car_pnr, best_f_bus, \
               best_x_e_car, best_x_e_truck, best_x_e_passenger, best_x_e_bus, best_tt_e_car, best_tt_e_truck, best_tt_e_passenger, best_tt_e_bus, \
               loss_list

    def estimate_demand_pytorch(self, init_scale_passenger=10, init_scale_truck=10, init_scale_bus=1,
                                car_driving_scale=10, truck_driving_scale=1, passenger_bustransit_scale=1, car_pnr_scale=5,
                                passenger_step_size=0.1, truck_step_size=0.01, bus_step_size=0.01,
                                link_car_flow_weight=1, link_truck_flow_weight=1, link_passenger_flow_weight=1, link_bus_flow_weight=1,
                                link_car_tt_weight=1, link_truck_tt_weight=1, link_passenger_tt_weight=1, link_bus_tt_weight=1,
                                max_epoch=100, algo="NAdam", fix_bus=True, column_generation=False, use_tdsp=False,
                                alpha_mode=(1., 1.5, 2.), beta_mode=1, alpha_path=1, beta_path=1, 
                                use_file_as_init=None, save_folder=None, starting_epoch=0):
        
        if save_folder is not None and not os.path.exists(save_folder):
            os.mkdir(save_folder)

        if fix_bus:
            init_scale_bus = None
            bus_step_size = None
        else:
            assert(init_scale_bus is not None)
            assert(bus_step_size is not None)

        if np.isscalar(column_generation):
            column_generation = np.ones(max_epoch, dtype=int) * column_generation
        assert(len(column_generation) == max_epoch)

        if np.isscalar(link_car_flow_weight):
            link_car_flow_weight = np.ones(max_epoch, dtype=bool) * link_car_flow_weight
        assert(len(link_car_flow_weight) == max_epoch)

        if np.isscalar(link_truck_flow_weight):
            link_truck_flow_weight = np.ones(max_epoch, dtype=bool) * link_truck_flow_weight
        assert(len(link_truck_flow_weight) == max_epoch)

        if np.isscalar(link_passenger_flow_weight):
            link_passenger_flow_weight = np.ones(max_epoch, dtype=bool) * link_passenger_flow_weight
        assert(len(link_passenger_flow_weight) == max_epoch)

        if np.isscalar(link_bus_flow_weight):
            link_bus_flow_weight = np.ones(max_epoch, dtype=bool) * link_bus_flow_weight
        assert(len(link_bus_flow_weight) == max_epoch)

        if np.isscalar(link_car_tt_weight):
            link_car_tt_weight = np.ones(max_epoch, dtype=bool) * link_car_tt_weight
        assert(len(link_car_tt_weight) == max_epoch)

        if np.isscalar(link_truck_tt_weight):
            link_truck_tt_weight = np.ones(max_epoch, dtype=bool) * link_truck_tt_weight
        assert(len(link_truck_tt_weight) == max_epoch)

        if np.isscalar(link_passenger_tt_weight):
            link_passenger_tt_weight = np.ones(max_epoch, dtype=bool) * link_passenger_tt_weight
        assert(len(link_passenger_tt_weight) == max_epoch)

        if np.isscalar(link_bus_tt_weight):
            link_bus_tt_weight = np.ones(max_epoch, dtype=bool) * link_bus_tt_weight
        assert(len(link_bus_tt_weight) == max_epoch)
        
        loss_list = list()
        best_epoch = starting_epoch
        best_f_car_driving, best_f_truck_driving, best_f_passenger_bustransit, best_f_car_pnr, best_f_bus, best_q_e_passenger, best_q_e_truck = 0, 0, 0, 0, 0, 0, 0
        best_x_e_car, best_x_e_truck, best_x_e_passenger, best_x_e_bus, best_tt_e_car, best_tt_e_truck, best_tt_e_passenger, best_tt_e_bus = 0, 0, 0, 0, 0, 0, 0, 0
        # read from files as init values
        if use_file_as_init is not None:
            # most recent
            _, _, loss_list, best_epoch, _, _, _, _, _, \
                _, _, _, _, _, \
                _, _, _, _, _, _, _, _ = pickle.load(open(use_file_as_init, 'rb'))
            # best
            use_file_as_init = os.path.join(save_folder, '{}_iteration_estimate_demand.pickle'.format(best_epoch))
            _, _, _, _, best_q_e_passenger, best_q_e_truck, _, _, _, \
                best_f_car_driving, best_f_truck_driving, best_f_passenger_bustransit, best_f_car_pnr, best_f_bus, \
                best_x_e_car, best_x_e_truck, best_x_e_passenger, best_x_e_bus, best_tt_e_car, best_tt_e_truck, best_tt_e_passenger, best_tt_e_bus \
                    = pickle.load(open(use_file_as_init, 'rb'))

            q_e_passenger, q_e_truck = best_q_e_passenger, best_q_e_truck
            f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus = \
                best_f_car_driving, best_f_truck_driving, best_f_passenger_bustransit, best_f_car_pnr, best_f_bus
            
            self.nb.update_demand_path_driving(f_car_driving, f_truck_driving)
            self.nb.update_demand_path_bustransit(f_passenger_bustransit)
            self.nb.update_demand_path_pnr(f_car_pnr)
        else:
            # q_e: num_OD x num_assign_interval flattened in F order
            q_e_passenger = self.init_demand_flow(len(self.demand_list_total_passenger), init_scale=init_scale_passenger)
            # use truck demand
            q_e_truck = self.init_demand_flow(len(self.demand_list_truck_driving), init_scale=init_scale_truck)
            
            if not fix_bus:
                f_bus = self.init_demand_vector(self.num_assign_interval, self.num_path_busroute, init_scale_bus)

            # uniform
            self.init_mode_route_portions()
            
        # fixed bus path flow
        if fix_bus:
            f_bus = self.nb.demand_bus.path_flow_matrix.flatten(order='F')
        else:
            self.nb.update_demand_path_busroute(f_bus)

        # relu
        q_e_passenger = np.maximum(q_e_passenger, 1e-6)
        q_e_truck = np.maximum(q_e_truck, 1e-6)
        f_bus = np.maximum(f_bus, 1e-6)

        q_e_passenger_tensor = torch.from_numpy(q_e_passenger / np.maximum(init_scale_passenger, 1e-6))
        q_e_truck_tensor = torch.from_numpy(q_e_truck / np.maximum(init_scale_truck, 1e-6))
        if not fix_bus:
            f_bus_tensor = torch.from_numpy(f_bus / np.maximum(init_scale_bus, 1e-6))

        q_e_passenger_tensor.requires_grad = True
        q_e_truck_tensor.requires_grad = True
        if not fix_bus:
            f_bus_tensor.requires_grad = True

        params = [
            {'params': q_e_passenger_tensor, 'lr': passenger_step_size},
            {'params': q_e_truck_tensor, 'lr': truck_step_size}
        ]
        if not fix_bus:
            params.append({'params': f_bus_tensor, 'lr': bus_step_size})

        algo_dict = {
            "SGD": torch.optim.SGD,
            "NAdam": torch.optim.NAdam,
            "Adam": torch.optim.Adam,
            "Adamax": torch.optim.Adamax,
            "AdamW": torch.optim.AdamW,
            "RAdam": torch.optim.RAdam,
            "Adagrad": torch.optim.Adagrad,
            "Adadelta": torch.optim.Adadelta
        }
        optimizer = algo_dict[algo](params)

        for i in range(max_epoch):
            seq = np.random.permutation(self.num_data)
            loss = float(0)
            loss_dict = {
                "car_count_loss": 0.,
                "truck_count_loss": 0.,
                "bus_count_loss": 0.,
                "passenger_count_loss": 0.,
                "car_tt_loss": 0.,
                "truck_tt_loss": 0.,
                "bus_tt_loss": 0.,
                "passenger_tt_loss": 0.
            }

            self.config['link_car_flow_weight'] = link_car_flow_weight[i] * (self.config['use_car_link_flow'] or self.config['compute_car_link_flow_loss'])
            self.config['link_truck_flow_weight'] = link_truck_flow_weight[i] * (self.config['use_truck_link_flow'] or self.config['compute_truck_link_flow_loss'])
            self.config['link_passenger_flow_weight'] = link_passenger_flow_weight[i] * (self.config['use_passenger_link_flow'] or self.config['compute_passenger_link_flow_loss'])
            self.config['link_bus_flow_weight'] = link_bus_flow_weight[i] * (self.config['use_bus_link_flow'] or self.config['compute_bus_link_flow_loss'])

            self.config['link_car_tt_weight'] = link_car_tt_weight[i] * (self.config['use_car_link_tt'] or self.config['compute_car_link_tt_loss'])
            self.config['link_truck_tt_weight'] = link_truck_tt_weight[i] * (self.config['use_truck_link_tt'] or self.config['compute_truck_link_tt_loss'])
            self.config['link_passenger_tt_weight'] = link_passenger_tt_weight[i] * (self.config['use_passenger_link_tt'] or self.config['compute_passenger_link_tt_loss'])
            self.config['link_bus_tt_weight'] = link_bus_tt_weight[i] * (self.config['use_bus_link_tt'] or self.config['compute_bus_link_tt_loss'])
            

            for j in seq:
                # retrieve one record of observed data
                one_data_dict = self._get_one_data(j)

                # P_mode: (num_OD_one_mode * num_assign_interval, num_OD * num_assign_interval)
                P_mode_driving, P_mode_bustransit, P_mode_pnr = self.nb.get_mode_portion_matrix()

                # q_e_mode: num_OD_one_mode x num_assign_interval flattened in F order
                q_e_mode_driving = P_mode_driving.dot(q_e_passenger)
                q_e_mode_bustransit = P_mode_bustransit.dot(q_e_passenger)
                q_e_mode_pnr = P_mode_pnr.dot(q_e_passenger)

                q_e_mode_driving = np.maximum(q_e_mode_driving, 1e-6)
                q_e_mode_bustransit = np.maximum(q_e_mode_bustransit, 1e-6)
                q_e_mode_pnr = np.maximum(q_e_mode_pnr, 1e-6)

                # P_path: (num_path * num_assign_interval, num_OD_one_mode * num_assign_interval)
                P_path_car_driving, P_path_truck_driving = self.nb.get_route_portion_matrix_driving()
                P_path_passenger_bustransit = self.nb.get_route_portion_matrix_bustransit()
                P_path_car_pnr = self.nb.get_route_portion_matrix_pnr()

                # f_e: num_path x num_assign_interval flattened in F order
                f_car_driving = P_path_car_driving.dot(q_e_mode_driving)
                f_truck_driving = P_path_truck_driving.dot(q_e_truck)
                f_passenger_bustransit = P_path_passenger_bustransit.dot(q_e_mode_bustransit)
                f_car_pnr = P_path_car_pnr.dot(q_e_mode_pnr)

                f_car_driving = np.maximum(f_car_driving, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])
                f_truck_driving = np.maximum(f_truck_driving, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])
                # this alleviate trapping in local minima, when f = 0 -> grad = 0 is not helping
                f_passenger_bustransit = np.maximum(f_passenger_bustransit, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])
                f_car_pnr = np.maximum(f_car_pnr, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])
                f_bus = np.maximum(f_bus, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])
                
                # f_grad: num_path * num_assign_interval
                f_car_driving_grad, f_truck_driving_grad, f_passenger_bustransit_grad, f_car_pnr_grad, f_passenger_pnr_grad, f_bus_grad, \
                    tmp_loss, tmp_loss_dict, dta, x_e_car, x_e_truck, x_e_passenger, x_e_bus, tt_e_car, tt_e_truck, tt_e_passenger, tt_e_bus = \
                        self.compute_path_flow_grad_and_loss(one_data_dict, f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus, fix_bus=fix_bus, counter=0, run_mmdta_adaptive=False)
                
                # q_mode_grad: num_OD_one_mode * num_assign_interval
                q_grad_car_driving = P_path_car_driving.T.dot(f_car_driving_grad)  # link_car_flow_weight, link_car_tt_weight
                q_grad_car_pnr = P_path_car_pnr.T.dot(f_car_pnr_grad)  # link_car_flow_weight, link_car_tt_weight
                q_truck_grad = P_path_truck_driving.T.dot(f_truck_driving_grad)  # link_truck_flow_weight, link_truck_tt_weight, link_bus_tt_weight
                q_grad_passenger_bustransit = P_path_passenger_bustransit.T.dot(f_passenger_bustransit_grad)  # link_passenger_flow_weight, link_bus_tt_weight
                q_grad_passenger_pnr = P_path_car_pnr.T.dot(f_passenger_pnr_grad)  # link_passenger_flow_weight, link_bus_tt_weight

                # q_grad: num_OD * num_assign_interval
                q_passenger_grad = P_mode_driving.T.dot(q_grad_car_driving) \
                                   + P_mode_bustransit.T.dot(q_grad_passenger_bustransit) \
                                   + P_mode_pnr.T.dot(q_grad_passenger_pnr + q_grad_car_pnr)
                
                optimizer.zero_grad()

                q_e_passenger_tensor.grad = torch.from_numpy(q_passenger_grad * init_scale_passenger)
                q_e_truck_tensor.grad = torch.from_numpy(q_truck_grad * init_scale_truck)
                if not fix_bus:
                    f_bus_tensor.grad = torch.from_numpy(f_bus_grad * init_scale_bus)

                optimizer.step()

                # release memory
                f_car_driving_grad, f_truck_driving_grad, f_passenger_bustransit_grad, f_car_pnr_grad, f_passenger_pnr_grad, f_bus_grad = 0, 0, 0, 0, 0, 0
                q_grad_car_driving, q_grad_car_pnr, q_truck_grad, q_grad_passenger_bustransit, q_grad_passenger_pnr, q_passenger_grad = 0, 0, 0, 0, 0, 0
                optimizer.zero_grad()

                q_e_passenger = q_e_passenger_tensor.data.cpu().numpy() * init_scale_passenger
                q_e_truck = q_e_truck_tensor.data.cpu().numpy() * init_scale_truck
                if not fix_bus:
                    f_bus = f_bus_tensor.data.cpu().numpy() * init_scale_bus

                # relu
                q_e_passenger = np.maximum(q_e_passenger, 1e-6)
                q_e_truck = np.maximum(q_e_truck, 1e-6)
                f_bus = np.maximum(f_bus, 1e-6)

                loss += tmp_loss / float(self.num_data)
                for loss_type, loss_value in tmp_loss_dict.items():
                    loss_dict[loss_type] += loss_value / float(self.num_data)

                if not self.config['use_car_link_tt'] and not self.config['use_truck_link_tt'] and not self.config['use_passenger_link_tt'] and not self.config['use_bus_link_tt']:
                    # if any of these is true, dta.build_link_cost_map(True) is already invoked in compute_path_flow_grad_and_loss()
                    dta.build_link_cost_map(False)

                self.compute_path_cost(dta)
                if column_generation[i]:
                    print("***************** generate new paths *****************")
                    self.update_path_table(dta, use_tdsp)
                    f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr = \
                        self.update_path_flow(f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr,
                                              car_driving_scale, truck_driving_scale, passenger_bustransit_scale, car_pnr_scale)
                    dta = 0
                # adjust modal split and path flow portion based on path cost and logit choice model
                self.assign_mode_route_portions(alpha_mode, beta_mode, alpha_path, beta_path)

            print("Epoch:", starting_epoch + i, "Loss:", loss, self.print_separate_accuracy(loss_dict))
            loss_list.append([loss, loss_dict])

            if (best_epoch == 0) or (loss_list[best_epoch][0] > loss_list[-1][0]):
                best_epoch = starting_epoch + i
                best_f_car_driving, best_f_truck_driving, best_f_passenger_bustransit, best_f_car_pnr, best_f_bus, best_q_e_passenger, best_q_e_truck = \
                    f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus, q_e_passenger, q_e_truck

                best_x_e_car, best_x_e_truck, best_x_e_passenger, best_x_e_bus, best_tt_e_car, best_tt_e_truck, best_tt_e_passenger, best_tt_e_bus = \
                    x_e_car, x_e_truck, x_e_passenger, x_e_bus, tt_e_car, tt_e_truck, tt_e_passenger, tt_e_bus

                if save_folder is not None:
                    self.save_simulation_input_files(os.path.join(save_folder, 'input_files_estimate_demand'), 
                                                     f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus=None if fix_bus else f_bus,
                                                     explicit_bus=1, historical_bus_waiting_time=0)

            if save_folder is not None:
                pickle.dump([loss, loss_dict, loss_list, best_epoch,
                             q_e_passenger, q_e_truck, 
                             q_e_mode_driving, q_e_mode_bustransit, q_e_mode_pnr,
                             f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus,
                             x_e_car, x_e_truck, x_e_passenger, x_e_bus, tt_e_car, tt_e_truck, tt_e_passenger, tt_e_bus], 
                            open(os.path.join(save_folder, str(starting_epoch + i) + '_iteration_estimate_demand.pickle'), 'wb'))

                # if column_generation[i]:
                #     self.save_simulation_input_files(os.path.join(save_folder, 'input_files_estimate_demand'), 
                #                                      f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus,
                #                                      explicit_bus=0, historical_bus_waiting_time=0)
        
        print("Best loss at Epoch:", best_epoch, "Loss:", loss_list[best_epoch][0], self.print_separate_accuracy(loss_list[best_epoch][1]))
        return best_f_car_driving, best_f_truck_driving, best_f_passenger_bustransit, best_f_car_pnr, best_f_bus, best_q_e_passenger, best_q_e_truck, \
               best_x_e_car, best_x_e_truck, best_x_e_passenger, best_x_e_bus, best_tt_e_car, best_tt_e_truck, best_tt_e_passenger, best_tt_e_bus, \
               loss_list

#*********************************************************************************************************************
    def update_demand_by_mode(self, q_e_truck, q_e_mode_driving, q_e_mode_bustransit, q_e_mode_pnr, random_init):
        nb = self.nb
        assign_interval = int(nb.config.config_dict['DTA']['max_interval'])
        if random_init:
            q_e_truck_init = np.ones(assign_interval) * q_e_truck
            q_e_mode_driving_init = np.ones(assign_interval) * q_e_mode_driving
            q_e_mode_bustransit_init = np.ones(assign_interval) * q_e_mode_bustransit
            q_e_mode_pnr_init = np.ones(assign_interval) * q_e_mode_pnr

            # route portions
            for path in nb.path_table_driving.ID2path.values():
                path.attach_route_choice_portions(np.ones(assign_interval))
                path.attach_route_choice_portions_truck(np.ones(assign_interval))
            for path in nb.path_table_bustransit.ID2path.values():
                path.attach_route_choice_portions_bustransit(np.ones(assign_interval))
            for path in nb.path_table_pnr.ID2path.values():
                path.attach_route_choice_portions_pnr(np.ones(assign_interval))

            # driving
            nb.demand_driving.demand_dict = dict()
            for O_node in nb.path_table_driving.path_dict.keys():
                O = nb.od.O_dict.inv[O_node]
                for D_node in nb.path_table_driving.path_dict[O_node].keys():
                    D = nb.od.D_dict.inv[D_node]
                    nb.path_table_driving.path_dict[O_node][D_node].normalize_route_portions(sum_to_OD = False)
                    nb.path_table_driving.path_dict[O_node][D_node].normalize_truck_route_portions(sum_to_OD = False)
                    nb.demand_driving.add_demand(O, D, q_e_mode_driving_init, q_e_truck_init, overwriting = True)

            # bustransit
            nb.demand_bustransit.demand_dict = dict()
            for O_node in nb.path_table_bustransit.path_dict.keys():
                O = nb.od.O_dict.inv[O_node]
                for D_node in nb.path_table_bustransit.path_dict[O_node].keys():
                    D = nb.od.D_dict.inv[D_node]
                    nb.path_table_bustransit.path_dict[O_node][D_node].normalize_route_portions(sum_to_OD = False)
                    nb.demand_bustransit.add_demand(O, D, q_e_mode_bustransit_init, overwriting = True)

            # pnr
            nb.demand_pnr.demand_dict = dict()
            for O_node in nb.path_table_pnr.path_dict.keys():
                O = nb.od.O_dict.inv[O_node]
                for D_node in nb.path_table_pnr.path_dict[O_node].keys():
                    D = nb.od.D_dict.inv[D_node]
                    nb.path_table_pnr.path_dict[O_node][D_node].normalize_route_portions(sum_to_OD = False)
                    nb.demand_pnr.add_demand(O, D, q_e_mode_pnr_init, overwriting = True)

            
        else:
            q_e_truck = q_e_truck.reshape((int(len(q_e_truck)/assign_interval), assign_interval), order='F')
            q_e_mode_driving = q_e_mode_driving.reshape((int(len(q_e_mode_driving)/assign_interval), assign_interval), order='F')
            q_e_mode_bustransit = q_e_mode_bustransit.reshape((int(len(q_e_mode_bustransit)/assign_interval), assign_interval), order='F')
            q_e_mode_pnr = q_e_mode_pnr.reshape((int(len(q_e_mode_pnr)/assign_interval), assign_interval), order='F')

            # driving
            nb.demand_driving.demand_dict = dict()
            i = 0
            for O_node in nb.path_table_driving.path_dict.keys():
                O = nb.od.O_dict.inv[O_node]
                for D_node in nb.path_table_driving.path_dict[O_node].keys():
                    D = nb.od.D_dict.inv[D_node]
                    nb.path_table_driving.path_dict[O_node][D_node].normalize_route_portions(sum_to_OD = False)
                    nb.path_table_driving.path_dict[O_node][D_node].normalize_truck_route_portions(sum_to_OD = False)
                    nb.demand_driving.add_demand(O, D, q_e_mode_driving[i,:], q_e_truck[i,:], overwriting = True)
                    i += 1

            # bustransit
            nb.demand_bustransit.demand_dict = dict()
            j = 0
            for O_node in nb.path_table_bustransit.path_dict.keys():
                O = nb.od.O_dict.inv[O_node]
                for D_node in nb.path_table_bustransit.path_dict[O_node].keys():
                    D = nb.od.D_dict.inv[D_node]
                    nb.path_table_bustransit.path_dict[O_node][D_node].normalize_route_portions(sum_to_OD = False)
                    nb.demand_bustransit.add_demand(O, D, q_e_mode_bustransit[j,:], overwriting = True)
                    j += 1

            # pnr
            nb.demand_pnr.demand_dict = dict()
            k = 0
            for O_node in nb.path_table_pnr.path_dict.keys():
                O = nb.od.O_dict.inv[O_node]
                for D_node in nb.path_table_pnr.path_dict[O_node].keys():
                    D = nb.od.D_dict.inv[D_node]
                    nb.path_table_pnr.path_dict[O_node][D_node].normalize_route_portions(sum_to_OD = False)
                    nb.demand_pnr.add_demand(O, D, q_e_mode_pnr[k,:], overwriting = True)
                    k += 1


    def estimate_demand_by_mode_pytorch(self, init_scale_bus=1,
                                    q_e_truck_scale = 0, q_e_mode_driving_scale=0.5, q_e_mode_bustransit_scale=0.2, q_e_mode_pnr_scale=0.2,
                                    car_driving_scale=1, truck_driving_scale=1, passenger_bustransit_scale=1, car_pnr_scale=5,
                                    driving_step_size=0.1, bustransit_step_size = 0.1, pnr_step_size = 0.1, truck_step_size=0.01, bus_step_size=0.01,
                                    gamma_truck = 0.9, gamma_driving = 0.9, gamma_bustransit = 0.9, gamma_pnr = 0.9,
                                    link_car_flow_weight=1, link_truck_flow_weight=1, link_passenger_flow_weight=1, link_bus_flow_weight=1,
                                    link_car_tt_weight=1, link_truck_tt_weight=1, link_passenger_tt_weight=1, link_bus_tt_weight=1, ODloss_weight=1, veh_run_boarding_alighting_weight=1,
                                    max_epoch=100, algo="NAdam", fix_bus=True, column_generation=False, use_tdsp=False, explicit_bus=1,
                                    # alpha_mode=(1., 1.5, 2.), beta_mode=1, alpha_path=1, beta_path=1, 
                                    use_file_as_init=None, save_folder=None, starting_epoch=0, random_init=True,
                                    q_driving_in_loss = None, q_truck_in_loss = None, q_bustransit_in_loss = None, q_pnr_in_loss = None,
                                    use_seperate_optimizer = True):
        
        if save_folder is not None and not os.path.exists(save_folder):
            os.mkdir(save_folder)

        if fix_bus:
            init_scale_bus = None
            bus_step_size = None
        else:
            assert(init_scale_bus is not None)
            assert(bus_step_size is not None)

        if np.isscalar(column_generation):
            column_generation = np.ones(max_epoch, dtype=int) * column_generation
        assert(len(column_generation) == max_epoch)

        if np.isscalar(link_car_flow_weight):
            link_car_flow_weight = np.ones(max_epoch, dtype=bool) * link_car_flow_weight
        assert(len(link_car_flow_weight) == max_epoch)

        if np.isscalar(link_truck_flow_weight):
            link_truck_flow_weight = np.ones(max_epoch, dtype=bool) * link_truck_flow_weight
        assert(len(link_truck_flow_weight) == max_epoch)

        if np.isscalar(link_passenger_flow_weight):
            link_passenger_flow_weight = np.ones(max_epoch, dtype=bool) * link_passenger_flow_weight
        assert(len(link_passenger_flow_weight) == max_epoch)

        if np.isscalar(link_bus_flow_weight):
            link_bus_flow_weight = np.ones(max_epoch, dtype=bool) * link_bus_flow_weight
        assert(len(link_bus_flow_weight) == max_epoch)

        if np.isscalar(link_car_tt_weight):
            link_car_tt_weight = np.ones(max_epoch, dtype=bool) * link_car_tt_weight
        assert(len(link_car_tt_weight) == max_epoch)

        if np.isscalar(link_truck_tt_weight):
            link_truck_tt_weight = np.ones(max_epoch, dtype=bool) * link_truck_tt_weight
        assert(len(link_truck_tt_weight) == max_epoch)

        if np.isscalar(link_passenger_tt_weight):
            link_passenger_tt_weight = np.ones(max_epoch, dtype=bool) * link_passenger_tt_weight
        assert(len(link_passenger_tt_weight) == max_epoch)

        if np.isscalar(link_bus_tt_weight):
            link_bus_tt_weight = np.ones(max_epoch, dtype=bool) * link_bus_tt_weight
        assert(len(link_bus_tt_weight) == max_epoch)

        if np.isscalar(ODloss_weight):
            ODloss_weight = np.ones(max_epoch, dtype=bool) * ODloss_weight
        assert(len(ODloss_weight) == max_epoch)

        if np.isscalar(veh_run_boarding_alighting_weight):
            veh_run_boarding_alighting_weight = np.ones(max_epoch, dtype=bool) * veh_run_boarding_alighting_weight
        assert(len(veh_run_boarding_alighting_weight) == max_epoch)
        
        loss_list = list()
        best_epoch = starting_epoch
        best_q_e_truck, best_f_bus = 0, 0
        best_q_e_mode_driving, best_q_e_mode_bustransit, best_q_e_mode_pnr = 0, 0, 0
        best_x_e_car, best_x_e_truck, best_x_e_passenger, best_x_e_bus, best_x_e_BoardingAlighting_count, best_tt_e_car, best_tt_e_truck, best_tt_e_passenger, best_tt_e_bus = 0, 0, 0, 0, 0, 0, 0, 0, 0
        # read from files as init values
        if use_file_as_init is not None:
            # most recent
            _, _, loss_list, best_epoch,\
                _, \
                _, _, _, \
                _, _, _, _, _, \
                _, _, _ , _, _, _, _, _, _ , _= pickle.load(open(use_file_as_init, 'rb'))
            # best
            use_file_as_init = os.path.join(save_folder, '{}_iteration_estimate_demand.pickle'.format(best_epoch))
            _, _, _, _, best_q_e_truck, best_q_e_mode_driving, best_q_e_mode_bustransit, best_q_e_mode_pnr, \
                _, _, _, _, best_f_bus, \
                best_x_e_car, best_x_e_truck, best_x_e_passenger, best_x_e_bus, best_x_e_BoardingAlighting_count, best_tt_e_car, best_tt_e_truck, best_tt_e_passenger, best_tt_e_bus,\
                     best_stop_arrival_departure_travel_time_df \
                    = pickle.load(open(use_file_as_init, 'rb'))

            q_e_truck = best_q_e_truck
            q_e_mode_driving, q_e_mode_bustransit, q_e_mode_pnr = best_q_e_mode_driving, best_q_e_mode_bustransit, best_q_e_mode_pnr
            f_bus = best_f_bus
            
            # update demand and route portions according to the original network builder process 
            self.update_demand_by_mode(q_e_truck, q_e_mode_driving, q_e_mode_bustransit, q_e_mode_pnr, False)
            self.nb.get_mode_portion_matrix()
                
        else:
            if random_init:
                self.update_demand_by_mode(q_e_truck_scale, q_e_mode_driving_scale, q_e_mode_bustransit_scale, q_e_mode_pnr_scale, True)
                # q_e: num_OD x num_assign_interval flattened in F order
                q_e_truck = np.ones(len(self.demand_list_truck_driving)* self.num_assign_interval) * q_e_truck_scale
                q_e_mode_driving = np.ones(self.nb.config.config_dict['DTA']['OD_pair_driving'] * self.num_assign_interval) * q_e_mode_driving_scale
                q_e_mode_bustransit = np.ones(self.nb.config.config_dict['DTA']['OD_pair_bustransit'] * self.num_assign_interval) * q_e_mode_bustransit_scale
                q_e_mode_pnr = np.ones(self.nb.config.config_dict['DTA']['OD_pair_pnr'] * self.num_assign_interval) * q_e_mode_pnr_scale
                self.nb.get_mode_portion_matrix()
                if not fix_bus:
                    f_bus = self.init_demand_vector(self.num_assign_interval, self.num_path_busroute, init_scale_bus)

            else: # use input files as init -  demand and route portions are already initialized in the network builder
                self.nb.get_mode_portion_matrix() # make the total demand consistent
                # get q 
                # driving
                q_e_mode_driving = np.empty((0, self.num_assign_interval)) 
                q_e_truck = np.empty((0, self.num_assign_interval)) 
                for O_node in self.nb.path_table_driving.path_dict.keys():
                    O = self.nb.od.O_dict.inv[O_node]
                    for D_node in self.nb.path_table_driving.path_dict[O_node].keys():
                        D = self.nb.od.D_dict.inv[D_node]
                        self.nb.path_table_driving.path_dict[O_node][D_node].normalize_route_portions(sum_to_OD = False)
                        self.nb.path_table_driving.path_dict[O_node][D_node].normalize_truck_route_portions(sum_to_OD = False)
                        car_demand = self.nb.demand_driving.demand_dict[O][D][0]
                        truck_demand = self.nb.demand_driving.demand_dict[O][D][1]
                        q_e_mode_driving = np.vstack((q_e_mode_driving, car_demand))
                        q_e_truck = np.vstack((q_e_truck, truck_demand))
                # bustransit
                q_e_mode_bustransit = np.empty((0, self.num_assign_interval)) 
                for O_node in self.nb.path_table_bustransit.path_dict.keys():
                    O = self.nb.od.O_dict.inv[O_node]
                    for D_node in self.nb.path_table_bustransit.path_dict[O_node].keys():
                        D = self.nb.od.D_dict.inv[D_node]
                        self.nb.path_table_bustransit.path_dict[O_node][D_node].normalize_route_portions(sum_to_OD = False)
                        demand = self.nb.demand_bustransit.demand_dict[O][D]
                        q_e_mode_bustransit = np.vstack((q_e_mode_bustransit, demand))
                # pnr
                q_e_mode_pnr = np.empty((0, self.num_assign_interval)) 
                for O_node in self.nb.path_table_pnr.path_dict.keys():
                    O = self.nb.od.O_dict.inv[O_node]
                    for D_node in self.nb.path_table_pnr.path_dict[O_node].keys():
                        D = self.nb.od.D_dict.inv[D_node]
                        self.nb.path_table_pnr.path_dict[O_node][D_node].normalize_route_portions(sum_to_OD = False)
                        demand = self.nb.demand_pnr.demand_dict[O][D]
                        q_e_mode_pnr = np.vstack((q_e_mode_pnr, demand))
                # flatten
                q_e_mode_driving = q_e_mode_driving.flatten(order='F')
                q_e_truck = q_e_truck.flatten(order='F')
                q_e_mode_bustransit = q_e_mode_bustransit.flatten(order='F')
                q_e_mode_pnr = q_e_mode_pnr.flatten(order='F')

                if not fix_bus:
                    f_bus = self.nb.demand_bus.path_flow_matrix.flatten(order='F')


        # fixed bus path flow
        if fix_bus:
            f_bus = self.nb.demand_bus.path_flow_matrix.flatten(order='F')
        else:
            self.nb.update_demand_path_busroute(f_bus)

        # relu
        q_e_truck = np.maximum(q_e_truck, 1e-6)
        q_e_mode_driving = np.maximum(q_e_mode_driving, 1e-6)
        q_e_mode_bustransit = np.maximum(q_e_mode_bustransit, 1e-6)
        q_e_mode_pnr = np.maximum(q_e_mode_pnr, 1e-6)
        f_bus = np.maximum(f_bus, 1e-6)

        q_e_truck_tensor = torch.from_numpy(q_e_truck)
        q_e_mode_driving_tensor = torch.from_numpy(q_e_mode_driving)
        q_e_mode_bustransit_tensor = torch.from_numpy(q_e_mode_bustransit)
        q_e_mode_pnr_tensor = torch.from_numpy(q_e_mode_pnr)

        if not fix_bus:
            f_bus_tensor = torch.from_numpy(f_bus)

        q_e_truck_tensor.requires_grad = True
        q_e_mode_driving_tensor.requires_grad = True
        q_e_mode_bustransit_tensor.requires_grad = True
        q_e_mode_pnr_tensor.requires_grad = True

        if not fix_bus:
            f_bus_tensor.requires_grad = True


        algo_dict = {
            "SGD": torch.optim.SGD,
            "NAdam": torch.optim.NAdam,
            "Adam": torch.optim.Adam,
            "Adamax": torch.optim.Adamax,
            "AdamW": torch.optim.AdamW,
            "RAdam": torch.optim.RAdam,
            "Adagrad": torch.optim.Adagrad,
            "Adadelta": torch.optim.Adadelta
        }

        if use_seperate_optimizer:
            optimizers = [
                algo_dict[algo]([{'params': q_e_truck_tensor}], lr=truck_step_size),
                algo_dict[algo]([{'params': q_e_mode_driving_tensor}], lr=driving_step_size),
                algo_dict[algo]([{'params': q_e_mode_bustransit_tensor}], lr=bustransit_step_size),
                algo_dict[algo]([{'params': q_e_mode_pnr_tensor}], lr=pnr_step_size)
            ]

            schedulers = [
                torch.optim.lr_scheduler.ExponentialLR(optimizers[0], gamma=gamma_truck),
                torch.optim.lr_scheduler.ExponentialLR(optimizers[1], gamma=gamma_driving),
                torch.optim.lr_scheduler.ExponentialLR(optimizers[2], gamma=gamma_bustransit),
                torch.optim.lr_scheduler.ExponentialLR(optimizers[3], gamma=gamma_pnr)
            ]

            if not fix_bus:
                optimizers.append(algo_dict[algo]([{'params': f_bus_tensor}], lr=bus_step_size))
                schedulers.append(torch.optim.lr_scheduler.ExponentialLR(optimizers[-1], gamma=0.9))
        else:
            params = [
                    {'params': q_e_truck_tensor, 'lr': truck_step_size},
                    {'params': q_e_mode_driving_tensor, 'lr': driving_step_size},
                    {'params': q_e_mode_bustransit_tensor, 'lr': bustransit_step_size},
                    {'params': q_e_mode_pnr_tensor, 'lr': pnr_step_size}
                        ]
            if not fix_bus:
                params.append({'params': f_bus_tensor, 'lr': bus_step_size})
            optimizers = [algo_dict[algo](params)]
            schedulers = [torch.optim.lr_scheduler.ExponentialLR(optimizers[0], gamma=gamma_driving)] # if use single optimizer, use gamma_driving for all

        for i in range(max_epoch):
            seq = np.random.permutation(self.num_data)
            loss = float(0)
            loss_dict = {
                "car_count_loss": 0.,
                "truck_count_loss": 0.,
                "bus_count_loss": 0.,
                "passenger_count_loss": 0.,
                "veh_run_boarding_alighting_loss": 0.,
                "car_tt_loss": 0.,
                "truck_tt_loss": 0.,
                "bus_tt_loss": 0.,
                "passenger_tt_loss": 0.,
                "car_count_loss_weighted": 0.,
                "truck_count_loss_weighted": 0.,
                "bus_count_loss_weighted": 0.,
                "passenger_count_loss_weighted": 0.,
                "car_tt_loss_weighted": 0.,
                "truck_tt_loss_weighted": 0.,
                "bus_tt_loss_weighted": 0.,
                "passenger_tt_loss_weighted": 0.,
                "veh_run_boarding_alighting_loss_weighted": 0.,
                "total_loss_weighted": 0.
            }

            self.config['link_car_flow_weight'] = link_car_flow_weight[i] * (self.config['use_car_link_flow'] or self.config['compute_car_link_flow_loss'])
            self.config['link_truck_flow_weight'] = link_truck_flow_weight[i] * (self.config['use_truck_link_flow'] or self.config['compute_truck_link_flow_loss'])
            self.config['link_passenger_flow_weight'] = link_passenger_flow_weight[i] * (self.config['use_passenger_link_flow'] or self.config['compute_passenger_link_flow_loss'])
            self.config['link_bus_flow_weight'] = link_bus_flow_weight[i] * (self.config['use_bus_link_flow'] or self.config['compute_bus_link_flow_loss'])
            self.config['veh_run_boarding_alighting_weight'] = veh_run_boarding_alighting_weight[i] * (self.config['use_veh_run_boarding_alighting'] or self.config['compute_veh_run_boarding_alighting_loss'])

            self.config['link_car_tt_weight'] = link_car_tt_weight[i] * (self.config['use_car_link_tt'] or self.config['compute_car_link_tt_loss'])
            self.config['link_truck_tt_weight'] = link_truck_tt_weight[i] * (self.config['use_truck_link_tt'] or self.config['compute_truck_link_tt_loss'])
            self.config['link_passenger_tt_weight'] = link_passenger_tt_weight[i] * (self.config['use_passenger_link_tt'] or self.config['compute_passenger_link_tt_loss'])
            self.config['link_bus_tt_weight'] = link_bus_tt_weight[i] * (self.config['use_bus_link_tt'] or self.config['compute_bus_link_tt_loss'])
            

            for j in seq:  # TODO: not update for each data record: update after all data records
                # retrieve one record of observed data
                one_data_dict = self._get_one_data(j)

                # P_path: (num_path * num_assign_interval, num_OD_one_mode * num_assign_interval)
                P_path_car_driving, P_path_truck_driving = self.nb.get_route_portion_matrix_driving()
                P_path_passenger_bustransit = self.nb.get_route_portion_matrix_bustransit()
                P_path_car_pnr = self.nb.get_route_portion_matrix_pnr()

                # f_e: num_path x num_assign_interval flattened in F order
                f_car_driving = P_path_car_driving.dot(q_e_mode_driving)
                f_truck_driving = P_path_truck_driving.dot(q_e_truck)
                f_passenger_bustransit = P_path_passenger_bustransit.dot(q_e_mode_bustransit)
                f_car_pnr = P_path_car_pnr.dot(q_e_mode_pnr)

                f_car_driving = np.maximum(f_car_driving, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])
                f_truck_driving = np.maximum(f_truck_driving, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])
                # this alleviate trapping in local minima, when f = 0 -> grad = 0 is not helping
                f_passenger_bustransit = np.maximum(f_passenger_bustransit, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])
                f_car_pnr = np.maximum(f_car_pnr, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])
                f_bus = np.maximum(f_bus, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])

                if self.config['use_ULP_f_transit'] and i > 0 and f_transit_ULP is not None:
                    f_passenger_bustransit = f_transit_ULP
                
                if loss_list:
                    init_loss = loss_list[0][1]
                else:
                    init_loss = None
                # f_grad: num_path * num_assign_interval
                f_car_driving_grad, f_truck_driving_grad, f_passenger_bustransit_grad, f_car_pnr_grad, f_passenger_pnr_grad, f_bus_grad, \
                    tmp_loss, tmp_loss_dict, dta, x_e_car, x_e_truck, x_e_passenger, x_e_bus, x_e_BoardingAlighting_count, tt_e_car, tt_e_truck, tt_e_passenger, tt_e_bus,\
                        f_transit_ULP, stop_arrival_departure_travel_time_df, _ = \
                        self.compute_path_flow_grad_and_loss(one_data_dict, f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus = None if fix_bus else f_bus, 
                        fix_bus=fix_bus, counter=0, run_mmdta_adaptive=False, init_loss = init_loss, explicit_bus=explicit_bus, isUsingbymode = False)
                
                # q_mode_grad: num_OD_one_mode * num_assign_interval
                q_grad_car_driving = P_path_car_driving.T.dot(f_car_driving_grad)  # link_car_flow_weight, link_car_tt_weight
                q_grad_car_pnr = P_path_car_pnr.T.dot(f_car_pnr_grad)  # link_car_flow_weight, link_car_tt_weight
                q_truck_grad = P_path_truck_driving.T.dot(f_truck_driving_grad)  # link_truck_flow_weight, link_truck_tt_weight, link_bus_tt_weight
                q_grad_passenger_bustransit = P_path_passenger_bustransit.T.dot(f_passenger_bustransit_grad)  # link_passenger_flow_weight, link_bus_tt_weight
                q_grad_passenger_pnr = P_path_car_pnr.T.dot(f_passenger_pnr_grad)  # link_passenger_flow_weight, link_bus_tt_weight

                # use OD loss or not
                eps = 1e-8
                if self.config['use_OD_loss']:
                    q_grad_car_driving += (q_e_mode_driving - q_driving_in_loss) / (np.linalg.norm(q_e_mode_driving - q_driving_in_loss) + eps) * ODloss_weight
                    q_grad_car_pnr += (q_e_mode_pnr - q_pnr_in_loss) / (np.linalg.norm(q_e_mode_pnr - q_pnr_in_loss) + eps) * ODloss_weight
                    q_truck_grad += (q_e_truck - q_truck_in_loss) / (np.linalg.norm(q_e_truck - q_truck_in_loss) + eps) * ODloss_weight
                    q_grad_passenger_bustransit += (q_e_mode_bustransit - q_bustransit_in_loss) / (np.linalg.norm(q_e_mode_bustransit - q_bustransit_in_loss) + eps) * ODloss_weight
                    # q_grad_passenger_pnr += (q_e_mode_pnr - q_pnr_in_loss) / (np.linalg.norm(q_e_mode_pnr - q_pnr_in_loss) + eps)
                
                for optimizer in optimizers:
                    optimizer.zero_grad()

                q_e_truck_tensor.grad = torch.from_numpy(q_truck_grad)
                q_e_mode_driving_tensor.grad = torch.from_numpy(q_grad_car_driving)
                q_e_mode_bustransit_tensor.grad = torch.from_numpy(q_grad_passenger_bustransit)
                q_e_mode_pnr_tensor.grad = torch.from_numpy(q_grad_passenger_pnr + q_grad_car_pnr)


                if not fix_bus:
                    f_bus_tensor.grad = torch.from_numpy(f_bus_grad)

                for optimizer in optimizers:
                    optimizer.step()
                for scheduler in schedulers:
                    scheduler.step()

                # release memory
                f_car_driving_grad, f_truck_driving_grad, f_passenger_bustransit_grad, f_car_pnr_grad, f_passenger_pnr_grad, f_bus_grad = 0, 0, 0, 0, 0, 0
                q_grad_car_driving, q_grad_car_pnr, q_truck_grad, q_grad_passenger_bustransit, q_grad_passenger_pnr = 0, 0, 0, 0, 0
                
                for optimizer in optimizers:
                    optimizer.zero_grad()

                # q_e_passenger = q_e_passenger_tensor.data.cpu().numpy() * init_scale_passenger
                q_e_truck = q_e_truck_tensor.data.cpu().numpy()
                q_e_mode_driving = q_e_mode_driving_tensor.data.cpu().numpy()
                q_e_mode_bustransit = q_e_mode_bustransit_tensor.data.cpu().numpy()
                q_e_mode_pnr = q_e_mode_pnr_tensor.data.cpu().numpy()

                if not fix_bus:
                    f_bus = f_bus_tensor.data.cpu().numpy()

                # note: here assume no pnr
                if self.config['use_ULP_f_transit'] and f_transit_ULP is not None:
                    f_passenger_bustransit = f_transit_ULP
                    self.nb.update_demand_path_bustransit(f_passenger_bustransit)
                    q_e_mode_bustransit = np.empty((0, self.num_assign_interval)) 
                    for O_node in self.nb.path_table_bustransit.path_dict.keys():
                        O = self.nb.od.O_dict.inv[O_node]
                        for D_node in self.nb.path_table_bustransit.path_dict[O_node].keys():
                            D = self.nb.od.D_dict.inv[D_node]
                            demand = self.nb.demand_bustransit.demand_dict[O][D]
                            q_e_mode_bustransit = np.vstack((q_e_mode_bustransit, demand))
                    q_e_mode_bustransit = q_e_mode_bustransit.flatten(order='F')
                    
                # relu
                # q_e_passenger = np.maximum(q_e_passenger, 1e-6)
                q_e_truck = np.maximum(q_e_truck, 1e-6)
                q_e_mode_driving = np.maximum(q_e_mode_driving, 1e-6)
                q_e_mode_bustransit = np.maximum(q_e_mode_bustransit, 1e-6)
                q_e_mode_pnr = np.maximum(q_e_mode_pnr, 1e-6)
                f_bus = np.maximum(f_bus, 1e-6)

                loss += tmp_loss / float(self.num_data)
                for loss_type, loss_value in tmp_loss_dict.items():
                    loss_dict[loss_type] += loss_value / float(self.num_data)

                if not self.config['use_car_link_tt'] and not self.config['use_truck_link_tt'] and not self.config['use_passenger_link_tt'] and not self.config['use_bus_link_tt']:
                    # if any of these is true, dta.build_link_cost_map(True) is already invoked in compute_path_flow_grad_and_loss()
                    dta.build_link_cost_map(False)

                # self.compute_path_cost(dta)
                if column_generation[i]:
                    print("***************** generate new paths *****************")
                    self.update_path_table(dta, use_tdsp)
                    f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr = \
                        self.update_path_flow(f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr,
                                              car_driving_scale, truck_driving_scale, passenger_bustransit_scale, car_pnr_scale) # xm: there might be problems with these scales
                    dta = 0

                # update demand 
                self.update_demand_by_mode(q_e_truck, q_e_mode_driving, q_e_mode_bustransit, q_e_mode_pnr, False)
                self.nb.get_mode_portion_matrix() # update total passenger demand

            print("Epoch:", starting_epoch + i, "Loss:", loss, self.print_separate_accuracy(loss_dict))
            loss_list.append([loss, loss_dict]) # xm: if use_file_as_init, wouldn't this be starting overwriting from the best_epoch?

            if (best_epoch == 0) or (loss_list[best_epoch][0] > loss_list[-1][0]):
                best_epoch = starting_epoch + i
                best_f_car_driving, best_f_truck_driving, best_f_passenger_bustransit, best_f_car_pnr, best_f_bus, best_q_e_truck = \
                    f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus, q_e_truck
                
                best_q_e_mode_driving, best_q_e_mode_bustransit, best_q_e_mode_pnr = q_e_mode_driving, q_e_mode_bustransit, q_e_mode_pnr

                best_x_e_car, best_x_e_truck, best_x_e_passenger, best_x_e_bus, best_x_e_BoardingAlighting_count, best_tt_e_car, best_tt_e_truck, best_tt_e_passenger, best_tt_e_bus,\
                     best_stop_arrival_departure_travel_time_df = \
                    x_e_car, x_e_truck, x_e_passenger, x_e_bus, x_e_BoardingAlighting_count, tt_e_car, tt_e_truck, tt_e_passenger, tt_e_bus, stop_arrival_departure_travel_time_df

                if save_folder is not None:
                    self.save_simulation_input_files(folder_path = os.path.join(save_folder, 'input_files_estimate_demand'), explicit_bus=explicit_bus, historical_bus_waiting_time=0)

            if save_folder is not None:
                pickle.dump([loss, loss_dict, loss_list, best_epoch,
                             q_e_truck, 
                             q_e_mode_driving, q_e_mode_bustransit, q_e_mode_pnr,
                             f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus,
                             x_e_car, x_e_truck, x_e_passenger, x_e_bus, x_e_BoardingAlighting_count, tt_e_car, tt_e_truck, tt_e_passenger, tt_e_bus, stop_arrival_departure_travel_time_df], 
                            open(os.path.join(save_folder, str(starting_epoch + i) + '_iteration_estimate_demand.pickle'), 'wb'))

                # if column_generation[i]:
                #     self.save_simulation_input_files(os.path.join(save_folder, 'input_files_estimate_demand'), 
                #                                      f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus,
                #                                      explicit_bus=explicit_bus, historical_bus_waiting_time=0)
        
        print("Best loss at Epoch:", best_epoch, "Loss:", loss_list[best_epoch][0], self.print_separate_accuracy(loss_list[best_epoch][1]))
        return best_f_car_driving, best_f_truck_driving, best_f_passenger_bustransit, best_f_car_pnr, best_f_bus, best_q_e_truck, \
                best_q_e_mode_driving, best_q_e_mode_bustransit, best_q_e_mode_pnr, best_x_e_car, best_x_e_truck, best_x_e_passenger, best_x_e_bus, best_x_e_BoardingAlighting_count, \
                best_tt_e_car, best_tt_e_truck, best_tt_e_passenger, best_tt_e_bus, best_stop_arrival_departure_travel_time_df, \
                loss_list
    
    def run_pure_simulation(self, f_car_driving, f_truck_driving, f_passenger_bustransit, f_pnr, counter =0, explicit_bus=1, historical_bus_waiting_time=0):

        hash1 = hashlib.sha1()
        hash1.update((str(time.time()) + str(counter)).encode('utf-8'))
        new_folder = str(hash1.hexdigest())

        self.save_simulation_input_files(folder_path = new_folder, f_car_driving=f_car_driving, f_truck_driving=f_truck_driving, 
                                            f_passenger_bustransit=f_passenger_bustransit, f_car_pnr=f_pnr, f_bus=None,
                                         explicit_bus=explicit_bus, historical_bus_waiting_time=historical_bus_waiting_time)

        a = macposts.mmdta_api()
        a.initialize(new_folder)

        link_driving_array = np.array([e.ID for e in self.nb.link_driving_list], dtype=int)
        a.register_links_driving(link_driving_array)
        link_bus_array = np.array([e.ID for e in self.nb.link_bus_list], dtype=int)
        a.register_links_bus(link_bus_array)
        link_walking_array = np.array([e.ID for e in self.nb.link_walking_list], dtype=int)
        a.register_links_walking(link_walking_array)

        a.register_paths(self.paths_list)
        a.register_paths_driving(self.paths_list_driving)
        a.register_paths_bustransit(self.paths_list_bustransit)
        a.register_paths_pnr(self.paths_list_pnr)
        a.register_paths_bus(self.paths_list_busroute)

        a.install_cc()
        a.install_cc_tree()
        a.run_whole(False)

        start_intervals = np.arange(self.num_loading_interval)

        car_link_tt = a.get_car_link_tt(start_intervals, False)
        assert(car_link_tt.shape[0] == len(link_driving_array))

        bus_link_tt = a.get_bus_link_tt(start_intervals, False, False)
        assert(bus_link_tt.shape[0] == len(link_bus_array))

        walking_link_tt = a.get_passenger_walking_link_tt(start_intervals)
        assert(walking_link_tt.shape[0] == len(link_walking_array))

        # change to be np.float32 to save memory
        car_link_tt = car_link_tt.astype(np.float32)
        bus_link_tt = bus_link_tt.astype(np.float32)
        walking_link_tt = walking_link_tt.astype(np.float32)

        shutil.rmtree(new_folder)
        a.delete_all_agents()

        return car_link_tt, bus_link_tt, walking_link_tt


    def compute_path_travel_waiting_walking_time(self, input_file_folder, f_car_driving, f_truck_driving, f_passenger_bustransit, f_pnr, use_robust = True):
        
        car_link_tt, bus_link_tt, walking_link_tt = self.run_pure_simulation(f_car_driving, f_truck_driving, f_passenger_bustransit, f_pnr)
        
        num_loading_interval = self.num_loading_interval
        unit_time = self.nb.config.config_dict['DTA']['unit_time']
        assign_frq = self.nb.config.config_dict['DTA']['assign_frq']
        num_assign_interval = self.num_assign_interval

        # replace huge values with the values of the previous interval - based on the setting of macposts
        car_link_tt = pd.DataFrame(car_link_tt)
        car_link_tt.replace(2*num_loading_interval*unit_time, np.nan, inplace=True)
        car_link_tt.ffill(axis=1, inplace=True)
        car_link_tt = car_link_tt.to_numpy()

        bus_link_tt = pd.DataFrame(bus_link_tt)
        bus_link_tt.replace(2*num_loading_interval*unit_time, np.nan, inplace=True)
        bus_link_tt.ffill(axis=1, inplace=True)
        bus_link_tt = bus_link_tt.to_numpy()
        bus_link_tt = np.nan_to_num(bus_link_tt, nan=num_loading_interval*unit_time) 

        # load dode_data.pickle
        with open(os.path.join(input_file_folder, 'dode_data.pickle'), 'rb') as f:
            [OD_pathset_list_driving, OD_pathset_list_transit, OD_pathset_list_pnr, car_link_tt_dict, bus_link_tt_dict, walking_link_tt_dict, transit_link_dict, \
            _, _, _] = pickle.load(f)

        for i in range(len(car_link_tt)):
            link_id = list(car_link_tt_dict.keys())[i]
            car_link_tt_dict[link_id] = car_link_tt[i]
        for i in range(len(bus_link_tt)):
            link_id = list(bus_link_tt_dict.keys())[i]
            bus_link_tt_dict[link_id] = bus_link_tt[i]
        for i in range(len(walking_link_tt)):
            link_id = list(walking_link_tt_dict.keys())[i]
            walking_link_tt_dict[link_id] = walking_link_tt[i]

        # compute path travel time, walking time and waiting time: by 5-s (loading intervel) first and then convert to assign-interval-based given use_robust
        OD_path_tt_list_driving = copy.deepcopy(OD_pathset_list_driving)
        OD_path_alltts_list_transit = copy.deepcopy(OD_pathset_list_transit)
        OD_path_alltts_list_pnr = copy.deepcopy(OD_pathset_list_pnr)

        for O in OD_pathset_list_driving.keys():
            for D in OD_pathset_list_driving[O].keys():
                # driving
                OD_path_tt_list_driving[O][D] = []
                for path in OD_pathset_list_driving[O][D]:
                    path_tt = []
                    for loading_interval in range(num_loading_interval):
                        arrival_time = loading_interval
                        finish = False
                        for link in path:
                            if arrival_time < num_loading_interval:
                                arrival_time += int(car_link_tt_dict[link][arrival_time] / unit_time)
                            else:
                                path_tt.append(np.inf)
                                finish = True
                                break
                        if not finish:
                            path_tt.append((arrival_time - loading_interval) * unit_time)
                    path_tt = np.array(path_tt)
                    if np.isinf(path_tt[0]):  # if the first interval is inf, then all the tts for all the loading intervals are large numbers
                        path_bus_tt =np.ones(num_loading_interval) * unit_time * num_loading_interval
                    elif np.isinf(path_tt).any():
                        for i in range(len(path_tt)):
                            if np.isinf(path_tt[i]):
                                path_tt[i] = path_tt[i-1]
                    # if np.isinf(path_tt[-1]):
                    #     last_valid = path_tt[np.isfinite(path_tt)][-1]
                    #     for i in range(len(path_tt)-1, -1, -1):
                    #         if np.isinf(path_tt[i]):
                    #             path_tt[i] = last_valid
                    #         else:
                    #             break
                    if use_robust:  # suggest use robust when init_demand_split = 1 in the config
                        # based on the rationale in macposts, the release interval if init_demand_split = 1 is 1 min (12 5-s intervals)
                        path_tt = path_tt[::12] #  pick the first 5-s interval of each minute, because in that 1 min, the demand is released at the first 5-s interval
                        num_mins_in_assign_interval = int(assign_frq * unit_time/60)
                        # compute the average travel time in each assign interval (among the 15 minutes)
                        path_tt = path_tt.reshape(int(len(path_tt)/num_mins_in_assign_interval), num_mins_in_assign_interval).mean(axis=1)
                        assert(len(path_tt) == num_assign_interval)
                    else:
                        # just pick the value of the first 5-s interval of each assignment interval
                        path_tt = path_tt[::int(assign_frq)]
                        assert(len(path_tt) == num_assign_interval)

                    OD_path_tt_list_driving[O][D].append(path_tt)

                # transit
                if O in OD_pathset_list_transit:
                    if D in OD_pathset_list_transit[O]:
                        OD_path_alltts_list_transit[O][D] = []
                        for path in OD_pathset_list_transit[O][D]:  
                            path_tt, path_bus_tt, path_metro_tt, path_bus_waiting_tt, path_metro_waiting_tt, path_walking_tt = [], [], [], [], [], []
                            for loading_interval in range(num_loading_interval):
                                arrival_time = loading_interval
                                bus_tt, metro_tt, bus_waiting_tt, metro_waiting_tt, walking_tt = 0, 0, 0, 0, 0
                                finish = False
                                for link in path:
                                    if arrival_time < num_loading_interval:
                                        if transit_link_dict[link] == 'bus':
                                            bus_tt += bus_link_tt_dict[link][arrival_time]
                                            arrival_time += int(bus_link_tt_dict[link][arrival_time] / unit_time)
                                        elif transit_link_dict[link] == 'metro':
                                            metro_tt += bus_link_tt_dict[link][arrival_time]
                                            arrival_time += int(bus_link_tt_dict[link][arrival_time] / unit_time)
                                        elif transit_link_dict[link] == 'bus_boarding':
                                            bus_waiting_tt += int(walking_link_tt_dict[link][arrival_time] / unit_time) * unit_time
                                            arrival_time += int(walking_link_tt_dict[link][arrival_time] / unit_time)
                                        elif transit_link_dict[link] == 'metro_boarding':
                                            metro_waiting_tt += int(walking_link_tt_dict[link][arrival_time] / unit_time) * unit_time
                                            arrival_time += int(walking_link_tt_dict[link][arrival_time] / unit_time)
                                        elif transit_link_dict[link] == 'walking':
                                            walking_tt += max(int(walking_link_tt_dict[link][arrival_time]/unit_time) , 1) * unit_time
                                            arrival_time += max(int(walking_link_tt_dict[link][arrival_time] / unit_time) , 1)
                                    else:
                                        path_tt.append(np.inf)
                                        path_bus_tt.append(np.inf)
                                        path_metro_tt.append(np.inf)
                                        path_bus_waiting_tt.append(np.inf)
                                        path_metro_waiting_tt.append(np.inf)
                                        path_walking_tt.append(np.inf)
                                        finish = True
                                        break
                                if not finish:
                                    path_tt.append((arrival_time - loading_interval) * unit_time)
                                    path_bus_tt.append(bus_tt)
                                    path_metro_tt.append(metro_tt)
                                    path_bus_waiting_tt.append(bus_waiting_tt)
                                    path_metro_waiting_tt.append(metro_waiting_tt)
                                    path_walking_tt.append(walking_tt)
                            path_tt = np.array(path_tt)
                            path_bus_tt = np.array(path_bus_tt)
                            path_metro_tt = np.array(path_metro_tt)
                            path_bus_waiting_tt = np.array(path_bus_waiting_tt)
                            path_metro_waiting_tt = np.array(path_metro_waiting_tt)
                            path_walking_tt = np.array(path_walking_tt)
                            if np.isinf(path_tt[0]):  # if the first interval is inf, then all the tts for all the loading intervals are large numbers
                                path_bus_tt =np.ones(num_loading_interval) * unit_time * num_loading_interval
                                path_metro_tt = np.ones(num_loading_interval) * unit_time * num_loading_interval
                                path_bus_waiting_tt = np.ones(num_loading_interval) * unit_time * num_loading_interval
                                path_metro_waiting_tt = np.ones(num_loading_interval) * unit_time * num_loading_interval
                                path_walking_tt = np.ones(num_loading_interval) * unit_time * num_loading_interval
                                path_tt = np.ones(num_loading_interval) * unit_time * num_loading_interval
                            # if np.isinf(path_tt[-1]):
                            #     last_valid = path_tt[np.isfinite(path_tt)][-1]
                            #     last_valid_bus_tt = path_bus_tt[np.isfinite(path_bus_tt)][-1]
                            #     last_valid_metro_tt = path_metro_tt[np.isfinite(path_metro_tt)][-1]
                            #     last_valid_bus_waiting_tt = path_bus_waiting_tt[np.isfinite(path_bus_waiting_tt)][-1]
                            #     last_valid_metro_waiting_tt = path_metro_waiting_tt[np.isfinite(path_metro_waiting_tt)][-1]
                            #     last_valid_walking_tt = path_walking_tt[np.isfinite(path_walking_tt)][-1]
                            #     for i in range(len(path_tt)-1, -1, -1):
                            #         if np.isinf(path_tt[i]):
                            #             path_tt[i] = last_valid
                            #             path_bus_tt[i] = last_valid_bus_tt
                            #             path_metro_tt[i] = last_valid_metro_tt
                            #             path_bus_waiting_tt[i] = last_valid_bus_waiting_tt
                            #             path_metro_waiting_tt[i] = last_valid_metro_waiting_tt
                            #             path_walking_tt[i] = last_valid_walking_tt
                            #         else:
                            #             break

                            elif np.isinf(path_tt).any():
                                for i in range(len(path_tt)):
                                    if np.isinf(path_tt[i]):
                                        path_tt[i] = path_tt[i-1]
                                        path_bus_tt[i] = path_bus_tt[i-1]
                                        path_metro_tt[i] = path_metro_tt[i-1]
                                        path_bus_waiting_tt[i] = path_bus_waiting_tt[i-1]
                                        path_metro_waiting_tt[i] = path_metro_waiting_tt[i-1]
                                        path_walking_tt[i] = path_walking_tt[i-1]
                            
                            if use_robust:
                                path_tt = path_tt[::12]
                                path_bus_tt = path_bus_tt[::12]
                                path_metro_tt = path_metro_tt[::12]
                                path_bus_waiting_tt = path_bus_waiting_tt[::12]
                                path_metro_waiting_tt = path_metro_waiting_tt[::12]
                                path_walking_tt = path_walking_tt[::12]
                                num_mins_in_assign_interval = int(assign_frq * unit_time/60)
                                path_tt = path_tt.reshape(int(len(path_tt)/num_mins_in_assign_interval), num_mins_in_assign_interval).mean(axis=1)
                                path_bus_tt = path_bus_tt.reshape(int(len(path_bus_tt)/num_mins_in_assign_interval), num_mins_in_assign_interval).mean(axis=1)
                                path_metro_tt = path_metro_tt.reshape(int(len(path_metro_tt)/num_mins_in_assign_interval), num_mins_in_assign_interval).mean(axis=1)
                                path_bus_waiting_tt = path_bus_waiting_tt.reshape(int(len(path_bus_waiting_tt)/num_mins_in_assign_interval), num_mins_in_assign_interval).mean(axis=1)
                                path_metro_waiting_tt = path_metro_waiting_tt.reshape(int(len(path_metro_waiting_tt)/num_mins_in_assign_interval), num_mins_in_assign_interval).mean(axis=1)
                                path_walking_tt = path_walking_tt.reshape(int(len(path_walking_tt)/num_mins_in_assign_interval), num_mins_in_assign_interval).mean(axis=1)
                                assert(len(path_tt) == num_assign_interval)
                                assert(len(path_bus_tt) == num_assign_interval)
                                assert(len(path_metro_tt) == num_assign_interval)
                                assert(len(path_bus_waiting_tt) == num_assign_interval)
                                assert(len(path_metro_waiting_tt) == num_assign_interval)
                                assert(len(path_walking_tt) == num_assign_interval)
                            else:
                                path_tt = path_tt[::int(assign_frq)]
                                path_bus_tt = path_bus_tt[::int(assign_frq)]
                                path_metro_tt = path_metro_tt[::int(assign_frq)]
                                path_bus_waiting_tt = path_bus_waiting_tt[::int(assign_frq)]
                                path_metro_waiting_tt = path_metro_waiting_tt[::int(assign_frq)]
                                path_walking_tt = path_walking_tt[::int(assign_frq)]
                                assert(len(path_tt) == num_assign_interval)
                                assert(len(path_bus_tt) == num_assign_interval)
                                assert(len(path_metro_tt) == num_assign_interval)
                                assert(len(path_bus_waiting_tt) == num_assign_interval)
                                assert(len(path_metro_waiting_tt) == num_assign_interval)
                                assert(len(path_walking_tt) == num_assign_interval)
                            tt_dict = {'path_tt': path_tt, 'bus': path_bus_tt, 'metro': path_metro_tt, \
                                    'bus_waiting': path_bus_waiting_tt, 'metro_waiting': path_metro_waiting_tt, 'walking': path_walking_tt}
                            for key, arr in tt_dict.items():
                                assert not np.isnan(arr).any(), f"NaN detected in array at key: {key}"
                            OD_path_alltts_list_transit[O][D].append(tt_dict)
                
                # pnr
                if O in OD_pathset_list_pnr:
                    if D in OD_pathset_list_pnr[O]:
                        OD_path_alltts_list_pnr[O][D] = []
                        for path in OD_pathset_list_pnr[O][D]:  
                            path_tt, path_car_tt, path_bus_tt, path_metro_tt, path_bus_waiting_tt, path_metro_waiting_tt, path_walking_tt = [], [], [], [], [], [], []
                            for loading_interval in range(num_loading_interval):
                                arrival_time = loading_interval
                                car_tt, bus_tt, metro_tt, bus_waiting_tt, metro_waiting_tt, walking_tt = 0, 0, 0, 0, 0, 0
                                finish = False
                                for link in path:
                                    if arrival_time < num_loading_interval:
                                        if link not in transit_link_dict:
                                            car_tt += int(car_link_tt_dict[link][arrival_time] / unit_time) * unit_time
                                            arrival_time += int(car_link_tt_dict[link][arrival_time] / unit_time)
                                        else:
                                            if transit_link_dict[link] == 'bus':
                                                bus_tt += bus_link_tt_dict[link][arrival_time]
                                                arrival_time += int(bus_link_tt_dict[link][arrival_time] / unit_time)
                                            elif transit_link_dict[link] == 'metro':
                                                metro_tt += bus_link_tt_dict[link][arrival_time]
                                                arrival_time += int(bus_link_tt_dict[link][arrival_time] / unit_time)
                                            elif transit_link_dict[link] == 'bus_boarding':
                                                bus_waiting_tt += int(walking_link_tt_dict[link][arrival_time] / unit_time) * unit_time
                                                arrival_time += int(walking_link_tt_dict[link][arrival_time] / unit_time)
                                            elif transit_link_dict[link] == 'metro_boarding':
                                                metro_waiting_tt += int(walking_link_tt_dict[link][arrival_time] / unit_time) * unit_time
                                                arrival_time += int(walking_link_tt_dict[link][arrival_time] / unit_time)
                                            elif transit_link_dict[link] == 'walking':
                                                walking_tt += max(int(walking_link_tt_dict[link][arrival_time] / unit_time) , 1) * unit_time
                                                arrival_time += max(int(walking_link_tt_dict[link][arrival_time] / unit_time) , 1)
                                            else: # parking type: tt out of parking lot includes cruising time so counted as car tt
                                                car_tt += max(int(walking_link_tt_dict[link][arrival_time] / unit_time) , 1) * unit_time
                                                arrival_time += max(int(walking_link_tt_dict[link][arrival_time] / unit_time) , 1)
                                    else:
                                        path_tt.append(np.inf)
                                        path_car_tt.append(np.inf)
                                        path_bus_tt.append(np.inf)
                                        path_metro_tt.append(np.inf)
                                        path_bus_waiting_tt.append(np.inf)
                                        path_metro_waiting_tt.append(np.inf)
                                        path_walking_tt.append(np.inf)
                                        finish = True
                                        break
                                if not finish:
                                    path_tt.append((arrival_time - loading_interval) * unit_time)
                                    path_car_tt.append(car_tt)
                                    path_bus_tt.append(bus_tt)
                                    path_metro_tt.append(metro_tt)
                                    path_bus_waiting_tt.append(bus_waiting_tt)
                                    path_metro_waiting_tt.append(metro_waiting_tt)
                                    path_walking_tt.append(walking_tt)
                            path_tt = np.array(path_tt)
                            path_car_tt = np.array(path_car_tt)
                            path_bus_tt = np.array(path_bus_tt)
                            path_metro_tt = np.array(path_metro_tt)
                            path_bus_waiting_tt = np.array(path_bus_waiting_tt)
                            path_metro_waiting_tt = np.array(path_metro_waiting_tt)
                            path_walking_tt = np.array(path_walking_tt)
                            if np.isinf(path_tt[0]):
                                path_car_tt =np.ones(num_loading_interval) * unit_time * num_loading_interval
                                path_bus_tt =np.ones(num_loading_interval) * unit_time * num_loading_interval
                                path_metro_tt = np.ones(num_loading_interval) * unit_time * num_loading_interval
                                path_bus_waiting_tt = np.ones(num_loading_interval) * unit_time * num_loading_interval
                                path_metro_waiting_tt = np.ones(num_loading_interval) * unit_time * num_loading_interval
                                path_walking_tt = np.ones(num_loading_interval) * unit_time * num_loading_interval
                                path_tt = np.ones(num_loading_interval) * unit_time * num_loading_interval
                            # if np.isinf(path_tt[-1]):
                            #     last_valid = path_tt[np.isfinite(path_tt)][-1]
                            #     last_valid_car_tt = path_car_tt[np.isfinite(path_car_tt)][-1]
                            #     last_valid_bus_tt = path_bus_tt[np.isfinite(path_bus_tt)][-1]
                            #     last_valid_metro_tt = path_metro_tt[np.isfinite(path_metro_tt)][-1]
                            #     last_valid_bus_waiting_tt = path_bus_waiting_tt[np.isfinite(path_bus_waiting_tt)][-1]
                            #     last_valid_metro_waiting_tt = path_metro_waiting_tt[np.isfinite(path_metro_waiting_tt)][-1]
                            #     last_valid_walking_tt = path_walking_tt[np.isfinite(path_walking_tt)][-1]
                            #     for i in range(len(path_tt)-1, -1, -1):
                            #         if np.isinf(path_tt[i]):
                            #             path_tt[i] = last_valid
                            #             path_car_tt[i] = last_valid_car_tt
                            #             path_bus_tt[i] = last_valid_bus_tt
                            #             path_metro_tt[i] = last_valid_metro_tt
                            #             path_bus_waiting_tt[i] = last_valid_bus_waiting_tt
                            #             path_metro_waiting_tt[i] = last_valid_metro_waiting_tt
                            #             path_walking_tt[i] = last_valid_walking_tt
                            #         else:
                            #             break

                            elif np.isinf(path_tt).any():
                                for i in range(len(path_tt)):
                                    if np.isinf(path_tt[i]):
                                        path_tt[i] = path_tt[i-1]
                                        path_car_tt[i] = path_car_tt[i-1]
                                        path_bus_tt[i] = path_bus_tt[i-1]
                                        path_metro_tt[i] = path_metro_tt[i-1]
                                        path_bus_waiting_tt[i] = path_bus_waiting_tt[i-1]
                                        path_metro_waiting_tt[i] = path_metro_waiting_tt[i-1]
                                        path_walking_tt[i] = path_walking_tt[i-1]

                            if use_robust:
                                path_tt = path_tt[::12]
                                path_car_tt = path_car_tt[::12]
                                path_bus_tt = path_bus_tt[::12]
                                path_metro_tt = path_metro_tt[::12]
                                path_bus_waiting_tt = path_bus_waiting_tt[::12]
                                path_metro_waiting_tt = path_metro_waiting_tt[::12]
                                path_walking_tt = path_walking_tt[::12]
                                num_mins_in_assign_interval = int(assign_frq * unit_time/60)
                                path_tt = path_tt.reshape(int(len(path_tt)/num_mins_in_assign_interval), num_mins_in_assign_interval).mean(axis=1)
                                path_car_tt = path_car_tt.reshape(int(len(path_car_tt)/num_mins_in_assign_interval), num_mins_in_assign_interval).mean(axis=1)
                                path_bus_tt = path_bus_tt.reshape(int(len(path_bus_tt)/num_mins_in_assign_interval), num_mins_in_assign_interval).mean(axis=1)
                                path_metro_tt = path_metro_tt.reshape(int(len(path_metro_tt)/num_mins_in_assign_interval), num_mins_in_assign_interval).mean(axis=1)
                                path_bus_waiting_tt = path_bus_waiting_tt.reshape(int(len(path_bus_waiting_tt)/num_mins_in_assign_interval), num_mins_in_assign_interval).mean(axis=1)
                                path_metro_waiting_tt = path_metro_waiting_tt.reshape(int(len(path_metro_waiting_tt)/num_mins_in_assign_interval), num_mins_in_assign_interval).mean(axis=1)
                                path_walking_tt = path_walking_tt.reshape(int(len(path_walking_tt)/num_mins_in_assign_interval), num_mins_in_assign_interval).mean(axis=1)
                                assert(len(path_tt) == num_assign_interval)
                                assert(len(path_car_tt) == num_assign_interval)
                                assert(len(path_bus_tt) == num_assign_interval)
                                assert(len(path_metro_tt) == num_assign_interval)
                                assert(len(path_bus_waiting_tt) == num_assign_interval)
                                assert(len(path_metro_waiting_tt) == num_assign_interval)
                                assert(len(path_walking_tt) == num_assign_interval)
                            else:
                                path_tt = path_tt[::int(assign_frq)]
                                path_car_tt = path_car_tt[::int(assign_frq)]
                                path_bus_tt = path_bus_tt[::int(assign_frq)]
                                path_metro_tt = path_metro_tt[::int(assign_frq)]
                                path_bus_waiting_tt = path_bus_waiting_tt[::int(assign_frq)]
                                path_metro_waiting_tt = path_metro_waiting_tt[::int(assign_frq)]
                                path_walking_tt = path_walking_tt[::int(assign_frq)]
                                assert(len(path_tt) == num_assign_interval)
                                assert(len(path_car_tt) == num_assign_interval)
                                assert(len(path_bus_tt) == num_assign_interval)
                                assert(len(path_metro_tt) == num_assign_interval)
                                assert(len(path_bus_waiting_tt) == num_assign_interval)
                                assert(len(path_metro_waiting_tt) == num_assign_interval)
                                assert(len(path_walking_tt) == num_assign_interval)
                            tt_dict = {'path_tt': path_tt, 'car': path_car_tt, 'bus': path_bus_tt, 'metro': path_metro_tt, \
                                    'bus_waiting': path_bus_waiting_tt, 'metro_waiting': path_metro_waiting_tt, 'walking': path_walking_tt}
                            for key, arr in tt_dict.items():
                                assert not np.isnan(arr).any(), f"NaN detected in array at key: {key}"
                            OD_path_alltts_list_pnr[O][D].append(tt_dict)
        return OD_path_tt_list_driving, OD_path_alltts_list_transit, OD_path_alltts_list_pnr                   
          
    def compute_path_cost_all(self, input_file_folder, OD_path_tt_list_driving, OD_path_alltts_list_transit, OD_path_alltts_list_pnr, beta_dict, alpha_dict, gamma_dict):
        
        bus_fare = self.nb.config.config_dict['MMDUE']['bus_fare']
        metro_fare =self.nb.config.config_dict['MMDUE']['metro_fare']

        # read median income and population density of OD pairs
        with open(os.path.join(input_file_folder, 'disutility_info_df.pickle'), 'rb') as f:
            disutility_info_df = pickle.load(f)
        # load dode_data.pickle
        with open(os.path.join(input_file_folder, 'dode_data.pickle'), 'rb') as f:
            [_, _, _, _, _, _, _, \
            _, OD_path_type_list_transit,OD_path_type_list_pnr] = pickle.load(f)

        OD_path_cost_list_driving = copy.deepcopy(OD_path_tt_list_driving)
        OD_path_cost_list_transit = copy.deepcopy(OD_path_alltts_list_transit)
        OD_path_cost_list_pnr = copy.deepcopy(OD_path_alltts_list_pnr)

        for O in OD_path_tt_list_driving.keys():
            for D in OD_path_tt_list_driving[O].keys():
                # driving
                OD_path_cost_list_driving[O][D] = []
                for path_tt in OD_path_tt_list_driving[O][D]:
                    path_cost = beta_dict['beta_tt_car'] * path_tt/60 + beta_dict['beta_money'] * disutility_info_df[disutility_info_df['D_id'] == D]['parking_fee'].values[0]\
                                + gamma_dict['gamma_income_car'] * disutility_info_df[disutility_info_df['O_id'] == O]['median_income'].values[0] \
                                + gamma_dict['gamma_Originpopden_car'] * disutility_info_df[disutility_info_df['O_id'] == O]['pop_den'].values[0] \
                                + gamma_dict['gamma_Destpopden_car'] * disutility_info_df[disutility_info_df['D_id'] == D]['pop_den'].values[0]
                    OD_path_cost_list_driving[O][D].append(path_cost)
                
                # transit
                if O in OD_path_alltts_list_transit:
                    if D in OD_path_alltts_list_transit[O]:
                        OD_path_cost_list_transit[O][D] = []
                        for alltts_dict in OD_path_alltts_list_transit[O][D]:
                            path_cost = beta_dict['beta_tt_bus'] * alltts_dict['bus'] /60 + beta_dict['beta_tt_metro'] * alltts_dict['metro'] /60 \
                                        + beta_dict['beta_waiting_bus'] * alltts_dict['bus_waiting'] /60 + beta_dict['beta_waiting_metro'] * alltts_dict['metro_waiting'] /60 \
                                        + beta_dict['beta_walking'] * alltts_dict['walking']/60
                            if OD_path_type_list_transit[O][D] == 'bus':
                                path_cost += beta_dict['beta_money'] * bus_fare + gamma_dict['gamma_income_bus'] * disutility_info_df[disutility_info_df['O_id'] == O]['median_income'].values[0] \
                                    + gamma_dict['gamma_Originpopden_bus'] * disutility_info_df[disutility_info_df['O_id'] == O]['pop_den'].values[0] \
                                    + gamma_dict['gamma_Destpopden_bus'] * disutility_info_df[disutility_info_df['D_id'] == D]['pop_den'].values[0] + alpha_dict['alpha_bus']
                            elif OD_path_type_list_transit[O][D] == 'metro':
                                path_cost += beta_dict['beta_money'] * metro_fare + gamma_dict['gamma_income_metro'] * disutility_info_df[disutility_info_df['O_id'] == O]['median_income'].values[0] \
                                    + gamma_dict['gamma_Originpopden_metro'] * disutility_info_df[disutility_info_df['O_id'] == O]['pop_den'].values[0] \
                                    + gamma_dict['gamma_Destpopden_metro'] * disutility_info_df[disutility_info_df['D_id'] == D]['pop_den'].values[0] + alpha_dict['alpha_metro']
                            else:
                                path_cost += beta_dict['beta_money'] * (bus_fare + metro_fare) + gamma_dict['gamma_income_busmetro'] * disutility_info_df[disutility_info_df['O_id'] == O]['median_income'].values[0] \
                                    + gamma_dict['gamma_Originpopden_busmetro'] * disutility_info_df[disutility_info_df['O_id'] == O]['pop_den'].values[0] \
                                    + gamma_dict['gamma_Destpopden_busmetro'] * disutility_info_df[disutility_info_df['D_id'] == D]['pop_den'].values[0] + alpha_dict['alpha_busmetro']
                            OD_path_cost_list_transit[O][D].append(path_cost)

                # pnr
                if O in OD_path_alltts_list_pnr:
                    if D in OD_path_alltts_list_pnr[O]:
                        OD_path_cost_list_pnr[O][D] = []
                        for alltts_dict in OD_path_alltts_list_pnr[O][D]:
                            path_cost = beta_dict['beta_tt_car'] * alltts_dict['car'] /60 + beta_dict['beta_tt_bus'] * alltts_dict['bus'] /60 + beta_dict['beta_tt_metro'] * alltts_dict['metro'] /60 \
                                        + beta_dict['beta_waiting_bus'] * alltts_dict['bus_waiting'] /60 + beta_dict['beta_waiting_metro'] * alltts_dict['metro_waiting'] /60 \
                                        + beta_dict['beta_walking'] * alltts_dict['walking']/60 + beta_dict['beta_money'] * disutility_info_df['pnr_parking_fee'].values[0]
                            # assume all the parking lots for pnr have the same parking fee
                            if OD_path_type_list_pnr[O][D] == 'car+bus':
                                path_cost += beta_dict['beta_money'] * bus_fare + gamma_dict['gamma_income_carbus'] * disutility_info_df[disutility_info_df['O_id'] == O]['median_income'].values[0] \
                                    + gamma_dict['gamma_Originpopden_carbus'] * disutility_info_df[disutility_info_df['O_id'] == O]['pop_den'].values[0] \
                                    + gamma_dict['gamma_Destpopden_carbus'] * disutility_info_df[disutility_info_df['D_id'] == D]['pop_den'].values[0] + alpha_dict['alpha_carbus']
                            elif OD_path_type_list_pnr[O][D] == 'car+metro':
                                path_cost += beta_dict['beta_money'] * metro_fare + gamma_dict['gamma_income_carmetro'] * disutility_info_df[disutility_info_df['O_id'] == O]['median_income'].values[0] \
                                    + gamma_dict['gamma_Originpopden_carmetro'] * disutility_info_df[disutility_info_df['O_id'] == O]['pop_den'].values[0] \
                                    + gamma_dict['gamma_Destpopden_carmetro'] * disutility_info_df[disutility_info_df['D_id'] == D]['pop_den'].values[0] + alpha_dict['alpha_carmetro']
                            else:
                                path_cost += beta_dict['beta_money'] * (bus_fare + metro_fare) + gamma_dict['gamma_income_carbusmetro'] * disutility_info_df[disutility_info_df['O_id'] == O]['median_income'].values[0] \
                                    + gamma_dict['gamma_Originpopden_carbusmetro'] * disutility_info_df[disutility_info_df['O_id'] == O]['pop_den'].values[0] \
                                    + gamma_dict['gamma_Destpopden_carbusmetro'] * disutility_info_df[disutility_info_df['D_id'] == D]['pop_den'].values[0] + alpha_dict['alpha_carbusmetro']
                            OD_path_cost_list_pnr[O][D].append(path_cost)

        return OD_path_cost_list_driving, OD_path_cost_list_transit, OD_path_cost_list_pnr
                        
                        
    def mode_route_choice_from_cost(self, input_file_folder, OD_path_cost_list_driving, OD_path_cost_list_transit, OD_path_cost_list_pnr, q_e_traveler, nested_type, theta_dict):
        
        '''theta_dict: for the 1st level: \
            theta_dict['theta_1_driving'], theta_dict['theta_1_transit'], theta_dict['theta_1_pnr'], theta_dict['theta_1_bus'], theta_dict['theta_1_metro'], \
            theta_dict['theta_1_combinedtransit'], theta_dict['theta_1_transit']\
            for the 2nd level: \
            theta_dict['theta_car'], theta_dict['theta_bus'], theta_dict['theta_metro'], theta_dict['theta_busmetro'], 
             theta_dict['theta_carbus'], theta_dict['theta_carmetro'], and theta_dict['theta_carbusmetro']'''
        
        q_e_traveler = q_e_traveler.reshape((int(len(q_e_traveler) / self.num_assign_interval), self.num_assign_interval), order='F')
        
        
        with open(os.path.join(input_file_folder, 'dode_data.pickle'), 'rb') as f:
            [_, _, _, _, _, _, _, \
            od_mode_connectivity, OD_path_type_list_transit,OD_path_type_list_pnr] = pickle.load(f)

        f_car_driving = []
        f_transit =[]
        f_pnr =[]

        k_probs_driving, k_probs_transit, k_probs_pnr = [], [], []
        gm_probs_driving, gm_probs_transit, gm_probs_pnr = [], [], []
        m_probs_driving, m_probs_transit, m_probs_pnr = [], [], []

        OD_index = -1
        for O in OD_path_cost_list_driving.keys():
            for D in OD_path_cost_list_driving[O].keys():
                OD_index += 1
                # driving is avaliable for all the OD pairs (assumed)
                pathcost_array_car = np.array(OD_path_cost_list_driving[O][D])
                exp_pathcost_array_car = np.maximum(np.exp((- 1/ theta_dict['theta_car']) * pathcost_array_car),1e-6)
                col_sum = np.sum(exp_pathcost_array_car, axis =0)
                path_prob_car = exp_pathcost_array_car / col_sum
                IV_car = np.log(col_sum)
                assert not np.isnan(path_prob_car).any()
                assert not np.isnan(IV_car).any()

                row = od_mode_connectivity[(od_mode_connectivity['OriginNodeID'] == O) & (od_mode_connectivity['DestNodeID'] == D)]
                if row['bus'].values[0] == 1:
                    path_indices_bus = [i for i, s in enumerate(OD_path_type_list_transit[O][D]) if s == 'bus']
                    pathcost_array_bus = np.array([OD_path_cost_list_transit[O][D][i] for i in path_indices_bus])
                    exp_pathcost_array_bus = np.maximum(np.exp((- 1/ theta_dict['theta_bus']) * pathcost_array_bus), 1e-6)
                    col_sum = np.sum(exp_pathcost_array_bus, axis =0)
                    path_prob_bus = exp_pathcost_array_bus / col_sum
                    IV_bus = np.log(col_sum)
                    has_IV_bus = 1
                    assert not np.isnan(pathcost_array_bus).any()
                    assert not np.isnan(exp_pathcost_array_bus).any()
                    assert not np.isnan(col_sum).any()
                    assert not np.isnan(path_prob_bus).any()
                    assert not np.isnan(IV_bus).any()
                else:
                    path_indices_bus = []
                    IV_bus = np.zeros(self.num_assign_interval)
                    has_IV_bus = 0
                if row['metro'].values[0] == 1:
                    path_indices_metro = [i for i, s in enumerate(OD_path_type_list_transit[O][D]) if s == 'metro']
                    pathcost_array_metro = np.array([OD_path_cost_list_transit[O][D][i] for i in path_indices_metro])
                    exp_pathcost_array_metro = np.maximum(np.exp((- 1/ theta_dict['theta_metro']) * pathcost_array_metro), 1e-6)
                    col_sum = np.sum(exp_pathcost_array_metro, axis =0)
                    path_prob_metro = exp_pathcost_array_metro / col_sum
                    IV_metro = np.log(col_sum)
                    has_IV_metro = 1
                    assert not np.isnan(path_prob_metro).any()
                    assert not np.isnan(IV_metro).any()
                else:
                    path_indices_metro = []
                    IV_metro = np.zeros(self.num_assign_interval)
                    has_IV_metro = 0
                if row['bus+metro'].values[0] == 1:
                    path_indices_busmetro = [i for i, s in enumerate(OD_path_type_list_transit[O][D]) if s == 'bus+metro']
                    pathcost_array_busmetro = np.array([OD_path_cost_list_transit[O][D][i] for i in path_indices_busmetro])
                    exp_pathcost_array_busmetro = np.maximum(np.exp((- 1/ theta_dict['theta_busmetro']) * pathcost_array_busmetro), 1e-6)
                    col_sum = np.sum(exp_pathcost_array_busmetro, axis =0)
                    path_prob_busmetro = exp_pathcost_array_busmetro / col_sum
                    IV_busmetro = np.log(col_sum)
                    has_IV_busmetro = 1
                    assert not np.isnan(path_prob_busmetro).any()
                    assert not np.isnan(IV_busmetro).any()
                else:
                    path_indices_busmetro = []
                    IV_busmetro = np.zeros(self.num_assign_interval)
                    has_IV_busmetro = 0

                if row['car+bus'].values[0] == 1:
                    path_indices_carbus = [i for i, s in enumerate(OD_path_type_list_pnr[O][D]) if s == 'car+bus']
                    pathcost_array_carbus = np.array([OD_path_cost_list_pnr[O][D][i] for i in path_indices_carbus])
                    exp_pathcost_array_carbus = np.maximum(np.exp((- 1/ theta_dict['theta_carbus']) * pathcost_array_carbus), 1e-6)
                    col_sum = np.sum(exp_pathcost_array_carbus, axis =0)
                    path_prob_carbus = exp_pathcost_array_carbus / col_sum
                    IV_carbus = np.log(col_sum)
                    has_IV_carbus = 1
                    assert not np.isnan(path_prob_carbus).any()
                    assert not np.isnan(IV_carbus).any()
                else:
                    path_indices_carbus = []
                    IV_carbus = np.zeros(self.num_assign_interval)
                    has_IV_carbus = 0
                if row['car+metro'].values[0] == 1:
                    path_indices_carmetro = [i for i, s in enumerate(OD_path_type_list_pnr[O][D]) if s == 'car+metro']
                    pathcost_array_carmetro = np.array([OD_path_cost_list_pnr[O][D][i] for i in path_indices_carmetro])
                    exp_pathcost_array_carmetro = np.maximum(np.exp((- 1/ theta_dict['theta_carmetro']) * pathcost_array_carmetro), 1e-6)
                    col_sum = np.sum(exp_pathcost_array_carmetro, axis =0)
                    path_prob_carmetro = exp_pathcost_array_carmetro / col_sum
                    IV_carmetro = np.log(col_sum)
                    has_IV_carmetro = 1
                    assert not np.isnan(path_prob_carmetro).any()
                    assert not np.isnan(IV_carmetro).any()
                else:
                    path_indices_carmetro = []
                    IV_carmetro = np.zeros(self.num_assign_interval)
                    has_IV_carmetro = 0
                if row['car+bus+metro'].values[0] == 1:
                    path_indices_carbusmetro = [i for i, s in enumerate(OD_path_type_list_pnr[O][D]) if s == 'car+bus+metro']
                    pathcost_array_carbusmetro = np.array([OD_path_cost_list_pnr[O][D][i] for i in path_indices_carbusmetro])
                    exp_pathcost_array_carbusmetro = np.maximum(np.exp((- 1/ theta_dict['theta_carbusmetro']) * pathcost_array_carbusmetro), 1e-6)
                    col_sum = np.sum(exp_pathcost_array_carbusmetro, axis =0)
                    path_prob_carbusmetro = exp_pathcost_array_carbusmetro / col_sum
                    IV_carbusmetro = np.log(col_sum)
                    has_IV_carbusmetro = 1
                    assert not np.isnan(path_prob_carbusmetro).any()
                    assert not np.isnan(IV_carbusmetro).any()
                else:
                    path_indices_carbusmetro = []
                    IV_carbusmetro = np.zeros(self.num_assign_interval)
                    has_IV_carbusmetro = 0

                if nested_type == 1:
                    IV_1_driving = np.log(np.exp((theta_dict['theta_car'] / theta_dict['theta_1_driving']) * IV_car))
                    if row['BusTransit'].values[0] == 1: #note here is the general transit not only bus
                        has_transit = 1
                        IV_1_transit = np.log(has_IV_bus * np.exp((theta_dict['theta_bus']/theta_dict['theta_1_transit'])*IV_bus) + has_IV_metro * np.exp((theta_dict['theta_metro']/theta_dict['theta_1_transit'])*IV_metro) + \
                            has_IV_busmetro * np.exp((theta_dict['theta_busmetro']/theta_dict['theta_1_transit'])*IV_busmetro))
                        assert not np.isnan(IV_1_transit).any()
                    else:
                        has_transit = 0 
                        IV_1_transit = np.zeros(self.num_assign_interval)
                    if row['PNR'].values[0] == 1:
                        has_pnr = 1
                        IV_1_pnr = np.log(has_IV_carbus * np.exp((theta_dict['theta_carbus']/theta_dict['theta_1_pnr'])*IV_carbus) + \
                                          has_IV_carmetro * np.exp((theta_dict['theta_carmetro']/theta_dict['theta_1_pnr'])*IV_carmetro) + \
                            has_IV_carbusmetro * np.exp((theta_dict['theta_carbusmetro']/theta_dict['theta_1_pnr'])*IV_carbusmetro))
                        assert not np.isnan(IV_1_pnr).any()
                    else:
                        has_pnr = 0
                        IV_1_pnr = np.zeros(self.num_assign_interval)
                    
                    prob_1_driving = np.exp(theta_dict['theta_1_driving'] * IV_1_driving) / (np.exp(theta_dict['theta_1_driving'] * IV_1_driving) + \
                                                                                             has_transit * np.exp(theta_dict['theta_1_transit'] * IV_1_transit) + \
                                                                                                has_pnr * np.exp(theta_dict['theta_1_pnr'] * IV_1_pnr))
                    assert not np.isnan(prob_1_driving).any()
                    prob_car = np.ones(self.num_assign_interval)
                    final_path_prob_car = prob_1_driving * prob_car * path_prob_car # note the dimensions: path_prob_car is a 2D array
                    path_flow = q_e_traveler[OD_index] * final_path_prob_car
                    f_car_driving = f_car_driving + list(path_flow)
                    k_probs_driving = k_probs_driving + list(path_prob_car)
                    gm_probs_driving = gm_probs_driving + [prob_car]*len(path_prob_car)
                    m_probs_driving = m_probs_driving + [prob_1_driving]*len(path_prob_car)

                    if row['BusTransit'].values[0] == 1:
                        path_flow = list(np.zeros((len(OD_path_type_list_transit[O][D]),self.num_assign_interval)))
                        k_prob_list, gm_prob_list, m_prob_list = list(np.zeros(len(OD_path_type_list_transit[O][D]))), \
                            list(np.zeros(len(OD_path_type_list_transit[O][D]))), list(np.zeros(len(OD_path_type_list_transit[O][D])))
                        prob_1_transit = np.exp(theta_dict['theta_1_transit'] * IV_1_transit) / (np.exp(theta_dict['theta_1_driving'] * IV_1_driving) + \
                                                                                                has_transit * np.exp(theta_dict['theta_1_transit'] * IV_1_transit) + \
                                                                                                    has_pnr * np.exp(theta_dict['theta_1_pnr'] * IV_1_pnr))
                        assert not np.isnan(prob_1_transit).any()
                        second_level_denom = has_IV_bus * np.exp((theta_dict['theta_bus']/theta_dict['theta_1_transit'])*IV_bus) + \
                            has_IV_metro * np.exp((theta_dict['theta_metro']/theta_dict['theta_1_transit'])*IV_metro) + \
                            has_IV_busmetro * np.exp((theta_dict['theta_busmetro']/theta_dict['theta_1_transit'])*IV_busmetro)
                        if has_IV_bus:
                            prob_bus = np.exp((theta_dict['theta_bus']/theta_dict['theta_1_transit'])*IV_bus) / second_level_denom  
                            assert not np.isnan(prob_bus).any()
                            final_path_prob_bus = prob_1_transit * prob_bus * path_prob_bus
                            path_flow_bus = q_e_traveler[OD_index] * final_path_prob_bus
                            for i in range(len(path_indices_bus)):
                                path_flow[path_indices_bus[i]] = path_flow_bus[i]
                                k_prob_list[path_indices_bus[i]] = path_prob_bus[i]
                                gm_prob_list[path_indices_bus[i]] = prob_bus
                                m_prob_list[path_indices_bus[i]] = prob_1_transit
                        if has_IV_metro:
                            prob_metro = np.exp((theta_dict['theta_metro']/theta_dict['theta_1_transit'])*IV_metro) / second_level_denom
                            final_path_prob_metro = prob_1_transit * prob_metro * path_prob_metro
                            path_flow_metro = q_e_traveler[OD_index] * final_path_prob_metro
                            assert not np.isnan(prob_metro).any()
                            for i in range(len(path_indices_metro)):
                                path_flow[path_indices_metro[i]] = path_flow_metro[i]
                                k_prob_list[path_indices_metro[i]] = path_prob_metro[i]
                                gm_prob_list[path_indices_metro[i]] = prob_metro
                                m_prob_list[path_indices_metro[i]] = prob_1_transit
                        if has_IV_busmetro:
                            prob_busmetro = np.exp((theta_dict['theta_busmetro']/theta_dict['theta_1_transit'])*IV_busmetro) / second_level_denom
                            final_path_prob_busmetro = prob_1_transit * prob_busmetro * path_prob_busmetro
                            path_flow_busmetro = q_e_traveler[OD_index] * final_path_prob_busmetro
                            assert not np.isnan(prob_busmetro).any()
                            for i in range(len(path_indices_busmetro)):
                                path_flow[path_indices_busmetro[i]] = path_flow_busmetro[i]
                                k_prob_list[path_indices_busmetro[i]] = path_prob_busmetro[i]
                                gm_prob_list[path_indices_busmetro[i]] = prob_busmetro
                                m_prob_list[path_indices_busmetro[i]] = prob_1_transit
                        f_transit = f_transit + path_flow
                        k_probs_transit = k_probs_transit + k_prob_list
                        gm_probs_transit = gm_probs_transit + gm_prob_list
                        m_probs_transit = m_probs_transit + m_prob_list

                    
                    if row['PNR'].values[0] == 1:
                        path_flow = list(np.zeros((len(OD_path_type_list_pnr[O][D]),self.num_assign_interval)))
                        k_prob_list, gm_prob_list, m_prob_list = list(np.zeros(len(OD_path_type_list_pnr[O][D]))), \
                            list(np.zeros(len(OD_path_type_list_pnr[O][D]))), list(np.zeros(len(OD_path_type_list_pnr[O][D])))
                        prob_1_pnr = np.exp(theta_dict['theta_1_pnr'] * IV_1_pnr) / (np.exp(theta_dict['theta_1_driving'] * IV_1_driving) + \
                                                                                     has_transit * np.exp(theta_dict['theta_1_transit'] * IV_1_transit) + \
                                                                                        has_pnr * np.exp(theta_dict['theta_1_pnr'] * IV_1_pnr))
                        assert not np.isnan(prob_1_pnr).any()
                        second_level_denom = has_IV_carbus * np.exp((theta_dict['theta_carbus']/theta_dict['theta_1_pnr'])*IV_carbus) + \
                            has_IV_carmetro * np.exp((theta_dict['theta_carmetro']/theta_dict['theta_1_pnr'])*IV_carmetro) + \
                            has_IV_carbusmetro * np.exp((theta_dict['theta_carbusmetro']/theta_dict['theta_1_pnr'])*IV_carbusmetro)
                        if has_IV_carbus:
                            prob_carbus = np.exp((theta_dict['theta_carbus']/theta_dict['theta_1_pnr'])*IV_carbus) / second_level_denom
                            final_path_prob_carbus = prob_1_pnr * prob_carbus * path_prob_carbus
                            path_flow_carbus = q_e_traveler[OD_index] * final_path_prob_carbus
                            assert not np.isnan(prob_carbus).any()
                            for i in range(len(path_indices_carbus)):
                                path_flow[path_indices_carbus[i]] = path_flow_carbus[i]
                                k_prob_list[path_indices_carbus[i]] = path_prob_carbus[i]
                                gm_prob_list[path_indices_carbus[i]] = prob_carbus
                                m_prob_list[path_indices_carbus[i]] = prob_1_pnr
                        if has_IV_carmetro:
                            prob_carmetro = np.exp((theta_dict['theta_carmetro']/theta_dict['theta_1_pnr'])*IV_carmetro) / second_level_denom
                            final_path_prob_carmetro = prob_1_pnr * prob_carmetro * path_prob_carmetro
                            path_flow_carmetro = q_e_traveler[OD_index] * final_path_prob_carmetro
                            assert not np.isnan(prob_carmetro).any()
                            for i in range(len(path_indices_carmetro)):
                                path_flow[path_indices_carmetro[i]] = path_flow_carmetro[i]
                                k_prob_list[path_indices_carmetro[i]] = path_prob_carmetro[i]
                                gm_prob_list[path_indices_carmetro[i]] = prob_carmetro
                                m_prob_list[path_indices_carmetro[i]] = prob_1_pnr
                        if has_IV_carbusmetro:
                            prob_carbusmetro = np.exp((theta_dict['theta_carbusmetro']/theta_dict['theta_1_pnr'])*IV_carbusmetro) / second_level_denom
                            final_path_prob_carbusmetro = prob_1_pnr * prob_carbusmetro * path_prob_carbusmetro
                            path_flow_carbusmetro = q_e_traveler[OD_index] * final_path_prob_carbusmetro
                            assert not np.isnan(prob_carbusmetro).any()
                            for i in range(len(path_indices_carbusmetro)):
                                path_flow[path_indices_carbusmetro[i]] = path_flow_carbusmetro[i]
                                k_prob_list[path_indices_carbusmetro[i]] = path_prob_carbusmetro[i]
                                gm_prob_list[path_indices_carbusmetro[i]] = prob_carbusmetro
                                m_prob_list[path_indices_carbusmetro[i]] = prob_1_pnr
                        f_pnr = f_pnr + path_flow
                        k_probs_pnr = k_probs_pnr + k_prob_list
                        gm_probs_pnr = gm_probs_pnr + gm_prob_list
                        m_probs_pnr = m_probs_pnr + m_prob_list
                
                elif nested_type == 2:
                    IV_1_driving = np.log(np.exp((theta_dict['theta_car'] / theta_dict['theta_1_driving']) * IV_car))
                    if has_IV_bus or has_IV_carbus:
                        has_bus_nest = 1
                        IV_1_bus = np.log(has_IV_bus * np.exp((theta_dict['theta_bus']/theta_dict['theta_1_bus'])*IV_bus) +\
                                           has_IV_carbus * np.exp((theta_dict['theta_carbus']/theta_dict['theta_1_bus'])*IV_carbus))
                    else:
                        has_bus_nest = 0
                        IV_1_bus = np.zeros(self.num_assign_interval)
                    if has_IV_metro or has_IV_carmetro:
                        has_metro_nest = 1
                        IV_1_metro = np.log(has_IV_metro * np.exp((theta_dict['theta_metro']/theta_dict['theta_1_metro'])*IV_metro) +\
                                           has_IV_carmetro * np.exp((theta_dict['theta_carmetro']/theta_dict['theta_1_metro'])*IV_carmetro))
                    else:
                        has_metro_nest = 0
                        IV_1_metro = np.zeros(self.num_assign_interval)
                    if has_IV_busmetro or has_IV_carbusmetro:
                        has_combinedtransit_nest = 1
                        IV_1_combinedtransit = np.log(np.exp((theta_dict['theta_busmetro']/theta_dict['theta_1_combinedtransit'])*IV_busmetro) + \
                                                      np.exp((theta_dict['theta_carbusmetro']/theta_dict['theta_1_combinedtransit'])*IV_carbusmetro))
                    else:
                        has_combinedtransit_nest = 0
                        IV_1_combinedtransit = np.zeros(self.num_assign_interval)
                    
                    prob_1_driving = np.exp(theta_dict['theta_1_driving'] * IV_1_driving) / (np.exp(theta_dict['theta_1_driving'] * IV_1_driving) + \
                                                                                             has_bus_nest * np.exp(theta_dict['theta_1_bus'] * IV_1_bus) + \
                                                                                             has_metro_nest * np.exp(theta_dict['theta_1_metro'] * IV_1_metro) + \
                                                                                             has_combinedtransit_nest * np.exp(theta_dict['theta_1_combinedtransit'] * IV_1_combinedtransit))
                    prob_car = np.ones(self.num_assign_interval)
                    final_path_prob_car = prob_1_driving * prob_car * path_prob_car # note the dimensions: path_prob_car is a 2D array
                    path_flow = q_e_traveler[OD_index] * final_path_prob_car
                    f_car_driving = f_car_driving + list(path_flow)
                    k_probs_driving = k_probs_driving + list(path_prob_car)
                    gm_probs_driving = gm_probs_driving + [prob_car]*len(path_prob_car)
                    m_probs_driving = m_probs_driving + [prob_1_driving]*len(path_prob_car)

                    if has_IV_bus or has_IV_metro or has_IV_busmetro:
                        path_flow_f_transit = list(np.zeros((len(OD_path_type_list_transit[O][D]),self.num_assign_interval)))
                        k_prob_list_transit, gm_prob_list_transit, m_prob_list_transit = list(np.zeros(len(OD_path_type_list_transit[O][D]))), \
                        list(np.zeros(len(OD_path_type_list_transit[O][D]))), list(np.zeros(len(OD_path_type_list_transit[O][D])))
                    if has_IV_carbus or has_IV_carmetro or has_IV_carbusmetro:
                        path_flow_f_pnr = list(np.zeros((len(OD_path_type_list_pnr[O][D]),self.num_assign_interval)))
                        k_prob_list_pnr, gm_prob_list_pnr, m_prob_list_pnr = list(np.zeros(len(OD_path_type_list_pnr[O][D]))), \
                        list(np.zeros(len(OD_path_type_list_pnr[O][D]))), list(np.zeros(len(OD_path_type_list_pnr[O][D])))
                    
                    if has_IV_bus or has_IV_carbus:
                        prob_1_bus = np.exp(theta_dict['theta_1_bus'] * IV_1_bus) / (np.exp(theta_dict['theta_1_driving'] * IV_1_driving) + \
                                                                                     has_bus_nest * np.exp(theta_dict['theta_1_bus'] * IV_1_bus) + \
                                                                                     has_metro_nest * np.exp(theta_dict['theta_1_metro'] * IV_1_metro) + \
                                                                                     has_combinedtransit_nest * np.exp(theta_dict['theta_1_combinedtransit'] * IV_1_combinedtransit))
                        second_level_denom = has_IV_bus * np.exp((theta_dict['theta_bus']/theta_dict['theta_1_bus'])*IV_bus) + \
                            has_IV_carbus * np.exp((theta_dict['theta_carbus']/theta_dict['theta_1_bus'])*IV_carbus)
                        if has_IV_bus:
                            prob_bus = np.exp((theta_dict['theta_bus']/theta_dict['theta_1_bus'])*IV_bus) / second_level_denom
                            final_path_prob_bus = prob_1_bus * prob_bus * path_prob_bus
                            path_flow_bus = q_e_traveler[OD_index] * final_path_prob_bus
                            for i in range(len(path_indices_bus)):
                                path_flow_f_transit[path_indices_bus[i]] = path_flow_bus[i]
                                k_prob_list_transit[path_indices_bus[i]] = path_prob_bus[i]
                                gm_prob_list_transit[path_indices_bus[i]] = prob_bus
                                m_prob_list_transit[path_indices_bus[i]] = prob_1_bus
                        if has_IV_carbus:
                            prob_carbus = np.exp((theta_dict['theta_carbus']/theta_dict['theta_1_bus'])*IV_carbus) / second_level_denom
                            final_path_prob_carbus = prob_1_bus * prob_carbus * path_prob_carbus
                            path_flow_carbus = q_e_traveler[OD_index] * final_path_prob_carbus
                            for i in range(len(path_indices_carbus)):
                                path_flow_f_pnr[path_indices_carbus[i]] = path_flow_carbus[i]
                                k_prob_list_pnr[path_indices_carbus[i]] = path_prob_carbus[i]
                                gm_prob_list_pnr[path_indices_carbus[i]] = prob_carbus
                                m_prob_list_pnr[path_indices_carbus[i]] = prob_1_bus

                    if has_IV_metro or has_IV_carmetro:
                        prob_1_metro = np.exp(theta_dict['theta_1_metro'] * IV_1_metro) / (np.exp(theta_dict['theta_1_driving'] * IV_1_driving) + \
                                                                                         has_bus_nest * np.exp(theta_dict['theta_1_bus'] * IV_1_bus) + \
                                                                                         has_metro_nest * np.exp(theta_dict['theta_1_metro'] * IV_1_metro) + \
                                                                                         has_combinedtransit_nest * np.exp(theta_dict['theta_1_combinedtransit'] * IV_1_combinedtransit))
                        second_level_denom = has_IV_metro * np.exp((theta_dict['theta_metro']/theta_dict['theta_1_metro'])*IV_metro) + \
                            has_IV_carmetro * np.exp((theta_dict['theta_carmetro']/theta_dict['theta_1_metro'])*IV_carmetro)
                        if has_IV_metro:
                            prob_metro = np.exp((theta_dict['theta_metro']/theta_dict['theta_1_metro'])*IV_metro) / second_level_denom
                            final_path_prob_metro = prob_1_metro * prob_metro * path_prob_metro
                            path_flow_metro = q_e_traveler[OD_index] * final_path_prob_metro
                            for i in range(len(path_indices_metro)):
                                path_flow_f_transit[path_indices_metro[i]] = path_flow_metro[i]
                                k_prob_list_transit[path_indices_metro[i]] = path_prob_metro[i]
                                gm_prob_list_transit[path_indices_metro[i]] = prob_metro
                                m_prob_list_transit[path_indices_metro[i]] = prob_1_metro
                        if has_IV_carmetro:
                            prob_carmetro = np.exp((theta_dict['theta_carmetro']/theta_dict['theta_1_metro'])*IV_carmetro) / second_level_denom
                            final_path_prob_carmetro = prob_1_metro * prob_carmetro * path_prob_carmetro
                            path_flow_carmetro = q_e_traveler[OD_index] * final_path_prob_carmetro
                            for i in range(len(path_indices_carmetro)):
                                path_flow_f_pnr[path_indices_carmetro[i]] = path_flow_carmetro[i]
                                k_prob_list_pnr[path_indices_carmetro[i]] = path_prob_carmetro[i]
                                gm_prob_list_pnr[path_indices_carmetro[i]] = prob_carmetro
                                m_prob_list_pnr[path_indices_carmetro[i]] = prob_1_metro

                    if has_IV_busmetro or has_IV_carbusmetro:
                        prob_1_combinedtransit = np.exp(theta_dict['theta_1_combinedtransit'] * IV_1_combinedtransit) / (np.exp(theta_dict['theta_1_driving'] * IV_1_driving) + \
                                                                                                                 has_bus_nest * np.exp(theta_dict['theta_1_bus'] * IV_1_bus) + \
                                                                                                                 has_metro_nest * np.exp(theta_dict['theta_1_metro'] * IV_1_metro) + \
                                                                                                                 has_combinedtransit_nest * np.exp(theta_dict['theta_1_combinedtransit'] * IV_1_combinedtransit))
                        second_level_denom = has_IV_busmetro * np.exp((theta_dict['theta_busmetro']/theta_dict['theta_1_combinedtransit'])*IV_busmetro) + \
                            has_IV_carbusmetro * np.exp((theta_dict['theta_carbusmetro']/theta_dict['theta_1_combinedtransit'])*IV_carbusmetro)
                        if has_IV_busmetro:
                            prob_busmetro = np.exp((theta_dict['theta_busmetro']/theta_dict['theta_1_combinedtransit'])*IV_busmetro) / second_level_denom
                            final_path_prob_busmetro = prob_1_combinedtransit * prob_busmetro * path_prob_busmetro
                            path_flow_busmetro = q_e_traveler[OD_index] * final_path_prob_busmetro
                            for i in range(len(path_indices_busmetro)):
                                path_flow_f_transit[path_indices_busmetro[i]] = path_flow_busmetro[i]
                                k_prob_list_transit[path_indices_busmetro[i]] = path_prob_busmetro[i]
                                gm_prob_list_transit[path_indices_busmetro[i]] = prob_busmetro
                                m_prob_list_transit[path_indices_busmetro[i]] = prob_1_combinedtransit
                        if has_IV_carbusmetro:
                            prob_carbusmetro = np.exp((theta_dict['theta_carbusmetro']/theta_dict['theta_1_combinedtransit'])*IV_carbusmetro) / second_level_denom
                            final_path_prob_carbusmetro = prob_1_combinedtransit * prob_carbusmetro * path_prob_carbusmetro
                            path_flow_carbusmetro = q_e_traveler[OD_index] * final_path_prob_carbusmetro
                            for i in range(len(path_indices_carbusmetro)):
                                path_flow_f_pnr[path_indices_carbusmetro[i]] = path_flow_carbusmetro[i]
                                k_prob_list_pnr[path_indices_carbusmetro[i]] = path_prob_carbusmetro[i]
                                gm_prob_list_pnr[path_indices_carbusmetro[i]] = prob_carbusmetro
                                m_prob_list_pnr[path_indices_carbusmetro[i]] = prob_1_combinedtransit

                    if row['BusTransit'].values[0] == 1:
                        f_transit = f_transit + path_flow_f_transit
                        k_probs_transit = k_probs_transit + k_prob_list_transit
                        gm_probs_transit = gm_probs_transit + gm_prob_list_transit
                        m_probs_transit = m_probs_transit + m_prob_list_transit

                    if row['PNR'].values[0] == 1:
                        f_pnr = f_pnr + list(path_flow_f_pnr)
                        k_probs_pnr = k_probs_pnr + k_prob_list_pnr
                        gm_probs_pnr = gm_probs_pnr + gm_prob_list_pnr
                        m_probs_pnr = m_probs_pnr + m_prob_list_pnr
                
                else: # nested_type == 3
                    IV_1_driving = np.log(np.exp((theta_dict['theta_car'] / theta_dict['theta_1_driving']) * IV_car) + has_IV_carbus * np.exp((theta_dict['theta_carbus'] / \
                                                                                                                                               theta_dict['theta_1_driving']) * IV_carbus) + 
                                        has_IV_carmetro * np.exp((theta_dict['theta_carmetro'] / theta_dict['theta_1_driving']) * IV_carmetro) )
                    if row['BusTransit'].values[0] == 1:
                        has_transit_nest = 1
                        IV_1_transit = np.log(has_IV_bus * np.exp((theta_dict['theta_bus']/theta_dict['theta_1_transit'])*IV_bus) + \
                                             has_IV_metro * np.exp((theta_dict['theta_metro']/theta_dict['theta_1_transit'])*IV_metro) + \
                                             has_IV_busmetro * np.exp((theta_dict['theta_busmetro']/theta_dict['theta_1_transit'])*IV_busmetro))
                    else:
                        has_transit_nest = 0
                        IV_1_transit = np.zeros(self.num_assign_interval)

                    if row['PNR'].values[0] == 1:
                        path_flow_f_pnr = list(np.zeros((len(OD_path_type_list_pnr[O][D]),self.num_assign_interval)))
                        k_prob_list_pnr, gm_prob_list_pnr, m_prob_list_pnr = list(np.zeros(len(OD_path_type_list_pnr[O][D]))), \
                            list(np.zeros(len(OD_path_type_list_pnr[O][D]))), list(np.zeros(len(OD_path_type_list_pnr[O][D])))
                    if row['BusTransit'].values[0] == 1:
                        path_flow_f_transit = list(np.zeros((len(OD_path_type_list_transit[O][D]),self.num_assign_interval)))
                        k_prob_list_transit, gm_prob_list_transit, m_prob_list_transit = list(np.zeros(len(OD_path_type_list_transit[O][D]))), \
                        list(np.zeros(len(OD_path_type_list_transit[O][D]))), list(np.zeros(len(OD_path_type_list_transit[O][D])))
                    
                    prob_1_driving = np.exp(theta_dict['theta_1_driving'] * IV_1_driving) / (np.exp(theta_dict['theta_1_driving'] * IV_1_driving) + \
                                                                                             has_transit_nest * np.exp(theta_dict['theta_1_transit'] * IV_1_transit))
                    second_level_denom = np.exp(theta_dict['theta_car'] / theta_dict['theta_1_driving'] * IV_car) + \
                        has_IV_carbus * np.exp((theta_dict['theta_carbus'] / theta_dict['theta_1_driving']) * IV_carbus) + \
                            has_IV_carmetro * np.exp((theta_dict['theta_carmetro'] / theta_dict['theta_1_driving']) * IV_carmetro) +\
                            has_IV_carbusmetro * np.exp((theta_dict['theta_carbusmetro'] / theta_dict['theta_1_driving']) * IV_carbusmetro)
                    prob_car = np.exp((theta_dict['theta_car'] / theta_dict['theta_1_driving']) * IV_car) / second_level_denom
                    final_path_prob_car = prob_1_driving * prob_car * path_prob_car
                    path_flow = q_e_traveler[OD_index] * final_path_prob_car
                    f_car_driving = f_car_driving +list(path_flow)
                    k_probs_driving = k_probs_driving + list(path_prob_car)
                    gm_probs_driving = gm_probs_driving + [prob_car]*len(path_prob_car)
                    m_probs_driving = m_probs_driving + [prob_1_driving]*len(path_prob_car)
                    if has_IV_carbus:
                        prob_carbus = np.exp((theta_dict['theta_carbus'] / theta_dict['theta_1_driving']) * IV_carbus) / second_level_denom
                        final_path_prob_carbus = prob_1_driving * prob_carbus * path_prob_carbus
                        path_flow_carbus = q_e_traveler[OD_index] * final_path_prob_carbus
                        for i in range(len(path_indices_carbus)):
                            path_flow_f_pnr[path_indices_carbus[i]] = path_flow_carbus[i]
                            k_prob_list_pnr[path_indices_carbus[i]] = path_prob_carbus[i]
                            gm_prob_list_pnr[path_indices_carbus[i]] = prob_carbus
                            m_prob_list_pnr[path_indices_carbus[i]] = prob_1_driving
                    if has_IV_carmetro:
                        prob_carmetro = np.exp((theta_dict['theta_carmetro'] / theta_dict['theta_1_driving']) * IV_carmetro) / second_level_denom
                        final_path_prob_carmetro = prob_1_driving * prob_carmetro * path_prob_carmetro
                        path_flow_carmetro = q_e_traveler[OD_index] * final_path_prob_carmetro
                        for i in range(len(path_indices_carmetro)):
                            path_flow_f_pnr[path_indices_carmetro[i]] = path_flow_carmetro[i]
                            k_prob_list_pnr[path_indices_carmetro[i]] = path_prob_carmetro[i]
                            gm_prob_list_pnr[path_indices_carmetro[i]] = prob_carmetro
                            m_prob_list_pnr[path_indices_carmetro[i]] = prob_1_driving
                    if has_IV_carbusmetro:
                        prob_carbusmetro = np.exp((theta_dict['theta_carbusmetro'] / theta_dict['theta_1_driving']) * IV_carbusmetro) / second_level_denom
                        final_path_prob_carbusmetro = prob_1_driving * prob_carbusmetro * path_prob_carbusmetro
                        path_flow_carbusmetro = q_e_traveler[OD_index] * final_path_prob_carbusmetro
                        for i in range(len(path_indices_carbusmetro)):
                            path_flow_f_pnr[path_indices_carbusmetro[i]] = path_flow_carbusmetro[i]
                            k_prob_list_pnr[path_indices_carbusmetro[i]] = path_prob_carbusmetro[i]
                            gm_prob_list_pnr[path_indices_carbusmetro[i]] = prob_carbusmetro
                            m_prob_list_pnr[path_indices_carbusmetro[i]] = prob_1_driving

                    if has_transit_nest == 1:
                        prob_1_transit = np.exp(theta_dict['theta_1_transit'] * IV_1_transit) / (np.exp(theta_dict['theta_1_driving'] * IV_1_driving) + \
                                                                                                 np.exp(theta_dict['theta_1_transit'] * IV_1_transit))
                        second_level_denom = has_IV_bus * np.exp((theta_dict['theta_bus']/theta_dict['theta_1_transit'])*IV_bus) + \
                            has_IV_metro * np.exp((theta_dict['theta_metro']/theta_dict['theta_1_transit'])*IV_metro) + \
                            has_IV_busmetro * np.exp((theta_dict['theta_busmetro']/theta_dict['theta_1_transit'])*IV_busmetro)
                        if has_IV_bus:
                            prob_bus = np.exp((theta_dict['theta_bus']/theta_dict['theta_1_transit'])*IV_bus) / second_level_denom
                            final_path_prob_bus = prob_1_transit * prob_bus * path_prob_bus
                            path_flow_bus = q_e_traveler[OD_index] * final_path_prob_bus
                            for i in range(len(path_indices_bus)):
                                path_flow_f_transit[path_indices_bus[i]] = path_flow_bus[i]
                                k_prob_list_transit[path_indices_bus[i]] = path_prob_bus[i]
                                gm_prob_list_transit[path_indices_bus[i]] = prob_bus
                                m_prob_list_transit[path_indices_bus[i]] = prob_1_transit
                        if has_IV_metro:
                            prob_metro = np.exp((theta_dict['theta_metro']/theta_dict['theta_1_transit'])*IV_metro) / second_level_denom
                            final_path_prob_metro = prob_1_transit * prob_metro * path_prob_metro
                            path_flow_metro = q_e_traveler[OD_index] * final_path_prob_metro
                            for i in range(len(path_indices_metro)):
                                path_flow_f_transit[path_indices_metro[i]] = path_flow_metro[i]
                                k_prob_list_transit[path_indices_metro[i]] = path_prob_metro[i]
                                gm_prob_list_transit[path_indices_metro[i]] = prob_metro
                                m_prob_list_transit[path_indices_metro[i]] = prob_1_transit
                        if has_IV_busmetro:
                            prob_busmetro = np.exp((theta_dict['theta_busmetro']/theta_dict['theta_1_transit'])*IV_busmetro) / second_level_denom
                            final_path_prob_busmetro = prob_1_transit * prob_busmetro * path_prob_busmetro
                            path_flow_busmetro = q_e_traveler[OD_index] * final_path_prob_busmetro
                            for i in range(len(path_indices_busmetro)):
                                path_flow_f_transit[path_indices_busmetro[i]] = path_flow_busmetro[i]
                                k_prob_list_transit[path_indices_busmetro[i]] = path_prob_busmetro[i]
                                gm_prob_list_transit[path_indices_busmetro[i]] = prob_busmetro
                                m_prob_list_transit[path_indices_busmetro[i]] = prob_1_transit
                        f_transit = f_transit + path_flow_f_transit
                        k_probs_transit = k_probs_transit + k_prob_list_transit
                        gm_probs_transit = gm_probs_transit + gm_prob_list_transit
                        m_probs_transit = m_probs_transit + m_prob_list_transit

                    if row['PNR'].values[0] == 1:
                        f_pnr = f_pnr + path_flow_f_pnr
                        k_probs_pnr = k_probs_pnr + k_prob_list_pnr
                        gm_probs_pnr = gm_probs_pnr + gm_prob_list_pnr
                        m_probs_pnr = m_probs_pnr + m_prob_list_pnr
        f_car_driving = np.array(f_car_driving)
        f_transit = np.array(f_transit)
        f_pnr = np.array(f_pnr)
        assert(f_car_driving.shape == (self.num_path_driving, self.num_assign_interval))
        assert(f_transit.shape == (self.num_path_bustransit, self.num_assign_interval))
        assert(f_pnr.shape == (self.num_path_pnr, self.num_assign_interval))
        f_car_driving = f_car_driving.flatten(order='F')
        f_transit = f_transit.flatten(order='F')
        f_pnr = f_pnr.flatten(order='F')

        assert all(not np.isnan(arr).any() for arr in k_probs_driving)
        assert all(not np.isnan(arr).any() for arr in k_probs_transit)
        assert all(not np.isnan(arr).any() for arr in k_probs_pnr)
        assert all(not np.isnan(arr).any() for arr in gm_probs_driving)
        assert all(not np.isnan(arr).any() for arr in gm_probs_transit)
        assert all(not np.isnan(arr).any() for arr in gm_probs_pnr)
        assert all(not np.isnan(arr).any() for arr in m_probs_driving)
        assert all(not np.isnan(arr).any() for arr in m_probs_transit)
        assert all(not np.isnan(arr).any() for arr in m_probs_pnr)

        probs_record_dict = {'k_probs_driving': k_probs_driving, 'k_probs_transit': k_probs_transit, 'k_probs_pnr': k_probs_pnr,
                             'gm_probs_driving': gm_probs_driving, 'gm_probs_transit': gm_probs_transit, 'gm_probs_pnr': gm_probs_pnr,
                             'm_probs_driving': m_probs_driving, 'm_probs_transit': m_probs_transit, 'm_probs_pnr': m_probs_pnr}
        
        return f_car_driving, f_transit, f_pnr, probs_record_dict
    
    def compute_para_derivatives(self, input_file_folder, q_e_traveler, nested_type, theta_dict, OD_path_tt_list_driving, OD_path_alltts_list_transit, OD_path_alltts_list_pnr,
                                 probs_record_dict):
        
        # with open(os.path.join(input_file_folder, 'probs_record_dict.pickle'), 'wb') as f:
        #     pickle.dump(probs_record_dict, f)

        with open(os.path.join(input_file_folder, 'disutility_info_df.pickle'), 'rb') as f:
            disutility_info_df = pickle.load(f)
        with open(os.path.join(input_file_folder, 'dode_data.pickle'), 'rb') as f:
            [_, _, _, _, _, _, _, \
            _, OD_path_type_list_transit,OD_path_type_list_pnr] = pickle.load(f)
        bus_fare = self.nb.config.config_dict['MMDUE']['bus_fare']
        metro_fare =self.nb.config.config_dict['MMDUE']['metro_fare']
        q_e_traveler = q_e_traveler.reshape((int(len(q_e_traveler) / self.num_assign_interval), self.num_assign_interval), order='F')

        beta_der_dict = dict()
        for string1 in ['beta_tt_car', 'beta_tt_bus', 'beta_tt_metro', 'beta_walking', 'beta_waiting_bus', 'beta_waiting_metro', 'beta_money']:
            beta_der_dict[string1] = dict()
        beta_der_dict['beta_tt_car']['driving'], beta_der_dict['beta_tt_car']['pnr'] = np.zeros((self.num_path_driving, self.num_assign_interval)), np.zeros((self.num_path_pnr, self.num_assign_interval))
        beta_der_dict['beta_tt_bus']['transit'], beta_der_dict['beta_tt_bus']['pnr'] = np.zeros((self.num_path_bustransit, self.num_assign_interval)), np.zeros((self.num_path_pnr, self.num_assign_interval))
        beta_der_dict['beta_tt_metro']['transit'], beta_der_dict['beta_tt_metro']['pnr'] = np.zeros((self.num_path_bustransit, self.num_assign_interval)), np.zeros((self.num_path_pnr, self.num_assign_interval))
        beta_der_dict['beta_walking']['transit'], beta_der_dict['beta_walking']['pnr'] = np.zeros((self.num_path_bustransit, self.num_assign_interval)), np.zeros((self.num_path_pnr, self.num_assign_interval))
        beta_der_dict['beta_waiting_bus']['transit'], beta_der_dict['beta_waiting_bus']['pnr'] = np.zeros((self.num_path_bustransit, self.num_assign_interval)), np.zeros((self.num_path_pnr, self.num_assign_interval))
        beta_der_dict['beta_waiting_metro']['transit'], beta_der_dict['beta_waiting_metro']['pnr'] = np.zeros((self.num_path_bustransit, self.num_assign_interval)), np.zeros((self.num_path_pnr, self.num_assign_interval))
        beta_der_dict['beta_money']['driving'], beta_der_dict['beta_money']['transit'], beta_der_dict['beta_money']['pnr'] = np.zeros((self.num_path_driving, self.num_assign_interval)), np.zeros((self.num_path_bustransit, \
                                                                                                                                                                                                    self.num_assign_interval)), np.zeros((self.num_path_pnr, self.num_assign_interval))
        gamma_der_dict = dict()
        for string1 in ['income', 'Originpopden', 'Destpopden']:
            gamma_der_dict['gamma_' + string1 + '_car'] = np.zeros((self.num_path_driving, self.num_assign_interval))
            for string2 in ['bus', 'metro', 'busmetro']:
                gamma_der_dict['gamma_' + string1 + '_' + string2] = np.zeros((self.num_path_bustransit, self.num_assign_interval))
            for string2 in ['carbus', 'carmetro', 'carbusmetro']:
                gamma_der_dict['gamma_' + string1 + '_' + string2] = np.zeros((self.num_path_pnr, self.num_assign_interval))

        alpha_der_dict = dict()
        for string1 in ['bus', 'metro', 'busmetro']:
            alpha_der_dict['alpha_' + string1] = np.zeros((self.num_path_bustransit, self.num_assign_interval))
        for string1 in ['carbus', 'carmetro', 'carbusmetro']:
            alpha_der_dict['alpha_' + string1] = np.zeros((self.num_path_pnr, self.num_assign_interval))


        # second derivatives (Hessian): inside each dict, the first key is the name of the parameter, the second key is the name of the parameter that it is derived with respect to
        beta_2nd_der_dict = dict()
        for string1 in ['beta_tt_car', 'beta_tt_bus', 'beta_tt_metro', 'beta_walking', 'beta_waiting_bus', 'beta_waiting_metro', 'beta_money']:
            beta_2nd_der_dict[string1] = dict()
        gamma_2nd_der_dict = dict()
        for string1 in ['income', 'Originpopden', 'Destpopden']:
            gamma_2nd_der_dict['gamma_' + string1 + '_car'] = dict()
            for string2 in ['bus', 'metro', 'busmetro']:
                gamma_2nd_der_dict['gamma_' + string1 + '_' + string2] = dict()
            for string2 in ['carbus', 'carmetro', 'carbusmetro']:
                gamma_2nd_der_dict['gamma_' + string1 + '_' + string2] = dict()
        alpha_2nd_der_dict = dict()
        for string1 in ['bus', 'metro', 'busmetro']:
            alpha_2nd_der_dict['alpha_' + string1] = dict()
        for string1 in ['carbus', 'carmetro', 'carbusmetro']:
            alpha_2nd_der_dict['alpha_' + string1] = dict()
        # driving
        for string1 in ['beta_tt_car', 'beta_money']:
            for string2 in ['beta_tt_car', 'beta_money', 'gamma_income_car', 'gamma_Originpopden_car', 'gamma_Destpopden_car']:
                beta_2nd_der_dict[string1][string2] = dict()
                beta_2nd_der_dict[string1][string2]['driving'] = np.zeros((self.num_path_driving, self.num_assign_interval))
                beta_2nd_der_dict[string1][string2]['pnr'] = np.zeros((self.num_path_pnr, self.num_assign_interval))
                if string1 == 'beta_money':
                    beta_2nd_der_dict[string1][string2]['transit'] = np.zeros((self.num_path_bustransit, self.num_assign_interval))
        for string1 in ['gamma_income_car', 'gamma_Originpopden_car', 'gamma_Destpopden_car']:
            for string2 in ['beta_tt_car', 'beta_money', 'gamma_income_car', 'gamma_Originpopden_car', 'gamma_Destpopden_car']:
                gamma_2nd_der_dict[string1][string2] = np.zeros((self.num_path_driving, self.num_assign_interval))
        # bustransit
        for string2 in ['beta_tt_bus', 'beta_tt_metro', 'beta_walking', 'beta_waiting_bus', 'beta_waiting_metro', 'beta_money', 'gamma_income_bus', 'gamma_Originpopden_bus', 'gamma_Destpopden_bus',
                            'gamma_income_metro', 'gamma_Originpopden_metro', 'gamma_Destpopden_metro', 'gamma_income_busmetro', 'gamma_Originpopden_busmetro', 'gamma_Destpopden_busmetro',\
                                'alpha_bus', 'alpha_metro', 'alpha_busmetro']:
            for string1 in ['beta_tt_bus', 'beta_tt_metro', 'beta_walking', 'beta_waiting_bus', 'beta_waiting_metro', 'beta_money']:
                # if beta_2nd_der_dict[string1][string2] has not been defined as dict, then define it as dict
                if beta_2nd_der_dict[string1].get(string2) == None:
                    beta_2nd_der_dict[string1][string2] = dict()
                beta_2nd_der_dict[string1][string2]['transit'] = np.zeros((self.num_path_bustransit, self.num_assign_interval))
                beta_2nd_der_dict[string1][string2]['pnr'] = np.zeros((self.num_path_pnr, self.num_assign_interval))
                if string1 == 'beta_money':
                    beta_2nd_der_dict[string1][string2]['driving'] = np.zeros((self.num_path_driving, self.num_assign_interval))
            for string1 in ['gamma_income_bus', 'gamma_Originpopden_bus', 'gamma_Destpopden_bus', 'gamma_income_metro', 'gamma_Originpopden_metro', 'gamma_Destpopden_metro', \
                        'gamma_income_busmetro', 'gamma_Originpopden_busmetro', 'gamma_Destpopden_busmetro']:
                gamma_2nd_der_dict[string1][string2] = np.zeros((self.num_path_bustransit, self.num_assign_interval))
            for string1 in ['alpha_bus', 'alpha_metro', 'alpha_busmetro']:
                alpha_2nd_der_dict[string1][string2] = np.zeros((self.num_path_bustransit, self.num_assign_interval))
        # pnr
        for string2 in ['beta_tt_car', 'beta_tt_bus', 'beta_tt_metro', 'beta_walking', 'beta_waiting_bus', 'beta_waiting_metro', 'beta_money', 'gamma_income_carbus', 'gamma_Originpopden_carbus', 'gamma_Destpopden_carbus',
                            'gamma_income_carmetro', 'gamma_Originpopden_carmetro', 'gamma_Destpopden_carmetro', 'gamma_income_carbusmetro', 'gamma_Originpopden_carbusmetro', 'gamma_Destpopden_carbusmetro', \
                                'alpha_carbus', 'alpha_carmetro', 'alpha_carbusmetro']:
            for string1 in ['beta_tt_car', 'beta_tt_bus', 'beta_tt_metro', 'beta_walking', 'beta_waiting_bus', 'beta_waiting_metro', 'beta_money']:
                if beta_2nd_der_dict[string1].get(string2) == None:
                    beta_2nd_der_dict[string1][string2] = dict()
                beta_2nd_der_dict[string1][string2]['pnr'] = np.zeros((self.num_path_pnr, self.num_assign_interval))
                beta_2nd_der_dict[string1][string2]['transit'] = np.zeros((self.num_path_bustransit, self.num_assign_interval))
                if string1 == 'beta_money' or string1 == 'beta_tt_car':
                    beta_2nd_der_dict[string1][string2]['driving'] = np.zeros((self.num_path_driving, self.num_assign_interval))
            for string1 in ['gamma_income_carbus', 'gamma_Originpopden_carbus', 'gamma_Destpopden_carbus', 'gamma_income_carmetro', 'gamma_Originpopden_carmetro', 'gamma_Destpopden_carmetro', \
                        'gamma_income_carbusmetro', 'gamma_Originpopden_carbusmetro', 'gamma_Destpopden_carbusmetro']:
                gamma_2nd_der_dict[string1][string2] = np.zeros((self.num_path_pnr, self.num_assign_interval))
            for string1 in ['alpha_carbus', 'alpha_carmetro', 'alpha_carbusmetro']:
                alpha_2nd_der_dict[string1][string2] = np.zeros((self.num_path_pnr, self.num_assign_interval))

                
        #****************** for testing purpose only ********************
        if probs_record_dict == None:
            return beta_der_dict, gamma_der_dict, alpha_der_dict, beta_2nd_der_dict, gamma_2nd_der_dict, alpha_2nd_der_dict

        #****************************************************************

        OD_idx, path_driving_idx, path_transit_idx, path_pnr_idx = -1, -1, -1, -1
        for O in OD_path_tt_list_driving:
            for D in OD_path_tt_list_driving[O]:
                OD_idx += 1
                for i in range(len(OD_path_tt_list_driving[O][D])):
                    path_driving_idx += 1
                    A = probs_record_dict['m_probs_driving'][path_driving_idx]
                    B = probs_record_dict['gm_probs_driving'][path_driving_idx]
                    C = probs_record_dict['k_probs_driving'][path_driving_idx]
                    # same for all the nested types
                    temp_fi = -1/theta_dict['theta_car']* (1-C) - 1/theta_dict['theta_1_driving'] * C * (1-B) - C*B * (1-A)
                    f_der_c = q_e_traveler[OD_idx] * A * B * C * temp_fi
                    f_2nd_der_c = q_e_traveler[OD_idx] * A * B * C * (temp_fi**2 + C*(1-C)*(-1/(theta_dict['theta_car']**2) + (1-B)/(theta_dict['theta_car']*theta_dict['theta_1_driving']) + B*(1-A)/theta_dict['theta_car']) \
                                                                      + C**2*(-B*(1-B)/(theta_dict['theta_1_driving']**2) + B*(1-B)*(1-A)/theta_dict['theta_1_driving'] - B**2 * A * (1-A)))
                    assert not np.isnan(f_der_c).any()
                    beta_der_dict['beta_tt_car']['driving'][path_driving_idx] = f_der_c * OD_path_tt_list_driving[O][D][i] / 60
                    beta_der_dict['beta_money']['driving'][path_driving_idx] = f_der_c * disutility_info_df[disutility_info_df['D_id'] == D]['parking_fee'].values[0]
                    gamma_der_dict['gamma_income_car'][path_driving_idx] = f_der_c * disutility_info_df[disutility_info_df['O_id'] == O]['median_income'].values[0]
                    gamma_der_dict['gamma_Originpopden_car'][path_driving_idx] = f_der_c * disutility_info_df[disutility_info_df['O_id'] == O]['pop_den'].values[0]
                    gamma_der_dict['gamma_Destpopden_car'][path_driving_idx] = f_der_c * disutility_info_df[disutility_info_df['D_id'] == D]['pop_den'].values[0]

                    coef = dict()
                    coef['beta_tt_car'], coef['beta_money'], coef['gamma_income_car'], coef['gamma_Originpopden_car'], coef['gamma_Destpopden_car']  = OD_path_tt_list_driving[O][D][i] / 60,\
                        disutility_info_df[disutility_info_df['D_id'] == D]['parking_fee'].values[0], disutility_info_df[disutility_info_df['O_id'] == O]['median_income'].values[0], \
                        disutility_info_df[disutility_info_df['O_id'] == O]['pop_den'].values[0], disutility_info_df[disutility_info_df['D_id'] == D]['pop_den'].values[0]
                    for string2 in ['beta_tt_car', 'beta_money', 'gamma_income_car', 'gamma_Originpopden_car', 'gamma_Destpopden_car']:
                        for string1 in ['beta_tt_car', 'beta_money']:
                            beta_2nd_der_dict[string1][string2]['driving'][path_driving_idx] = f_2nd_der_c * coef[string1] * coef[string2]
                        for string1 in ['gamma_income_car', 'gamma_Originpopden_car', 'gamma_Destpopden_car']:
                            gamma_2nd_der_dict[string1][string2][path_driving_idx] = f_2nd_der_c * coef[string1] * coef[string2]
                    
                if O in OD_path_type_list_transit:
                    if D in OD_path_type_list_transit[O]:
                        for i in range(len(OD_path_type_list_transit[O][D])):
                            path_transit_idx += 1
                            A = probs_record_dict['m_probs_transit'][path_transit_idx]
                            B = probs_record_dict['gm_probs_transit'][path_transit_idx]
                            C = probs_record_dict['k_probs_transit'][path_transit_idx]
                            assert not np.isnan(A).any()
                            assert not np.isnan(B).any()
                            assert not np.isnan(C).any()
                            assert not np.isinf(A).any()
                            assert not np.isinf(B).any()
                            assert not np.isinf(C).any()
                            string1 = re.sub(r'[^a-zA-Z]', '', OD_path_type_list_transit[O][D][i])
                            if nested_type == 1 or nested_type == 3:
                                temp_fi = -1/theta_dict['theta_' + string1]* (1-C) - 1/theta_dict['theta_1_transit'] * C * (1-B) - C*B * (1-A)
                                f_der_c = q_e_traveler[OD_idx] * A * B * C * temp_fi
                                f_2nd_der_c = q_e_traveler[OD_idx] * A * B * C * (temp_fi**2 + C*(1-C)*(-1/(theta_dict['theta_' + string1]**2) + (1-B)/(theta_dict['theta_' + string1]*theta_dict['theta_1_transit']) + B*(1-A)/theta_dict['theta_' + string1]) \
                                                                      + C**2*(-B*(1-B)/(theta_dict['theta_1_transit']**2) + B*(1-B)*(1-A)/theta_dict['theta_1_transit'] - B**2 * A * (1-A)))
                            else:
                                if OD_path_type_list_transit[O][D][i] == 'bus+metro':
                                    temp_fi = -1/theta_dict['theta_' + string1]* (1-C) - 1/theta_dict['theta_1_combinedtransit'] * C * (1-B) - C*B * (1-A)
                                    f_der_c = q_e_traveler[OD_idx] * A * B * C * temp_fi
                                    f_2nd_der_c = q_e_traveler[OD_idx] * A * B * C * (temp_fi**2 + C*(1-C)*(-1/(theta_dict['theta_' + string1]**2) + (1-B)/(theta_dict['theta_' + string1]*theta_dict['theta_1_combinedtransit']) + B*(1-A)/theta_dict['theta_' + string1]) \
                                                                      + C**2*(-B*(1-B)/(theta_dict['theta_1_combinedtransit']**2) + B*(1-B)*(1-A)/theta_dict['theta_1_combinedtransit'] - B**2 * A * (1-A)))
                                else:
                                    temp_fi = -1/theta_dict['theta_' + string1]* (1-C) - 1/theta_dict['theta_1_' + string1] * C * (1-B) - C*B * (1-A)
                                    f_der_c = q_e_traveler[OD_idx] * A * B * C * temp_fi
                                    f_2nd_der_c = q_e_traveler[OD_idx] * A * B * C * (temp_fi**2 + C*(1-C)*(-1/(theta_dict['theta_' + string1]**2) + (1-B)/(theta_dict['theta_' + string1]*theta_dict['theta_1_' + string1]) + B*(1-A)/theta_dict['theta_' + string1]) \
                                                                      + C**2*(-B*(1-B)/(theta_dict['theta_1_' + string1]**2) + B*(1-B)*(1-A)/theta_dict['theta_1_' + string1] - B**2 * A * (1-A)))
                            assert not np.isnan(q_e_traveler[OD_idx]).any()
                            assert not np.isinf(q_e_traveler[OD_idx]).any()
                            assert not np.isnan(f_der_c).any()
                            assert not np.isinf(f_der_c).any()
                            assert not np.isinf(OD_path_alltts_list_transit[O][D][i]['bus']).any()
                            assert not np.isnan(OD_path_alltts_list_transit[O][D][i]['bus']).any()
                            beta_der_dict['beta_tt_bus']['transit'][path_transit_idx] = f_der_c * OD_path_alltts_list_transit[O][D][i]['bus'] / 60
                            beta_der_dict['beta_tt_metro']['transit'][path_transit_idx] = f_der_c * OD_path_alltts_list_transit[O][D][i]['metro'] / 60
                            beta_der_dict['beta_waiting_bus']['transit'][path_transit_idx] = f_der_c * OD_path_alltts_list_transit[O][D][i]['bus_waiting'] /60
                            beta_der_dict['beta_waiting_metro']['transit'][path_transit_idx] = f_der_c * OD_path_alltts_list_transit[O][D][i]['metro_waiting'] /60
                            beta_der_dict['beta_walking']['transit'][path_transit_idx] = f_der_c * OD_path_alltts_list_transit[O][D][i]['walking'] /60
                            has_bus = 1 if 'bus' in string1 else 0
                            has_metro = 1 if 'metro' in string1 else 0
                            beta_der_dict['beta_money']['transit'][path_transit_idx] = f_der_c * (bus_fare * has_bus + metro_fare * has_metro)
                            gamma_der_dict['gamma_income_' + string1][path_transit_idx] = f_der_c * disutility_info_df[disutility_info_df['O_id'] == O]['median_income'].values[0]
                            gamma_der_dict['gamma_Originpopden_' + string1][path_transit_idx] = f_der_c * disutility_info_df[disutility_info_df['O_id'] == O]['pop_den'].values[0]
                            gamma_der_dict['gamma_Destpopden_' + string1][path_transit_idx] = f_der_c * disutility_info_df[disutility_info_df['D_id'] == D]['pop_den'].values[0]
                            alpha_der_dict['alpha_' + string1][path_transit_idx] = f_der_c

                            coef = dict()
                            coef['beta_tt_bus'], coef['beta_tt_metro'], coef['beta_waiting_bus'], coef['beta_waiting_metro'], coef['beta_walking'], coef['beta_money'], \
                                coef['gamma_income_' + string1], coef['gamma_Originpopden_' + string1], coef['gamma_Destpopden_' + string1], coef['alpha_' + string1] = OD_path_alltts_list_transit[O][D][i]['bus'] / 60, \
                                OD_path_alltts_list_transit[O][D][i]['metro'] / 60, OD_path_alltts_list_transit[O][D][i]['bus_waiting'] / 60, OD_path_alltts_list_transit[O][D][i]['metro_waiting'] / 60, \
                                OD_path_alltts_list_transit[O][D][i]['walking'] / 60, bus_fare * has_bus + metro_fare * has_metro, \
                                disutility_info_df[disutility_info_df['O_id'] == O]['median_income'].values[0], disutility_info_df[disutility_info_df['O_id'] == O]['pop_den'].values[0], \
                                disutility_info_df[disutility_info_df['D_id'] == D]['pop_den'].values[0], 1
                            for string3 in ['beta_tt_bus', 'beta_tt_metro', 'beta_waiting_bus', 'beta_waiting_metro', 'beta_walking', 'beta_money', 'gamma_income_' + string1, 'gamma_Originpopden_' + string1, \
                                            'gamma_Destpopden_' + string1, 'alpha_' + string1]:
                                for string2 in ['beta_tt_bus', 'beta_tt_metro', 'beta_waiting_bus', 'beta_waiting_metro', 'beta_walking', 'beta_money']:
                                    beta_2nd_der_dict[string2][string3]['transit'][path_transit_idx] = f_2nd_der_c * coef[string2] * coef[string3]
                                for string2 in ['gamma_income_' + string1, 'gamma_Originpopden_' + string1, 'gamma_Destpopden_' + string1]:
                                    gamma_2nd_der_dict[string2][string3][path_transit_idx] = f_2nd_der_c * coef[string2] * coef[string3]
                                for string2 in ['alpha_' + string1]:
                                    alpha_2nd_der_dict[string2][string3][path_transit_idx] = f_2nd_der_c * coef[string2] * coef[string3] 

                if O in OD_path_type_list_pnr:
                    if D in OD_path_type_list_pnr[O]:
                        for i in range(len(OD_path_type_list_pnr[O][D])):
                            path_pnr_idx += 1
                            A = probs_record_dict['m_probs_pnr'][path_pnr_idx]
                            B = probs_record_dict['gm_probs_pnr'][path_pnr_idx]
                            C = probs_record_dict['k_probs_pnr'][path_pnr_idx]
                            string1 = re.sub(r'[^a-zA-Z]', '', OD_path_type_list_pnr[O][D][i])
                            if nested_type == 1:
                                temp_fi = -1/theta_dict['theta_' + string1]* (1-C) - 1/theta_dict['theta_1_pnr'] * C * (1-B) - C*B * (1-A)
                                f_der_c = q_e_traveler[OD_idx] * A * B * C * temp_fi
                                f_2nd_der_c = q_e_traveler[OD_idx] * A * B * C * (temp_fi**2 + C*(1-C)*(-1/(theta_dict['theta_' + string1]**2) + (1-B)/(theta_dict['theta_' + string1]*theta_dict['theta_1_pnr']) + B*(1-A)/theta_dict['theta_' + string1]) \
                                                                      + C**2*(-B*(1-B)/(theta_dict['theta_1_pnr']**2) + B*(1-B)*(1-A)/theta_dict['theta_1_pnr'] - B**2 * A * (1-A)))
                            elif nested_type == 2:
                                if OD_path_type_list_pnr[O][D][i] == 'car+bus+metro':
                                    temp_fi = -1/theta_dict['theta_' + string1]* (1-C) - 1/theta_dict['theta_1_combinedtransit'] * C * (1-B) - C*B * (1-A)
                                    f_der_c = q_e_traveler[OD_idx] * A * B * C * temp_fi
                                    f_2nd_der_c = q_e_traveler[OD_idx] * A * B * C * (temp_fi**2 + C*(1-C)*(-1/(theta_dict['theta_' + string1]**2) + (1-B)/(theta_dict['theta_' + string1]*theta_dict['theta_1_combinedtransit']) + B*(1-A)/theta_dict['theta_' + string1]) \
                                                                      + C**2*(-B*(1-B)/(theta_dict['theta_1_combinedtransit']**2) + B*(1-B)*(1-A)/theta_dict['theta_1_combinedtransit'] - B**2 * A * (1-A)))
                                elif OD_path_type_list_pnr[O][D][i] == 'car+bus':
                                    temp_fi = -1/theta_dict['theta_' + string1]* (1-C) - 1/theta_dict['theta_1_bus'] * C * (1-B) - C*B * (1-A)
                                    f_der_c = q_e_traveler[OD_idx] * A * B * C * temp_fi
                                    f_2nd_der_c = q_e_traveler[OD_idx] * A * B * C * (temp_fi**2 + C*(1-C)*(-1/(theta_dict['theta_' + string1]**2) + (1-B)/(theta_dict['theta_' + string1]*theta_dict['theta_1_bus']) + B*(1-A)/theta_dict['theta_' + string1]) \
                                                                      + C**2*(-B*(1-B)/(theta_dict['theta_1_bus']**2) + B*(1-B)*(1-A)/theta_dict['theta_1_bus'] - B**2 * A * (1-A)))
                                else:
                                    temp_fi = -1/theta_dict['theta_' + string1]* (1-C) - 1/theta_dict['theta_1_metro'] * C * (1-B) - C*B * (1-A)
                                    f_der_c = q_e_traveler[OD_idx] * A * B * C * temp_fi
                                    f_2nd_der_c = q_e_traveler[OD_idx] * A * B * C * (temp_fi**2 + C*(1-C)*(-1/(theta_dict['theta_' + string1]**2) + (1-B)/(theta_dict['theta_' + string1]*theta_dict['theta_1_metro']) + B*(1-A)/theta_dict['theta_' + string1]) \
                                                                      + C**2*(-B*(1-B)/(theta_dict['theta_1_metro']**2) + B*(1-B)*(1-A)/theta_dict['theta_1_metro'] - B**2 * A * (1-A)))
                            else:
                                temp_fi = -1/theta_dict['theta_' + string1]* (1-C) - 1/theta_dict['theta_1_driving'] * C * (1-B) - C*B * (1-A)
                                f_der_c = q_e_traveler[OD_idx] * A * B * C * temp_fi
                                f_2nd_der_c = q_e_traveler[OD_idx] * A * B * C * (temp_fi**2 + C*(1-C)*(-1/(theta_dict['theta_' + string1]**2) + (1-B)/(theta_dict['theta_' + string1]*theta_dict['theta_1_driving']) + B*(1-A)/theta_dict['theta_' + string1]) \
                                                                      + C**2*(-B*(1-B)/(theta_dict['theta_1_driving']**2) + B*(1-B)*(1-A)/theta_dict['theta_1_driving'] - B**2 * A * (1-A)))
                            assert not np.isnan(f_der_c).any()
                            beta_der_dict['beta_tt_car']['pnr'][path_pnr_idx] = f_der_c * OD_path_alltts_list_pnr[O][D][i]['car'] / 60
                            beta_der_dict['beta_tt_bus']['pnr'][path_pnr_idx] = f_der_c * OD_path_alltts_list_pnr[O][D][i]['bus'] / 60
                            beta_der_dict['beta_tt_metro']['pnr'][path_pnr_idx] = f_der_c * OD_path_alltts_list_pnr[O][D][i]['metro'] /60
                            beta_der_dict['beta_waiting_bus']['pnr'][path_pnr_idx] = f_der_c * OD_path_alltts_list_pnr[O][D][i]['bus_waiting'] /60
                            beta_der_dict['beta_waiting_metro']['pnr'][path_pnr_idx] = f_der_c * OD_path_alltts_list_pnr[O][D][i]['metro_waiting'] /60
                            beta_der_dict['beta_walking']['pnr'][path_pnr_idx] = f_der_c * OD_path_alltts_list_pnr[O][D][i]['walking'] / 60
                            has_bus = 1 if 'bus' in string1 else 0
                            has_metro = 1 if 'metro' in string1 else 0
                            beta_der_dict['beta_money']['pnr'][path_pnr_idx] = f_der_c * (bus_fare * has_bus + metro_fare * has_metro + disutility_info_df['pnr_parking_fee'].values[0]) # again, assume pnr parking fee is the same for all pnr parking lots
                            gamma_der_dict['gamma_income_' + string1][path_pnr_idx] = f_der_c * disutility_info_df[disutility_info_df['O_id'] == O]['median_income'].values[0]
                            gamma_der_dict['gamma_Originpopden_' + string1][path_pnr_idx] = f_der_c * disutility_info_df[disutility_info_df['O_id'] == O]['pop_den'].values[0]
                            gamma_der_dict['gamma_Destpopden_' + string1][path_pnr_idx] = f_der_c * disutility_info_df[disutility_info_df['D_id'] == D]['pop_den'].values[0]
                            alpha_der_dict['alpha_' + string1][path_pnr_idx] = f_der_c

                            coef = dict()
                            coef['beta_tt_car'], coef['beta_tt_bus'], coef['beta_tt_metro'], coef['beta_waiting_bus'], coef['beta_waiting_metro'], coef['beta_walking'], coef['beta_money'], \
                                coef['gamma_income_' + string1], coef['gamma_Originpopden_' + string1], coef['gamma_Destpopden_' + string1], coef['alpha_' + string1] = OD_path_alltts_list_pnr[O][D][i]['car'] / 60, \
                                OD_path_alltts_list_pnr[O][D][i]['bus'] / 60, OD_path_alltts_list_pnr[O][D][i]['metro'] / 60, OD_path_alltts_list_pnr[O][D][i]['bus_waiting'] / 60, \
                                OD_path_alltts_list_pnr[O][D][i]['metro_waiting'] / 60, OD_path_alltts_list_pnr[O][D][i]['walking'] / 60, bus_fare * has_bus + metro_fare * has_metro + disutility_info_df['pnr_parking_fee'].values[0], \
                                disutility_info_df[disutility_info_df['O_id'] == O]['median_income'].values[0], disutility_info_df[disutility_info_df['O_id'] == O]['pop_den'].values[0], \
                                disutility_info_df[disutility_info_df['D_id'] == D]['pop_den'].values[0], 1
                            for string3 in ['beta_tt_car', 'beta_tt_bus', 'beta_tt_metro', 'beta_waiting_bus', 'beta_waiting_metro', 'beta_walking', 'beta_money', 'gamma_income_' + string1, 'gamma_Originpopden_' + string1, \
                                            'gamma_Destpopden_' + string1, 'alpha_' + string1]:
                                for string2 in ['beta_tt_car', 'beta_tt_bus', 'beta_tt_metro', 'beta_waiting_bus', 'beta_waiting_metro', 'beta_walking', 'beta_money']:
                                    beta_2nd_der_dict[string2][string3]['pnr'][path_pnr_idx] = f_2nd_der_c * coef[string2] * coef[string3]
                                for string2 in ['gamma_income_' + string1, 'gamma_Originpopden_' + string1, 'gamma_Destpopden_' + string1]:
                                    gamma_2nd_der_dict[string2][string3][path_pnr_idx] = f_2nd_der_c * coef[string2] * coef[string3]
                                for string2 in ['alpha_' + string1]:
                                    alpha_2nd_der_dict[string2][string3][path_pnr_idx] = f_2nd_der_c * coef[string2] * coef[string3]
        
        assert all(not np.isnan(arr).any() for arr in beta_der_dict['beta_tt_bus']['transit'])
        assert all(not np.isnan(arr).any() for arr in beta_der_dict['beta_tt_metro']['transit'])
        assert all(not np.isnan(arr).any() for arr in beta_der_dict['beta_tt_bus']['pnr'])
        assert all(not np.isnan(arr).any() for arr in beta_der_dict['beta_tt_metro']['pnr'])
        assert all(not np.isnan(arr).any() for arr in beta_der_dict['beta_walking']['transit'])
        assert all(not np.isnan(arr).any() for arr in beta_der_dict['beta_walking']['pnr'])
        assert all(not np.isnan(arr).any() for arr in beta_der_dict['beta_waiting_bus']['transit'])
        assert all(not np.isnan(arr).any() for arr in beta_der_dict['beta_waiting_bus']['pnr'])
        assert all(not np.isnan(arr).any() for arr in beta_der_dict['beta_waiting_metro']['transit'])
        assert all(not np.isnan(arr).any() for arr in beta_der_dict['beta_waiting_metro']['pnr'])


        return beta_der_dict, gamma_der_dict, alpha_der_dict, beta_2nd_der_dict, gamma_2nd_der_dict, alpha_2nd_der_dict
                                        

    def solve_DSUE(self, input_file_folder, q_e_traveler, nested_type, theta_dict, beta_dict, alpha_dict, gamma_dict, use_robust_tt, MSA_max_inter, MSA_step_size_gamma,
                   MSA_step_size_Gamma, MSA_convergence_threshold, init_route_portion, use_file_as_init_and_first_inter, f_car_driving, f_transit, f_pnr, is_testing=False):
        
        if not use_file_as_init_and_first_inter:
            with open(os.path.join(input_file_folder, 'dode_data.pickle'), 'rb') as f:
                [_, _, _, _, _, _, _, od_mode_connectivity, _,_] = pickle.load(f)
            if init_route_portion is None:
                q_e_traveler = q_e_traveler.reshape((int(len(q_e_traveler) / self.num_assign_interval), self.num_assign_interval), order='F')
                f_car_driving, f_transit, f_pnr = [], [], []
                for idx, row in od_mode_connectivity.iterrows():
                    O = row['OriginNodeID']
                    D = row['DestNodeID']
                    assigned_rate = 0
                    if row['BusTransit'] == 1:
                        for _ in range(len(self.nb.path_table_bustransit.path_dict[O][D].path_list)):
                            f_transit.append(q_e_traveler[idx] * 0.4 / len(self.nb.path_table_bustransit.path_dict[O][D].path_list))
                        assigned_rate += 0.4
                    if row['PNR'] == 1:
                        for _ in range(len(self.nb.path_table_pnr.path_dict[O][D].path_list)):
                            f_pnr.append(q_e_traveler[idx] * 0.1 / len(self.nb.path_table_pnr.path_dict[O][D].path_list))
                        assigned_rate += 0.1
                    for _ in range(len(self.nb.path_table_driving.path_dict[O][D].path_list)):
                        f_car_driving.append(q_e_traveler[idx] * (1 - assigned_rate) / len(self.nb.path_table_driving.path_dict[O][D].path_list))
                f_car_driving = np.array(f_car_driving).flatten(order='F')
                f_transit = np.array(f_transit).flatten(order='F')
                f_pnr = np.array(f_pnr).flatten(order='F')
                q_e_traveler = q_e_traveler.flatten(order='F')
            else:
                f_car_driving = init_route_portion['driving'].dot(q_e_traveler)
                f_transit = init_route_portion['transit'].dot(q_e_traveler)
                f_pnr = init_route_portion['pnr'].dot(q_e_traveler)
        f_truck_driving = np.zeros((self.num_path_driving, self.num_assign_interval)).flatten(order='F')

        k = 0
        gap_list, f_car_driving_list, f_transit_list, f_pnr_list = [], [], [], []
        f_car_driving_list.append(f_car_driving)
        f_transit_list.append(f_transit)
        f_pnr_list.append(f_pnr)
        while k < MSA_max_inter:
            OD_path_tt_list_driving, OD_path_alltts_list_transit, OD_path_alltts_list_pnr = self.compute_path_travel_waiting_walking_time(input_file_folder, f_car_driving, f_truck_driving, f_transit, f_pnr, use_robust_tt)
            OD_path_cost_list_driving, OD_path_cost_list_transit, OD_path_cost_list_pnr = self.compute_path_cost_all(input_file_folder, OD_path_tt_list_driving, OD_path_alltts_list_transit, OD_path_alltts_list_pnr, beta_dict, alpha_dict, gamma_dict)
            y_f_car_driving, y_f_transit, y_f_pnr, probs_record_dict = self.mode_route_choice_from_cost(input_file_folder, OD_path_cost_list_driving, OD_path_cost_list_transit, OD_path_cost_list_pnr, q_e_traveler, nested_type, theta_dict)
            
            assert not np.isnan(y_f_car_driving).any()
            assert not np.isnan(y_f_transit).any()
            assert not np.isnan(y_f_pnr).any()

            gap = (np.linalg.norm(y_f_car_driving-f_car_driving) ** 2 + np.linalg.norm(y_f_transit-f_transit) ** 2 + np.linalg.norm(y_f_pnr-f_pnr) ** 2) / \
                    (np.linalg.norm(f_car_driving) ** 2 + np.linalg.norm(f_transit) ** 2 + np.linalg.norm(f_pnr) ** 2)
            if is_testing:
                gap_list.append(gap)
            if gap < MSA_convergence_threshold:
                break
            if k == 0:
                MSA_beta = 1
            elif np.linalg.norm(y_f_car_driving-f_car_driving) + np.linalg.norm(y_f_transit-f_transit) + np.linalg.norm(y_f_pnr-f_pnr) >= \
                    np.linalg.norm(y_f_car_driving_old-f_car_driving_old) + np.linalg.norm(y_f_transit_old-f_transit_old) + np.linalg.norm(y_f_pnr_old-f_pnr_old):
                MSA_beta = MSA_beta_old + MSA_step_size_Gamma
            else:
                MSA_beta = MSA_beta_old + MSA_step_size_gamma
            
            f_car_driving_old = f_car_driving
            f_transit_old = f_transit
            f_pnr_old = f_pnr
            MSA_beta_old = MSA_beta
            y_f_car_driving_old = y_f_car_driving
            y_f_transit_old = y_f_transit
            y_f_pnr_old = y_f_pnr
        
            f_car_driving = f_car_driving_old + (1/MSA_beta) * (y_f_car_driving - f_car_driving_old)
            f_transit = f_transit_old + (1/MSA_beta) * (y_f_transit - f_transit_old)
            f_pnr = f_pnr_old + (1/MSA_beta) * (y_f_pnr - f_pnr_old)

            if is_testing:
                f_car_driving_list.append(f_car_driving)
                f_transit_list.append(f_transit)
                f_pnr_list.append(f_pnr)

            k += 1

        if is_testing:
            return gap_list, f_car_driving_list, f_transit_list, f_pnr_list
        
        # k_probs_driving = probs_record_dict['k_probs_driving']
        # k_probs_transit = probs_record_dict['k_probs_transit']
        # k_probs_pnr = probs_record_dict['k_probs_pnr']
        # gm_probs_driving = probs_record_dict['gm_probs_driving']
        # gm_probs_transit = probs_record_dict['gm_probs_transit']
        # gm_probs_pnr = probs_record_dict['gm_probs_pnr']
        # m_probs_driving = probs_record_dict['m_probs_driving']
        # m_probs_transit = probs_record_dict['m_probs_transit']
        # m_probs_pnr = probs_record_dict['m_probs_pnr']
        # assert all(not np.isnan(arr).any() for arr in k_probs_driving)
        # assert all(not np.isnan(arr).any() for arr in k_probs_transit)
        # assert all(not np.isnan(arr).any() for arr in k_probs_pnr)
        # assert all(not np.isnan(arr).any() for arr in gm_probs_driving)
        # assert all(not np.isnan(arr).any() for arr in gm_probs_transit)
        # assert all(not np.isnan(arr).any() for arr in gm_probs_pnr)
        # assert all(not np.isnan(arr).any() for arr in m_probs_driving)
        # assert all(not np.isnan(arr).any() for arr in m_probs_transit)
        # assert all(not np.isnan(arr).any() for arr in m_probs_pnr)
        
        #********************for testing purpose only********************
        # if 'probs_record_dict' not in locals():
        #     probs_record_dict = None
        #     OD_path_tt_list_driving  = None
        #     OD_path_alltts_list_transit = None
        #     OD_path_alltts_list_pnr = None
        #     gap = None
        #*********************************************************************

        # compute the derivatives of disutility parameters. Note: if not achieving DSUE, the derivatives are not (cannot be) accurate
        beta_der_dict, gamma_der_dict, alpha_der_dict, beta_2nd_der_dict, gamma_2nd_der_dict, alpha_2nd_der_dict \
              = self.compute_para_derivatives(input_file_folder, q_e_traveler, nested_type, theta_dict, OD_path_tt_list_driving, OD_path_alltts_list_transit, OD_path_alltts_list_pnr,
                                 probs_record_dict)
        assert not np.isnan(f_car_driving).any()
        assert not np.isnan(f_transit).any()
        assert not np.isnan(f_pnr).any()

        
        return f_car_driving, f_transit, f_pnr, beta_der_dict, gamma_der_dict, alpha_der_dict, k, gap, beta_2nd_der_dict, gamma_2nd_der_dict, alpha_2nd_der_dict


    def estimate_JointDemandDisutility(self, init_q_e_traveler_scale = 5, init_q_e_traveler = None,
                                    traveler_step_size = 0.1, truck_step_size=0.01, 
                                    gamma_truck = 0.9, gamma_traveler = 0.9, gamma_disutility_paras = 0.9,
                                    link_car_flow_weight=1, link_truck_flow_weight=1, link_passenger_flow_weight=1, link_bus_flow_weight=1,
                                    link_car_tt_weight=1, link_truck_tt_weight=1, link_passenger_tt_weight=1, link_bus_tt_weight=1, ODloss_weight=1,
                                    max_epoch=100, algo="NAdam", explicit_bus=1, 
                                    use_file_as_init=None, save_folder=None, starting_epoch=0, random_init=True,
                                    q_e_traveler_in_loss = None, q_truck_in_loss = None,
                                    input_file_folder=None, nested_type =1, theta_dict=None, beta_dict=None, alpha_dict=None, gamma_dict=None, # if estimate disutility paras, these will be initial values
                                    use_robust_tt=True, MSA_max_inter=10, MSA_step_size_gamma=0.2, MSA_step_size_Gamma=1.1, MSA_convergence_threshold=0.0001,
                                    beta_step_size=0.1, alpha_step_size=0.1, gamma_step_size=0.1, removed_para_idx=None):
        
        if save_folder is not None and not os.path.exists(save_folder):
            os.mkdir(save_folder)

        if np.isscalar(link_car_flow_weight):
            link_car_flow_weight = np.ones(max_epoch, dtype=bool) * link_car_flow_weight
        assert(len(link_car_flow_weight) == max_epoch)

        if np.isscalar(link_truck_flow_weight):
            link_truck_flow_weight = np.ones(max_epoch, dtype=bool) * link_truck_flow_weight
        assert(len(link_truck_flow_weight) == max_epoch)

        if np.isscalar(link_passenger_flow_weight):
            link_passenger_flow_weight = np.ones(max_epoch, dtype=bool) * link_passenger_flow_weight
        assert(len(link_passenger_flow_weight) == max_epoch)

        if np.isscalar(link_bus_flow_weight):
            link_bus_flow_weight = np.ones(max_epoch, dtype=bool) * link_bus_flow_weight
        assert(len(link_bus_flow_weight) == max_epoch)

        if np.isscalar(link_car_tt_weight):
            link_car_tt_weight = np.ones(max_epoch, dtype=bool) * link_car_tt_weight
        assert(len(link_car_tt_weight) == max_epoch)

        if np.isscalar(link_truck_tt_weight):
            link_truck_tt_weight = np.ones(max_epoch, dtype=bool) * link_truck_tt_weight
        assert(len(link_truck_tt_weight) == max_epoch)

        if np.isscalar(link_passenger_tt_weight):
            link_passenger_tt_weight = np.ones(max_epoch, dtype=bool) * link_passenger_tt_weight
        assert(len(link_passenger_tt_weight) == max_epoch)

        if np.isscalar(link_bus_tt_weight):
            link_bus_tt_weight = np.ones(max_epoch, dtype=bool) * link_bus_tt_weight
        assert(len(link_bus_tt_weight) == max_epoch)
        
        loss_list = list()
        DSUE_k_list, DSUE_gap_list = list(), list()
        best_epoch = starting_epoch
        best_q_e_truck = 0
        best_q_e_traveler = 0
        best_x_e_car, best_x_e_truck, best_x_e_passenger, best_x_e_bus, best_tt_e_car, best_tt_e_truck, best_tt_e_passenger, best_tt_e_bus = 0, 0, 0, 0, 0, 0, 0, 0
        # read from files as init values
        if use_file_as_init is not None:
            # most recent
            _, _, loss_list, best_epoch,\
                _, _, \
                _, _, _, _, \
                _, _, _ , _, _, _, _, _, \
                _, _, _, DSUE_k_list, DSUE_gap_list, _ ,_, _ = pickle.load(open(use_file_as_init, 'rb'))
            # best
            use_file_as_init = os.path.join(save_folder, '{}_iteration_estimate_demand.pickle'.format(best_epoch))
            _, _, _, _, best_q_e_traveler, best_q_e_truck, \
                best_f_car_driving, best_f_truck_driving, best_f_transit, best_f_pnr, \
                best_x_e_car, best_x_e_truck, best_x_e_passenger, best_x_e_bus, best_tt_e_car, best_tt_e_truck, best_tt_e_passenger, best_tt_e_bus, \
                best_beta_dict, best_gamma_dict, best_alpha_dict, _ ,_ , _ , _, _= pickle.load(open(use_file_as_init, 'rb'))

            q_e_truck = best_q_e_truck
            q_e_traveler = best_q_e_traveler
            f_car_driving, f_truck_driving, f_transit, f_pnr = best_f_car_driving, best_f_truck_driving, best_f_transit, best_f_pnr
            beta_dict, gamma_dict, alpha_dict = best_beta_dict, best_gamma_dict, best_alpha_dict
            
            # update demand and route portions according to the path flow (maybe not necessary, since in solve_DSUE, we will update the demand based on path flow again)
            self.nb.update_demand_path_driving(f_car_driving, f_truck_driving)
            self.nb.update_demand_path_bustransit(f_transit)
            self.nb.update_demand_path_pnr(f_pnr)
            self.nb.get_mode_portion_matrix() # update total demand
                
        else:
            if random_init: # for demand only, init_values parameters of disutility functions are always from the input of this function if not using file as init
                # q_e: num_OD x num_assign_interval flattened in F order
                q_e_traveler = self.init_demand_flow(len(self.demand_list_total_passenger), init_scale=init_q_e_traveler_scale)
                q_e_truck = self.init_demand_flow(len(self.demand_list_truck_driving), init_scale=0)
                # uniform
                self.init_mode_route_portions()
                f_car_driving, f_transit, f_pnr = None, None, None # this is because these are part of input in solve_DSUE if use_file_as_init is True and if the 1st iteration, 
                # will not be used actually but we need to have these variables
                f_truck_driving = np.zeros((self.num_path_driving, self.num_assign_interval)).flatten(order='F') # if estimate truck, make sure this is consistent with q_e_truck, route portion will be fixed
            else: # use input files as init -  demand and route portions are already initialized in the network builder
                # note: it will use the by_mode demand from the input files: need to make sure the summation of by_mode demand is consistent with the init total demand in the input (init_q_e_traveler)
                # the specific mode split in the input files will not be used since in the solve_DSUE process later, it has an inherent mode split setting
                # Note: if we want to use a specific mode split pattern as init, define it well in the input files, and modify the setting in the solve_DSUE function (simply
                #  change the the calculation of init path flows, get them from the inintalized network builder)
                q_e_traveler = init_q_e_traveler
                self.nb.get_mode_portion_matrix() # make the total demand consistent
                q_e_truck = self.init_demand_flow(len(self.demand_list_truck_driving), init_scale=0)
                f_car_driving, f_transit, f_pnr = None, None, None
                f_truck_driving = np.zeros((self.num_path_driving, self.num_assign_interval)).flatten(order='F') # if estimate truck, make sure this is consistent with q_e_truck, route portion will be fixed
                # if want the route portion of truck to be from input files, here use: _, P_path_truck_driving = self.nb.get_route_portion_matrix_driving(); f_truck_driving = P_path_truck_driving.dot(q_e_truck)
            
                
        # relu
        q_e_truck = np.maximum(q_e_truck, 1e-6)
        q_e_traveler = np.maximum(q_e_traveler, 1e-6)
        beta_array, gamma_array, alpha_array = swap_dict_and_array(beta_dict, gamma_dict, alpha_dict)
        beta_array, gamma_array, alpha_array = np.maximum(beta_array, 1e-6), np.maximum(gamma_array, 1e-6), np.maximum(alpha_array, 1e-6)
        if removed_para_idx is not None:
            beta_array[removed_para_idx['beta']] = 0
            gamma_array[removed_para_idx['gamma']] = 0
            alpha_array[removed_para_idx['alpha']] = 0

        # q_e_truck_tensor = torch.from_numpy(q_e_truck)
        q_e_traveler_tensor = torch.from_numpy(q_e_traveler)
        beta_tensor, gamma_tensor, alpha_tensor = torch.from_numpy(beta_array), torch.from_numpy(gamma_array), torch.from_numpy(alpha_array)

        # q_e_truck_tensor.requires_grad = True
        q_e_traveler_tensor.requires_grad = True
        beta_tensor.requires_grad, gamma_tensor.requires_grad, alpha_tensor.requires_grad = True, True, True

        algo_dict = {
            "SGD": torch.optim.SGD,
            "NAdam": torch.optim.NAdam,
            "Adam": torch.optim.Adam,
            "Adamax": torch.optim.Adamax,
            "AdamW": torch.optim.AdamW,
            "RAdam": torch.optim.RAdam,
            "Adagrad": torch.optim.Adagrad,
            "Adadelta": torch.optim.Adadelta
        }

        optimizers = [
        algo_dict[algo]([{'params': q_e_traveler_tensor}], lr=traveler_step_size),
        algo_dict[algo]([{'params': beta_tensor, 'lr': beta_step_size}, {'params': gamma_tensor, 'lr': gamma_step_size}, {'params': alpha_tensor, 'lr': alpha_step_size}]),
        # algo_dict[algo]([{'params': q_e_truck_tensor}], lr=truck_step_size)
        ]

        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(optimizers[0], gamma=gamma_traveler),
            torch.optim.lr_scheduler.ExponentialLR(optimizers[1], gamma=gamma_disutility_paras),
            # torch.optim.lr_scheduler.ExponentialLR(optimizers[2], gamma=gamma_truck)
        ]


        for i in range(max_epoch):
            seq = np.random.permutation(self.num_data)
            loss = float(0)
            loss_dict = {
                "car_count_loss": 0.,
                "truck_count_loss": 0.,
                "bus_count_loss": 0.,
                "passenger_count_loss": 0.,
                "car_tt_loss": 0.,
                "truck_tt_loss": 0.,
                "bus_tt_loss": 0.,
                "passenger_tt_loss": 0.,
                "car_count_loss_weighted": 0.,
                "truck_count_loss_weighted": 0.,
                "bus_count_loss_weighted": 0.,
                "passenger_count_loss_weighted": 0.,
                "car_tt_loss_weighted": 0.,
                "truck_tt_loss_weighted": 0.,
                "bus_tt_loss_weighted": 0.,
                "passenger_tt_loss_weighted": 0.,
                "total_loss_weighted": 0.
            }

            self.config['link_car_flow_weight'] = link_car_flow_weight[i] * (self.config['use_car_link_flow'] or self.config['compute_car_link_flow_loss'])
            self.config['link_truck_flow_weight'] = link_truck_flow_weight[i] * (self.config['use_truck_link_flow'] or self.config['compute_truck_link_flow_loss'])
            self.config['link_passenger_flow_weight'] = link_passenger_flow_weight[i] * (self.config['use_passenger_link_flow'] or self.config['compute_passenger_link_flow_loss'])
            self.config['link_bus_flow_weight'] = link_bus_flow_weight[i] * (self.config['use_bus_link_flow'] or self.config['compute_bus_link_flow_loss'])

            self.config['link_car_tt_weight'] = link_car_tt_weight[i] * (self.config['use_car_link_tt'] or self.config['compute_car_link_tt_loss'])
            self.config['link_truck_tt_weight'] = link_truck_tt_weight[i] * (self.config['use_truck_link_tt'] or self.config['compute_truck_link_tt_loss'])
            self.config['link_passenger_tt_weight'] = link_passenger_tt_weight[i] * (self.config['use_passenger_link_tt'] or self.config['compute_passenger_link_tt_loss'])
            self.config['link_bus_tt_weight'] = link_bus_tt_weight[i] * (self.config['use_bus_link_tt'] or self.config['compute_bus_link_tt_loss'])
            

            for j in seq:  # TODO: not update for each data record: update after all data records
                # retrieve one record of observed data
                one_data_dict = self._get_one_data(j)

                # forward, get path flow with q_e_traveler
                if i == 0 and not use_file_as_init: 
                    init_route_portion = None
                if use_file_as_init and i == 0:
                    use_file_as_init_and_first_inter = True
                else:
                    use_file_as_init_and_first_inter = False
                f_car_driving, f_transit, f_pnr, beta_der_dict, gamma_der_dict, alpha_der_dict, k, gap, beta_2nd_der_dict, gamma_2nd_der_dict, alpha_2nd_der_dict\
                      = self.solve_DSUE(input_file_folder, q_e_traveler, nested_type, theta_dict, beta_dict, alpha_dict, 
                                                                  gamma_dict, use_robust_tt, MSA_max_inter, MSA_step_size_gamma,
                                                                    MSA_step_size_Gamma, MSA_convergence_threshold, init_route_portion, use_file_as_init_and_first_inter,
                                                                    f_car_driving, f_transit, f_pnr)
                
                # if 'P_path_truck_driving' in locals():
                #     f_truck_driving = P_path_truck_driving.dot(q_e_truck) # assume truck route portion is fixed

                # Note: below for solving the mode/route portion matrices, follow the previous code and logic for ease, 
                # theorectically equivalent to our case where we estimate total demand and have nested mode such as metro, etc..

                # update demand and route portions according to the new path flow, for computing mode/route portion matrices
                self.nb.update_demand_path_driving(f_car_driving, f_truck_driving)
                self.nb.update_demand_path_bustransit(f_transit)
                self.nb.update_demand_path_pnr(f_pnr)

                # P_mode: (num_OD_one_mode * num_assign_interval, num_OD * num_assign_interval)
                P_mode_driving, P_mode_transit, P_mode_pnr = self.nb.get_mode_portion_matrix() # also at the same time update total demand

                # P_path: (num_path * num_assign_interval, num_OD_one_mode * num_assign_interval)
                P_path_car_driving, P_path_truck_driving = self.nb.get_route_portion_matrix_driving()
                P_path_passenger_transit = self.nb.get_route_portion_matrix_bustransit()
                P_path_pnr = self.nb.get_route_portion_matrix_pnr()
                
                # init_route_portion = None
                init_route_portion = dict()
                init_route_portion['driving'] = P_path_car_driving.dot(P_mode_driving)
                init_route_portion['transit'] = P_path_passenger_transit.dot(P_mode_transit)
                init_route_portion['pnr'] = P_path_pnr.dot(P_mode_pnr)

                f_car_driving = np.maximum(f_car_driving, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])
                # f_truck_driving = np.maximum(f_truck_driving, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])
                # this alleviate trapping in local minima, when f = 0 -> grad = 0 is not helping
                f_transit = np.maximum(f_transit, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])
                f_pnr = np.maximum(f_pnr, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])
                
                if loss_list:
                    init_loss = loss_list[0][1]
                else:
                    init_loss = None
                # f_grad: num_path * num_assign_interval
                # f_car_driving_grad, f_truck_driving_grad, f_passenger_transit_grad, f_car_pnr_grad, f_passenger_pnr_grad, _, \
                #     tmp_loss, tmp_loss_dict, _, x_e_car, x_e_truck, x_e_passenger, x_e_bus, tt_e_car, tt_e_truck, tt_e_passenger, tt_e_bus = \
                f_car_driving_grad, f_truck_driving_grad, f_passenger_bustransit_grad, f_car_pnr_grad, f_passenger_pnr_grad, _, \
                    tmp_loss, tmp_loss_dict, _, x_e_car, x_e_truck, x_e_passenger, x_e_bus, x_e_BoardingAlighting_count, tt_e_car, tt_e_truck, tt_e_passenger, tt_e_bus,\
                        f_transit_ULP, stop_arrival_departure_travel_time_df, Jacobian_dict = \
                        self.compute_path_flow_grad_and_loss(one_data_dict, f_car_driving, f_truck_driving, f_transit, f_pnr, f_bus=None, 
                        fix_bus=True, counter=0, run_mmdta_adaptive=False, init_loss = init_loss, explicit_bus=explicit_bus, isUsingbymode = False)
                assert not np.isnan(f_car_driving_grad).any()
                assert not np.isnan(f_passenger_transit_grad).any()
                assert not np.isnan(f_car_pnr_grad).any()
                assert not np.isnan(f_passenger_pnr_grad).any()
                
                # q_mode_grad: num_OD_one_mode * num_assign_interval
                q_grad_car_driving = P_path_car_driving.T.dot(f_car_driving_grad)  # link_car_flow_weight, link_car_tt_weight
                q_grad_car_pnr = P_path_pnr.T.dot(f_car_pnr_grad)  # link_car_flow_weight, link_car_tt_weight
                # q_truck_grad = P_path_truck_driving.T.dot(f_truck_driving_grad)  # link_truck_flow_weight, link_truck_tt_weight, link_bus_tt_weight
                q_grad_passenger_transit = P_path_passenger_transit.T.dot(f_passenger_transit_grad)  # link_passenger_flow_weight, link_bus_tt_weight
                q_grad_passenger_pnr = P_path_pnr.T.dot(f_passenger_pnr_grad)  # link_passenger_flow_weight, link_bus_tt_weight

                # q_grad: num_OD * num_assign_interval
                q_traveler_grad = P_mode_driving.T.dot(q_grad_car_driving) \
                                   + P_mode_transit.T.dot(q_grad_passenger_transit) \
                                   + P_mode_pnr.T.dot(q_grad_passenger_pnr + q_grad_car_pnr)
                
                assert not np.isnan(q_traveler_grad).any()
                assert not np.isnan(q_grad_car_driving).any()
                assert not np.isnan(q_grad_car_pnr).any()
                assert not np.isnan(q_grad_passenger_transit).any()
                assert not np.isnan(q_grad_passenger_pnr).any()

                # use OD loss or not
                eps = 1e-8
                if self.config['use_OD_loss']:
                    q_traveler_grad += (q_e_traveler - q_e_traveler_in_loss) / (np.linalg.norm(q_e_traveler - q_e_traveler_in_loss) + eps) * ODloss_weight
                    # q_truck_grad += (q_e_truck - q_truck_in_loss) / (np.linalg.norm(q_e_truck - q_truck_in_loss) + eps) * ODloss_weight

                # grads for disutility parameters
                beta_grad, gamma_grad, alpha_grad = dict(), dict(), dict()
                beta_2nd_grad, gamma_2nd_grad, alpha_2nd_grad = dict(), dict(), dict()
                for key in beta_dict.keys():
                    if key == 'beta_tt_car':
                        beta_grad[key] = np.dot(beta_der_dict[key]['driving'].flatten(order='F'), f_car_driving_grad) + \
                                        np.dot(beta_der_dict[key]['pnr'].flatten(order='F'), f_car_pnr_grad) + \
                                        np.dot(beta_der_dict[key]['pnr'].flatten(order='F'), f_passenger_pnr_grad)
                        beta_2nd_grad[key] = dict()
                        for k in beta_2nd_der_dict[key].keys():
                            beta_2nd_grad[key][k] = np.dot(beta_2nd_der_dict[key][k]['driving'].flatten(order='F'), f_car_driving_grad) + \
                                        np.dot(beta_2nd_der_dict[key][k]['pnr'].flatten(order='F'), f_car_pnr_grad) + \
                                        np.dot(beta_2nd_der_dict[key][k]['pnr'].flatten(order='F'), f_passenger_pnr_grad)
                    elif key == 'beta_money':
                        beta_grad[key] = np.dot(beta_der_dict[key]['driving'].flatten(order='F'), f_car_driving_grad) + \
                                        np.dot(beta_der_dict[key]['pnr'].flatten(order='F'), f_car_pnr_grad) + \
                                        np.dot(beta_der_dict[key]['pnr'].flatten(order='F'), f_passenger_pnr_grad) + np.dot(beta_der_dict[key]['transit'].flatten(order='F'), \
                                                                                                                            f_passenger_transit_grad)
                        beta_2nd_grad[key] = dict()
                        for k in beta_2nd_der_dict[key].keys():
                            beta_2nd_grad[key][k] = np.dot(beta_2nd_der_dict[key][k]['driving'].flatten(order='F'), f_car_driving_grad) + \
                                        np.dot(beta_2nd_der_dict[key][k]['pnr'].flatten(order='F'), f_car_pnr_grad) + \
                                        np.dot(beta_2nd_der_dict[key][k]['pnr'].flatten(order='F'), f_passenger_pnr_grad) + np.dot(beta_2nd_der_dict[key][k]['transit'].flatten(order='F'), \
                                                                                                                            f_passenger_transit_grad)
                    else:
                        beta_grad[key] = np.dot(beta_der_dict[key]['transit'].flatten(order='F'), f_passenger_transit_grad) + \
                                        np.dot(beta_der_dict[key]['pnr'].flatten(order='F'), f_passenger_pnr_grad) + \
                                        np.dot(beta_der_dict[key]['pnr'].flatten(order='F'), f_car_pnr_grad)
                        beta_2nd_grad[key] = dict()
                        for k in beta_2nd_der_dict[key].keys():
                            beta_2nd_grad[key][k] = np.dot(beta_2nd_der_dict[key][k]['transit'].flatten(order='F'), f_passenger_transit_grad) + \
                                        np.dot(beta_2nd_der_dict[key][k]['pnr'].flatten(order='F'), f_passenger_pnr_grad) + \
                                        np.dot(beta_2nd_der_dict[key][k]['pnr'].flatten(order='F'), f_car_pnr_grad)
                        if np.isnan(beta_grad[key]).any():
                            print(f"NaN detected in beta_grad[{key}]")
                            pickle.dump([beta_grad,beta_der_dict,f_passenger_transit_grad,f_passenger_pnr_grad,f_car_pnr_grad], open(os.path.join(save_folder, 'errors.pickle'), 'wb'))
                            raise ValueError(f"NaN detected in beta_grad[{key}]")
                for string1 in ['income', 'Originpopden', 'Destpopden']:
                    gamma_grad['gamma_' + string1 + '_car'] = np.dot(gamma_der_dict['gamma_' + string1 + '_car'].flatten(order='F'), f_car_driving_grad)
                    gamma_2nd_grad['gamma_' + string1 + '_car'] = dict()
                    for k in gamma_2nd_der_dict['gamma_' + string1 + '_car'].keys():
                        gamma_2nd_grad['gamma_' + string1 + '_car'][k] = np.dot(gamma_2nd_der_dict['gamma_' + string1 + '_car'][k].flatten(order='F'), f_car_driving_grad)
                    for string2 in ['bus', 'metro', 'busmetro']:
                        gamma_grad['gamma_' + string1 + '_' + string2] = np.dot(gamma_der_dict['gamma_' + string1 + '_' + string2].flatten(order='F'), f_passenger_transit_grad)
                        gamma_2nd_grad['gamma_' + string1 + '_' + string2] = dict()
                        for k in gamma_2nd_der_dict['gamma_' + string1 + '_' + string2].keys():
                            gamma_2nd_grad['gamma_' + string1 + '_' + string2][k] = np.dot(gamma_2nd_der_dict['gamma_' + string1 + '_' + string2][k].flatten(order='F'), f_passenger_transit_grad)
                    for string2 in ['carbus', 'carmetro', 'carbusmetro']:
                        gamma_grad['gamma_' + string1 + '_' + string2] = np.dot(gamma_der_dict['gamma_' + string1 + '_' + string2].flatten(order='F'), f_car_pnr_grad) + \
                                        np.dot(gamma_der_dict['gamma_' + string1 + '_' + string2].flatten(order='F'), f_passenger_pnr_grad)
                        gamma_2nd_grad['gamma_' + string1 + '_' + string2] = dict()
                        for k in gamma_2nd_der_dict['gamma_' + string1 + '_' + string2].keys():
                            gamma_2nd_grad['gamma_' + string1 + '_' + string2][k] = np.dot(gamma_2nd_der_dict['gamma_' + string1 + '_' + string2][k].flatten(order='F'), f_car_pnr_grad) + \
                                        np.dot(gamma_2nd_der_dict['gamma_' + string1 + '_' + string2][k].flatten(order='F'), f_passenger_pnr_grad)
                for string1 in ['bus', 'metro', 'busmetro']:
                    alpha_grad['alpha_' + string1] = np.dot(alpha_der_dict['alpha_' + string1].flatten(order='F'), f_passenger_transit_grad)
                    alpha_2nd_grad['alpha_' + string1] = dict()
                    for k in alpha_2nd_der_dict['alpha_' + string1].keys():
                        alpha_2nd_grad['alpha_' + string1][k] = np.dot(alpha_2nd_der_dict['alpha_' + string1][k].flatten(order='F'), f_passenger_transit_grad)
                for string1 in ['carbus', 'carmetro', 'carbusmetro']:
                    alpha_grad['alpha_' + string1] = np.dot(alpha_der_dict['alpha_' + string1].flatten(order='F'), f_car_pnr_grad) + \
                                        np.dot(alpha_der_dict['alpha_' + string1].flatten(order='F'), f_passenger_pnr_grad)
                    alpha_2nd_grad['alpha_' + string1] = dict()
                    for k in alpha_2nd_der_dict['alpha_' + string1].keys():
                        alpha_2nd_grad['alpha_' + string1][k] = np.dot(alpha_2nd_der_dict['alpha_' + string1][k].flatten(order='F'), f_car_pnr_grad) + \
                                        np.dot(alpha_2nd_der_dict['alpha_' + string1][k].flatten(order='F'), f_passenger_pnr_grad)
                
                
                assert not np.isnan(beta_der_dict['beta_tt_bus']['transit']).any()
                assert not np.isnan(beta_der_dict['beta_tt_bus']['pnr']).any()
                assert not np.isnan(beta_der_dict['beta_tt_metro']['transit']).any()
                assert not np.isnan(beta_der_dict['beta_tt_metro']['pnr']).any()
                assert not np.isnan(beta_der_dict['beta_walking']['transit']).any()
                assert not np.isnan(beta_der_dict['beta_walking']['pnr']).any()
                assert not np.isnan(beta_der_dict['beta_waiting_bus']['transit']).any()
                assert not np.isnan(beta_der_dict['beta_waiting_bus']['pnr']).any()
                assert not np.isnan(beta_der_dict['beta_waiting_metro']['transit']).any()
                assert not np.isnan(beta_der_dict['beta_waiting_metro']['pnr']).any()
                for key, arr in beta_grad.items():
                    assert not np.isnan(arr).any(), f"NaN detected in array at key: {key}"
                
                beta_grad, gamma_grad, alpha_grad = swap_dict_and_array(beta_grad, gamma_grad, alpha_grad)
                assert not np.isnan(beta_grad).any()
                assert not np.isnan(gamma_grad).any()
                assert not np.isnan(alpha_grad).any()

                for optimizer in optimizers:
                    optimizer.zero_grad()

                q_e_traveler_tensor.grad = torch.from_numpy(q_traveler_grad)
                # q_e_truck_tensor.grad = torch.from_numpy(q_truck_grad)
                beta_tensor.grad, gamma_tensor.grad, alpha_tensor.grad = torch.from_numpy(beta_grad), torch.from_numpy(gamma_grad), torch.from_numpy(alpha_grad)
                
                for optimizer in optimizers:
                    optimizer.step()
                for scheduler in schedulers:
                    scheduler.step()

                # release memory
                f_car_driving_grad, f_truck_driving_grad, f_passenger_transit_grad, f_car_pnr_grad, f_passenger_pnr_grad = 0, 0, 0, 0, 0
                q_grad_car_driving, q_grad_car_pnr, q_truck_grad, q_grad_passenger_transit, q_grad_passenger_pnr, q_traveler_grad = 0, 0, 0, 0, 0, 0
                beta_grad, gamma_grad, alpha_grad = 0, 0, 0
                beta_der_dict, gamma_der_dict, alpha_der_dict = 0, 0, 0
                for optimizer in optimizers:
                    optimizer.zero_grad()

                q_e_traveler = q_e_traveler_tensor.data.cpu().numpy()
                # q_e_truck = q_e_truck_tensor.data.cpu().numpy()
                beta_array, gamma_array, alpha_array = beta_tensor.data.cpu().numpy(), gamma_tensor.data.cpu().numpy(), alpha_tensor.data.cpu().numpy()
                

                # relu
                q_e_traveler = np.maximum(q_e_traveler, 1e-6)
                q_e_truck = np.maximum(q_e_truck, 1e-6)
                beta_array, gamma_array, alpha_array = np.maximum(beta_array, 1e-6), np.maximum(gamma_array, 1e-6), np.maximum(alpha_array, 1e-6)
                if removed_para_idx is not None:
                    beta_array[removed_para_idx['beta']] = 0
                    gamma_array[removed_para_idx['gamma']] = 0
                    alpha_array[removed_para_idx['alpha']] = 0
                beta_dict, gamma_dict, alpha_dict = swap_dict_and_array(beta_array, gamma_array, alpha_array)

                loss += tmp_loss / float(self.num_data)
                for loss_type, loss_value in tmp_loss_dict.items():
                    loss_dict[loss_type] += loss_value / float(self.num_data)

                #  temporary not in use, since we do not need to compute path cost later on from macposts c++
                # if not self.config['use_car_link_tt'] and not self.config['use_truck_link_tt'] and not self.config['use_passenger_link_tt'] and not self.config['use_bus_link_tt']:
                #     # if any of these is true, dta.build_link_cost_map(True) is already invoked in compute_path_flow_grad_and_loss()
                #     dta.build_link_cost_map(False)

            print("Epoch:", starting_epoch + i, "Loss:", loss, self.print_separate_accuracy(loss_dict))
            loss_list.append([loss, loss_dict]) # xm: if use_file_as_init, wouldn't this be starting overwriting from the best_epoch?

            DSUE_k_list.append(k)
            DSUE_gap_list.append(gap)

            if (best_epoch == 0) or (loss_list[best_epoch][0] > loss_list[-1][0]):
                best_epoch = starting_epoch + i
                best_f_car_driving, best_f_truck_driving, best_f_transit, best_f_pnr, best_q_e_truck, best_q_e_traveler = \
                    f_car_driving, f_truck_driving, f_transit, f_pnr, q_e_truck, q_e_traveler

                best_x_e_car, best_x_e_truck, best_x_e_passenger, best_x_e_bus, best_tt_e_car, best_tt_e_truck, best_tt_e_passenger, best_tt_e_bus = \
                    x_e_car, x_e_truck, x_e_passenger, x_e_bus, tt_e_car, tt_e_truck, tt_e_passenger, tt_e_bus
                
                best_beta_dict, best_gamma_dict, best_alpha_dict = beta_dict, gamma_dict, alpha_dict

                best_DSUE_k, best_DSUE_gap = k, gap

                if save_folder is not None:
                    self.save_simulation_input_files(os.path.join(save_folder, 'input_files_estimate_demand'), 
                                                     f_car_driving, f_truck_driving, f_transit, f_pnr, f_bus=None,
                                                     explicit_bus=explicit_bus, historical_bus_waiting_time=0)

            if save_folder is not None:
                pickle.dump([loss, loss_dict, loss_list, best_epoch,
                             q_e_traveler, q_e_truck, 
                             f_car_driving, f_truck_driving, f_transit, f_pnr,
                             x_e_car, x_e_truck, x_e_passenger, x_e_bus, tt_e_car, tt_e_truck, tt_e_passenger, tt_e_bus,
                             beta_dict, gamma_dict, alpha_dict, DSUE_k_list, DSUE_gap_list, beta_2nd_grad, gamma_2nd_grad, alpha_2nd_grad], 
                            open(os.path.join(save_folder, str(starting_epoch + i) + '_iteration_estimate_demand.pickle'), 'wb'))
        
        print("Best loss at Epoch:", best_epoch, "Loss:", loss_list[best_epoch][0], self.print_separate_accuracy(loss_list[best_epoch][1]))
        return best_f_car_driving, best_f_truck_driving, best_f_transit, best_f_pnr, best_q_e_traveler, best_q_e_truck, \
                best_x_e_car, best_x_e_truck, best_x_e_passenger, best_x_e_bus, best_tt_e_car, best_tt_e_truck, best_tt_e_passenger, best_tt_e_bus, \
                best_beta_dict, best_gamma_dict, best_alpha_dict, best_DSUE_k, best_DSUE_gap, DSUE_k_list, DSUE_gap_list, loss_list, beta_2nd_grad, gamma_2nd_grad, alpha_2nd_grad


    def estimate_demand(self, init_scale_passenger=10, init_scale_truck=10, init_scale_bus=1,
                        car_driving_scale=10, truck_driving_scale=1, passenger_bustransit_scale=1, car_pnr_scale=5,
                        passenger_step_size=0.1, truck_step_size=0.01, bus_step_size=0.01,
                        link_car_flow_weight=1, link_truck_flow_weight=1, link_passenger_flow_weight=1, link_bus_flow_weight=1,
                        link_car_tt_weight=1, link_truck_tt_weight=1, link_passenger_tt_weight=1, link_bus_tt_weight=1,
                        max_epoch=100, adagrad=False, fix_bus=True, column_generation=False, use_tdsp=False,
                        alpha_mode=(1., 1.5, 2.), beta_mode=1, alpha_path=1, beta_path=1, 
                        use_file_as_init=None, save_folder=None, starting_epoch=0):
        if save_folder is not None and not os.path.exists(save_folder):
            os.mkdir(save_folder)
            
        if fix_bus:
            init_scale_bus = None
            bus_step_size = None
        else:
            assert(init_scale_bus is not None)
            assert(bus_step_size is not None)

        if np.isscalar(passenger_step_size):
            passenger_step_size = np.ones(max_epoch) * passenger_step_size
        if np.isscalar(truck_step_size):
            truck_step_size = np.ones(max_epoch) * truck_step_size
        if np.isscalar(bus_step_size):
            bus_step_size = np.ones(max_epoch) * bus_step_size
        assert(len(passenger_step_size) == max_epoch)
        assert(len(truck_step_size) == max_epoch)
        assert(len(bus_step_size) == max_epoch)

        if np.isscalar(column_generation):
            column_generation = np.ones(max_epoch, dtype=int) * column_generation
        assert(len(column_generation) == max_epoch)

        if np.isscalar(link_car_flow_weight):
            link_car_flow_weight = np.ones(max_epoch, dtype=bool) * link_car_flow_weight
        assert(len(link_car_flow_weight) == max_epoch)

        if np.isscalar(link_truck_flow_weight):
            link_truck_flow_weight = np.ones(max_epoch, dtype=bool) * link_truck_flow_weight
        assert(len(link_truck_flow_weight) == max_epoch)

        if np.isscalar(link_passenger_flow_weight):
            link_passenger_flow_weight = np.ones(max_epoch, dtype=bool) * link_passenger_flow_weight
        assert(len(link_passenger_flow_weight) == max_epoch)

        if np.isscalar(link_bus_flow_weight):
            link_bus_flow_weight = np.ones(max_epoch, dtype=bool) * link_bus_flow_weight
        assert(len(link_bus_flow_weight) == max_epoch)

        if np.isscalar(link_car_tt_weight):
            link_car_tt_weight = np.ones(max_epoch, dtype=bool) * link_car_tt_weight
        assert(len(link_car_tt_weight) == max_epoch)

        if np.isscalar(link_truck_tt_weight):
            link_truck_tt_weight = np.ones(max_epoch, dtype=bool) * link_truck_tt_weight
        assert(len(link_truck_tt_weight) == max_epoch)

        if np.isscalar(link_passenger_tt_weight):
            link_passenger_tt_weight = np.ones(max_epoch, dtype=bool) * link_passenger_tt_weight
        assert(len(link_passenger_tt_weight) == max_epoch)

        if np.isscalar(link_bus_tt_weight):
            link_bus_tt_weight = np.ones(max_epoch, dtype=bool) * link_bus_tt_weight
        assert(len(link_bus_tt_weight) == max_epoch)
        
        loss_list = list()
        best_epoch = starting_epoch
        best_f_car_driving, best_f_truck_driving, best_f_passenger_bustransit, best_f_car_pnr, best_f_bus, best_q_e_passenger, best_q_e_truck = 0, 0, 0, 0, 0, 0, 0
        best_x_e_car, best_x_e_truck, best_x_e_passenger, best_x_e_bus, best_tt_e_car, best_tt_e_truck, best_tt_e_passenger, best_tt_e_bus = 0, 0, 0, 0, 0, 0, 0, 0
        # read from files as init values
        if use_file_as_init is not None:
            # most recent
            _, _, loss_list, best_epoch, _, _, _, _, _, \
                _, _, _, _, _, \
                _, _, _, _, _, _, _, _ = pickle.load(open(use_file_as_init, 'rb'))
            # best
            use_file_as_init = os.path.join(save_folder, '{}_iteration_estimate_demand.pickle'.format(best_epoch))
            _, _, _, _, best_q_e_passenger, best_q_e_truck, _, _, _, \
                best_f_car_driving, best_f_truck_driving, best_f_passenger_bustransit, best_f_car_pnr, best_f_bus, \
                best_x_e_car, best_x_e_truck, best_x_e_passenger, best_x_e_bus, best_tt_e_car, best_tt_e_truck, best_tt_e_passenger, best_tt_e_bus \
                     = pickle.load(open(use_file_as_init, 'rb'))

            q_e_passenger, q_e_truck = best_q_e_passenger, best_q_e_truck
            f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus = \
                best_f_car_driving, best_f_truck_driving, best_f_passenger_bustransit, best_f_car_pnr, best_f_bus
            
            self.nb.update_demand_path_driving(f_car_driving, f_truck_driving)
            self.nb.update_demand_path_bustransit(f_passenger_bustransit)
            self.nb.update_demand_path_pnr(f_car_pnr)
        else:
            # q_e: num_OD x num_assign_interval flattened in F order
            q_e_passenger = self.init_demand_flow(len(self.demand_list_total_passenger), init_scale=init_scale_passenger)
            q_e_truck = self.init_demand_flow(len(self.demand_list_truck_driving), init_scale=init_scale_truck)

            # uniform
            self.init_mode_route_portions()

        # fixed bus path flow
        if fix_bus:
            f_bus = self.nb.demand_bus.path_flow_matrix.flatten(order='F')
        else:
            f_bus = self.init_demand_vector(self.num_assign_interval, self.num_path_busroute, init_scale_bus)
            self.nb.update_demand_path_busroute(f_bus)

        # relu
        q_e_passenger = np.maximum(q_e_passenger, 1e-6)
        q_e_truck = np.maximum(q_e_truck, 1e-6)
        f_bus = np.maximum(f_bus, 1e-6)

        for i in range(max_epoch):
            seq = np.random.permutation(self.num_data)
            loss = float(0)
            loss_dict = {
                "car_count_loss": 0.,
                "truck_count_loss": 0.,
                "bus_count_loss": 0.,
                "passenger_count_loss": 0.,
                "car_tt_loss": 0.,
                "truck_tt_loss": 0.,
                "bus_tt_loss": 0.,
                "passenger_tt_loss": 0.
            }

            self.config['link_car_flow_weight'] = link_car_flow_weight[i] * (self.config['use_car_link_flow'] or self.config['compute_car_link_flow_loss'])
            self.config['link_truck_flow_weight'] = link_truck_flow_weight[i] * (self.config['use_truck_link_flow'] or self.config['compute_truck_link_flow_loss'])
            self.config['link_passenger_flow_weight'] = link_passenger_flow_weight[i] * (self.config['use_passenger_link_flow'] or self.config['compute_passenger_link_flow_loss'])
            self.config['link_bus_flow_weight'] = link_bus_flow_weight[i] * (self.config['use_bus_link_flow'] or self.config['compute_bus_link_flow_loss'])

            self.config['link_car_tt_weight'] = link_car_tt_weight[i] * (self.config['use_car_link_tt'] or self.config['compute_car_link_tt_loss'])
            self.config['link_truck_tt_weight'] = link_truck_tt_weight[i] * (self.config['use_truck_link_tt'] or self.config['compute_truck_link_tt_loss'])
            self.config['link_passenger_tt_weight'] = link_passenger_tt_weight[i] * (self.config['use_passenger_link_tt'] or self.config['compute_passenger_link_tt_loss'])
            self.config['link_bus_tt_weight'] = link_bus_tt_weight[i] * (self.config['use_bus_link_tt'] or self.config['compute_bus_link_tt_loss'])

            if adagrad:
                sum_g_square_passenger = 1e-6
                sum_g_square_truck = 1e-6
                if not fix_bus:
                    sum_g_square_bus = 1e-6
            for j in seq:
                # retrieve one record of observed data
                one_data_dict = self._get_one_data(j)

                # P_mode: (num_OD_one_mode * num_assign_interval, num_OD * num_assign_interval)
                P_mode_driving, P_mode_bustransit, P_mode_pnr = self.nb.get_mode_portion_matrix()

                # q_e_mode: num_OD_one_mode x num_assign_interval flattened in F order
                q_e_mode_driving = P_mode_driving.dot(q_e_passenger)
                q_e_mode_bustransit = P_mode_bustransit.dot(q_e_passenger)
                q_e_mode_pnr = P_mode_pnr.dot(q_e_passenger)

                q_e_mode_driving = np.maximum(q_e_mode_driving, 1e-6)
                q_e_mode_bustransit = np.maximum(q_e_mode_bustransit, 1e-6)
                q_e_mode_pnr = np.maximum(q_e_mode_pnr, 1e-6)

                # P_path: (num_path * num_assign_interval, num_OD_one_mode * num_assign_interval)
                P_path_car_driving, P_path_truck_driving = self.nb.get_route_portion_matrix_driving()
                P_path_passenger_bustransit = self.nb.get_route_portion_matrix_bustransit()
                P_path_car_pnr = self.nb.get_route_portion_matrix_pnr()

                # f_e: num_path x num_assign_interval flattened in F order
                f_car_driving = P_path_car_driving.dot(q_e_mode_driving)
                f_truck_driving = P_path_truck_driving.dot(q_e_truck)
                f_passenger_bustransit = P_path_passenger_bustransit.dot(q_e_mode_bustransit)
                f_car_pnr = P_path_car_pnr.dot(q_e_mode_pnr)

                f_car_driving = np.maximum(f_car_driving, 1 / self.nb.config.config_dict['DTA']['flow_scalar'])
                f_truck_driving = np.maximum(f_truck_driving, 1 / self.nb.config.config_dict['DTA']['flow_scalar'])
                # this alleviate trapping in local minima, when f = 0 -> grad = 0 is not helping
                f_passenger_bustransit = np.maximum(f_passenger_bustransit, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])
                f_car_pnr = np.maximum(f_car_pnr, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])
                f_bus = np.maximum(f_bus, 1e-3 / self.nb.config.config_dict['DTA']['flow_scalar'])
                
                # f_grad: num_path * num_assign_interval
                f_car_driving_grad, f_truck_driving_grad, f_passenger_bustransit_grad, f_car_pnr_grad, f_passenger_pnr_grad, f_bus_grad, \
                    tmp_loss, tmp_loss_dict, dta, x_e_car, x_e_truck, x_e_passenger, x_e_bus, tt_e_car, tt_e_truck, tt_e_passenger, tt_e_bus = \
                        self.compute_path_flow_grad_and_loss(one_data_dict, f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus, fix_bus=fix_bus, counter=0, run_mmdta_adaptive=False)
                
                # q_mode_grad: num_OD_one_mode * num_assign_interval
                q_grad_car_driving = P_path_car_driving.T.dot(f_car_driving_grad)
                q_grad_car_pnr = P_path_car_pnr.T.dot(f_car_pnr_grad)
                q_truck_grad = P_path_truck_driving.T.dot(f_truck_driving_grad)
                q_grad_passenger_bustransit = P_path_passenger_bustransit.T.dot(f_passenger_bustransit_grad)
                q_grad_passenger_pnr = P_path_car_pnr.T.dot(f_passenger_pnr_grad)

                # q_grad: num_OD * num_assign_interval
                q_passenger_grad = P_mode_driving.T.dot(q_grad_car_driving) \
                                   + P_mode_bustransit.T.dot(q_grad_passenger_bustransit) \
                                   + P_mode_pnr.T.dot(q_grad_passenger_pnr + q_grad_car_pnr)
                
                if adagrad:
                    sum_g_square_passenger = sum_g_square_passenger + np.power(q_passenger_grad, 2)
                    q_e_passenger -= q_passenger_grad * passenger_step_size[i] / np.sqrt(sum_g_square_passenger)
                    sum_g_square_truck = sum_g_square_truck + np.power(q_truck_grad, 2)
                    q_e_truck -= q_truck_grad * truck_step_size[i] / np.sqrt(sum_g_square_truck)
                    if not fix_bus:
                        sum_g_square_bus = sum_g_square_bus + np.power(f_bus_grad, 2)
                        f_bus -= f_bus_grad * bus_step_size[i] / np.sqrt(sum_g_square_bus)
                else:
                    q_e_passenger -= q_passenger_grad * passenger_step_size[i] / np.sqrt(i+1)
                    q_e_truck -= q_truck_grad * truck_step_size[i] / np.sqrt(i+1)
                    if not fix_bus:
                        f_bus -= f_bus_grad * bus_step_size[i] / np.sqrt(i+1)

                # release memory
                f_car_driving_grad, f_truck_driving_grad, f_passenger_bustransit_grad, f_car_pnr_grad, f_passenger_pnr_grad, f_bus_grad = 0, 0, 0, 0, 0, 0
                q_grad_car_driving, q_grad_car_pnr, q_truck_grad, q_grad_passenger_bustransit, q_grad_passenger_pnr, q_passenger_grad = 0, 0, 0, 0, 0, 0
                
                # relu
                q_e_passenger = np.maximum(q_e_passenger, 1e-6)
                q_e_truck = np.maximum(q_e_truck, 1e-6)
                f_bus = np.maximum(f_bus, 1e-6)

                loss += tmp_loss / float(self.num_data)
                for loss_type, loss_value in tmp_loss_dict.items():
                    loss_dict[loss_type] += loss_value / float(self.num_data)

                if not self.config['use_car_link_tt'] and not self.config['use_truck_link_tt'] and not self.config['use_passenger_link_tt'] and not self.config['use_bus_link_tt']:
                    # if any of these is true, dta.build_link_cost_map(True) is already invoked in compute_path_flow_grad_and_loss()
                    dta.build_link_cost_map(False)
                self.compute_path_cost(dta)
                if column_generation[i]:
                    print("***************** generate new paths *****************")
                    self.update_path_table(dta, use_tdsp)
                    f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr = \
                        self.update_path_flow(f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr,
                                              car_driving_scale, truck_driving_scale, passenger_bustransit_scale, car_pnr_scale)
                    dta = 0
                # adjust modal split and path flow portion based on path cost and logit choice model
                self.assign_mode_route_portions(alpha_mode, beta_mode, alpha_path, beta_path)

            print("Epoch:", starting_epoch + i, "Loss:", loss, self.print_separate_accuracy(loss_dict))
            loss_list.append([loss, loss_dict])

            if (best_epoch == 0) or (loss_list[best_epoch][0] > loss_list[-1][0]):
                best_epoch = starting_epoch + i
                best_f_car_driving, best_f_truck_driving, best_f_passenger_bustransit, best_f_car_pnr, best_f_bus, best_q_e_passenger, best_q_e_truck = \
                    f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus, q_e_passenger, q_e_truck

                best_x_e_car, best_x_e_truck, best_x_e_passenger, best_x_e_bus, best_tt_e_car, best_tt_e_truck, best_tt_e_passenger, best_tt_e_bus = \
                    x_e_car, x_e_truck, x_e_passenger, x_e_bus, tt_e_car, tt_e_truck, tt_e_passenger, tt_e_bus

                if save_folder is not None:
                    self.save_simulation_input_files(os.path.join(save_folder, 'input_files_estimate_demand'), 
                                                     f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus=None if fix_bus else f_bus,
                                                     explicit_bus=1, historical_bus_waiting_time=0)

            if save_folder is not None:
                pickle.dump([loss, loss_dict, loss_list, best_epoch,
                             q_e_passenger, q_e_truck, 
                             q_e_mode_driving, q_e_mode_bustransit, q_e_mode_pnr,
                             f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus,
                             x_e_car, x_e_truck, x_e_passenger, x_e_bus, tt_e_car, tt_e_truck, tt_e_passenger, tt_e_bus], 
                            open(os.path.join(save_folder, str(starting_epoch + i) + '_iteration_estimate_demand.pickle'), 'wb'))

                # if column_generation[i]:
                #     self.save_simulation_input_files(os.path.join(save_folder, 'input_files_estimate_demand'), 
                #                                      f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr, f_bus,
                #                                      explicit_bus=0, historical_bus_waiting_time=0)
        
        print("Best loss at Epoch:", best_epoch, "Loss:", loss_list[best_epoch][0], self.print_separate_accuracy(loss_list[best_epoch][1]))
        return best_f_car_driving, best_f_truck_driving, best_f_passenger_bustransit, best_f_car_pnr, best_f_bus, best_q_e_passenger, best_q_e_truck, \
               best_x_e_car, best_x_e_truck, best_x_e_passenger, best_x_e_bus, best_tt_e_car, best_tt_e_truck, best_tt_e_passenger, best_tt_e_bus, \
               loss_list

    def update_path_table(self, dta, use_tdsp=True):
        # dta.build_link_cost_map() should be called before this method

        start_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)
        self.nb.update_path_table(dta, start_intervals, use_tdsp)

        self.num_path_driving = self.nb.config.config_dict['FIXED']['num_driving_path']
        self.num_path_bustransit = self.nb.config.config_dict['FIXED']['num_bustransit_path']
        self.num_path_pnr = self.nb.config.config_dict['FIXED']['num_pnr_path']
        self.num_path_busroute = self.nb.config.config_dict['FIXED']['num_bus_routes']

        # observed path IDs, np.array
        self.config['paths_list_driving'] = np.array(list(self.nb.path_table_driving.ID2path.keys()), dtype=int)
        self.config['paths_list_bustransit'] = np.array(list(self.nb.path_table_bustransit.ID2path.keys()), dtype=int)
        self.config['paths_list_pnr'] = np.array(list(self.nb.path_table_pnr.ID2path.keys()), dtype=int)
        self.config['paths_list_busroute'] = np.array(list(self.nb.path_table_bus.ID2path.keys()), dtype=int)
        self.config['paths_list'] = np.concatenate((self.config['paths_list_driving'], self.config['paths_list_bustransit'],
                                                    self.config['paths_list_pnr'], self.config['paths_list_busroute']))
        assert(len(np.unique(self.config['paths_list'])) == len(self.config['paths_list']))
        self.paths_list_driving = self.config['paths_list_driving']
        self.paths_list_bustransit = self.config['paths_list_bustransit']
        self.paths_list_pnr = self.config['paths_list_pnr']
        self.paths_list_busroute = self.config['paths_list_busroute']
        self.paths_list = self.config['paths_list']
        assert (len(self.paths_list_driving) == self.num_path_driving)
        assert (len(self.paths_list_bustransit) == self.num_path_bustransit)
        assert (len(self.paths_list_pnr) == self.num_path_pnr)
        assert (len(self.paths_list_busroute) == self.num_path_busroute)
        assert (len(self.paths_list) == self.num_path_driving + self.num_path_bustransit + self.num_path_pnr + self.num_path_busroute)

    def update_path_flow(self, f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr,
                         car_driving_scale=1, truck_driving_scale=0.1, passenger_bustransit_scale=1, car_pnr_scale=0.5):
        max_interval = self.nb.config.config_dict['DTA']['max_interval']
        # reshape path flow into ndarrays with dimensions of intervals x number of total paths
        f_car_driving = f_car_driving.reshape(max_interval, -1)
        f_truck_driving = f_truck_driving.reshape(max_interval, -1)
        f_passenger_bustransit = f_passenger_bustransit.reshape(max_interval, -1)
        f_car_pnr = f_car_pnr.reshape(max_interval, -1)

        if len(self.nb.path_table_driving.ID2path) > f_car_driving.shape[1]:
            _add_f = self.init_demand_vector(max_interval, len(self.nb.path_table_driving.ID2path) - f_car_driving.shape[1], car_driving_scale)
            _add_f = _add_f.reshape(max_interval, -1)
            f_car_driving = np.concatenate((f_car_driving, _add_f), axis=1)
            assert(f_car_driving.shape[1] == len(self.nb.path_table_driving.ID2path))

            _add_f = self.init_demand_vector(max_interval, len(self.nb.path_table_driving.ID2path) - f_truck_driving.shape[1], truck_driving_scale)
            _add_f = _add_f.reshape(max_interval, -1)
            f_truck_driving = np.concatenate((f_truck_driving, _add_f), axis=1)
            assert(f_truck_driving.shape[1] == len(self.nb.path_table_driving.ID2path))

        if len(self.nb.path_table_bustransit.ID2path) > f_passenger_bustransit.shape[1]:
            _add_f = self.init_demand_vector(max_interval, len(self.nb.path_table_bustransit.ID2path) - f_passenger_bustransit.shape[1], passenger_bustransit_scale)
            _add_f = _add_f.reshape(max_interval, -1)
            f_passenger_bustransit = np.concatenate((f_passenger_bustransit, _add_f), axis=1)
            assert(f_passenger_bustransit.shape[1] == len(self.nb.path_table_bustransit.ID2path))

        if len(self.nb.path_table_pnr.ID2path) > f_car_pnr.shape[1]:
            _add_f = self.init_demand_vector(max_interval, len(self.nb.path_table_pnr.ID2path) - f_car_pnr.shape[1], car_pnr_scale)
            _add_f = _add_f.reshape(max_interval, -1)
            f_car_pnr = np.concatenate((f_car_pnr, _add_f), axis=1)
            assert(f_car_pnr.shape[1] == len(self.nb.path_table_pnr.ID2path))

        f_car_driving = f_car_driving.flatten(order='C')
        f_truck_driving = f_truck_driving.flatten(order='C')
        f_passenger_bustransit = f_passenger_bustransit.flatten(order='C')
        f_car_pnr = f_car_pnr.flatten(order='C')

        return f_car_driving, f_truck_driving, f_passenger_bustransit, f_car_pnr

    def print_separate_accuracy(self, loss_dict):
        tmp_str = ""
        for loss_type, loss_value in loss_dict.items():
            tmp_str += loss_type + ": " + str(np.round(loss_value, 2)) + "|"
        return tmp_str

    def assign_mode_route_portions(self, alpha_mode=(1., 1.5, 2.), beta_mode=1, alpha_path=1, beta_path=1):
        for OD_idx, (O, D) in enumerate(self.nb.demand_total_passenger.demand_list):
            O_node = self.nb.od.O_dict[O]
            D_node = self.nb.od.D_dict[D]
            
            min_mode_cost = OrderedDict()
            alpha_mode_existed = list()
            if self.nb.od_mode_connectivity.loc[OD_idx, 'driving'] == 1:
                tmp_path_set = self.nb.path_table_driving.path_dict[O_node][D_node]
                cost_array = np.zeros((len(tmp_path_set.path_list), self.num_assign_interval))
                truck_tt_array = np.zeros((len(tmp_path_set.path_list), self.num_assign_interval))
                for tmp_path_idx, tmp_path in enumerate(tmp_path_set.path_list):
                    cost_array[tmp_path_idx, :] = tmp_path.path_cost_car
                    truck_tt_array[tmp_path_idx, :] = tmp_path.path_cost_truck 
                p_array = generate_portion_array(cost_array, alpha_path, beta_path)
                p_array_truck = generate_portion_array(truck_tt_array, alpha_path, beta_path)
                for tmp_path_idx, tmp_path in enumerate(tmp_path_set.path_list):
                    tmp_path.attach_route_choice_portions(p_array[tmp_path_idx, :])
                    tmp_path.attach_route_choice_portions_truck(p_array_truck[tmp_path_idx, :])

                min_mode_cost["driving"] = np.min(cost_array, axis=0)
                alpha_mode_existed.append(alpha_mode[0])

            if self.nb.od_mode_connectivity.loc[OD_idx, 'bustransit'] == 1:
                tmp_path_set = self.nb.path_table_bustransit.path_dict[O_node][D_node]
                cost_array = np.zeros((len(tmp_path_set.path_list), self.num_assign_interval))
                for tmp_path_idx, tmp_path in enumerate(tmp_path_set.path_list):
                    cost_array[tmp_path_idx, :] = tmp_path.path_cost
                p_array = generate_portion_array(cost_array, alpha_path, beta_path)
                for tmp_path_idx, tmp_path in enumerate(tmp_path_set.path_list):
                    tmp_path.attach_route_choice_portions_bustransit(p_array[tmp_path_idx, :])

                min_mode_cost["bustransit"] = np.min(cost_array, axis=0)
                alpha_mode_existed.append(alpha_mode[1])

            if self.nb.od_mode_connectivity.loc[OD_idx, 'pnr'] == 1:
                tmp_path_set = self.nb.path_table_pnr.path_dict[O_node][D_node]
                cost_array = np.zeros((len(tmp_path_set.path_list), self.num_assign_interval))
                for tmp_path_idx, tmp_path in enumerate(tmp_path_set.path_list):
                    cost_array[tmp_path_idx, :] = tmp_path.path_cost
                p_array = generate_portion_array(cost_array, alpha_path, beta_path)
                for tmp_path_idx, tmp_path in enumerate(tmp_path_set.path_list):
                    tmp_path.attach_route_choice_portions_pnr(p_array[tmp_path_idx, :])
                
                min_mode_cost["pnr"] = np.min(cost_array, axis=0)
                alpha_mode_existed.append(alpha_mode[2])

            mode_p_dict = generate_mode_portion_array(min_mode_cost, np.array(alpha_mode_existed), beta_mode)

            if self.nb.od_mode_connectivity.loc[OD_idx, 'driving'] == 1:
                self.nb.demand_driving.demand_dict[O][D][0] = mode_p_dict["driving"]
            if self.nb.od_mode_connectivity.loc[OD_idx, 'bustransit'] == 1:
                self.nb.demand_bustransit.demand_dict[O][D] = mode_p_dict["bustransit"]
            if self.nb.od_mode_connectivity.loc[OD_idx, 'pnr'] == 1:
                self.nb.demand_pnr.demand_dict[O][D] = mode_p_dict["pnr"]

    def init_mode_route_portions(self):
        for O in self.nb.demand_driving.demand_dict.keys():
            for D in self.nb.demand_driving.demand_dict[O].keys():
                self.nb.demand_driving.demand_dict[O][D] = [np.ones(self.num_assign_interval), np.ones(self.num_assign_interval)] 
        for O in self.nb.demand_bustransit.demand_dict.keys():
            for D in self.nb.demand_bustransit.demand_dict[O].keys():
                self.nb.demand_bustransit.demand_dict[O][D] = np.ones(self.num_assign_interval)
        for O in self.nb.demand_pnr.demand_dict.keys():
            for D in self.nb.demand_pnr.demand_dict[O].keys():
                self.nb.demand_pnr.demand_dict[O][D] = np.ones(self.num_assign_interval) 

        for path in self.nb.path_table_driving.ID2path.values():
            path.attach_route_choice_portions(np.ones(self.num_assign_interval))
            path.attach_route_choice_portions_truck(np.ones(self.num_assign_interval))
        for path in self.nb.path_table_bustransit.ID2path.values():
            path.attach_route_choice_portions_bustransit(np.ones(self.num_assign_interval))
        for path in self.nb.path_table_pnr.ID2path.values():
            path.attach_route_choice_portions_pnr(np.ones(self.num_assign_interval))
        
                
###  Behavior related function
def generate_mode_portion_array(mode_cost_dict, alpha, beta=1.):
    assert(len(mode_cost_dict) == len(alpha))
    mode_cost_array = np.stack([v for _, v in mode_cost_dict.items()], axis=0)
    p_array = np.zeros(mode_cost_array.shape)
    for i in range(mode_cost_array.shape[1]):
        p_array[:, i] = logit_fn(mode_cost_array[:,i], alpha, beta)
    
    mode_p_dict = OrderedDict()
    for i, k in enumerate(mode_cost_dict.keys()):
        mode_p_dict[k] = p_array[i, :]
    return mode_p_dict

def generate_portion_array(cost_array, alpha=1.5, beta=1.):
    p_array = np.zeros(cost_array.shape)
    for i in range(cost_array.shape[1]):  # time
        p_array[:, i] = logit_fn(cost_array[:,i], alpha, beta)
    return p_array

def logit_fn(cost, alpha, beta, max_cut=True):
    # given alpha >=0, beta > 0, decreasing alpha and beta will leading to a more evenly distribution; otherwise more concentrated distribution
    scale_cost = - (alpha + beta * cost)
    if max_cut:
        e_x = np.exp(scale_cost - np.max(scale_cost))
    else:
        e_x = np.exp(scale_cost)
    p = np.maximum(e_x / e_x.sum(), 1e-6)
    return p


class PostProcessing:
    def __init__(self, dode, dta=None, 
                 estimated_car_count=None, estimated_truck_count=None, estimated_passenger_count=None, estimated_bus_count=None, estimated_BoardingAlighting_count=None,
                 estimated_car_cost=None, estimated_truck_cost=None, estimated_passenger_cost=None, estimated_bus_cost=None, 
                 estimated_stop_arrival_departure_travel_time_df = None,
                 result_folder=None):
        self.dode = dode
        self.dta = dta
        self.result_folder = result_folder
        self.one_data_dict = None

        self.color_list = ['teal', 'tomato', 'blue', 'sienna', 'plum', 'red', 'yellowgreen', 'khaki', 'lightpink']
        self.marker_list = ["o", "v", "^", "<", ">", "p", "D", "*", "s", "D", "p"]

        self.r2_car_count, self.r2_truck_count, self.r2_passenger_count, self.r2_bus_count, self.r2_BoardingAlighting_count = "NA", "NA", "NA", "NA", "NA"
        self.true_car_count, self.estimated_car_count = None, estimated_car_count
        self.true_truck_count, self.estimated_truck_count = None, estimated_truck_count
        self.true_passenger_count, self.estimated_passenger_count = None, estimated_passenger_count
        self.true_bus_count, self.estimated_bus_count = None, estimated_bus_count
        self.true_BoardingAlighting_count, self.estimated_BoardingAlighting_count = None, estimated_BoardingAlighting_count
        self.true_stop_arrival_departure_travel_time_df, self.estimated_stop_arrival_departure_travel_time_df = None, estimated_stop_arrival_departure_travel_time_df

        self.r2_car_cost, self.r2_truck_cost, self.r2_passenger_cost, self.r2_bus_cost, self.r2_travel_and_dwell_time = "NA", "NA", "NA", "NA", "NA"
        self.true_car_cost, self.estimated_car_cost = None, estimated_car_cost
        self.true_truck_cost, self.estimated_truck_cost = None, estimated_truck_cost
        self.true_passenger_cost, self.estimated_passenger_cost = None, estimated_passenger_cost
        self.true_bus_cost, self.estimated_bus_cost = None, estimated_bus_cost
        
        plt.rc('font', size=20)          # controls default text sizes
        plt.rc('axes', titlesize=20)     # fontsize of the axes title
        plt.rc('axes', labelsize=20)     # fontsize of the x and y labels
        plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
        plt.rc('legend', fontsize=20)    # legend fontsize
        plt.rc('figure', titlesize=20)   # fontsize of the figure title

        sns.set_theme()
        plt.style.use(plt_style)

    def plot_total_loss(self, loss_list, fig_name = 'total_loss_pathflow.png'):

        plt.figure(figsize = (16, 9), dpi=300)
        plt.plot(np.arange(len(loss_list))+1, list(map(lambda x: x[0], loss_list)), color = self.color_list[0], marker = self.marker_list[0], linewidth = 3)
        # plt.plot(range(len(l_list)), list(map(lambda x: x[0], l_list)),
        #          color = color_list[4], linewidth = 3, label = "Total cost")

        plt.ylabel('Loss', fontsize = 20)
        plt.xlabel('Iteration', fontsize = 20)
        # plt.legend()
        # plt.ylim([0, 1])
        plt.xlim([1, len(loss_list)])
        step = max(1, len(loss_list) // 10)  # Adjust step based on size (e.g., every ~10% of data)
        plt.xticks(np.arange(1, len(loss_list) + 1, step))

        plt.savefig(os.path.join(self.result_folder, fig_name), bbox_inches='tight')

        plt.show()

    def plot_breakdown_loss(self, loss_list, fig_name = 'breakdown_loss_pathflow.png'):

        if self.dode.config['use_car_link_flow'] + self.dode.config['use_truck_link_flow'] + self.dode.config['use_passenger_link_flow'] + self.dode.config['use_bus_link_flow'] + \
            (self.dode.config['use_veh_run_boarding_alighting'] or  self.dode.config['use_ULP_f_transit']) + \
            self.dode.config['use_car_link_tt'] + self.dode.config['use_truck_link_tt'] + self.dode.config['use_passenger_link_tt'] + self.dode.config['use_bus_link_tt']:

            plt.figure(figsize = (16, 9), dpi=300)

            i = 0

            if self.dode.config['use_car_link_flow']:
                plt.plot(np.arange(len(loss_list))+1, list(map(lambda x: x[1]['car_count_loss']/loss_list[0][1]['car_count_loss'], loss_list)),
                        color = self.color_list[i],  marker = self.marker_list[i], linewidth = 3, label = "Car flow")
            i += self.dode.config['use_car_link_flow']

            if self.dode.config['use_truck_link_flow']:
                plt.plot(np.arange(len(loss_list))+1, list(map(lambda x: x[1]['truck_count_loss']/loss_list[0][1]['truck_count_loss'], loss_list)),
                        color = self.color_list[i],  marker = self.marker_list[i], linewidth = 3, label = "Truck flow")
            i += self.dode.config['use_truck_link_flow']

            if self.dode.config['use_passenger_link_flow']:
                plt.plot(np.arange(len(loss_list))+1, list(map(lambda x: x[1]['passenger_count_loss']/loss_list[0][1]['passenger_count_loss'], loss_list)),
                        color = self.color_list[i],  marker = self.marker_list[i], linewidth = 3, label = "Passenger flow")
            i += self.dode.config['use_passenger_link_flow']

            if self.dode.config['use_bus_link_flow']:
                plt.plot(np.arange(len(loss_list))+1, list(map(lambda x: x[1]['bus_count_loss']/loss_list[0][1]['bus_count_loss'], loss_list)),
                        color = self.color_list[i],  marker = self.marker_list[i], linewidth = 3, label = "Bus/Metro flow")
            i += self.dode.config['use_bus_link_flow']

            if self.dode.config['use_veh_run_boarding_alighting'] or self.dode.config['use_ULP_f_transit']:
                plt.plot(np.arange(len(loss_list))+1, list(map(lambda x: x[1]['veh_run_boarding_alighting_loss']/loss_list[0][1]['veh_run_boarding_alighting_loss'], loss_list)),
                        color = self.color_list[i],  marker = self.marker_list[i], linewidth = 3, label = "Boarding/Alighting flow")
            i += (self.dode.config['use_veh_run_boarding_alighting'] or self.dode.config['use_ULP_f_transit'])

            if self.dode.config['use_car_link_tt']:
                plt.plot(np.arange(len(loss_list))+1, list(map(lambda x: x[1]['car_tt_loss']/loss_list[0][1]['car_tt_loss'], loss_list)),
                    color = self.color_list[i],  marker = self.marker_list[i], linewidth = 3, label = "Car travel time")
            i += self.dode.config['use_car_link_tt']

            if self.dode.config['use_truck_link_tt']:
                plt.plot(np.arange(len(loss_list))+1, list(map(lambda x: x[1]['truck_tt_loss']/loss_list[0][1]['truck_tt_loss'], loss_list)), 
                        color = self.color_list[i],  marker = self.marker_list[i], linewidth = 3, label = "Truck travel time")
            i += self.dode.config['use_truck_link_tt']

            if self.dode.config['use_passenger_link_tt']:
                plt.plot(np.arange(len(loss_list))+1, list(map(lambda x: x[1]['passenger_tt_loss']/loss_list[0][1]['passenger_tt_loss'], loss_list)), 
                        color = self.color_list[i],  marker = self.marker_list[i], linewidth = 3, label = "Passenger travel time")
            i += self.dode.config['use_passenger_link_tt']

            if self.dode.config['use_bus_link_tt']:
                plt.plot(np.arange(len(loss_list))+1, list(map(lambda x: x[1]['bus_tt_loss']/loss_list[0][1]['bus_tt_loss'], loss_list)), 
                        color = self.color_list[i],  marker = self.marker_list[i], linewidth = 3, label = "Bus travel time")
                
            

            plt.ylabel('Loss')
            plt.xlabel('Iteration')
            plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 1))
            # plt.ylim([0, 1.1])
            plt.xlim([1, len(loss_list)])
            step = max(1, len(loss_list) // 10)  # Adjust step based on size (e.g., every ~10% of data)
            plt.xticks(np.arange(1, len(loss_list) + 1, step))

            plt.savefig(os.path.join(self.result_folder, fig_name), bbox_inches='tight')

            # plt.show()

    def plot_NMSE(self, loss_list, fig_name = 'NMSE.png'):
        if self.dode.config['use_car_link_flow'] + self.dode.config['use_truck_link_flow'] + self.dode.config['use_passenger_link_flow'] + self.dode.config['use_bus_link_flow'] + \
            (self.dode.config['use_veh_run_boarding_alighting'] or  self.dode.config['use_ULP_f_transit']) + \
            self.dode.config['use_car_link_tt'] + self.dode.config['use_truck_link_tt'] + self.dode.config['use_passenger_link_tt'] + self.dode.config['use_bus_link_tt']:

            plt.figure(figsize = (16, 9), dpi=300)

            i = 0

            if self.dode.config['use_car_link_flow']:
                nmse_list = []
                for j in range(len(loss_list)):
                    [_, _, _, _, _, _, _, _, _, _, _, _, _,
                             x_e_car, _, _, _, _, _, _, _, _, _] = pickle.load(open(os.path.join(self.result_folder, str(j) + '_iteration_estimate_demand.pickle'), 'rb'))
                    if self.dode.config['car_count_agg']:
                        L_car = self.one_data_dict['car_count_agg_L']
                        x_e_car = L_car.dot(x_e_car)
                    x_e_car = x_e_car[self.one_data_dict['mask_driving_link']]
                    ind = ~(np.isinf(self.true_car_count) + np.isinf(x_e_car) + np.isnan(self.true_car_count) + np.isnan(x_e_car))
                    nmse = np.mean(np.square(self.true_car_count[ind] - x_e_car[ind])) / np.mean(np.square(self.true_car_count[ind]))
                    nmse_list.append(nmse)
                plt.plot(np.arange(len(loss_list))+1, nmse_list, color = self.color_list[i],  marker = self.marker_list[i], linewidth = 3, label = "Car flow")
            i += self.dode.config['use_car_link_flow']

            if self.dode.config['use_truck_link_flow']:
                nmse_list = []
                for j in range(len(loss_list)):
                    [_, _, _, _, _, _, _, _, _, _, _, _, _,
                             _, x_e_truck, _, _, _, _, _, _, _, _] = pickle.load(open(os.path.join(self.result_folder, str(j) + '_iteration_estimate_demand.pickle'), 'rb'))
                    if self.dode.config['truck_count_agg']:
                        L_truck = self.one_data_dict['truck_count_agg_L']
                        x_e_truck = L_truck.dot(x_e_truck)
                    x_e_truck = x_e_truck[self.one_data_dict['mask_driving_link']]
                    ind = ~(np.isinf(self.true_truck_count) + np.isinf(x_e_truck) + np.isnan(self.true_truck_count) + np.isnan(x_e_truck))
                    nmse = np.mean(np.square(self.true_truck_count[ind] - x_e_truck[ind])) / np.mean(np.square(self.true_truck_count[ind]))
                    nmse_list.append(nmse)
                plt.plot(np.arange(len(loss_list))+1, nmse_list, color = self.color_list[i],  marker = self.marker_list[i], linewidth = 3, label = "Truck flow")
            i += self.dode.config['use_truck_link_flow']

            if self.dode.config['use_passenger_link_flow']:
                nmse_list = []
                for j in range(len(loss_list)):
                    [_, _, _, _, _, _, _, _, _, _, _, _, _,
                             _, _, x_e_passenger, _, _, _, _, _, _, _] = pickle.load(open(os.path.join(self.result_folder, str(j) + '_iteration_estimate_demand.pickle'), 'rb'))
                    if self.dode.config['passenger_count_agg']:
                        L_passenger = self.one_data_dict['passenger_count_agg_L']
                        x_e_passenger = L_passenger.dot(x_e_passenger)
                    if len(self.one_data_dict['mask_walking_link']) > 0:
                        mask_passenger = np.concatenate(
                            (self.one_data_dict['mask_bus_link'].reshape(-1, len(self.dode.observed_links_bus)), 
                            self.one_data_dict['mask_walking_link'].reshape(-1, len(self.dode.observed_links_walking))), 
                            axis=1
                        )
                        mask_passenger = mask_passenger.flatten()
                    else:
                        mask_passenger = self.one_data_dict['mask_bus_link']
                    x_e_passenger = x_e_passenger[mask_passenger]
                    ind = ~(np.isinf(self.true_passenger_count) + np.isinf(x_e_passenger) + np.isnan(self.true_passenger_count) + np.isnan(x_e_passenger))
                    nmse = np.mean(np.square(self.true_passenger_count[ind] - x_e_passenger[ind])) / np.mean(np.square(self.true_passenger_count[ind]))
                    nmse_list.append(nmse)
                plt.plot(np.arange(len(loss_list))+1, nmse_list, color = self.color_list[i],  marker = self.marker_list[i], linewidth = 3, label = "Passenger flow")
            i += self.dode.config['use_passenger_link_flow']

            if self.dode.config['use_bus_link_flow']:
                nmse_list = []
                for j in range(len(loss_list)):
                    [_, _, _, _, _, _, _, _, _, _, _, _, _,
                             _, _, _, x_e_bus, _, _, _, _, _, _] = pickle.load(open(os.path.join(self.result_folder, str(j) + '_iteration_estimate_demand.pickle'), 'rb'))
                    if self.dode.config['bus_count_agg']:
                        L_bus = self.one_data_dict['bus_count_agg_L']
                        x_e_bus = L_bus.dot(x_e_bus)
                    x_e_bus = x_e_bus[self.one_data_dict['mask_bus_link']]
                    ind = ~(np.isinf(self.true_bus_count) + np.isinf(x_e_bus) + np.isnan(self.true_bus_count) + np.isnan(x_e_bus))
                    nmse = np.mean(np.square(self.true_bus_count[ind] - x_e_bus[ind])) / np.mean(np.square(self.true_bus_count[ind]))
                    nmse_list.append(nmse)
                plt.plot(np.arange(len(loss_list))+1, nmse_list, color = self.color_list[i],  marker = self.marker_list[i], linewidth = 3, label = "Bus/Metro flow")
            i += self.dode.config['use_bus_link_flow']

            if self.dode.config['use_veh_run_boarding_alighting'] or self.dode.config['use_ULP_f_transit']:
                nmse_list = []
                for j in range(len(loss_list)):
                    [_, _, _, _, _, _, _, _, _, _, _, _, _,
                             _, _, _, _, x_e_BoardingAlighting_count, _, _, _, _, _] = pickle.load(open(os.path.join(self.result_folder, str(j) + '_iteration_estimate_demand.pickle'), 'rb'))
                    x_e_BoardingAlighting_count = x_e_BoardingAlighting_count[self.one_data_dict['mask_observed_stops_vehs_record']]
                    ind = ~(np.isinf(self.true_BoardingAlighting_count) + np.isinf(x_e_BoardingAlighting_count) + np.isnan(self.true_BoardingAlighting_count) + np.isnan(x_e_BoardingAlighting_count))
                    nmse = np.mean(np.square(self.true_BoardingAlighting_count[ind] - x_e_BoardingAlighting_count[ind])) / np.mean(np.square(self.true_BoardingAlighting_count[ind]))
                    nmse_list.append(nmse)
                plt.plot(np.arange(len(loss_list))+1, nmse_list, color = self.color_list[i],  marker = self.marker_list[i], linewidth = 3, label = "Boarding/Alighting flow")
            i += (self.dode.config['use_veh_run_boarding_alighting'] or self.dode.config['use_ULP_f_transit'])

            if self.dode.config['use_car_link_tt']:
                nmse_list = []
                for j in range(len(loss_list)):
                    [_, _, _, _, _, _, _, _, _, _, _, _, _,
                             _, _, _, _, _, tt_e_car, _, _, _, _] = pickle.load(open(os.path.join(self.result_folder, str(j) + '_iteration_estimate_demand.pickle'), 'rb'))
                    tt_e_car = tt_e_car[self.one_data_dict['mask_driving_link']]
                    ind = ~(np.isinf(self.true_car_cost) + np.isinf(tt_e_car) + np.isnan(self.true_car_cost) + np.isnan(tt_e_car))
                    nmse = np.mean(np.square(self.true_car_cost[ind] - tt_e_car[ind])) / np.mean(np.square(self.true_car_cost[ind]))
                    nmse_list.append(nmse)
                plt.plot(np.arange(len(loss_list))+1, nmse_list, color = self.color_list[i],  marker = self.marker_list[i], linewidth = 3, label = "Car travel time")
            i += self.dode.config['use_car_link_tt']

            if self.dode.config['use_truck_link_tt']:
                nmse_list = []
                for j in range(len(loss_list)):
                    [_, _, _, _, _, _, _, _, _, _, _, _, _,
                             _, _, _, _, _, _, tt_e_truck, _, _, _] = pickle.load(open(os.path.join(self.result_folder, str(j) + '_iteration_estimate_demand.pickle'), 'rb'))
                    tt_e_truck = tt_e_truck[self.one_data_dict['mask_driving_link']]
                    ind = ~(np.isinf(self.true_truck_cost) + np.isinf(tt_e_truck) + np.isnan(self.true_truck_cost) + np.isnan(tt_e_truck))
                    nmse = np.mean(np.square(self.true_truck_cost[ind] - tt_e_truck[ind])) / np.mean(np.square(self.true_truck_cost[ind]))
                    nmse_list.append(nmse)
                plt.plot(np.arange(len(loss_list))+1, nmse_list, color = self.color_list[i],  marker = self.marker_list[i], linewidth = 3, label = "Truck travel time")
            i += self.dode.config['use_truck_link_tt']

            if self.dode.config['use_passenger_link_tt']:
                nmse_list = []
                for j in range(len(loss_list)):
                    [_, _, _, _, _, _, _, _, _, _, _, _, _,
                             _, _, _, _, _, _, _, tt_e_passenger, _, _] = pickle.load(open(os.path.join(self.result_folder, str(j) + '_iteration_estimate_demand.pickle'), 'rb'))
                    if len(self.one_data_dict['mask_walking_link']) > 0:
                        mask_passenger = np.concatenate(
                            (self.one_data_dict['mask_bus_link'].reshape(-1, len(self.dode.observed_links_bus)), 
                            self.one_data_dict['mask_walking_link'].reshape(-1, len(self.dode.observed_links_walking))), 
                            axis=1
                        )
                        mask_passenger = mask_passenger.flatten()
                    else:
                        mask_passenger = self.one_data_dict['mask_bus_link']

                    tt_e_passenger = tt_e_passenger[mask_passenger]
                    ind = ~(np.isinf(self.true_passenger_cost) + np.isinf(tt_e_passenger) + np.isnan(self.true_passenger_cost) + np.isnan(tt_e_passenger))
                    nmse = np.mean(np.square(self.true_passenger_cost[ind] - tt_e_passenger[ind])) / np.mean(np.square(self.true_passenger_cost[ind]))
                    nmse_list.append(nmse)
                plt.plot(np.arange(len(loss_list))+1, nmse_list, color = self.color_list[i],  marker = self.marker_list[i], linewidth = 3, label = "Passenger travel time")
            i += self.dode.config['use_passenger_link_tt']

            if self.dode.config['use_bus_link_tt']:
                nmse_list = []
                for j in range(len(loss_list)):
                    [_, _, _, _, _, _, _, _, _, _, _, _, _,
                             _, _, _, _, _, _, _, _, tt_e_bus, _] = pickle.load(open(os.path.join(self.result_folder, str(j) + '_iteration_estimate_demand.pickle'), 'rb'))
                    tt_e_bus = tt_e_bus[self.one_data_dict['mask_bus_link']]
                    ind = ~(np.isinf(self.true_bus_cost) + np.isinf(tt_e_bus) + np.isnan(self.true_bus_cost) + np.isnan(tt_e_bus))
                    nmse = np.mean(np.square(self.true_bus_cost[ind] - tt_e_bus[ind])) / np.mean(np.square(self.true_bus_cost[ind]))
                    nmse_list.append(nmse)
                plt.plot(np.arange(len(loss_list))+1, nmse_list, color = self.color_list[i],  marker = self.marker_list[i], linewidth = 3, label = "Bus travel time")


            plt.ylabel('Normalized Mean Squared Error (NMSE)')
            plt.xlabel('Iteration')
            plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 1))
            # plt.ylim([0, 1.1])
            plt.xlim([1, len(loss_list)])
            step = max(1, len(loss_list) // 10)  # Adjust step based on size (e.g., every ~10% of data)
            plt.xticks(np.arange(1, len(loss_list) + 1, step))

            plt.savefig(os.path.join(self.result_folder, fig_name), bbox_inches='tight')


    def get_one_data(self, start_intervals, end_intervals, j=0):
        assert(len(start_intervals) == len(end_intervals))

        # assume only one observation exists
        self.one_data_dict = self.dode._get_one_data(j)  

        if 'mask_driving_link' not in self.one_data_dict:
            self.one_data_dict['mask_driving_link'] = np.ones(len(self.dode.observed_links_driving) * len(start_intervals), dtype=bool)

        if 'mask_bus_link' not in self.one_data_dict:
            self.one_data_dict['mask_bus_link'] = np.ones(len(self.dode.observed_links_bus) * len(start_intervals), dtype=bool)

        if 'mask_walking_link' not in self.one_data_dict:
            self.one_data_dict['mask_walking_link'] = np.ones(len(self.dode.observed_links_walking) * len(start_intervals), dtype=bool)

        if 'mask_observed_stops_vehs_record' not in self.one_data_dict:
            self.one_data_dict['mask_observed_stops_vehs_record'] = np.ones(len(self.dode.observed_stops_vehs_list), dtype=bool)
        
        # count
        if self.dode.config['use_car_link_flow']:
            self.true_car_count = self.one_data_dict['car_link_flow']
            if self.estimated_car_count is None:
                L_car = self.one_data_dict['car_count_agg_L']
                estimated_car_x = self.dta.get_link_car_inflow(start_intervals, end_intervals).flatten(order='F')
                self.estimated_car_count = L_car.dot(estimated_car_x)

            self.true_car_count, self.estimated_car_count = self.true_car_count[self.one_data_dict['mask_driving_link']], self.estimated_car_count[self.one_data_dict['mask_driving_link']]

        if self.dode.config['use_truck_link_flow']:
            self.true_truck_count = self.one_data_dict['truck_link_flow']
            if self.estimated_truck_count is None:
                L_truck = self.one_data_dict['truck_count_agg_L']
                estimated_truck_x = self.dta.get_link_truck_inflow(start_intervals, end_intervals).flatten(order='F')
                self.estimated_truck_count = L_truck.dot(estimated_truck_x)
            
            self.true_truck_count, self.estimated_truck_count = self.true_truck_count[self.one_data_dict['mask_driving_link']], self.estimated_truck_count[self.one_data_dict['mask_driving_link']]

        if self.dode.config['use_passenger_link_flow']:
            self.true_passenger_count = self.one_data_dict['passenger_link_flow']
            if self.estimated_passenger_count is None:
                L_passenger = self.one_data_dict['passenger_count_agg_L']
                estimated_passenger_x_bus = self.dta.get_link_bus_passenger_inflow(start_intervals, end_intervals)
                estimated_passenger_x_walking = self.dta.get_link_walking_passenger_inflow(start_intervals, end_intervals)
                estimated_passenger_x = np.concatenate((estimated_passenger_x_bus, estimated_passenger_x_walking), axis=0).flatten(order='F')
                self.estimated_passenger_count = L_passenger.dot(estimated_passenger_x)

            if len(self.one_data_dict['mask_walking_link']) > 0:
                mask_passenger = np.concatenate(
                    (self.one_data_dict['mask_bus_link'].reshape(-1, len(self.dode.observed_links_bus)), 
                    self.one_data_dict['mask_walking_link'].reshape(-1, len(self.dode.observed_links_walking))), 
                    axis=1
                )
                mask_passenger = mask_passenger.flatten()
            else:
                mask_passenger = self.one_data_dict['mask_bus_link']
            
            self.true_passenger_count, self.estimated_passenger_count = self.true_passenger_count[mask_passenger], self.estimated_passenger_count[mask_passenger]
        
        if self.dode.config['use_bus_link_flow']:
            self.true_bus_count = self.one_data_dict['bus_link_flow']
            if self.estimated_bus_count is None:
                L_bus = self.one_data_dict['bus_count_agg_L']
                estimated_bus_x = self.dta.get_link_bus_inflow(start_intervals, end_intervals).flatten(order='F')
                self.estimated_bus_count = L_bus.dot(estimated_bus_x)

            self.true_bus_count, self.estimated_bus_count = self.true_bus_count[self.one_data_dict['mask_bus_link']], self.estimated_bus_count[self.one_data_dict['mask_bus_link']]

        if self.dode.config['use_veh_run_boarding_alighting'] or self.dode.config['use_ULP_f_transit']:
            self.true_BoardingAlighting_count = self.one_data_dict['veh_run_boarding_alighting_record'][1]
            if self.estimated_BoardingAlighting_count is None:
                raw_boarding_alighting_record = self.dta.get_bus_boarding_alighting_record()
                simulated_boarding_alighting_record_for_observed = self._massage_boarding_alighting_record(raw_boarding_alighting_record)
                self.estimated_BoardingAlighting_count = simulated_boarding_alighting_record_for_observed[1]
            self.true_BoardingAlighting_count, self.estimated_BoardingAlighting_count = \
                self.true_BoardingAlighting_count[self.one_data_dict['mask_observed_stops_vehs_record']], \
                self.estimated_BoardingAlighting_count[self.one_data_dict['mask_observed_stops_vehs_record']]
            
            if 'stop_arrival_departure_travel_time' in self.one_data_dict:
                self.true_stop_arrival_departure_travel_time_df = self.one_data_dict['stop_arrival_departure_travel_time']
                if self.estimated_stop_arrival_departure_travel_time_df is None:
                    raw_boarding_alighting_record = self.dta.get_bus_boarding_alighting_record()
                    stop_arrival_departure_travel_time_df = self.get_stop_arrival_departure_travel_time_df(raw_boarding_alighting_record)
                    self.estimated_stop_arrival_departure_travel_time_df = stop_arrival_departure_travel_time_df


        # travel cost
        if self.dode.config['use_car_link_tt']:
            self.true_car_cost = self.one_data_dict['car_link_tt']

            if self.estimated_car_cost is None:
                self.estimated_car_cost = self.dta.get_car_link_tt_robust(start_intervals, end_intervals, self.dode.ass_freq, True).flatten(order = 'F')
                # self.estimated_car_cost = self.dta.get_car_link_tt(start_intervals, True).flatten(order = 'F')

            self.true_car_cost, self.estimated_car_cost = self.true_car_cost[self.one_data_dict['mask_driving_link']], self.estimated_car_cost[self.one_data_dict['mask_driving_link']]

        if self.dode.config['use_truck_link_tt']:
            self.true_truck_cost = self.one_data_dict['truck_link_tt']

            if self.estimated_truck_cost is None:
                self.estimated_truck_cost = self.dta.get_truck_link_tt_robust(start_intervals, end_intervals, self.dode.ass_freq, True).flatten(order = 'F')
                # self.estimated_truck_cost = self.dta.get_truck_link_tt(start_intervals, True).flatten(order = 'F')

            self.true_truck_cost, self.estimated_truck_cost = self.true_truck_cost[self.one_data_dict['mask_driving_link']], self.estimated_truck_cost[self.one_data_dict['mask_driving_link']]

        if self.dode.config['use_passenger_link_tt']:
            self.true_passenger_cost = self.one_data_dict['passenger_link_tt']

            if self.estimated_passenger_cost is None:
                estimated_bus_tt = self.dta.get_bus_link_tt_robust(start_intervals, end_intervals, self.dode.ass_freq, True, True)
                estimated_walking_tt = self.dta.get_passenger_walking_link_tt_robust(start_intervals, end_intervals, self.dode.ass_freq)
                # estimated_bus_tt = self.dta.get_bus_link_tt(start_intervals, True, True)
                # estimated_walking_tt = self.dta.get_passenger_walking_link_tt(start_intervals)
                self.estimated_passenger_cost = np.concatenate((estimated_bus_tt, estimated_walking_tt), axis=0).flatten(order='F')

            # fill in the inf value
            # self.true_passenger_cost = np.nan_to_num(self.one_data_dict['passenger_link_tt'], posinf = 5 * self.dode.num_loading_interval)
            # self.estimated_passenger_cost = np.nan_to_num(self.estimated_passenger_cost, posinf = 5 * self.dode.num_loading_interval)

            if len(self.one_data_dict['mask_walking_link']) > 0:
                mask_passenger = np.concatenate(
                    (self.one_data_dict['mask_bus_link'].reshape(-1, len(self.dode.observed_links_bus)), 
                    self.one_data_dict['mask_walking_link'].reshape(-1, len(self.dode.observed_links_walking))), 
                    axis=1
                )
                mask_passenger = mask_passenger.flatten()
            else:
                mask_passenger = self.one_data_dict['mask_bus_link']
            self.true_passenger_cost, self.estimated_passenger_cost = self.true_passenger_cost[mask_passenger], self.estimated_passenger_cost[mask_passenger]
            
        if self.dode.config['use_bus_link_tt']:
            self.true_bus_cost = self.one_data_dict['bus_link_tt']

            if self.estimated_bus_cost is None:
                self.estimated_bus_cost = self.dta.get_bus_link_tt_robust(start_intervals, end_intervals, self.dode.ass_freq, True, True).flatten(order='F')
                # self.estimated_bus_cost = self.dta.get_bus_link_tt(start_intervals, True, True).flatten(order='F')

            # fill in the inf value
            # self.true_bus_cost = np.nan_to_num(self.one_data_dict['bus_link_tt'], posinf = 5 * self.dode.num_loading_interval)
            # self.estimated_bus_cost = np.nan_to_num(self.estimated_bus_cost, posinf = 5 * self.dode.num_loading_interval)

            self.true_bus_cost, self.estimated_bus_cost = self.true_bus_cost[self.one_data_dict['mask_bus_link']], self.estimated_bus_cost[self.one_data_dict['mask_bus_link']]
        

    def cal_r2_count(self):

        if self.dode.config['use_car_link_flow']:
            # print('----- car count -----')
            # print(self.true_car_count)
            # print(self.estimated_car_count)
            # print('----- car count -----')
            ind = ~(np.isinf(self.true_car_count) + np.isinf(self.estimated_car_count) + np.isnan(self.true_car_count) + np.isnan(self.estimated_car_count))
            self.r2_car_count = r2_score(self.true_car_count[ind], self.estimated_car_count[ind])

        if self.dode.config['use_truck_link_flow']:
            # print('----- truck count -----')
            # print(self.true_truck_count)
            # print(self.estimated_truck_count)
            # print('----- truck count -----')
            ind = ~(np.isinf(self.true_truck_count) + np.isinf(self.estimated_truck_count) + np.isnan(self.true_truck_count) + np.isnan(self.estimated_truck_count))
            self.r2_truck_count = r2_score(self.true_truck_count[ind], self.estimated_truck_count[ind])

        if self.dode.config['use_passenger_link_flow']:
            # print('----- passenger count -----')
            # print(self.true_passenger_count)
            # print(self.estimated_passenger_count)
            # print('----- passenger count -----')
            ind = ~(np.isinf(self.true_passenger_count) + np.isinf(self.estimated_passenger_count) + np.isnan(self.true_passenger_count) + np.isnan(self.estimated_passenger_count))
            self.r2_passenger_count = r2_score(self.true_passenger_count[ind], self.estimated_passenger_count[ind])

        if self.dode.config['use_bus_link_flow']:
            # print('----- bus count -----')
            # print(self.true_bus_count)
            # print(self.estimated_bus_count)
            # print('----- bus count -----')
            ind = ~(np.isinf(self.true_bus_count) + np.isinf(self.estimated_bus_count) + np.isnan(self.true_bus_count) + np.isnan(self.estimated_bus_count))
            self.r2_bus_count = r2_score(np.round(self.true_bus_count[ind], decimals=0), np.round(self.estimated_bus_count[ind], decimals=0))

        if self.dode.config['use_veh_run_boarding_alighting'] or self.dode.config['use_ULP_f_transit']:
            ind = ~(np.isinf(self.true_BoardingAlighting_count) + np.isinf(self.estimated_BoardingAlighting_count) + 
                    np.isnan(self.true_BoardingAlighting_count) + np.isnan(self.estimated_BoardingAlighting_count))
            # assert there is no false in ind
            assert np.all(ind)
            self.r2_BoardingAlighting_count = r2_score(self.true_BoardingAlighting_count[ind], self.estimated_BoardingAlighting_count[ind])

        print("r2 count --- r2_car_count: {}, r2_truck_count: {}, r2_passenger_count: {}, r2_bus_count: {}, r2_BoardingAlighting_count: {}".format(
                self.r2_car_count,
                self.r2_truck_count,
                self.r2_passenger_count,
                self.r2_bus_count,
                self.r2_BoardingAlighting_count
                ))

        return self.r2_car_count, self.r2_truck_count, self.r2_passenger_count, self.r2_bus_count, self.r2_BoardingAlighting_count

    def scatter_plot_ODdemand(self, true_q_e_mode_driving, q_e_mode_driving,true_q_e_mode_bustransit, q_e_mode_bustransit, true_q_e_mode_pnr, q_e_mode_pnr, fig_name = 'OD_demand_scatterplot.png'):
        driving = 0 if true_q_e_mode_driving is None else 1
        bustransit = 0 if true_q_e_mode_bustransit is None else 1
        pnr = 0 if true_q_e_mode_pnr is None else 1

        fig, axes = plt.subplots(1, driving + bustransit + pnr, figsize=(36, 9), dpi=300, squeeze=False)
        i = 0
        if driving:
            ind = ~(np.isinf(true_q_e_mode_driving) + np.isinf(q_e_mode_driving) + np.isnan(true_q_e_mode_driving) + np.isnan(q_e_mode_driving))
            m_max = int(np.max((np.max(true_q_e_mode_driving[ind]), np.max(q_e_mode_driving[ind]))) + 1)
            axes[0, i].scatter(true_q_e_mode_driving[ind], q_e_mode_driving[ind], color = self.color_list[i], marker = self.marker_list[i], s = 100)
            axes[0, i].plot(range(m_max + 1), range(m_max + 1), color = 'gray')
            axes[0, i].set_ylabel('Estimated driving demand')
            axes[0, i].set_xlabel('True driving demand')
            axes[0, i].set_xlim([0, m_max])
            axes[0, i].set_ylim([0, m_max])
            axes[0, i].text(0, 1, 'r2 = {}'.format(r2_score(true_q_e_mode_driving[ind], q_e_mode_driving[ind])),
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform=axes[0, i].transAxes)
            i += 1

        if bustransit:
            ind = ~(np.isinf(true_q_e_mode_bustransit) + np.isinf(q_e_mode_bustransit) + np.isnan(true_q_e_mode_bustransit) + np.isnan(q_e_mode_bustransit))
            m_max = int(np.max((np.max(true_q_e_mode_bustransit[ind]), np.max(q_e_mode_bustransit[ind]))) + 1)
            axes[0, i].scatter(true_q_e_mode_bustransit[ind], q_e_mode_bustransit[ind], color = self.color_list[i], marker = self.marker_list[i], s = 100)
            axes[0, i].plot(range(m_max + 1), range(m_max + 1), color = 'gray')
            axes[0, i].set_ylabel('Estimated transit passenger demand')
            axes[0, i].set_xlabel('True transit passenger demand')
            axes[0, i].set_xlim([0, m_max])
            axes[0, i].set_ylim([0, m_max])
            axes[0, i].text(0, 1, 'r2 = {}'.format(r2_score(true_q_e_mode_bustransit[ind], q_e_mode_bustransit[ind])),
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform=axes[0, i].transAxes)
            i += 1

        if pnr:
            ind = ~(np.isinf(true_q_e_mode_pnr) + np.isinf(q_e_mode_pnr) + np.isnan(true_q_e_mode_pnr) + np.isnan(q_e_mode_pnr))
            m_max = int(np.max((np.max(true_q_e_mode_pnr[ind]), np.max(q_e_mode_pnr[ind]))) + 1)
            axes[0, i].scatter(true_q_e_mode_pnr[ind], q_e_mode_pnr[ind], color = self.color_list[i], marker = self.marker_list[i], s = 100)
            axes[0, i].plot(range(m_max + 1), range(m_max + 1), color = 'gray')
            axes[0, i].set_ylabel('Estimated PNR demand')
            axes[0, i].set_xlabel('True PNR demand')
            axes[0, i].set_xlim([0, m_max])
            axes[0, i].set_ylim([0, m_max])
            axes[0, i].text(0, 1, 'r2 = {}'.format(r2_score(true_q_e_mode_pnr[ind], q_e_mode_pnr[ind])),
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform=axes[0, i].transAxes)

        plt.savefig(os.path.join(self.result_folder, fig_name), bbox_inches='tight')
        # plt.show()

    def scatter_plot_totalODdemand_only(self, true_q_e_traveler, q_e_traveler, fig_name = 'OD_demand_scatterplot.png'):
        plt.figure(figsize = (16, 9), dpi=300)
        ax = plt.gca()
        ind = ~(np.isinf(true_q_e_traveler) + np.isinf(q_e_traveler) + np.isnan(true_q_e_traveler) + np.isnan(q_e_traveler))
        m_max = int(np.max((np.max(true_q_e_traveler[ind]), np.max(q_e_traveler[ind]))) + 1)
        plt.scatter(true_q_e_traveler[ind], q_e_traveler[ind], color = self.color_list[0], marker = self.marker_list[0], s = 100)
        plt.plot(range(m_max + 1), range(m_max + 1), color = 'gray')
        plt.ylabel('Estimated demand')
        plt.xlabel('True demand')
        plt.xlim([0, m_max])
        plt.ylim([0, m_max])
        ax.text(0, 1, 'r2 = {}'.format(r2_score(true_q_e_traveler[ind], q_e_traveler[ind])),
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=ax.transAxes)

        plt.savefig(os.path.join(self.result_folder, fig_name), bbox_inches='tight')
        # plt.show()

    def scatter_plot_count(self, fig_name =  'link_flow_scatterplot_pathflow.png'):
        if self.dode.config['use_car_link_flow'] + self.dode.config['use_truck_link_flow'] + \
            self.dode.config['use_passenger_link_flow'] + self.dode.config['use_bus_link_flow'] + \
                (self.dode.config['use_veh_run_boarding_alighting'] or self.dode.config['use_ULP_f_transit']):

            fig, axes = plt.subplots(1,
                                    self.dode.config['use_car_link_flow'] + self.dode.config['use_truck_link_flow'] + 
                                    self.dode.config['use_passenger_link_flow'] + self.dode.config['use_bus_link_flow'] + 
                                    (self.dode.config['use_veh_run_boarding_alighting'] or self.dode.config['use_ULP_f_transit']), 
                                    figsize=(36, 9), dpi=300, squeeze=False)

            i = 0

            if self.dode.config['use_car_link_flow']:
                ind = ~(np.isinf(self.true_car_count) + np.isinf(self.estimated_car_count) + np.isnan(self.true_car_count) + np.isnan(self.estimated_car_count))
                m_car_max = int(np.max((np.max(self.true_car_count[ind]), np.max(self.estimated_car_count[ind]))) + 1)
                axes[0, i].scatter(self.true_car_count[ind], self.estimated_car_count[ind], color = self.color_list[i], marker = self.marker_list[i], s = 100)
                axes[0, i].plot(range(m_car_max + 1), range(m_car_max + 1), color = 'gray')
                axes[0, i].set_ylabel('Estimated flow for car')
                axes[0, i].set_xlabel('Observed flow for car')
                axes[0, i].set_xlim([0, m_car_max])
                axes[0, i].set_ylim([0, m_car_max])
                axes[0, i].text(0, 1, 'r2 = {}'.format(self.r2_car_count),
                            horizontalalignment='left',
                            verticalalignment='top',
                            transform=axes[0, i].transAxes)

            i += self.dode.config['use_car_link_flow']

            if self.dode.config['use_truck_link_flow']:
                ind = ~(np.isinf(self.true_truck_count) + np.isinf(self.estimated_truck_count) + np.isnan(self.true_truck_count) + np.isnan(self.estimated_truck_count))
                m_truck_max = int(np.max((np.max(self.true_truck_count[ind]), np.max(self.estimated_truck_count[ind]))) + 1)
                axes[0, i].scatter(self.true_truck_count[ind], self.estimated_truck_count[ind], color = self.color_list[i], marker = self.marker_list[i], s = 100)
                axes[0, i].plot(range(m_truck_max + 1), range(m_truck_max + 1), color = 'gray')
                axes[0, i].set_ylabel('Estimated flow for truck')
                axes[0, i].set_xlabel('Observed flow for truck')
                axes[0, i].set_xlim([0, m_truck_max])
                axes[0, i].set_ylim([0, m_truck_max])
                axes[0, i].text(0, 1, 'r2 = {}'.format(self.r2_truck_count),
                            horizontalalignment='left',
                            verticalalignment='top',
                            transform=axes[0, i].transAxes)

            i += self.dode.config['use_truck_link_flow']

            if self.dode.config['use_passenger_link_flow']:
                ind = ~(np.isinf(self.true_passenger_count) + np.isinf(self.estimated_passenger_count) + np.isnan(self.true_passenger_count) + np.isnan(self.estimated_passenger_count))
                m_passenger_max = int(np.max((np.max(self.true_passenger_count[ind]), np.max(self.estimated_passenger_count[ind]))) + 1)
                axes[0, i].scatter(self.true_passenger_count[ind], self.estimated_passenger_count[ind], color = self.color_list[i], marker = self.marker_list[i], s = 100)
                axes[0, i].plot(range(m_passenger_max + 1), range(m_passenger_max + 1), color = 'gray')
                axes[0, i].set_ylabel('Estimated flow for passenger')
                axes[0, i].set_xlabel('Observed flow for passenger')
                axes[0, i].set_xlim([0, m_passenger_max])
                axes[0, i].set_ylim([0, m_passenger_max])
                axes[0, i].text(0, 1, 'r2 = {}'.format(self.r2_passenger_count),
                            horizontalalignment='left',
                            verticalalignment='top',
                            transform=axes[0, i].transAxes)
            
            i += self.dode.config['use_passenger_link_flow']

            if self.dode.config['use_bus_link_flow']:
                ind = ~(np.isinf(self.true_bus_count) + np.isinf(self.estimated_bus_count) + np.isnan(self.true_bus_count) + np.isnan(self.estimated_bus_count))
                m_bus_max = int(np.max((np.max(self.true_bus_count[ind]), np.max(self.estimated_bus_count[ind]))) + 1)
                axes[0, i].scatter(self.true_bus_count[ind], self.estimated_bus_count[ind], color = self.color_list[i], marker = self.marker_list[i], s = 100)
                axes[0, i].plot(range(m_bus_max + 1), range(m_bus_max + 1), color = 'gray')
                axes[0, i].set_ylabel('Estimated flow for bus/metro')
                axes[0, i].set_xlabel('Observed flow for bus/metro')
                axes[0, i].set_xlim([0, m_bus_max])
                axes[0, i].set_ylim([0, m_bus_max])
                axes[0, i].text(0, 1, 'r2 = {}'.format(self.r2_bus_count),
                            horizontalalignment='left',
                            verticalalignment='top',
                            transform=axes[0, i].transAxes)
                
            i += self.dode.config['use_bus_link_flow']
                
            if self.dode.config['use_veh_run_boarding_alighting'] or self.dode.config['use_ULP_f_transit']:
                ind = ~(np.isinf(self.true_BoardingAlighting_count) + np.isinf(self.estimated_BoardingAlighting_count) + 
                        np.isnan(self.true_BoardingAlighting_count) + np.isnan(self.estimated_BoardingAlighting_count))
                m_boarding_alighting_max = int(np.max((np.max(self.true_BoardingAlighting_count[ind]), np.max(self.estimated_BoardingAlighting_count[ind]))) + 1)
                axes[0, i].scatter(self.true_BoardingAlighting_count[ind], self.estimated_BoardingAlighting_count[ind], color = self.color_list[i], marker = self.marker_list[i], s = 100)
                axes[0, i].plot(range(m_boarding_alighting_max + 1), range(m_boarding_alighting_max + 1), color = 'gray')
                axes[0, i].set_ylabel('Estimated boarding/alighting flow')
                axes[0, i].set_xlabel('Observed boarding/alighting flow')
                axes[0, i].set_xlim([0, m_boarding_alighting_max])
                axes[0, i].set_ylim([0, m_boarding_alighting_max])
                axes[0, i].text(0, 1, 'r2 = {}'.format(self.r2_BoardingAlighting_count),
                            horizontalalignment='left',
                            verticalalignment='top',
                            transform=axes[0, i].transAxes)

            plt.savefig(os.path.join(self.result_folder, fig_name), bbox_inches='tight')

            # plt.show()

    def cal_r2_cost(self):
        if self.dode.config['use_car_link_tt']:
            # print('----- car cost -----')
            # print(self.true_car_cost)
            # print(self.estimated_car_cost)
            # print('----- car cost -----')
            ind = ~(np.isinf(self.true_car_cost) + np.isinf(self.estimated_car_cost) + np.isnan(self.true_car_cost) + np.isnan(self.estimated_car_cost))
            self.r2_car_cost = r2_score(self.true_car_cost[ind], self.estimated_car_cost[ind])

        if self.dode.config['use_truck_link_tt']:
            # print('----- truck cost -----')
            # print(self.true_truck_cost)
            # print(self.estimated_truck_cost)
            # print('----- truck cost -----')
            ind = ~(np.isinf(self.true_truck_cost) + np.isinf(self.estimated_truck_cost) + np.isnan(self.true_truck_cost) + np.isnan(self.estimated_truck_cost))
            self.r2_truck_cost = r2_score(self.true_truck_cost[ind], self.estimated_truck_cost[ind])

        if self.dode.config['use_passenger_link_tt']:
            # print('----- passenger cost -----')
            # print(self.true_passenger_cost)
            # print(self.estimated_passenger_cost)
            # print('----- passenger cost -----')
            ind = ~(np.isinf(self.true_passenger_cost) + np.isinf(self.estimated_passenger_cost) + np.isnan(self.true_passenger_cost) + np.isnan(self.estimated_passenger_cost))
            self.r2_passenger_cost = r2_score(self.true_passenger_cost[ind], self.estimated_passenger_cost[ind])

        if self.dode.config['use_bus_link_tt']:
            # print('----- bus cost -----')
            # print(self.true_bus_cost)
            # print(self.estimated_bus_cost)
            # print('----- bus cost -----')
            ind = ~(np.isinf(self.true_bus_cost) + np.isinf(self.estimated_bus_cost) + np.isnan(self.true_bus_cost) + np.isnan(self.estimated_bus_cost))
            self.r2_bus_cost = r2_score(self.true_bus_cost[ind], self.estimated_bus_cost[ind])

        if (self.dode.config['use_veh_run_boarding_alighting'] or self.dode.config['use_ULP_f_transit']) and 'stop_arrival_departure_travel_time' in self.one_data_dict:
            merged_df = pd.merge(self.true_stop_arrival_departure_travel_time_df, self.estimated_stop_arrival_departure_travel_time_df, 
                how='inner', on=['route_id','veh_order', 'stop_id'], 
                suffixes=('_true', '_estimated'))
            # drop rows with NaN values in either true or estimated travel time
            merged_df = merged_df.dropna(subset=['pure_travel_time_true', 'pure_travel_time_estimated', 'travel_and_dwell_time_true', 'travel_and_dwell_time_estimated'])
            # drop rows with non-positive travel times
            merged_df = merged_df[(merged_df['pure_travel_time_true'] > 0) & (merged_df['pure_travel_time_estimated'] > 0) &
                                  (merged_df['travel_and_dwell_time_true'] > 0) & (merged_df['travel_and_dwell_time_estimated'] > 0)]
            merged_df.reset_index(inplace=True, drop=True)

            estimated_travel_time = np.array(merged_df['travel_and_dwell_time_estimated'])
            true_travel_time = np.array(merged_df['travel_and_dwell_time_true'])
            self.estimated_stop_travel_dwell_time = estimated_travel_time
            self.true_stop_travel_dwell_time = true_travel_time
            self.r2_travel_and_dwell_time = r2_score(true_travel_time, estimated_travel_time)
        

        print("r2 cost --- r2_car_cost: {}, r2_truck_cost: {}, r2_passenger_cost: {}, r2_bus_cost: {}, r2_travel_and_dwell_time: {}"
            .format(
                self.r2_car_cost, 
                self.r2_truck_cost, 
                self.r2_passenger_cost, 
                self.r2_bus_cost,
                self.r2_travel_and_dwell_time
                ))

        return self.r2_car_cost, self.r2_truck_cost, self.r2_passenger_cost, self.r2_bus_cost, self.r2_travel_and_dwell_time

    def scatter_plot_cost(self, fig_name = 'link_cost_scatterplot_pathflow.png'):
        if self.dode.config['use_car_link_tt'] + self.dode.config['use_truck_link_tt'] + \
            self.dode.config['use_passenger_link_tt'] + self.dode.config['use_bus_link_tt'] + \
                (self.dode.config['use_veh_run_boarding_alighting'] or self.dode.config['use_ULP_f_transit']):

            if self.dode.config['use_car_link_tt'] + self.dode.config['use_truck_link_tt'] + \
                self.dode.config['use_passenger_link_tt'] + self.dode.config['use_bus_link_tt'] + \
                ((self.dode.config['use_veh_run_boarding_alighting'] or self.dode.config['use_ULP_f_transit']) and 'stop_arrival_departure_travel_time' in self.one_data_dict) == 1:
                    fig, axes = plt.subplots(1, 1, figsize=(18, 9), dpi=300, squeeze=False)
            else:
                fig, axes = plt.subplots(1, 
                                        self.dode.config['use_car_link_tt'] + self.dode.config['use_truck_link_tt'] + 
                                        self.dode.config['use_passenger_link_tt'] + self.dode.config['use_bus_link_tt'] + 
                                        (self.dode.config['use_veh_run_boarding_alighting'] or self.dode.config['use_ULP_f_transit']), 
                                        figsize=(36, 9), dpi=300, squeeze=False)
            
            i = 0

            if self.dode.config['use_car_link_tt']:
                ind = ~(np.isinf(self.true_car_cost) + np.isinf(self.estimated_car_cost) + np.isnan(self.true_car_cost) + np.isnan(self.estimated_car_cost))
                car_tt_min = np.min((np.min(self.true_car_cost[ind]), np.min(self.estimated_car_cost[ind]))) - 1
                car_tt_max = np.max((np.max(self.true_car_cost[ind]), np.max(self.estimated_car_cost[ind]))) + 1
                axes[0, i].scatter(self.true_car_cost[ind], self.estimated_car_cost[ind], color = self.color_list[i], marker = self.marker_list[i], s = 100)
                axes[0, i].plot(np.linspace(car_tt_min, car_tt_max, 20), np.linspace(car_tt_min, car_tt_max, 20), color = 'gray')
                axes[0, i].set_ylabel('Estimated travel time for car (s)')
                axes[0, i].set_xlabel('Observed travel time for car (s)')
                axes[0, i].set_xlim([car_tt_min, car_tt_max])
                axes[0, i].set_ylim([car_tt_min, car_tt_max])
                axes[0, i].text(0, 1, 'r2 = {}'.format(self.r2_car_cost),
                            horizontalalignment='left',
                            verticalalignment='top',
                            transform=axes[0, i].transAxes)

            i += self.dode.config['use_car_link_tt']

            if self.dode.config['use_truck_link_tt']:
                ind = ~(np.isinf(self.true_truck_cost) + np.isinf(self.estimated_truck_cost) + np.isnan(self.true_truck_cost) + np.isnan(self.estimated_truck_cost))
                truck_tt_min = np.min((np.min(self.true_truck_cost[ind]), np.min(self.estimated_truck_cost[ind]))) - 1
                truck_tt_max = np.max((np.max(self.true_truck_cost[ind]), np.max(self.estimated_truck_cost[ind]))) + 1
                axes[0, i].scatter(self.true_truck_cost[ind], self.estimated_truck_cost[ind], color = self.color_list[i], marker = self.marker_list[i], s = 100)
                axes[0, i].plot(np.linspace(truck_tt_min, truck_tt_max, 20), np.linspace(truck_tt_min, truck_tt_max, 20), color = 'gray')
                axes[0, i].set_ylabel('Estimated travel time for truck (s)')
                axes[0, i].set_xlabel('Observed travel time for truck (s)')
                axes[0, i].set_xlim([truck_tt_min, truck_tt_max])
                axes[0, i].set_ylim([truck_tt_min, truck_tt_max])
                axes[0, i].text(0, 1, 'r2 = {}'.format(self.r2_truck_cost),
                            horizontalalignment='left',
                            verticalalignment='top',
                            transform=axes[0, i].transAxes)

            i += self.dode.config['use_truck_link_tt']

            if self.dode.config['use_passenger_link_tt']:
                ind = ~(np.isinf(self.true_passenger_cost) + np.isinf(self.estimated_passenger_cost) + np.isnan(self.true_passenger_cost) + np.isnan(self.estimated_passenger_cost))
                passenger_tt_min = np.min((np.min(self.true_passenger_cost[ind]), np.min(self.estimated_passenger_cost[ind]))) - 1
                passenger_tt_max = np.max((np.max(self.true_passenger_cost[ind]), np.max(self.estimated_passenger_cost[ind]))) + 1
                axes[0, i].scatter(self.true_passenger_cost[ind], self.estimated_passenger_cost[ind], color = self.color_list[i], marker = self.marker_list[i], s = 100)
                axes[0, i].plot(np.linspace(passenger_tt_min, passenger_tt_max, 20), np.linspace(passenger_tt_min, passenger_tt_max, 20), color = 'gray')
                axes[0, i].set_ylabel('Estimated travel time for passenger (s)')
                axes[0, i].set_xlabel('Observed travel time for passenger (s)')
                axes[0, i].set_xlim([passenger_tt_min, passenger_tt_max])
                axes[0, i].set_ylim([passenger_tt_min, passenger_tt_max])
                axes[0, i].text(0, 1, 'r2 = {}'.format(self.r2_passenger_cost),
                            horizontalalignment='left',
                            verticalalignment='top',
                            transform=axes[0, i].transAxes)
            
            i += self.dode.config['use_passenger_link_tt']

            if self.dode.config['use_bus_link_tt']:
                ind = ~(np.isinf(self.true_bus_cost) + np.isinf(self.estimated_bus_cost) + np.isnan(self.true_bus_cost) + np.isnan(self.estimated_bus_cost))
                bus_tt_min = np.min((np.min(self.true_bus_cost[ind]), np.min(self.estimated_bus_cost[ind]))) - 1
                bus_tt_max = np.max((np.max(self.true_bus_cost[ind]), np.max(self.estimated_bus_cost[ind]))) + 1
                axes[0, i].scatter(self.true_bus_cost[ind], self.estimated_bus_cost[ind], color = self.color_list[i], marker = self.marker_list[i], s = 100)
                axes[0, i].plot(np.linspace(bus_tt_min, bus_tt_max, 20), np.linspace(bus_tt_min, bus_tt_max, 20), color = 'gray')
                axes[0, i].set_ylabel('Estimated travel time for bus/metro (s)')
                axes[0, i].set_xlabel('Observed travel time for bus/metro (s)')
                axes[0, i].set_xlim([bus_tt_min, bus_tt_max])
                axes[0, i].set_ylim([bus_tt_min, bus_tt_max])
                axes[0, i].text(0, 1, 'r2 = {}'.format(self.r2_bus_cost),
                            horizontalalignment='left',
                            verticalalignment='top',
                            transform=axes[0, i].transAxes)
                
            i += self.dode.config['use_bus_link_tt']
                
            if (self.dode.config['use_veh_run_boarding_alighting'] or self.dode.config['use_ULP_f_transit']) and 'stop_arrival_departure_travel_time' in self.one_data_dict:
                m_travel_dwell_time_max = int(np.max((np.max(self.true_stop_travel_dwell_time), np.max(self.estimated_stop_travel_dwell_time))) + 1)
                axes[0, i].scatter(self.true_stop_travel_dwell_time, self.estimated_stop_travel_dwell_time, color = self.color_list[i], marker = self.marker_list[i], s = 100)
                axes[0, i].plot(range(m_travel_dwell_time_max + 1), range(m_travel_dwell_time_max + 1), color = 'gray')
                axes[0, i].set_ylabel('Estimated stop-level travel+dwelling time')
                axes[0, i].set_xlabel('Observed stop-level travel+dwelling time')
                axes[0, i].set_xlim([0, m_travel_dwell_time_max])
                axes[0, i].set_ylim([0, m_travel_dwell_time_max])
                axes[0, i].text(0, 1, 'r2 = {}'.format(self.r2_travel_and_dwell_time),
                            horizontalalignment='left',
                            verticalalignment='top',
                            transform=axes[0, i].transAxes)

            plt.savefig(os.path.join(self.result_folder, fig_name), bbox_inches='tight')

            # plt.show()


# def r2_score(y_true, y_hat):
#     y_bar = np.mean(y_true)
#     ss_total = np.sum((y_true - y_bar) ** 2)
#     # ss_explained = np.sum((y_hat - y_bar) ** 2)
#     ss_residual = np.sum((y_true - y_hat) ** 2)
#     # scikit_r2 = r2_score(y_true, y_hat)
#     if ss_residual <= 1e-6:
#         return 1
#     if ss_total <= 1e-6:
#         ss_total = 1e-6 
#     return 1 - (ss_residual / ss_total)



# directly use from sklearn.metrics import r2_score, this compute the fitting for y=x
# def r2_score(y_true, y_hat):
#     slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_true, y_hat)
#     # linear regression without intercept
#     # slope, _, _, _ = np.linalg.lstsq(y_true[:, np.newaxis], y_hat)
#     return r_value**2

def swap_dict_and_array(beta, gamma, alpha):
    beta_keys = ['beta_tt_car', 'beta_tt_bus', 'beta_tt_metro', 'beta_walking', 'beta_waiting_bus', 'beta_waiting_metro', 'beta_money']
    gamma_keys = ['gamma_income_car', 'gamma_income_bus', 'gamma_income_metro', 'gamma_income_busmetro', 'gamma_income_carbus', 'gamma_income_carmetro', 'gamma_income_carbusmetro', \
                  'gamma_Originpopden_car', 'gamma_Originpopden_bus', 'gamma_Originpopden_metro', 'gamma_Originpopden_busmetro', 'gamma_Originpopden_carbus', 'gamma_Originpopden_carmetro', 'gamma_Originpopden_carbusmetro', \
                  'gamma_Destpopden_car', 'gamma_Destpopden_bus', 'gamma_Destpopden_metro', 'gamma_Destpopden_busmetro', 'gamma_Destpopden_carbus', 'gamma_Destpopden_carmetro', 'gamma_Destpopden_carbusmetro']
    alpha_keys = ['alpha_bus', 'alpha_metro', 'alpha_busmetro', 'alpha_carbus', 'alpha_carmetro', 'alpha_carbusmetro']
    if isinstance(beta, dict) and isinstance(gamma, dict) and isinstance(alpha, dict):
        beta_array = np.zeros(len(beta))
        for i in range(len(beta_keys)):
            beta_array[i] = beta[beta_keys[i]]
        gamma_array = np.zeros(len(gamma))
        for i in range(len(gamma_keys)):
            gamma_array[i] = gamma[gamma_keys[i]]
        alpha_array = np.zeros(len(alpha))
        for i in range(len(alpha_keys)):
            alpha_array[i] = alpha[alpha_keys[i]]
        return beta_array, gamma_array, alpha_array
    elif isinstance(beta, np.ndarray) and isinstance(gamma, np.ndarray) and isinstance(alpha, np.ndarray):
        beta_dict = dict()
        for i in range(len(beta)):
            beta_dict[beta_keys[i]] = beta[i]
        gamma_dict = dict()
        for i in range(len(gamma)):
            gamma_dict[gamma_keys[i]] = gamma[i]
        alpha_dict = dict()
        for i in range(len(alpha)):
            alpha_dict[alpha_keys[i]] = alpha[i]
        return beta_dict, gamma_dict, alpha_dict
    

def compute_ULP_f(dar, b, lr, iter):
    dar = dar.tocoo()
    values = torch.tensor(dar.data, dtype=torch.float32)
    indices = torch.tensor(np.vstack((dar.row, dar.col)), dtype=torch.int64)
    A_sparse = torch.sparse_coo_tensor(indices, values, dar.shape).coalesce()

    b = torch.tensor(b, dtype=torch.float32)

    # Optimization variable (raw, unconstrained)
    x_raw = torch.randn(A_sparse.shape[1], requires_grad=True)
    optimizer = torch.optim.Adam([x_raw], lr=lr)

    for i in range(iter):
        optimizer.zero_grad()
        x = torch.nn.functional.relu(x_raw)  # enforce x ≥ 0

        # Use sparse-dense multiplication
        Ax = torch.sparse.mm(A_sparse, x.unsqueeze(1)).squeeze()

        loss = torch.norm(Ax - b) ** 2
        loss.backward()
        optimizer.step()

    x_final = torch.nn.functional.relu(x_raw).detach().cpu().numpy()
    return x_final


def compute_ULP_f_Gurobi(A, b):
    m, n = A.shape

    # -------- STEP 1: Try to solve Ax = b, x >= 0 -------- #
    model_lp = gp.Model("feasibility_sparse")
    x = model_lp.addVars(n, lb=0.0, vtype=GRB.CONTINUOUS, name="x")
    model_lp.setObjective(0, GRB.MINIMIZE)

    # Convert to COO for easier row/column indexing
    A_coo = A.tocoo()
    rows = [[] for _ in range(m)]

    for i, j, v in zip(A_coo.row, A_coo.col, A_coo.data):
        rows[i].append((j, v))

    # Add constraints: Ax = b
    for i in range(m):
        expr = gp.LinExpr()
        for j, val in rows[i]:
            expr.add(x[j], val)
        model_lp.addConstr(expr == b[i], name=f"eq_{i}")

    model_lp.setParam('OutputFlag', 0)
    model_lp.optimize()

    if model_lp.status == GRB.OPTIMAL:
        solution = np.array([x[i].X for i in range(n)])
        # print("Feasible solution found (Ax = b, x >= 0):")
        # print("x =", solution)
    else:
        # print("No feasible solution. Solving least-squares with x >= 0...")

        # -------- STEP 2: Solve min ||Ax - b||^2 with x >= 0 -------- #
        model_qp = gp.Model("least_squares_sparse")
        x_qp = model_qp.addVars(n, lb=0.0, vtype=GRB.CONTINUOUS, name="x_qp")

        obj = gp.QuadExpr()
        for i in range(m):
            expr = gp.LinExpr()
            for j, val in rows[i]:
                expr.add(x_qp[j], val)
            expr.addConstant(-b[i])
            obj += expr * expr

        model_qp.setObjective(obj, GRB.MINIMIZE)
        model_qp.setParam('OutputFlag', 0)
        model_qp.optimize()

        if model_qp.status == GRB.OPTIMAL:
            solution = np.array([x_qp[i].X for i in range(n)])
            # print("Least-squares solution (min ||Ax - b||^2, x >= 0):")
            # print("x =", solution)
            # print("Residual norm ||Ax - b|| =", np.linalg.norm(A @ solution - b))
            # print('R2 score:', r2_score(b, A @ solution))
        else:
            #print("Quadratic program failed.")
            solution = None
    return solution


def compute_Jacobian(Jacobian_dict, beta_dict, beta_der_dict, gamma_der_dict, alpha_der_dict):
    car_count_J_beta, car_count_J_gamma, car_count_J_alpha = dict(), dict(), dict()
    car_time_J_beta, car_time_J_gamma, car_time_J_alpha = dict(), dict(), dict()
    BoardingAlightingCount_J_beta, BoardingAlightingCount_J_gamma, BoardingAlightingCount_J_alpha = dict(), dict(), dict()
    for key in beta_dict.keys():
        if key == 'beta_tt_car':
            car_count_J_beta[key] = np.dot(Jacobian_dict['car_count_J_f_driving'], beta_der_dict[key]['driving'].flatten(order='F').reshape(-1, 1)) + \
                            np.dot(Jacobian_dict['car_count_J_f_pnr'], beta_der_dict[key]['pnr'].flatten(order='F').reshape(-1, 1))
            car_time_J_beta[key] = np.dot(Jacobian_dict['car_time_J_f_driving'], beta_der_dict[key]['driving'].flatten(order='F').reshape(-1, 1)) + \
                            np.dot(Jacobian_dict['car_time_J_f_pnr'], beta_der_dict[key]['pnr'].flatten(order='F').reshape(-1, 1))
            BoardingAlightingCount_J_beta[key] = np.dot(Jacobian_dict['BoardingAlightingCount_J_f_pnr'], beta_der_dict[key]['pnr'].flatten(order='F').reshape(-1, 1))
        elif key == 'beta_money':
            car_count_J_beta[key] = np.dot(Jacobian_dict['car_count_J_f_driving'], beta_der_dict[key]['driving'].flatten(order='F').reshape(-1, 1)) + \
                            np.dot(Jacobian_dict['car_count_J_f_pnr'], beta_der_dict[key]['pnr'].flatten(order='F').reshape(-1, 1)) + \
                            np.dot(Jacobian_dict['car_count_J_f_transit'], beta_der_dict[key]['transit'].flatten(order='F').reshape(-1, 1))
            car_time_J_beta[key] = np.dot(Jacobian_dict['car_time_J_f_driving'], beta_der_dict[key]['driving'].flatten(order='F').reshape(-1, 1)) + \
                            np.dot(Jacobian_dict['car_time_J_f_pnr'], beta_der_dict[key]['pnr'].flatten(order='F').reshape(-1, 1)) 
            BoardingAlightingCount_J_beta[key] = np.dot(Jacobian_dict['BoardingAlightingCount_J_f_pnr'], beta_der_dict[key]['pnr'].flatten(order='F').reshape(-1, 1)) + \
                            np.dot(Jacobian_dict['BoardingAlightingCount_J_f_transit'], beta_der_dict[key]['transit'].flatten(order='F').reshape(-1, 1))
        else:
            car_count_J_beta[key] = np.dot(Jacobian_dict['car_count_J_f_pnr'], beta_der_dict[key]['pnr'].flatten(order='F').reshape(-1, 1))
            car_time_J_beta[key] = np.dot(Jacobian_dict['car_time_J_f_pnr'], beta_der_dict[key]['pnr'].flatten(order='F').reshape(-1, 1))
            BoardingAlightingCount_J_beta[key] = np.dot(Jacobian_dict['BoardingAlightingCount_J_f_transit'], beta_der_dict[key]['transit'].flatten(order='F').reshape(-1, 1)) +\
                np.dot(Jacobian_dict['BoardingAlightingCount_J_f_pnr'], beta_der_dict[key]['pnr'].flatten(order='F').reshape(-1, 1))
    for string1 in ['income', 'Originpopden', 'Destpopden']:
        car_count_J_gamma['gamma_' + string1 + '_car'] = np.dot(Jacobian_dict['car_count_J_f_driving'], gamma_der_dict['gamma_' + string1 + '_car'].flatten(order='F').reshape(-1, 1))
        car_time_J_gamma['gamma_' + string1 + '_car'] = np.dot(Jacobian_dict['car_time_J_f_driving'], gamma_der_dict['gamma_' + string1 + '_car'].flatten(order='F').reshape(-1, 1))
        BoardingAlightingCount_J_gamma['gamma_' + string1 + '_car'] = np.zeros((Jacobian_dict['BoardingAlightingCount_J_f_pnr'].shape[0], 1))
        for string2 in ['bus', 'metro', 'busmetro']:
            car_count_J_gamma['gamma_' + string1 + '_' + string2] = np.zeros((Jacobian_dict['car_count_J_f_pnr'].shape[0], 1))  
            car_time_J_gamma['gamma_' + string1 + '_' + string2] = np.zeros((Jacobian_dict['car_time_J_f_pnr'].shape[0], 1))
            BoardingAlightingCount_J_gamma['gamma_' + string1 + '_' + string2] = np.dot(Jacobian_dict['BoardingAlightingCount_J_f_transit'],\
                                                                                         gamma_der_dict['gamma_' + string1 + '_' + string2].flatten(order='F').reshape(-1, 1))    
        for string2 in ['carbus', 'carmetro', 'carbusmetro']:
            car_count_J_gamma['gamma_' + string1 + '_' + string2] = np.dot(Jacobian_dict['car_count_J_f_pnr'], gamma_der_dict['gamma_' + string1 + '_' + string2].flatten(order='F').reshape(-1, 1))
            car_time_J_gamma['gamma_' + string1 + '_' + string2] = np.dot(Jacobian_dict['car_time_J_f_pnr'], gamma_der_dict['gamma_' + string1 + '_' + string2].flatten(order='F').reshape(-1, 1))
            BoardingAlightingCount_J_gamma['gamma_' + string1 + '_' + string2] = np.dot(Jacobian_dict['BoardingAlightingCount_J_f_pnr'], \
                                                                                        gamma_der_dict['gamma_' + string1 + '_' + string2].flatten(order='F').reshape(-1, 1))
    for string1 in ['bus', 'metro', 'busmetro']:
        car_count_J_alpha['alpha_' + string1] = np.zeros((Jacobian_dict['car_count_J_f_pnr'].shape[0], 1))
        car_time_J_alpha['alpha_' + string1] = np.zeros((Jacobian_dict['car_time_J_f_pnr'].shape[0], 1))
        BoardingAlightingCount_J_alpha['alpha_' + string1] = np.dot(Jacobian_dict['BoardingAlightingCount_J_f_transit'], alpha_der_dict['alpha_' + string1].flatten(order='F').reshape(-1, 1))
    for string1 in ['carbus', 'carmetro', 'carbusmetro']:
        car_count_J_alpha['alpha_' + string1] = np.dot(Jacobian_dict['car_count_J_f_pnr'], alpha_der_dict['alpha_' + string1].flatten(order='F').reshape(-1, 1))
        car_time_J_alpha['alpha_' + string1] = np.dot(Jacobian_dict['car_time_J_f_pnr'], alpha_der_dict['alpha_' + string1].flatten(order='F').reshape(-1, 1))
        BoardingAlightingCount_J_alpha['alpha_' + string1] = np.dot(Jacobian_dict['BoardingAlightingCount_J_f_pnr'], alpha_der_dict['alpha_' + string1].flatten(order='F').reshape(-1, 1))
