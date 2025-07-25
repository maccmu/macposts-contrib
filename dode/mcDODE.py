import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import time
import shutil
from scipy.sparse import coo_matrix, csr_matrix
import pickle
import multiprocessing as mp
from typing import Union
import torch
import torch.nn as nn
import scipy
from sklearn.metrics import r2_score
import re
matches = [s for s in plt.style.available if re.search(r"seaborn.*poster", s)]
assert(len(matches) > 0)
plt_style = matches[0]

# def r2_score(y_true, y_hat):
#     slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_true, y_hat)
#     # linear regression without intercept
#     # slope, _, _, _ = np.linalg.lstsq(y_true[:, np.newaxis], y_hat)
#     return r_value**2

import macposts

# from numba import jit

# @jit(nopython=True)
def find_first(vec, item):
    for i, v in enumerate(vec):
         if v == item: 
            return i
    return -1


def tensor_to_numpy(x: torch.Tensor):
    return x.data.cpu().numpy()

def groupby_sum(value:torch.Tensor, labels:torch.LongTensor):
    # https://discuss.pytorch.org/t/groupby-aggregate-mean-in-pytorch/45335/9
    """Group-wise sum(average) for (sparse) grouped tensors
    
    Args:
        value (torch.Tensor): values to average (# samples, latent dimension)
        labels (torch.LongTensor): labels for embedding parameters (# samples,)
    
    Returns: 
        result (torch.Tensor): (# unique labels, latent dimension)
        new_labels (torch.LongTensor): (# unique labels,)
        
    Examples:
        >>> samples = torch.Tensor([
                             [0.15, 0.15, 0.15],    #-> group / class 1
                             [0.2, 0.2, 0.2],    #-> group / class 5
                             [0.4, 0.4, 0.4],    #-> group / class 5
                             [0.0, 0.0, 0.0]     #-> group / class 0
                      ])
        >>> labels = torch.LongTensor([1, 5, 5, 0])
        >>> result, new_labels = groupby_sum(samples, labels)
        
        >>> result
        tensor([[0.0000, 0.0000, 0.0000],
            [0.1500, 0.1500, 0.1500],
            [0.6000, 0.6000, 0.6000]])
            
        >>> new_labels
        tensor([0, 1, 5])
    """
    uniques = labels.unique().tolist()
    labels = labels.tolist()

    key_val = {key: val for key, val in zip(uniques, range(len(uniques)))}
    val_key = {val: key for key, val in zip(uniques, range(len(uniques)))}
    
    labels = torch.LongTensor(list(map(key_val.get, labels)))
    
    labels = labels.view(labels.size(0), 1).expand(-1, value.size(1))
    
    unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
    result = torch.zeros_like(unique_labels, dtype=value.dtype).scatter_add_(0, labels, value)
    # result = result / labels_count.float().unsqueeze(1)
    new_labels = torch.LongTensor(list(map(val_key.get, unique_labels[:, 0].tolist())))
    return result, new_labels

class torch_pathflow_solver(nn.Module):
    def __init__(self, num_assign_interval, num_path,
               car_scale=1, truck_scale=0.1, use_file_as_init=None):
        super(torch_pathflow_solver, self).__init__()

        self.num_assign_interval = num_assign_interval

        self.car_scale = car_scale if car_scale is not None else 1.
        self.truck_scale = truck_scale if truck_scale is not None else 1.

        self.num_path = num_path

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
        log_f_car, log_f_truck = None, None

        if use_file_as_init is not None:
            if use_file_as_init.split(os.sep)[-1] == 'init_path_flow.pkl':
                (log_f_car, log_f_truck) = pickle.load(open(use_file_as_init, 'rb'))
                assert(np.all(log_f_car >= 0) and np.all(log_f_truck >= 0))
                # inverse of softplus, default threshold of nn.Softplus is 20
                log_f_car = np.where(np.log(np.exp(log_f_car) - 1) < 20., np.log(np.exp(log_f_car) - 1), log_f_car)
                log_f_truck = np.where(np.log(np.exp(log_f_truck) - 1) < 20., np.log(np.exp(log_f_truck) - 1), log_f_truck)
                self.car_scale, self.truck_scale = 1., 1.
            else:
                # log f numpy
                _, _, _, _, log_f_car, log_f_truck, _, _,\
                    _, _, _, _, _ = pickle.load(open(use_file_as_init, 'rb'))

        if log_f_car is None:
            self.log_f_car = nn.Parameter(self.init_tensor(torch.Tensor(self.num_assign_interval * self.num_path, 1)).squeeze(), requires_grad=True)
        else:
            assert(np.prod(log_f_car.shape) == self.num_assign_interval * self.num_path)
            self.log_f_car = nn.Parameter(torch.from_numpy(log_f_car).squeeze(), requires_grad=True)
        
        if log_f_truck is None:
            self.log_f_truck = nn.Parameter(self.init_tensor(torch.Tensor(self.num_assign_interval * self.num_path, 1)).squeeze(), requires_grad=True)
        else:
            assert(np.prod(log_f_truck.shape) == self.num_assign_interval * self.num_path)
            self.log_f_truck = nn.Parameter(torch.from_numpy(log_f_truck).squeeze(), requires_grad=True)

    def get_log_f_tensor(self):
        return self.log_f_car, self.log_f_truck

    def get_log_f_numpy(self):
        return tensor_to_numpy(self.log_f_car).flatten(), tensor_to_numpy(self.log_f_truck).flatten()

    def add_pathflow(self, num_path_add: int):
        self.num_path += num_path_add

        if num_path_add > 0:
            self.log_f_car = nn.Parameter(self.log_f_car.reshape(self.num_assign_interval, -1))
            log_f_car_add = nn.Parameter(self.init_tensor(torch.Tensor(self.num_assign_interval * num_path_add, 1)).squeeze().reshape(self.num_assign_interval, -1), requires_grad=True)
            self.log_f_car = nn.Parameter(torch.cat([self.log_f_car, log_f_car_add], dim=1))
            assert(self.log_f_car.shape[1] == self.num_path)
            self.log_f_car = nn.Parameter(self.log_f_car.reshape(-1))

            self.log_f_truck = nn.Parameter(self.log_f_truck_driving.reshape(self.num_assign_interval, -1))
            log_f_truck_add = nn.Parameter(self.init_tensor(torch.Tensor(self.num_assign_interval * num_path_add, 1)).squeeze().reshape(self.num_assign_interval, -1), requires_grad=True)
            self.log_f_truck = nn.Parameter(torch.cat([self.log_f_truck, log_f_truck_add], dim=1))
            assert(self.log_f_truck.shape[1] == self.num_path)
            self.log_f_truck = nn.Parameter(self.log_f_truck.reshape(-1))

    def generate_pathflow_tensor(self):
        # softplus
        f_car = torch.nn.functional.softplus(self.log_f_car) * self.car_scale
        f_truck = torch.nn.functional.softplus(self.log_f_truck) * self.truck_scale

        # f_car = torch.clamp(f_car, max=5e3)
        # f_truck = torch.clamp(f_truck, max=5e3)

        # relu
        # f_car = torch.clamp(self.log_f_car * self.car_scale, min=1e-6)
        # f_truck = torch.clamp(self.log_f_truck * self.truck_scale, min=1e-6)

        return f_car, f_truck

    def generate_pathflow_numpy(self, f_car: Union[torch.Tensor, None] = None, f_truck: Union[torch.Tensor, None] = None):
        if (f_car is None) and (f_truck is None):
            f_car, f_truck = self.generate_pathflow_tensor()
       
        return tensor_to_numpy(f_car).flatten(), tensor_to_numpy(f_truck).flatten()

    def set_params_with_lr(self, car_step_size=1e-2, truck_step_size=1e-3):
        self.params = [
            {'params': self.log_f_car, 'lr': car_step_size},
            {'params': self.log_f_truck, 'lr': truck_step_size}
        ]
    
    def set_optimizer(self, algo='NAdam'):
        self.optimizer = self.algo_dict[algo](self.params)

    def compute_gradient(self, f_car: torch.Tensor, f_truck: torch.Tensor, 
                         f_car_grad: np.ndarray, f_truck_grad: np.ndarray, l2_coeff: float = 1e-5):

        f_car.backward(torch.from_numpy(f_car_grad) + l2_coeff * f_car.data)
        f_truck.backward(torch.from_numpy(f_truck_grad) + l2_coeff * f_truck.data)

        # torch.nn.utils.clip_grad_value_(self.parameters(), 0.4)
    
    def set_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode='min', factor=0.75, patience=5, 
        #     threshold=0.15, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)


class MCDODE():
    def __init__(self, nb, config, num_procs=1):
        self.config = config
        if 'origin_registration_data_car_weight' not in self.config:
            self.config['origin_registration_data_car_weight'] = 1
        if 'origin_registration_data_truck_weight' not in self.config:
            self.config['origin_registration_data_truck_weight'] = 1
        if 'od_demand_data_car_weight' not in self.config:
            self.config['od_demand_data_car_weight'] = 1
        if 'od_demand_data_truck_weight' not in self.config:
            self.config['od_demand_data_truck_weight'] = 1

        self.nb = nb
        self.num_assign_interval = nb.config.config_dict['DTA']['max_interval']
        self.ass_freq = nb.config.config_dict['DTA']['assign_frq']
        self.num_link = nb.config.config_dict['DTA']['num_of_link']
        self.num_path = nb.config.config_dict['FIXED']['num_path']
        # if nb.config.config_dict['DTA']['total_interval'] > 0 and nb.config.config_dict['DTA']['total_interval'] > self.num_assign_interval * self.ass_freq:
        #   self.num_loading_interval = nb.config.config_dict['DTA']['total_interval']
        # else:
        #   self.num_loading_interval = self.num_assign_interval * self.ass_freq  # not long enough
        self.num_loading_interval = self.num_assign_interval * self.ass_freq
        self.data_dict = dict()
        self.num_data = self.config['num_data']
        self.observed_links = self.config['observed_links']
        self.paths_list = self.config['paths_list']
        self.car_count_agg_L_list = None
        self.truck_count_agg_L_list = None
        assert (len(self.paths_list) == self.num_path)

        self.num_procs = num_procs

    def get_registered_links_coverage_by_registered_paths(self, folder):
        a = macposts.mcdta_api()
        a.initialize(folder)
        a.register_links(self.observed_links)
        a.register_paths(self.paths_list)

        link_coverage = a.get_registered_links_coverage_in_registered_paths()
        assert((link_coverage.shape[0] == len(self.paths_list)) & (link_coverage.shape[1] == len(self.observed_links)))
        return link_coverage

    def check_registered_links_covered_by_registered_paths(self, folder, add=False):
        self.save_simulation_input_files(folder)

        a = macposts.mcdta_api()
        a.initialize(folder)

        a.register_links(self.observed_links)

        a.register_paths(self.paths_list)

        is_driving_link_covered = a.are_registered_links_in_registered_paths()

        is_updated = 0
        if add:
            if np.any(is_driving_link_covered == False):
                is_driving_link_covered = a.generate_paths_to_cover_registered_links()
                is_updated = is_driving_link_covered[0]
                is_driving_link_covered = is_driving_link_covered[1:]
            else:
                is_updated = 0
        return is_updated, is_driving_link_covered
    
    def generate_paths_to_cover_links(self, folder, links, max_iter):
        self.save_simulation_input_files(folder)
        a = macposts.mcdta_api()
        a.initialize(folder)
        a.register_links(self.observed_links)
        a.register_paths(self.paths_list)
        related_paths_added = a.generate_paths_to_cover_links(links, max_iter)
        return related_paths_added

    def _add_car_link_flow_data(self, link_flow_df_list):
        # assert (self.config['use_car_link_flow'])
        assert (self.num_data == len(link_flow_df_list))
        self.data_dict['car_link_flow'] = link_flow_df_list

    def _add_truck_link_flow_data(self, link_flow_df_list):
        # assert (self.config['use_truck_link_flow'])
        assert (self.num_data == len(link_flow_df_list))
        self.data_dict['truck_link_flow'] = link_flow_df_list

    def _add_car_link_tt_data(self, link_spd_df_list):
        # assert (self.config['use_car_link_tt'])
        assert (self.num_data == len(link_spd_df_list))
        self.data_dict['car_link_tt'] = link_spd_df_list

    def _add_truck_link_tt_data(self, link_spd_df_list):
        # assert (self.config['use_truck_link_tt'])
        assert (self.num_data == len(link_spd_df_list))
        self.data_dict['truck_link_tt'] = link_spd_df_list

    def _add_origin_vehicle_registration_data(self, origin_vehicle_registration_data_list):
        # assert (self.config['use_truck_link_tt'])
        assert (self.num_data == len(origin_vehicle_registration_data_list))
        self.data_dict['origin_vehicle_registration_data'] = origin_vehicle_registration_data_list
    
    def _add_od_demand_data(self, od_demand_data_list):
        assert (self.num_data == len(od_demand_data_list))
        self.data_dict['od_demand_data'] = od_demand_data_list

    def add_data(self, data_dict):
        if self.config['car_count_agg']:
            self.car_count_agg_L_list = data_dict['car_count_agg_L_list']
        if self.config['truck_count_agg']:
            self.truck_count_agg_L_list = data_dict['truck_count_agg_L_list']
        if self.config['use_car_link_flow'] or self.config['compute_car_link_flow_loss']:
            self._add_car_link_flow_data(data_dict['car_link_flow'])
        if self.config['use_truck_link_flow'] or self.config['compute_truck_link_flow_loss'] :
            self._add_truck_link_flow_data(data_dict['truck_link_flow'])
        if self.config['use_car_link_tt'] or self.config['compute_car_link_tt_loss']:
            self._add_car_link_tt_data(data_dict['car_link_tt'])
        if self.config['use_truck_link_tt'] or self.config['compute_car_link_tt_loss']:
            self._add_truck_link_tt_data(data_dict['truck_link_tt'])
        if self.config['use_origin_vehicle_registration_data'] or self.config['compute_origin_vehicle_registration_loss']:
            self._add_origin_vehicle_registration_data(data_dict['origin_vehicle_registration_data'])
        if self.config['use_od_demand_data'] or self.config['compute_od_demand_loss']:
            self._add_od_demand_data(data_dict['od_demand_data'])

        if 'mask_driving_link' in data_dict:
            self.data_dict['mask_driving_link'] = np.tile(data_dict['mask_driving_link'], self.num_assign_interval)
        else:
            self.data_dict['mask_driving_link'] = np.ones(len(self.observed_links) * self.num_assign_interval, dtype=bool)

        if 'link_loss_weight_count_car' in data_dict:
            self.data_dict['link_loss_weight_count_car'] = np.tile(data_dict['link_loss_weight_count_car'], self.num_assign_interval)
        if 'link_loss_weight_count_truck' in data_dict:
            self.data_dict['link_loss_weight_count_truck'] = np.tile(data_dict['link_loss_weight_count_truck'], self.num_assign_interval)

    def save_simulation_input_files(self, folder_path, f_car=None, f_truck=None):

        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        # modify demand based on input path flows
        if (f_car is not None) and (f_truck is not None):
            self.nb.update_demand_path2(f_car, f_truck)
   
        # self.nb.config.config_dict['DTA']['flow_scalar'] = 3
        if self.config['use_car_link_tt'] or self.config['use_truck_link_tt']:
            self.nb.config.config_dict['DTA']['total_interval'] = self.num_loading_interval # * 2  # hopefully this is sufficient 
        else:
            self.nb.config.config_dict['DTA']['total_interval'] = self.num_loading_interval  # if only count data is used

        self.nb.config.config_dict['DTA']['routing_type'] = 'Biclass_Hybrid'

        # no output files saved from DNL
        self.nb.config.config_dict['STAT']['rec_volume'] = 1
        self.nb.config.config_dict['STAT']['volume_load_automatic_rec'] = 0
        self.nb.config.config_dict['STAT']['volume_record_automatic_rec'] = 0
        self.nb.config.config_dict['STAT']['rec_tt'] = 1
        self.nb.config.config_dict['STAT']['tt_load_automatic_rec'] = 0
        self.nb.config.config_dict['STAT']['tt_record_automatic_rec'] = 0
        # save modified files in new_folder
        self.nb.dump_to_folder(folder_path)

    def _run_simulation(self, f_car, f_truck, counter=0, show_loading=False):
        # create a new_folder with a unique name
        hash1 = hashlib.sha1()
        # python 2
        # hash1.update(str(time.time()) + str(counter))
        # python 3
        hash1.update((str(time.time()) + str(counter)).encode('utf-8'))

        new_folder = str(hash1.hexdigest())

        self.save_simulation_input_files(new_folder, f_car, f_truck)

        # invoke macposts
        a = macposts.mcdta_api()
        # read all files in new_folder
        a.initialize(new_folder)
        
        # register links and paths
        a.register_links(self.observed_links)
        a.register_paths(self.paths_list)
        # install cc and cc_tree on registered links
        a.install_cc()
        a.install_cc_tree()
        # run DNL
        a.run_whole(show_loading)
        # print("Finish simulation", time.time())

        travel_stats = a.get_travel_stats()
        print("\n************ travel stats ************")
        print("Total released vehicles: {}".format(travel_stats[0]))
        print("Total enroute vehicles: {}".format(travel_stats[1]))
        print("Total finished vehicles: {}".format(travel_stats[2]))
        print("Total released VMT (miles): {}".format(travel_stats[3]))
        print("Total released VHT (hours): {}".format(travel_stats[4]))
        print("Total released average delay (minutes): {}".format(travel_stats[5]))
        print("Total released cars: {}".format(travel_stats[6]))
        print("Total enroute cars: {}".format(travel_stats[7]))
        print("Total finished cars: {}".format(travel_stats[8]))
        print("Total released car VMT (miles): {}".format(travel_stats[9]))
        print("Total released car VHT (hours): {}".format(travel_stats[10]))
        print("Total released car average delay (minutes): {}".format(travel_stats[11]))
        print("Total released trucks: {}".format(travel_stats[12]))
        print("Total enroute trucks: {}".format(travel_stats[13]))
        print("Total finished trucks: {}".format(travel_stats[14]))
        print("Total released truck VMT (miles): {}".format(travel_stats[15]))
        print("Total released truck VHT (hours): {}".format(travel_stats[16]))
        print("Total released truck average delay (minutes): {}".format(travel_stats[17]))
        print("************ travel stats ************\n")

        # print_emission_stats() only works if folder is not removed, cannot find reason
        a.print_emission_stats()

        # delete new_folder and all files and subdirectories below it.
        shutil.rmtree(new_folder)

        a.delete_all_agents()

        return a

    def get_dar3(self, dta, f_car, f_truck, engine='pyarrow'):

        if self.config['use_car_link_flow'] or self.config['use_car_link_tt']:
            # (num_assign_timesteps x num_links x num_path x num_assign_timesteps) x 5
            file_name = os.path.join(self.nb.folder_path, 'record', 'car_dar_matrix.txt')
            dta.save_car_dar_matrix(np.arange(0, self.num_loading_interval, self.ass_freq), 
                                    np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq,
                                    f_car, file_name)
            # print("raw car dar", raw_car_dar)
            # num_assign_interval * num_e_link, num_assign_interval * num_e_path
            if os.path.exists(file_name):
                # require pandas > 1.4 and pyarrow
                try:
                    dar_triplets = pd.read_csv(file_name, header=None, engine=engine, dtype={0: int, 1: int, 2: float})
                except pd.errors.EmptyDataError:
                    car_dar = csr_matrix((self.num_assign_interval * len(self.observed_links), self.num_assign_interval * len(self.paths_list)))
                else:
                    car_dar = coo_matrix((dar_triplets[2], (dar_triplets[0], dar_triplets[1])), 
                                        shape=(self.num_assign_interval * len(self.observed_links), self.num_assign_interval * len(self.paths_list)))
                    car_dar = car_dar.tocsr()
                    dar_triplets = None

                os.remove(file_name)
                if car_dar.max() == 0.:
                    print("car_dar is empty!")
            else:
                raise Exception('No car_dar_matrix.txt')
            

        if self.config['use_truck_link_flow'] or self.config['use_truck_link_tt']:
            # (num_assign_timesteps x num_links x num_path x num_assign_timesteps) x 5
            file_name = os.path.join(self.nb.folder_path, 'record', 'truck_dar_matrix.txt')
            dta.save_truck_dar_matrix(np.arange(0, self.num_loading_interval, self.ass_freq), 
                                    np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq,
                                    f_truck, file_name)
            # print("raw car dar", raw_car_dar)
            # num_assign_interval * num_e_link, num_assign_interval * num_e_path
            if os.path.exists(file_name):
                # require pandas > 1.4 and pyarrow
                try:
                    dar_triplets = pd.read_csv(file_name, header=None, engine=engine, dtype={0: int, 1: int, 2: float})
                except pd.errors.EmptyDataError:
                    truck_dar = csr_matrix((self.num_assign_interval * len(self.observed_links), self.num_assign_interval * len(self.paths_list)))
                else:
                    truck_dar = coo_matrix((dar_triplets[2], (dar_triplets[0], dar_triplets[1])), 
                                        shape=(self.num_assign_interval * len(self.observed_links), self.num_assign_interval * len(self.paths_list)))
                    truck_dar = truck_dar.tocsr()
                    dar_triplets = None
                    
                os.remove(file_name)
                if truck_dar.max() == 0.:
                    print("truck_dar is empty!")
            else:
                raise Exception('No truck_dar_matrix.txt')

        # print("dar", car_dar, truck_dar)
        return (car_dar, truck_dar)
        
    def get_dar2(self, dta, f_car, f_truck):
        car_dar = csr_matrix((self.num_assign_interval * len(self.observed_links), self.num_assign_interval * len(self.paths_list)))
        truck_dar = csr_matrix((self.num_assign_interval * len(self.observed_links), self.num_assign_interval * len(self.paths_list)))
        if self.config['use_car_link_flow'] or self.config['use_car_link_tt']:
            # num_assign_interval * num_e_link, num_assign_interval * num_e_path
            car_dar = dta.get_complete_car_dar_matrix(np.arange(0, self.num_loading_interval, self.ass_freq), 
                                                      np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq,
                                                      self.num_assign_interval,
                                                      f_car)
            if car_dar.max() == 0.:
                print("car_dar is empty!")

        if self.config['use_truck_link_flow'] or self.config['use_truck_link_tt']:
            # num_assign_interval * num_e_link, num_assign_interval * num_e_path
            truck_dar = dta.get_complete_truck_dar_matrix(np.arange(0, self.num_loading_interval, self.ass_freq), 
                                                          np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq,
                                                          self.num_assign_interval,
                                                          f_truck)
            if truck_dar.max() == 0.:
                print("truck_dar is empty!")

        # print("dar", car_dar, truck_dar)
        return (car_dar, truck_dar)

    def get_dar(self, dta, f_car, f_truck):
        car_dar = csr_matrix((self.num_assign_interval * len(self.observed_links), self.num_assign_interval * len(self.paths_list)))
        truck_dar = csr_matrix((self.num_assign_interval * len(self.observed_links), self.num_assign_interval * len(self.paths_list)))
        if self.config['use_car_link_flow'] or self.config['use_car_link_tt']:
            # (num_assign_timesteps x num_links x num_path x num_assign_timesteps) x 5
            raw_car_dar = dta.get_car_dar_matrix(np.arange(0, self.num_loading_interval, self.ass_freq), 
                                                np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq)
            # print("raw car dar", raw_car_dar)
            # num_assign_interval * num_e_link, num_assign_interval * num_e_path
            car_dar = self._massage_raw_dar(raw_car_dar, self.ass_freq, f_car, self.num_assign_interval, self.paths_list, self.observed_links, self.num_procs)
            if car_dar.max() == 0.:
                print("car_dar is empty!")

        if self.config['use_truck_link_flow'] or self.config['use_truck_link_tt']:
            # (num_assign_timesteps x num_links x num_path x num_assign_timesteps) x 5
            raw_truck_dar = dta.get_truck_dar_matrix(np.arange(0, self.num_loading_interval, self.ass_freq), 
                                                    np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq)
            # num_assign_interval * num_e_link, num_assign_interval * num_e_path
            truck_dar = self._massage_raw_dar(raw_truck_dar, self.ass_freq, f_truck, self.num_assign_interval, self.paths_list, self.observed_links, self.num_procs)
            if truck_dar.max() == 0.:
                print("truck_dar is empty!")

        # print("dar", car_dar, truck_dar)
        return (car_dar, truck_dar)

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

        assert(set(raw_dar[:, 0]).issubset(set(paths_list)))

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

            # # if release_one_interval_biclass set the 1-min assign interval for vehicle for the multiclass modeling
            # path_seq = (raw_dar.loc[:, 0].astype(int).parallel_apply(lambda x: np.nonzero(paths_list == x)[0][0]) 
            #             + (raw_dar.loc[:, 1] / small_assign_freq).astype(int) * num_e_path).astype(int)
            
            # path_seq = (raw_dar.loc[:, 0].astype(int).parallel_apply(lambda x: find_first(paths_list, x)) 
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

            link_seq = (raw_dar.loc[:, 2].astype(int).parallel_apply(lambda x: np.nonzero(observed_links == x)[0][0])
                        + (raw_dar.loc[:, 3] / ass_freq).astype(int) * num_e_link).astype(int)

            # link_seq = (raw_dar.loc[:, 2].astype(int).parallel_apply(lambda x: find_first(observed_links, x))
            #             + (raw_dar.loc[:, 3] / ass_freq).astype(int) * num_e_link).astype(int)

            p = raw_dar.loc[:, 4] / f[path_seq]
            
        else:

            # raw_dar[:, 0]: path no.
            # raw_dar[:, 1]: originally the count of 1 min interval, if release_one_interval_biclass already set the correct assign interval for vehicle, then this is 15 min intervals
            # path_seq = (raw_dar[:, 0].astype(int) + (raw_dar[:, 1] / small_assign_freq).astype(int) * num_e_path).astype(int)
            # path_seq = (raw_dar[:, 0].astype(int) + raw_dar[:, 1].astype(int) * num_e_path).astype(int)

            if type(paths_list) == np.ndarray:
                # ind = np.array(list(map(lambda x: True if len(np.nonzero(paths_list == x)[0]) > 0 else False, raw_dar[:, 0].astype(int)))).astype(bool)
                # assert(np.sum(ind) == len(ind))
                # # if release_one_interval_biclass set the 1-min assign interval for vehicle for the multiclass modeling
                # path_seq = (np.array(list(map(lambda x: np.nonzero(paths_list == x)[0][0], raw_dar[:, 0].astype(int))))
                #             + (raw_dar[:, 1] / small_assign_freq).astype(int) * num_e_path).astype(int)
                # path_seq = (np.array(list(map(lambda x: find_first(paths_list, x), raw_dar[:, 0].astype(int))))
                #             + (raw_dar[:, 1] / small_assign_freq).astype(int) * num_e_path).astype(int)
                # # if release_one_interval_biclass already set the correct assign interval for vehicle, then this is 15 min intervals
                path_seq = (np.array(list(map(lambda x: find_first(paths_list, x), raw_dar[:, 0].astype(int))))
                            + raw_dar[:, 1].astype(int) * num_e_path).astype(int)
                
            elif type(paths_list) == list:
                # ind = np.array(list(map(lambda x: True if x in paths_list else False, raw_dar[:, 0].astype(int)))).astype(bool)
                # assert(np.sum(ind) == len(ind))
                # # if release_one_interval_biclass set the 1-min assign interval for vehicle for the multiclass modeling
                # path_seq = (np.array(list(map(lambda x: paths_list.index(x), raw_dar[:, 0].astype(int))))
                #             + (raw_dar[:, 1] / small_assign_freq).astype(int) * num_e_path).astype(int)
                # path_seq = (np.array(list(map(lambda x: find_first(paths_list, x), raw_dar[:, 0].astype(int))))
                #             + (raw_dar[:, 1] / small_assign_freq).astype(int) * num_e_path).astype(int)
                # # if release_one_interval_biclass already set the correct assign interval for vehicle, then this is 15 min intervals
                path_seq = (np.array(list(map(lambda x: find_first(paths_list, x), raw_dar[:, 0].astype(int))))
                            + raw_dar[:, 1].astype(int) * num_e_path).astype(int)
            else:
                raise Exception('Wrong data type of paths_list')

            # raw_dar[:, 2]: link no.
            # raw_dar[:, 3]: the count of unit time interval (5s)
            # In Python 3, map() returns an iterable while, in Python 2, it returns a list.
            # link_seq = (np.array(list(map(lambda x: observed_links.index(x), raw_dar[:, 2].astype(int))))
            #             + (raw_dar[:, 3] / ass_freq).astype(int) * num_e_link).astype(int)
            
            if type(observed_links) == np.ndarray:
                # In Python 3, map() returns an iterable while, in Python 2, it returns a list.
                # link_seq = (np.array(list(map(lambda x: np.nonzero(observed_links == x)[0][0], raw_dar[:, 2].astype(int))))
                #             + (raw_dar[:, 3] / ass_freq).astype(int) * num_e_link).astype(int)
                link_seq = (np.array(list(map(lambda x: find_first(observed_links, x), raw_dar[:, 2].astype(int))))
                            + (raw_dar[:, 3] / ass_freq).astype(int) * num_e_link).astype(int)
            elif type(observed_links) == list:
                # link_seq = (np.array(list(map(lambda x: observed_links.index(x), raw_dar[:, 2].astype(int))))
                #             + (raw_dar[:, 3] / ass_freq).astype(int) * num_e_link).astype(int)
                link_seq = (np.array(list(map(lambda x: find_first(observed_links, x), raw_dar[:, 2].astype(int))))
                            + (raw_dar[:, 3] / ass_freq).astype(int) * num_e_link).astype(int)
            else:
                raise Exception('Wrong data type of observed_links')

            # print(path_seq)
            # raw_dar[:, 4]: flow
            p = raw_dar[:, 4] / f[path_seq]

        # print("Creating the coo matrix", time.time())
        mat = coo_matrix((p, (link_seq, path_seq)), 
                        shape=(num_assign_interval * num_e_link, num_assign_interval * num_e_path))
        # pickle.dump((p, link_seq, path_seq), open('test.pickle', 'wb'))
        # print('converting the csr', time.time())
        mat = mat.tocsr()
        # print('finish converting', time.time())
        return mat   

    def get_ltg(self, dta):
        car_ltg_matrix = csr_matrix((self.num_assign_interval * len(self.observed_links), 
                                             self.num_assign_interval * len(self.paths_list)))
     
        truck_ltg_matrix = csr_matrix((self.num_assign_interval * len(self.observed_links), 
                                               self.num_assign_interval * len(self.paths_list)))

        if self.config['use_car_link_tt']:
            car_ltg_matrix = self._compute_link_tt_grad_on_path_flow_car(dta)
            if car_ltg_matrix.max() == 0.:
                print("car_ltg_matrix is empty!")
            
        if self.config['use_truck_link_tt']:
            pass
            # TODO: issue
            # truck_ltg_matrix = self._compute_link_tt_grad_on_path_flow_truck(dta)
            # if truck_ltg_matrix.max() == 0.:
            #     print("truck_ltg_matrix is empty!")

        return car_ltg_matrix, truck_ltg_matrix

    def init_demand_vector(self, num_assign_interval, num_col, scale=1, dist_type='uniform'):
        
        if type(scale) == np.ndarray:
            if scale.ndim == 1:
                assert(len(scale) == num_col)
                scale = scale[np.newaxis, :]
            else:
                assert((scale.shape[0] == 1) & (scale.shape[1] == num_col))
        if dist_type == 'uniform':
            d = np.random.rand(num_assign_interval, num_col) * scale
        elif dist_type == 'normal':
            d = np.random.normal(0, 1, (num_assign_interval, num_col)) * 0.1 * scale + scale

        d = d.flatten(order='C')

        # Kaiming initialization (not working)
        # d = np.random.normal(0, 1, num_assign_interval * num_col) * scale
        # d *= np.sqrt(2 / len(d))
        # d = np.abs(d)

        # Xavier initialization
        # x = torch.Tensor(num_assign_interval * num_col, 1)
        # d = torch.abs(nn.init.xavier_uniform_(x)).squeeze().data.numpy() * scale
        # d = d.astype(float)
        return d

    def init_path_flow(self, car_scale=1, truck_scale=0.1, dist_type='uniform'):
        if car_scale is None or truck_scale is None:
            car_scale, truck_scale = self.estimate_path_flow_scale()

        f_car = self.init_demand_vector(self.num_assign_interval, self.num_path, car_scale, dist_type) 
        f_truck = self.init_demand_vector(self.num_assign_interval, self.num_path, truck_scale, dist_type) 

        f_car = np.maximum(f_car, 1e-6)
        f_truck = np.maximum(f_truck, 1e-6)
        return f_car, f_truck
    
    def estimate_path_flow_scale(self):
        # link_coverage = self.nb.get_link_coverage(self.observed_links, self.nb.folder_path, graph_file_name='Snap_graph', pathtable_file_name='path_table')
        link_coverage = self.get_registered_links_coverage_by_registered_paths(self.nb.folder_path)
        assert(link_coverage.shape[0] == len(self.paths_list))
        num_links = len(self.observed_links)
        # assume only one data point exists and no link aggregation
        one_data_dict = self._get_one_data(0)

        m_car = one_data_dict['car_link_flow']
        m_truck = one_data_dict['truck_link_flow']
        # link_mask = one_data_dict['mask_driving_link'][:num_links]
        
        m_car = m_car.reshape(-1, num_links) 
        m_truck = m_truck.reshape(-1, num_links) 

        # m_car_avg = np.nanmedian(m_car, axis=0, keepdims=True)
        # m_truck_avg = np.nanmedian(m_truck, axis=0, keepdims=True)

        m_car_avg = np.nanmax(m_car, axis=0, keepdims=True)
        m_truck_avg = np.nanmax(m_truck, axis=0, keepdims=True)

        car_scale = link_coverage.astype(float)/link_coverage.sum(axis=0, keepdims=True)*m_car_avg
        truck_scale = link_coverage.astype(float)/link_coverage.sum(axis=0, keepdims=True)*m_truck_avg

        car_scale = np.nanmax(car_scale, axis=1)
        truck_scale = np.nanmax(truck_scale, axis=1)

        car_scale = np.nan_to_num(car_scale)
        truck_scale = np.nan_to_num(truck_scale)

        car_scale = np.maximum(car_scale, 1.2)
        truck_scale = np.maximum(truck_scale, 1.2)

        car_scale = car_scale / self.nb.config.config_dict['DTA']['flow_scalar'] 
        truck_scale = truck_scale / self.nb.config.config_dict['DTA']['flow_scalar'] 

        return car_scale, truck_scale

        # f_car = np.random.rand(self.num_assign_interval, len(self.paths_list)) * car_scale[np.newaxis, :]
        # f_truck = np.random.rand(self.num_assign_interval, len(self.paths_list)) * truck_scale[np.newaxis, :]
        # f_car, f_truck = f_car.flatten(order='C'), f_truck.flatten(order='C')
        # return f_car, f_truck

    def compute_path_flow_grad_and_loss(self, one_data_dict, f_car, f_truck, counter=0):
        # print("Running simulation", time.time())
        dta = self._run_simulation(f_car, f_truck, counter, show_loading=True)

        if self.config['use_car_link_tt'] or self.config['use_truck_link_tt']:
            pass
            # dta.build_link_cost_map(True)
            # # IPMC method for tt
            # dta.get_link_queue_dissipated_time()
            # print("********************** Begin get_ltg **********************")
            # car_ltg_matrix, truck_ltg_matrix = self.get_ltg(dta)
            # print("********************** End get_ltg ************************")

        # print("Getting DAR", time.time())
        print("********************** Begin get_dar **********************")
        # get raw record and process using python
        # (car_dar, truck_dar) = self.get_dar(dta, f_car, f_truck)
        # directly construct sparse matrix
        # (car_dar2, truck_dar2) = self.get_dar2(dta, f_car, f_truck)
        # print(car_dar.toarray() - car_dar2.toarray())
        # save and read
        (car_dar, truck_dar) = self.get_dar3(dta, f_car, f_truck, engine='c')
        print("********************** End get_dar **********************")

        x_e_car, x_e_truck, tt_e_car, tt_e_truck, O_demand_est = None, None, None, None, None

        # print("Evaluating grad", time.time())
        # Count
        car_grad = np.zeros(len(self.observed_links) * self.num_assign_interval)
        truck_grad = np.zeros(len(self.observed_links) * self.num_assign_interval)
        if self.config['use_car_link_flow']:
            # print("car link flow", time.time())
            grad, x_e_car = self._compute_count_loss_grad_on_car_link_flow(dta, one_data_dict)
            car_grad += self.config['link_car_flow_weight'] * grad
        if self.config['use_truck_link_flow']:
            grad, x_e_truck = self._compute_count_loss_grad_on_truck_link_flow(dta, one_data_dict)
            truck_grad += self.config['link_truck_flow_weight'] * grad

        f_car_grad = car_dar.T.dot(car_grad)
        f_truck_grad = truck_dar.T.dot(truck_grad)

        # Travel time
        car_grad = np.zeros(len(self.observed_links) * self.num_assign_interval)
        truck_grad = np.zeros(len(self.observed_links) * self.num_assign_interval)
        if self.config['use_car_link_tt']:
            grad, tt_e_car = self._compute_tt_loss_grad_on_car_link_tt(dta, one_data_dict)
            car_grad += self.config['link_car_tt_weight'] * grad

            ## Original Wei's method
            f_car_grad += car_dar.T.dot(car_grad)

            ## MCDODE's method
            # _tt_loss_grad_on_car_link_flow = self._compute_tt_loss_grad_on_car_link_flow(dta, car_grad)
            # f_car_grad += car_dar.T.dot(_tt_loss_grad_on_car_link_flow)

            ## IPMC method
            # f_car_grad += car_ltg_matrix.T.dot(car_grad)

        if self.config['use_truck_link_tt']:
            grad, tt_e_truck = self._compute_tt_loss_grad_on_truck_link_tt(dta, one_data_dict)
            truck_grad += self.config['link_truck_tt_weight'] * grad

            ## Original Wei's method
            f_truck_grad += truck_dar.T.dot(truck_grad)

            ## MCDODE's method
            # _tt_loss_grad_on_truck_link_flow = self._compute_tt_loss_grad_on_truck_link_flow(dta, truck_grad)
            # f_truck_grad += truck_dar.T.dot(_tt_loss_grad_on_truck_link_flow)

            ## IPMC method
            # f_truck_grad += truck_ltg_matrix.T.dot(truck_grad)

        # Origin vehicle registration data
        if self.config['use_origin_vehicle_registration_data']:
            f_car_grad_add, f_truck_grad_add, O_demand_est = self._compute_grad_on_origin_vehicle_registration_data(one_data_dict, f_car, f_truck)
            f_car_grad += self.config['origin_vehicle_registration_weight'] * f_car_grad_add
            f_truck_grad += self.config['origin_vehicle_registration_weight'] * f_truck_grad_add

        if self.config['use_od_demand_data']:
            f_car_grad_add, f_truck_grad_add, _ = self._compute_grad_on_od_demand_data(one_data_dict, f_car, f_truck, od_demand_factor=self.config['od_demand_factor'])
            f_car_grad += self.config['od_demand_weight'] * f_car_grad_add
            f_truck_grad += self.config['od_demand_weight'] * f_truck_grad_add

        if 'regularization_weight' in self.config:
            f_car_grad += self.config['regularization_weight'] * f_car
            f_truck_grad += self.config['regularization_weight'] * f_truck

        # print("Getting Loss", time.time())
        total_loss, loss_dict = self._get_loss(one_data_dict, dta, f_car, f_truck)
        return f_car_grad, f_truck_grad, total_loss, loss_dict, dta, x_e_car, x_e_truck, tt_e_car, tt_e_truck, O_demand_est

    def _compute_count_loss_grad_on_car_link_flow(self, dta, one_data_dict):
        link_flow_array = one_data_dict['car_link_flow']
        x_e = dta.get_link_car_inflow(np.arange(0, self.num_loading_interval, self.ass_freq), 
                                      np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq).flatten(order='F')
        # print("x_e", x_e, link_flow_array)
        if self.config['car_count_agg']:
            x_e = one_data_dict['car_count_agg_L'].dot(x_e)
        discrepancy = np.nan_to_num(link_flow_array - x_e)
        if 'link_loss_weight_count_car' in one_data_dict:
            discrepancy = discrepancy * one_data_dict['link_loss_weight_count_car']
        grad = - discrepancy
        if self.config['car_count_agg']:
            grad = one_data_dict['car_count_agg_L'].T.dot(grad)
        # print("final link grad", grad)
        # assert(np.all(~np.isnan(grad)))
        return grad, x_e
  
    def _compute_count_loss_grad_on_truck_link_flow(self, dta, one_data_dict):
        link_flow_array = one_data_dict['truck_link_flow']
        x_e = dta.get_link_truck_inflow(np.arange(0, self.num_loading_interval, self.ass_freq), 
                                        np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq).flatten(order='F')
        if self.config['truck_count_agg']:
            x_e = one_data_dict['truck_count_agg_L'].dot(x_e)
        discrepancy = np.nan_to_num(link_flow_array - x_e)
        if 'link_loss_weight_count_truck' in one_data_dict:
            discrepancy = discrepancy * one_data_dict['link_loss_weight_count_truck']
        grad = - discrepancy
        if self.config['truck_count_agg']:
            grad = one_data_dict['truck_count_agg_L'].T.dot(grad)
        # assert(np.all(~np.isnan(grad)))
        return grad, x_e

    def _compute_tt_loss_grad_on_car_link_tt(self, dta, one_data_dict):
        # tt_e = dta.get_car_link_tt_robust(np.arange(0, self.num_loading_interval, self.ass_freq),
        #                                   np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq, self.ass_freq, False).flatten(order='F')
        tt_e = dta.get_car_link_tt(np.arange(0, self.num_loading_interval, self.ass_freq), False).flatten(order='F')
        # tt_free = np.tile(list(map(lambda x: self.nb.get_link(x).get_car_fft(), self.observed_links)), (self.num_assign_interval))
        tt_free = np.tile(dta.get_car_link_fftt(self.observed_links), (self.num_assign_interval))
        tt_e = np.maximum(tt_e, tt_free)
        tt_o = np.maximum(one_data_dict['car_link_tt'], tt_free) 
        # tt_o = one_data_dict['car_link_tt']
        # print('o-----', tt_o)
        # print('e-----', tt_e)

        # don't use inf value
        # ind = (np.isinf(tt_e) + np.isinf(tt_o))
        # tt_e[ind] = 0
        # tt_o[ind] = 0

        discrepancy = np.nan_to_num(tt_o - tt_e)
        grad = - discrepancy
        # print('g-----', grad)
        # if self.config['car_count_agg']:
        #   grad = one_data_dict['car_count_agg_L'].T.dot(grad)
        # print(tt_e, tt_o)
        # print("car_grad", grad)
        # assert(np.all(~np.isnan(grad)))

        # tt_e[ind] = np.inf
        return grad, tt_e

    def _compute_tt_loss_grad_on_truck_link_tt(self, dta, one_data_dict):
        # tt_e = dta.get_truck_link_tt_robust(np.arange(0, self.num_loading_interval, self.ass_freq),
        #                                     np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq, self.ass_freq, False).flatten(order='F')
        tt_e = dta.get_truck_link_tt(np.arange(0, self.num_loading_interval, self.ass_freq), False).flatten(order='F')
        # tt_free = np.tile(list(map(lambda x: self.nb.get_link(x).get_truck_fft(), self.observed_links)), (self.num_assign_interval))
        tt_free = np.tile(dta.get_truck_link_fftt(self.observed_links), (self.num_assign_interval))
        tt_e = np.maximum(tt_e, tt_free)
        tt_o = np.maximum(one_data_dict['truck_link_tt'], tt_free)

        # don't use inf value
        # ind = (np.isinf(tt_e) + np.isinf(tt_o))
        # tt_e[ind] = 0
        # tt_o[ind] = 0

        discrepancy = np.nan_to_num(tt_o - tt_e)
        grad = - discrepancy
        # if self.config['truck_count_agg']:
        #   grad = one_data_dict['truck_count_agg_L'].T.dot(grad)
        # print("truck_grad", grad)
        # assert(np.all(~np.isnan(grad)))

        # tt_e[ind] = np.inf
        return grad, tt_e

    def _compute_tt_loss_grad_on_car_link_flow(self, dta, tt_loss_grad_on_car_link_tt):
        _link_tt_grad_on_link_flow_car = self._compute_link_tt_grad_on_link_flow_car(dta)
        grad = _link_tt_grad_on_link_flow_car.dot(tt_loss_grad_on_car_link_tt)
        grad = grad / (np.linalg.norm(grad) + 1e-7)
        return grad

    def _compute_tt_loss_grad_on_truck_link_flow(self, dta, tt_loss_grad_on_truck_link_tt):
        _link_tt_grad_on_link_flow_truck = self._compute_link_tt_grad_on_link_flow_truck(dta)
        grad = _link_tt_grad_on_link_flow_truck.dot(tt_loss_grad_on_truck_link_tt)
        grad = grad / (np.linalg.norm(grad) + 1e-7)
        return grad

    def _compute_link_tt_grad_on_link_flow_car(self, dta):
        assign_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)
        num_assign_intervals = len(assign_intervals)
        num_links = len(self.observed_links)

        car_link_out_cc = dict()
        for link_ID in self.observed_links:
            car_link_out_cc[link_ID] = dta.get_car_link_out_cc(link_ID)

        # tt_e = dta.get_car_link_tt(np.arange(assign_intervals[-1] + self.ass_freq))
        # assert(tt_e.shape[0] == num_links and tt_e.shape[1] == assign_intervals[-1] + self.ass_freq)
        # # average link travel time
        # tt_e = np.stack(list(map(lambda i : np.mean(tt_e[:, i : i+self.ass_freq], axis=1), assign_intervals)), axis=1)
        # assert(tt_e.shape[0] == num_links and tt_e.shape[1] == num_assign_intervals)

        # tt_e = dta.get_car_link_tt(np.arange(0, self.num_loading_interval))

        # tt_e = dta.get_car_link_tt(assign_intervals, False)
        tt_e = dta.get_car_link_tt_robust(assign_intervals, assign_intervals + self.ass_freq, self.ass_freq, False)

        # tt_free = np.array(list(map(lambda x: self.nb.get_link_driving(x).get_car_fft(), self.observed_links)))
        tt_free = dta.get_car_link_fftt(self.observed_links)
        mask = tt_e > tt_free[:, np.newaxis]

        # no congestions
        if np.sum(mask) == 0:
            return csr_matrix((num_assign_intervals * num_links, 
                               num_assign_intervals * num_links))
            # return eye(num_assign_intervals * num_links)

        # cc = np.zeros((num_links, num_assign_intervals + 1), dtype=float)
        # for j, link_ID in enumerate(self.observed_links):
        #     cc[j, :] = np.array(list(map(lambda timestamp : self._get_flow_from_cc(timestamp, car_link_out_cc[link_ID]), 
        #                                  np.concatenate((assign_intervals, np.array([assign_intervals[-1] + self.ass_freq]))))))

        # outflow_rate = np.diff(cc, axis=1) / self.ass_freq / self.nb.config.config_dict['DTA']['unit_time']

        cc = np.zeros((num_links, assign_intervals[-1] + self.ass_freq + 1), dtype=float)
        for j, link_ID in enumerate(self.observed_links):
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
            assert(mask.shape[1] == outflow_avg_rate.shape[1])
            outflow_avg_rate *= mask
        assert(outflow_avg_rate.shape[0] == num_links and outflow_avg_rate.shape[1] == num_assign_intervals)

        val = list()
        row = list()
        col = list()

        for i, assign_interval in enumerate(assign_intervals):
            for j, link_ID in enumerate(self.observed_links):
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
        num_links = len(self.observed_links)

        truck_link_out_cc = dict()
        for link_ID in self.observed_links:
            truck_link_out_cc[link_ID] = dta.get_truck_link_out_cc(link_ID)

        # tt_e = dta.get_truck_link_tt(np.arange(assign_intervals[-1] + self.ass_freq))
        # assert(tt_e.shape[0] == num_links and tt_e.shape[1] == assign_intervals[-1] + self.ass_freq)
        # # average link travel time
        # tt_e = np.stack(list(map(lambda i : np.mean(tt_e[:, i : i+self.ass_freq], axis=1), assign_intervals)), axis=1)
        # assert(tt_e.shape[0] == num_links and tt_e.shape[1] == num_assign_intervals)

        # tt_e = dta.get_truck_link_tt(assign_intervals, False)
        tt_e = dta.get_truck_link_tt_robust(assign_intervals, assign_intervals + self.ass_freq, self.ass_freq, False)

        # tt_free = np.array(list(map(lambda x: self.nb.get_link_driving(x).get_truck_fft(), self.observed_links)))
        tt_free = dta.get_truck_link_fftt(self.observed_links)
        mask = tt_e > tt_free[:, np.newaxis]

        # no congestions
        if np.sum(mask) == 0:
            return csr_matrix((num_assign_intervals * num_links, 
                               num_assign_intervals * num_links))
            # return eye(num_assign_intervals * num_links)

        # cc = np.zeros((num_links, num_assign_intervals + 1), dtype=float)
        # for j, link_ID in enumerate(self.observed_links):
        #     cc[j, :] = np.array(list(map(lambda timestamp : self._get_flow_from_cc(timestamp, truck_link_out_cc[link_ID]), 
        #                                  np.concatenate((assign_intervals, np.array([assign_intervals[-1] + self.ass_freq]))))))

        # outflow_rate = np.diff(cc, axis=1) / self.ass_freq / self.nb.config.config_dict['DTA']['unit_time']

        cc = np.zeros((num_links, assign_intervals[-1] + self.ass_freq + 1), dtype=float)
        for j, link_ID in enumerate(self.observed_links):
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
            assert(mask.shape[1] == outflow_avg_rate.shape[1])
            outflow_avg_rate *= mask
        assert(outflow_avg_rate.shape[0] == num_links and outflow_avg_rate.shape[1] == num_assign_intervals)

        val = list()
        row = list()
        col = list()

        for i, assign_interval in enumerate(assign_intervals):
            for j, link_ID in enumerate(self.observed_links):
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

    def _get_flow_from_cc(self, timestamp, cc):
        # precision issue, consistent with C++
        cc = np.around(cc, decimals=4)
        if any(timestamp >= cc[:, 0]):
            ind = np.nonzero(timestamp >= cc[:, 0])[0][-1]
        else:
            ind = 0
        return cc[ind, 1]

    def _compute_link_tt_grad_on_path_flow_car(self, dta):
        # dta.build_link_cost_map() is invoked already
        assign_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)
        num_assign_intervals = len(assign_intervals)

        release_freq = 60
        # this is in terms of 5-s intervals
        release_intervals = np.arange(0, self.num_loading_interval, release_freq // self.nb.config.config_dict['DTA']['unit_time'])

        # # memory issue
        # # raw_ltg = dta.get_car_ltg_matrix(release_intervals, self.num_loading_interval)
        # raw_ltg = dta.get_car_ltg_matrix(assign_intervals, self.num_loading_interval)

        # ltg = self._massage_raw_ltg(raw_ltg, self.ass_freq, num_assign_intervals, self.paths_list, self.observed_links, self.num_procs)

        ltg = dta.get_complete_car_ltg_matrix(assign_intervals, self.num_loading_interval, num_assign_intervals)
        return ltg

    def _compute_link_tt_grad_on_path_flow_truck(self, dta):
        # dta.build_link_cost_map() is invoked already
        assign_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)
        num_assign_intervals = len(assign_intervals)

        release_freq = 60
        # this is in terms of 5-s intervals
        release_intervals = np.arange(0, self.num_loading_interval, release_freq // self.nb.config.config_dict['DTA']['unit_time'])

        # # memory issue
        # # raw_ltg = dta.get_truck_ltg_matrix(release_intervals, self.num_loading_interval)
        # raw_ltg = dta.get_truck_ltg_matrix(assign_intervals, self.num_loading_interval)

        # ltg = self._massage_raw_ltg(raw_ltg, self.ass_freq, num_assign_intervals, self.paths_list, self.observed_links, self.num_procs)

        ltg = dta.get_complete_truck_ltg_matrix(assign_intervals, self.num_loading_interval, num_assign_intervals)
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
        assert(set(raw_ltg[:, 0]).issubset(set(paths_list)))

        if num_procs > 1:
            raw_ltg = pd.DataFrame(data=raw_ltg)

            from pandarallel import pandarallel
            pandarallel.initialize(progress_bar=True, nb_workers=num_procs)

            # ind = raw_ltg.loc[:, 0].parallel_apply(lambda x: True if x in set(paths_list) else False)
            # assert(np.sum(ind) == len(ind))
            
            if type(paths_list) == list:
                paths_list = np.array(paths_list)
            elif type(paths_list) == np.ndarray:
                pass
            else:
                raise Exception('Wrong data type of paths_list')

            path_seq = (raw_ltg.loc[:, 0].astype(int).parallel_apply(lambda x: np.nonzero(paths_list == x)[0][0]) 
                        + (raw_ltg.loc[:, 1] / ass_freq).astype(int) * num_e_path).astype(int)
            # path_seq = (raw_ltg.loc[:, 0].astype(int).parallel_apply(lambda x: find_first(paths_list, x)) 
            #             + (raw_ltg.loc[:, 1] / ass_freq).astype(int) * num_e_path).astype(int)
            
            if type(observed_links) == list:
                observed_links = np.array(observed_links)
            elif type(observed_links) == np.ndarray:
                pass
            else:
                raise Exception('Wrong data type of observed_links')

            link_seq = (raw_ltg.loc[:, 2].astype(int).parallel_apply(lambda x: np.nonzero(observed_links == x)[0][0])
                        + (raw_ltg.loc[:, 3] / ass_freq).astype(int) * num_e_link).astype(int)
            # link_seq = (raw_ltg.loc[:, 2].astype(int).parallel_apply(lambda x: find_first(observed_links, x))
            #             + (raw_ltg.loc[:, 3] / ass_freq).astype(int) * num_e_link).astype(int)

            p = raw_ltg.loc[:, 4] / (ass_freq * small_assign_freq)

        else:

            # raw_ltg[:, 0]: path no.
            # raw_ltg[:, 1]: the count of 1 min interval in terms of 5s intervals
                    
            if type(paths_list) == np.ndarray:
                # ind = np.array(list(map(lambda x: True if len(np.nonzero(paths_list == x)[0]) > 0 else False, raw_ltg[:, 0].astype(int)))).astype(bool)
                # assert(np.sum(ind) == len(ind))
                # path_seq = (np.array(list(map(lambda x: np.nonzero(paths_list == x)[0][0], raw_ltg[:, 0].astype(int))))
                #             + (raw_ltg[:, 1] / ass_freq).astype(int) * num_e_path).astype(int)
                path_seq = (np.array(list(map(lambda x: find_first(paths_list, x), raw_ltg[:, 0].astype(int))))
                            + (raw_ltg[:, 1] / ass_freq).astype(int) * num_e_path).astype(int)
            elif type(paths_list) == list:
                # ind = np.array(list(map(lambda x: True if x in paths_list else False, raw_ltg[:, 0].astype(int)))).astype(bool)
                # assert(np.sum(ind) == len(ind))
                # path_seq = (np.array(list(map(lambda x: paths_list.index(x), raw_ltg[:, 0].astype(int))))
                #             + (raw_ltg[:, 1] / ass_freq).astype(int) * num_e_path).astype(int)
                path_seq = (np.array(list(map(lambda x: find_first(paths_list, x), raw_ltg[:, 0].astype(int))))
                            + (raw_ltg[:, 1] / ass_freq).astype(int) * num_e_path).astype(int)
            else:
                raise Exception('Wrong data type of paths_list')

            # raw_ltg[:, 2]: link no.
            # raw_ltg[:, 3]: the count of unit time interval (5s)
            if type(observed_links) == np.ndarray:
                # In Python 3, map() returns an iterable while, in Python 2, it returns a list.
                # link_seq = (np.array(list(map(lambda x: np.nonzero(observed_links == x)[0][0], raw_ltg[:, 2].astype(int))))
                #             + (raw_ltg[:, 3] / ass_freq).astype(int) * num_e_link).astype(int)
                link_seq = (np.array(list(map(lambda x: find_first(observed_links, x), raw_ltg[:, 2].astype(int))))
                            + (raw_ltg[:, 3] / ass_freq).astype(int) * num_e_link).astype(int)
            elif type(observed_links) == list:
                # link_seq = (np.array(list(map(lambda x: observed_links.index(x), raw_ltg[:, 2].astype(int))))
                #             + (raw_ltg[:, 3] / ass_freq).astype(int) * num_e_link).astype(int)
                link_seq = (np.array(list(map(lambda x: find_first(observed_links, x), raw_ltg[:, 2].astype(int))))
                            + (raw_ltg[:, 3] / ass_freq).astype(int) * num_e_link).astype(int)
            else:
                raise Exception('Wrong data type of observed_links')
                        
            # print(path_seq)
            # raw_ltg[:, 4]: gradient, to be averaged for each large assign interval 
            p = raw_ltg[:, 4] / (ass_freq * small_assign_freq)
        
        # print("Creating the coo matrix", time.time()), coo_matrix permits duplicate entries
        mat = coo_matrix((p, (link_seq, path_seq)), shape=(num_assign_interval * num_e_link, num_assign_interval * num_e_path))
        # pickle.dump((p, link_seq, path_seq), open('test.pickle', 'wb'))
        # print('converting the csr', time.time())
        
        # sum duplicate entries in coo_matrix
        mat = mat.tocsr()
        # print('finish converting', time.time())
        return mat

    def _compute_grad_on_origin_vehicle_registration_data(self, one_data_dict, f_car, f_truck):
        # reshape car_flow and truck_flow into ndarrays with dimensions of intervals x number of total paths
        # loss in terms of total counts of cars and trucks for some origin nodes
        f_car = torch.from_numpy(f_car)
        f_truck = torch.from_numpy(f_truck)
        f_car.requires_grad = True
        f_truck.requires_grad = True

        f_car_reshaped = f_car.reshape(self.num_assign_interval, -1)
        f_truck_reshaped = f_truck.reshape(self.num_assign_interval, -1)

        O_demand_est = dict()
        for i, path_ID in enumerate(self.nb.path_table.ID2path.keys()):
            path = self.nb.path_table.ID2path[path_ID]
            O_node = path.origin_node
            O = self.nb.od.O_dict.inv[O_node]
            if O not in O_demand_est:
                O_demand_est[O] = [torch.Tensor([0.]), torch.Tensor([0.])]
            O_demand_est[O][0] = O_demand_est[O][0] + f_car_reshaped[:, i].sum()
            O_demand_est[O][1] = O_demand_est[O][1] + f_truck_reshaped[:, i].sum()

        # pandas DataFrame
        def process_one_row(row):

            # loss = torch.Tensor([0.])
            # if not np.isnan(row['car']):
            #     loss += (sum(O_demand_est[Origin_ID][0] for Origin_ID in row['origin_ID']) - row['car'])**2
            # if not np.isnan(row['truck']):
            #     loss += (sum(O_demand_est[Origin_ID][1] for Origin_ID in row['origin_ID']) - row['truck'])**2
            # if (np.isnan(row['car'])) and (np.isnan(row['truck'])) and (not np.isnan(row['total'])):
            #     loss += (sum(O_demand_est[Origin_ID][0] + O_demand_est[Origin_ID][1] for Origin_ID in row['origin_ID']) - row['total'])**2

            loss = self.config['origin_registration_data_car_weight'] * (sum(O_demand_est[Origin_ID][0] for Origin_ID in row['origin_ID']) - row['car'])**2 + \
                   self.config['origin_registration_data_truck_weight'] * (sum(O_demand_est[Origin_ID][1] for Origin_ID in row['origin_ID']) - row['truck'])**2

            loss.backward()
                
        one_data_dict['origin_vehicle_registration_data'].apply(lambda row: process_one_row(row), axis=1)

        O_demand_est = {O: (O_demand_est[O][0].item(), O_demand_est[O][1].item()) for O in O_demand_est}
        return f_car.grad.data.cpu().numpy(), f_truck.grad.data.cpu().numpy(), O_demand_est

    def _compute_grad_on_origin_vehicle_registration_data2(self, one_data_dict, f_car, f_truck):
        # problematic!! in backward()
        # reshape car_flow and truck_flow into ndarrays with dimensions of intervals x number of total paths
        # loss in terms of total counts of cars and trucks for some origin nodes
        f_car = torch.from_numpy(f_car)
        f_truck = torch.from_numpy(f_truck)
        f_car.requires_grad = True
        f_truck.requires_grad = True

        f_car_reshaped = torch.transpose(f_car.reshape(self.num_assign_interval, -1), 0, 1)
        f_truck_reshaped = torch.transpose(f_truck.reshape(self.num_assign_interval, -1), 0, 1)
        f_car_reshaped = f_car_reshaped.sum(dim=1, keepdim=True)
        f_truck_reshaped = f_truck_reshaped.sum(dim=1, keepdim=True)
        f = torch.concat((f_car_reshaped, f_truck_reshaped), dim=1)

        O_ID = []
        for i, path_ID in enumerate(self.nb.path_table.ID2path.keys()):
            path = self.nb.path_table.ID2path[path_ID]
            O_node = path.origin_node
            O_ID.append(self.nb.od.O_dict.inv[O_node])
        O_ID = torch.LongTensor(O_ID)

        O_f, O_ID = groupby_sum(f, O_ID)
        # O_f_car, _ = groupby_sum(f_car_reshaped, O_ID)
        # O_f_truck, O_ID = groupby_sum(f_truck_reshaped, O_ID)

        O_demand_est = dict()
        for i in range(len(O_ID)):
            O_demand_est[O_ID[i].item()] = [O_f[i, 0], O_f[i, 1]]
            # O_demand_est[O_ID[i].item()] = [O_f_car[i, 0], O_f_truck[i, 0]]

        # pandas DataFrame
        def process_one_row(row):

            # loss = torch.Tensor([0.])
            # if not np.isnan(row['car']):
            #     loss += (sum(O_demand_est[Origin_ID][0] for Origin_ID in row['origin_ID']) - row['car'])**2
            # if not np.isnan(row['truck']):
            #     loss += (sum(O_demand_est[Origin_ID][1] for Origin_ID in row['origin_ID']) - row['truck'])**2
            # if (np.isnan(row['car'])) and (np.isnan(row['truck'])) and (not np.isnan(row['total'])):
            #     loss += (sum(O_demand_est[Origin_ID][0] + O_demand_est[Origin_ID][1] for Origin_ID in row['origin_ID']) - row['total'])**2

            loss = self.config['origin_registration_data_car_weight'] * (sum(O_demand_est[Origin_ID][0] for Origin_ID in row['origin_ID']) - row['car'])**2 + \
                   self.config['origin_registration_data_truck_weight'] * (sum(O_demand_est[Origin_ID][1] for Origin_ID in row['origin_ID']) - row['truck'])**2

            loss.backward()
                
        one_data_dict['origin_vehicle_registration_data'].apply(lambda row: process_one_row(row), axis=1)

        O_demand_est = {O: (O_demand_est[O][0].item(), O_demand_est[O][1].item()) for O in O_demand_est}
        return f_car.grad.data.cpu().numpy(), f_truck.grad.data.cpu().numpy(), O_demand_est

    def _compute_grad_on_od_demand_data(self, one_data_dict, f_car, f_truck, od_demand_factor=1.0):
        # reshape car_flow and truck_flow into ndarrays with dimensions of intervals x number of total paths
        f_car = torch.from_numpy(f_car).float()
        f_truck = torch.from_numpy(f_truck).float()
        f_car.requires_grad = True
        f_truck.requires_grad = True

        f_car_reshaped = f_car.reshape(self.num_assign_interval, -1)
        f_truck_reshaped = f_truck.reshape(self.num_assign_interval, -1)

        OD_demand_est = dict()
        for i, path_ID in enumerate(self.nb.path_table.ID2path.keys()):
            path = self.nb.path_table.ID2path[path_ID]
            O_node = path.origin_node
            D_node = path.destination_node
            O = self.nb.od.O_dict.inv[O_node]
            D = self.nb.od.D_dict.inv[D_node]
            if (O, D) not in OD_demand_est:
                OD_demand_est[(O, D)] = [torch.zeros(self.num_assign_interval), torch.zeros(self.num_assign_interval)]
            OD_demand_est[(O, D)][0] = OD_demand_est[(O, D)][0] + f_car_reshaped[:, i]
            OD_demand_est[(O, D)][1] = OD_demand_est[(O, D)][1] + f_truck_reshaped[:, i]

                # pandas DataFrame
        def process_one_row(row):
            car = torch.from_numpy(row['car']).float()
            truck = torch.from_numpy(row['truck']).float()
            car.requires_grad = False
            truck.requires_grad = False
            # loss = self.config['od_demand_data_car_weight'] * sum((OD_demand_est[(row['origin_ID'], row['destination_ID'])][0] - od_demand_factor * car)**2) + \
            #        self.config['od_demand_data_truck_weight'] * sum((OD_demand_est[(row['origin_ID'], row['destination_ID'])][1] - od_demand_factor * truck)**2) 

            loss = sum((OD_demand_est[(row['origin_ID'], row['destination_ID'])][0] + OD_demand_est[(row['origin_ID'], row['destination_ID'])][1] 
                        - od_demand_factor * car - od_demand_factor * truck)**2)
            loss.backward()

        one_data_dict['od_demand_data'].apply(lambda row: process_one_row(row), axis=1)
        return f_car.grad.data.cpu().numpy(), f_truck.grad.data.cpu().numpy(), OD_demand_est

    def _get_one_data(self, j):
        assert (self.num_data > j)
        one_data_dict = dict()
        if self.config['use_car_link_flow'] or self.config['compute_car_link_flow_loss']:
            one_data_dict['car_link_flow'] = self.data_dict['car_link_flow'][j]
        if self.config['use_truck_link_flow'] or self.config['compute_truck_link_flow_loss']:
            one_data_dict['truck_link_flow'] = self.data_dict['truck_link_flow'][j]
        if self.config['use_car_link_tt'] or self.config['compute_car_link_tt_loss']:
            one_data_dict['car_link_tt'] = self.data_dict['car_link_tt'][j]
        if self.config['use_truck_link_tt'] or self.config['compute_truck_link_tt_loss']:
            one_data_dict['truck_link_tt'] = self.data_dict['truck_link_tt'][j]
        if self.config['car_count_agg']:
            one_data_dict['car_count_agg_L'] = self.car_count_agg_L_list[j]
        if self.config['truck_count_agg']:
            one_data_dict['truck_count_agg_L'] = self.truck_count_agg_L_list[j]
        if self.config['use_origin_vehicle_registration_data'] or self.config['compute_origin_vehicle_registration_loss']:
            # pandas DataFrame
            one_data_dict['origin_vehicle_registration_data'] = self.data_dict['origin_vehicle_registration_data'][j]
        if self.config['use_od_demand_data'] or self.config['compute_od_demand_loss']:
            # pandas DataFrame
            one_data_dict['od_demand_data'] = self.data_dict['od_demand_data'][j]

        if 'mask_driving_link' in self.data_dict:
            one_data_dict['mask_driving_link'] = self.data_dict['mask_driving_link']
        else:
            one_data_dict['mask_driving_link'] = np.ones(len(self.observed_links) * self.num_assign_interval, dtype=bool)

        if 'link_loss_weight_count_car' in self.data_dict and self.config['compute_car_link_flow_loss']:
            one_data_dict['link_loss_weight_count_car'] = self.data_dict['link_loss_weight_count_car']
        if 'link_loss_weight_count_truck' in self.data_dict and self.config['compute_truck_link_flow_loss']:
            one_data_dict['link_loss_weight_count_truck'] = self.data_dict['link_loss_weight_count_truck']
        return one_data_dict

    def aggregate_f_to_O_demand(self, f_car, f_truck):
        # reshape car_flow and truck_flow into ndarrays with dimensions of intervals x number of total paths
        f_car = f_car.reshape(self.num_assign_interval, -1)
        f_truck = f_truck.reshape(self.num_assign_interval, -1)

        O_demand_est = dict()
        for i, path_ID in enumerate(self.nb.path_table.ID2path.keys()):
            path = self.nb.path_table.ID2path[path_ID]
            O_node = path.origin_node
            O = self.nb.od.O_dict.inv[O_node]
            if O not in O_demand_est:
                O_demand_est[O] = [0, 0]
            O_demand_est[O][0] = O_demand_est[O][0] + f_car[:, i].sum()
            O_demand_est[O][1] = O_demand_est[O][1] + f_truck[:, i].sum()
        f_car = f_car.flatten()
        f_truck = f_truck.flatten()
        return O_demand_est
    
    def aggregate_f_to_OD_demand(self, f_car, f_truck):
        # reshape car_flow and truck_flow into ndarrays with dimensions of intervals x number of total paths
        f_car = f_car.reshape(self.num_assign_interval, -1)
        f_truck = f_truck.reshape(self.num_assign_interval, -1)

        OD_demand_est = dict()
        for i, path_ID in enumerate(self.nb.path_table.ID2path.keys()):
            path = self.nb.path_table.ID2path[path_ID]
            O_node = path.origin_node
            D_node = path.destination_node
            O = self.nb.od.O_dict.inv[O_node]
            D = self.nb.od.D_dict.inv[D_node]
            if (O, D) not in OD_demand_est:
                OD_demand_est[(O,D)] = [np.zeros(self.num_assign_interval), np.zeros(self.num_assign_interval)]
            OD_demand_est[(O, D)][0] = OD_demand_est[(O, D)][0] + f_car[:, i]
            OD_demand_est[(O, D)][1] = OD_demand_est[(O, D)][1] + f_truck[:, i]
        f_car = f_car.flatten()
        f_truck = f_truck.flatten()
        return OD_demand_est

    def _get_loss(self, one_data_dict, dta, f_car, f_truck):
        loss_dict = dict()
        assign_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)

        if self.config['use_car_link_flow'] or self.config['compute_car_link_flow_loss']:
            x_e = dta.get_link_car_inflow(assign_intervals, assign_intervals + self.ass_freq).flatten(order='F')
            if self.config['car_count_agg']:
                x_e = one_data_dict['car_count_agg_L'].dot(x_e)
            # loss = self.config['link_car_flow_weight'] * np.linalg.norm(np.nan_to_num(x_e[one_data_dict['mask_driving_link']] - one_data_dict['car_link_flow'][one_data_dict['mask_driving_link']]))
            loss = np.linalg.norm(np.nan_to_num(x_e[one_data_dict['mask_driving_link']] - one_data_dict['car_link_flow'][one_data_dict['mask_driving_link']]))
            loss_dict['car_count_loss'] = loss

        if self.config['use_truck_link_flow'] or self.config['compute_truck_link_flow_loss']:
            x_e = dta.get_link_truck_inflow(assign_intervals, assign_intervals + self.ass_freq).flatten(order='F')
            if self.config['truck_count_agg']:
                x_e = one_data_dict['truck_count_agg_L'].dot(x_e)
            # loss = self.config['link_truck_flow_weight'] * np.linalg.norm(np.nan_to_num(x_e[one_data_dict['mask_driving_link']] - one_data_dict['truck_link_flow'][one_data_dict['mask_driving_link']]))
            loss = np.linalg.norm(np.nan_to_num(x_e[one_data_dict['mask_driving_link']] - one_data_dict['truck_link_flow'][one_data_dict['mask_driving_link']]))
            loss_dict['truck_count_loss'] = loss

        if self.config['use_car_link_tt'] or self.config['compute_car_link_tt_loss']:
            x_tt_e = dta.get_car_link_tt(assign_intervals, False).flatten(order='F')
            # x_tt_e = dta.get_car_link_tt_robust(assign_intervals, assign_intervals + self.ass_freq, self.ass_freq, False).flatten(order='F')
            # loss = self.config['link_car_tt_weight'] * np.linalg.norm(np.nan_to_num(x_tt_e[one_data_dict['mask_driving_link']] - one_data_dict['car_link_tt'][one_data_dict['mask_driving_link']]))
            loss = np.linalg.norm(np.nan_to_num(x_tt_e[one_data_dict['mask_driving_link']] - one_data_dict['car_link_tt'][one_data_dict['mask_driving_link']]))
            loss_dict['car_tt_loss'] = loss

        if self.config['use_truck_link_tt'] or self.config['compute_truck_link_tt_loss']:
            x_tt_e = dta.get_truck_link_tt(assign_intervals, False).flatten(order='F')
            # x_tt_e = dta.get_truck_link_tt_robust(assign_intervals, assign_intervals + self.ass_freq, self.ass_freq, False).flatten(order='F')
            # loss = self.config['link_truck_tt_weight'] * np.linalg.norm(np.nan_to_num(x_tt_e[one_data_dict['mask_driving_link']] - one_data_dict['truck_link_tt'][one_data_dict['mask_driving_link']]))
            loss = np.linalg.norm(np.nan_to_num(x_tt_e[one_data_dict['mask_driving_link']] - one_data_dict['truck_link_tt'][one_data_dict['mask_driving_link']]))
            loss_dict['truck_tt_loss'] = loss

        if self.config['use_origin_vehicle_registration_data'] or self.config['compute_origin_vehicle_registration_loss']:
            O_demand_est = self.aggregate_f_to_O_demand(f_car, f_truck)
            def process_one_row(row):
                return (sum(O_demand_est[Origin_ID][0] for Origin_ID in row['origin_ID']) - row['car'])**2 + \
                       (sum(O_demand_est[Origin_ID][1] for Origin_ID in row['origin_ID']) - row['truck'])**2
            loss = np.sqrt(np.nansum(one_data_dict['origin_vehicle_registration_data'].apply(lambda row: process_one_row(row), axis=1)))
            # loss_dict['origin_vehicle_registration_loss'] = self.config['origin_vehicle_registration_weight'] * loss
            loss_dict['origin_vehicle_registration_loss'] = loss

        if self.config['use_od_demand_data'] or self.config['compute_od_demand_loss']:
            OD_demand_est = self.aggregate_f_to_OD_demand(f_car, f_truck)
            def process_one_row(row):
                # return np.linalg.norm(OD_demand_est[(row['origin_ID'], row['destination_ID'])][0] - row['car'])**2 + \
                #        np.linalg.norm(OD_demand_est[(row['origin_ID'], row['destination_ID'])][1] - row['truck'])**2
                return np.linalg.norm(OD_demand_est[(row['origin_ID'], row['destination_ID'])][0] + OD_demand_est[(row['origin_ID'], row['destination_ID'])][1]
                                      - self.config['od_demand_factor'] * row['car'] - self.config['od_demand_factor'] * row['truck'])**2 
            loss = np.sqrt(np.nansum(one_data_dict['od_demand_data'].apply(lambda row: process_one_row(row), axis=1)))
            loss_dict['od_demand_loss'] = loss

        total_loss = 0.0
        for loss_type, loss_value in loss_dict.items():
            total_loss += loss_value
        return total_loss, loss_dict

    def estimate_path_flow_pytorch(self, car_step_size=0.1, truck_step_size=0.1, 
                                    link_car_flow_weight=1, link_truck_flow_weight=1, 
                                    link_car_tt_weight=1, link_truck_tt_weight=1, origin_vehicle_registration_weight=1, od_demand_weight=1,
                                    max_epoch=10, algo='NAdam',
                                    l2_coeff=1e-6,
                                    car_init_scale=10,
                                    truck_init_scale=1, 
                                    store_folder=None, 
                                    use_file_as_init=None,
                                    starting_epoch=0,
                                    column_generation=False):

        if np.isscalar(column_generation):
            column_generation = np.ones(max_epoch, dtype=int) * column_generation
        assert(len(column_generation) == max_epoch)
    
        if np.isscalar(link_car_flow_weight):
            link_car_flow_weight = np.ones(max_epoch, dtype=bool) * link_car_flow_weight
        assert(len(link_car_flow_weight) == max_epoch)

        if np.isscalar(link_truck_flow_weight):
            link_truck_flow_weight = np.ones(max_epoch, dtype=bool) * link_truck_flow_weight
        assert(len(link_truck_flow_weight) == max_epoch)

        if np.isscalar(link_car_tt_weight):
            link_car_tt_weight = np.ones(max_epoch, dtype=bool) * link_car_tt_weight
        assert(len(link_car_tt_weight) == max_epoch)

        if np.isscalar(link_truck_tt_weight):
            link_truck_tt_weight = np.ones(max_epoch, dtype=bool) * link_truck_tt_weight
        assert(len(link_truck_tt_weight) == max_epoch)

        if np.isscalar(origin_vehicle_registration_weight):
            origin_vehicle_registration_weight = np.ones(max_epoch, dtype=bool) * origin_vehicle_registration_weight
        assert(len(origin_vehicle_registration_weight) == max_epoch)

        if np.isscalar(od_demand_weight):
            od_demand_weight = np.ones(max_epoch, dtype=bool) * od_demand_weight
        assert(len(od_demand_weight) == max_epoch)

        loss_list = list()
        best_epoch = starting_epoch
        best_log_f_car, best_log_f_truck = 0, 0
        best_f_car, best_f_truck = 0, 0
        best_x_e_car, best_x_e_truck, best_tt_e_car, best_tt_e_truck, best_O_demand = 0, 0, 0, 0, 0
        # here the basic variables to be estimated are path flows, not OD demand, so no route choice model, unlike in sDODE.py
        if use_file_as_init is not None:
            if use_file_as_init.split(os.sep)[-1] == 'init_path_flow.pkl':
                pass
            else:
                # most recent
                _, _, loss_list, best_epoch, _, _, _, _, \
                    _, _, _, _, _ = pickle.load(open(use_file_as_init, 'rb'))
                # best 
                use_file_as_init = os.path.join(store_folder, '{}_iteration.pickle'.format(best_epoch))
                _, _, _, _, best_log_f_car, best_log_f_truck, best_f_car, best_f_truck, \
                    best_x_e_car, best_x_e_truck, best_tt_e_car, best_tt_e_truck, best_O_demand = pickle.load(open(use_file_as_init, 'rb'))

        assert((len(loss_list) >= best_epoch) and (starting_epoch >= best_epoch))

        pathflow_solver = torch_pathflow_solver(self.num_assign_interval, self.num_path,
                                                car_init_scale, truck_init_scale, use_file_as_init=use_file_as_init)

        # f, tensor
        f_car, f_truck = pathflow_solver.generate_pathflow_tensor()

        pathflow_solver.set_params_with_lr(car_step_size, truck_step_size)
        pathflow_solver.set_optimizer(algo)
        
        for i in range(max_epoch):
      
            seq = np.random.permutation(self.num_data)
            loss = float(0)
            # print("Start iteration", time.time())
            loss_dict = {'car_count_loss': 0.0, 'truck_count_loss': 0.0, 'car_tt_loss': 0.0, 'truck_tt_loss': 0.0, 
                         'origin_vehicle_registration_loss': 0.0, 'od_demand_loss': 0.0}

            self.config['link_car_flow_weight'] = link_car_flow_weight[i] * (self.config['use_car_link_flow'] or self.config['compute_car_link_flow_loss'])
            self.config['link_truck_flow_weight'] = link_truck_flow_weight[i] * (self.config['use_truck_link_flow'] or self.config['compute_truck_link_flow_loss'])
            
            self.config['link_car_tt_weight'] = link_car_tt_weight[i] * (self.config['use_car_link_tt'] or self.config['compute_car_link_tt_loss'])
            self.config['link_truck_tt_weight'] = link_truck_tt_weight[i] * (self.config['use_truck_link_tt'] or self.config['compute_truck_link_tt_loss'])

            self.config['origin_vehicle_registration_weight'] = origin_vehicle_registration_weight[i] * (self.config['use_origin_vehicle_registration_data'] or self.config['compute_origin_vehicle_registration_loss'])
            self.config['od_demand_weight'] = od_demand_weight[i] * (self.config['use_od_demand_data'] or self.config['compute_od_demand_loss'])
            for j in seq:
                one_data_dict = self._get_one_data(j)

                # f tensor -> f numpy
                f_car_numpy, f_truck_numpy = pathflow_solver.generate_pathflow_numpy(f_car, f_truck)

                f_car_grad, f_truck_grad, tmp_loss, tmp_loss_dict, dta, x_e_car, x_e_truck, tt_e_car, tt_e_truck, O_demand = self.compute_path_flow_grad_and_loss(one_data_dict, f_car_numpy, f_truck_numpy)
                # print("gradient", car_grad, truck_grad)

                if column_generation[i] == 0:
                    dta = 0

                pathflow_solver.optimizer.zero_grad()
                
                # grad of loss wrt log_f = grad of f wrt log_f (pytorch autograd) * grad of loss wrt f (manually)
                pathflow_solver.compute_gradient(f_car, f_truck,
                                                 f_car_grad=f_car_grad, f_truck_grad=f_truck_grad, l2_coeff=l2_coeff)

                # update log_f
                pathflow_solver.optimizer.step()

                # release memory
                f_car_grad, f_truck_grad = 0, 0
                pathflow_solver.optimizer.zero_grad()

                # f tensor
                f_car, f_truck = pathflow_solver.generate_pathflow_tensor()

                if column_generation[i]:
                    print("***************** generate new paths *****************")
                    if not self.config['use_car_link_tt'] and not self.config['use_truck_link_tt']:
                        # if any of these is true, dta.build_link_cost_map(True) is already invoked in compute_path_flow_grad_and_loss()
                        dta.build_link_cost_map(False)
                    
                    self.update_path_table(dta)
                    dta = 0

                    # update log_f tensor
                    pathflow_solver.add_pathflow(
                        len(self.nb.path_table.ID2path) - len(f_car) // self.num_assign_interval
                    )
                    # print(pathflow_solver.state_dict())

                    # update f tensor from updated log_f tensor
                    f_car, f_truck = pathflow_solver.generate_pathflow_tensor()

                    # update params and optimizer
                    pathflow_solver.set_params_with_lr(car_step_size, truck_step_size)
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
                best_log_f_car, best_log_f_truck = pathflow_solver.get_log_f_numpy()

                # f numpy
                best_f_car, best_f_truck = pathflow_solver.generate_pathflow_numpy(f_car, f_truck)
                
                best_x_e_car, best_x_e_truck, best_tt_e_car, best_tt_e_truck, best_O_demand = x_e_car, x_e_truck, tt_e_car, tt_e_truck, O_demand

                if store_folder is not None:
                    self.save_simulation_input_files(os.path.join(store_folder, 'input_files_estimate_path_flow'), 
                                                     best_f_car, best_f_truck)
            
            # print(f_car, f_truck)
            # break
            if store_folder is not None:
                # log_f numpy
                log_f_car_numpy, log_f_truck_numpy = pathflow_solver.get_log_f_numpy()
                # f numpy
                f_car_numpy, f_truck_numpy = pathflow_solver.generate_pathflow_numpy(f_car, f_truck)
                pickle.dump((loss, loss_dict, loss_list, best_epoch, log_f_car_numpy, log_f_truck_numpy, f_car_numpy, f_truck_numpy,
                                x_e_car, x_e_truck, tt_e_car, tt_e_truck, O_demand), 
                                open(os.path.join(store_folder, str(starting_epoch + i) + '_iteration.pickle'), 'wb'))

        print("Best loss at Epoch:", best_epoch, "Loss:", loss_list[best_epoch][0], self.print_separate_accuracy(loss_list[best_epoch][1]))
        return best_f_car, best_f_truck, best_x_e_car, best_x_e_truck, best_tt_e_car, best_tt_e_truck, best_O_demand, loss_list


    def estimate_path_flow_pytorch2(self, car_step_size=0.1, truck_step_size=0.1, 
                                    link_car_flow_weight=1, link_truck_flow_weight=1, 
                                    link_car_tt_weight=1, link_truck_tt_weight=1, origin_vehicle_registration_weight=1, od_demand_weight=1,
                                    max_epoch=10, algo='NAdam', normalized_by_scale = True,
                                    car_init_scale=10,
                                    truck_init_scale=1, 
                                    store_folder=None, 
                                    use_file_as_init=None,
                                    starting_epoch=0,
                                    column_generation=False):

        if np.isscalar(column_generation):
            column_generation = np.ones(max_epoch, dtype=int) * column_generation
        assert(len(column_generation) == max_epoch)
    
        if np.isscalar(link_car_flow_weight):
            link_car_flow_weight = np.ones(max_epoch, dtype=bool) * link_car_flow_weight
        assert(len(link_car_flow_weight) == max_epoch)

        if np.isscalar(link_truck_flow_weight):
            link_truck_flow_weight = np.ones(max_epoch, dtype=bool) * link_truck_flow_weight
        assert(len(link_truck_flow_weight) == max_epoch)

        if np.isscalar(link_car_tt_weight):
            link_car_tt_weight = np.ones(max_epoch, dtype=bool) * link_car_tt_weight
        assert(len(link_car_tt_weight) == max_epoch)

        if np.isscalar(link_truck_tt_weight):
            link_truck_tt_weight = np.ones(max_epoch, dtype=bool) * link_truck_tt_weight
        assert(len(link_truck_tt_weight) == max_epoch)

        if np.isscalar(origin_vehicle_registration_weight):
            origin_vehicle_registration_weight = np.ones(max_epoch, dtype=bool) * origin_vehicle_registration_weight
        assert(len(origin_vehicle_registration_weight) == max_epoch)

        if np.isscalar(od_demand_weight):
            od_demand_weight = np.ones(max_epoch, dtype=bool) * od_demand_weight
        assert(len(od_demand_weight) == max_epoch)

        loss_list = list()
        best_epoch = starting_epoch
        best_f_car, best_f_truck = 0, 0
        best_x_e_car, best_x_e_truck, best_tt_e_car, best_tt_e_truck, best_O_demand = 0, 0, 0, 0, 0
        # here the basic variables to be estimated are path flows, not OD demand, so no route choice model, unlike in sDODE.py
        if use_file_as_init is None:
            (f_car, f_truck) = self.init_path_flow(car_scale=car_init_scale, truck_scale=truck_init_scale)
        elif use_file_as_init.split(os.sep)[-1] == 'init_path_flow.pkl':
            (f_car, f_truck) = pickle.load(open(use_file_as_init, 'rb'))
        else:
            # most recent
            _, _, loss_list, best_epoch, _, _, \
                _, _, _, _, _ = pickle.load(open(use_file_as_init, 'rb'))
            # best 
            use_file_as_init = os.path.join(store_folder, '{}_iteration.pickle'.format(best_epoch))
            _, _, _, _, best_f_car, best_f_truck, \
                best_x_e_car, best_x_e_truck, best_tt_e_car, best_tt_e_truck, best_O_demand = pickle.load(open(use_file_as_init, 'rb'))
            
            f_car, f_truck = best_f_car, best_f_truck

        assert((len(loss_list) >= best_epoch) and (starting_epoch >= best_epoch))

        if normalized_by_scale:
            f_car_tensor = torch.from_numpy(f_car / np.maximum(car_init_scale, 1e-6))
            f_truck_tensor = torch.from_numpy(f_truck / np.maximum(truck_init_scale, 1e-6))
        else:
            f_car_tensor = torch.from_numpy(f_car)
            f_truck_tensor = torch.from_numpy(f_truck)

        f_car_tensor.requires_grad = True
        f_truck_tensor.requires_grad = True

        params = [
            {'params': f_car_tensor, 'lr': car_step_size},
            {'params': f_truck_tensor, 'lr': truck_step_size}
        ]

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
            # print("Start iteration", time.time())
            loss_dict = {'car_count_loss': 0.0, 'truck_count_loss': 0.0, 'car_tt_loss': 0.0, 'truck_tt_loss': 0.0, 
                         'origin_vehicle_registration_loss': 0.0, 'od_demand_loss': 0.0}

            self.config['link_car_flow_weight'] = link_car_flow_weight[i] * (self.config['use_car_link_flow'] or self.config['compute_car_link_flow_loss'])
            self.config['link_truck_flow_weight'] = link_truck_flow_weight[i] * (self.config['use_truck_link_flow'] or self.config['compute_truck_link_flow_loss'])
            
            self.config['link_car_tt_weight'] = link_car_tt_weight[i] * (self.config['use_car_link_tt'] or self.config['compute_car_link_tt_loss'])
            self.config['link_truck_tt_weight'] = link_truck_tt_weight[i] * (self.config['use_truck_link_tt'] or self.config['compute_truck_link_tt_loss'])

            self.config['origin_vehicle_registration_weight'] = origin_vehicle_registration_weight[i] * (self.config['use_origin_vehicle_registration_data'] or self.config['compute_origin_vehicle_registration_loss'])
            self.config['od_demand_weight'] = od_demand_weight[i] * (self.config['use_od_demand_data'] or self.config['compute_od_demand_loss'])
            for j in seq:
                one_data_dict = self._get_one_data(j)
                car_grad, truck_grad, tmp_loss, tmp_loss_dict, dta, x_e_car, x_e_truck, tt_e_car, tt_e_truck, O_demand = self.compute_path_flow_grad_and_loss(one_data_dict, f_car, f_truck)
                # print("gradient", car_grad, truck_grad)

                if column_generation[i] == 0:
                    dta = 0

                optimizer.zero_grad()

                if normalized_by_scale:
                    f_car_tensor.grad = torch.from_numpy(car_grad * car_init_scale)
                    f_truck_tensor.grad = torch.from_numpy(truck_grad * truck_init_scale)
                else:
                    f_car_tensor.grad = torch.from_numpy(car_grad)
                    f_truck_tensor.grad = torch.from_numpy(truck_grad)

                optimizer.step()

                car_grad, truck_grad = 0, 0
                optimizer.zero_grad()

                if normalized_by_scale:
                    f_car = f_car_tensor.data.cpu().numpy() * car_init_scale
                    f_truck = f_truck_tensor.data.cpu().numpy() * truck_init_scale
                else:
                    f_car = f_car_tensor.data.cpu().numpy()
                    f_truck = f_truck_tensor.data.cpu().numpy()

                if column_generation[i]:
                    print("***************** generate new paths *****************")
                    if not self.config['use_car_link_tt'] and not self.config['use_truck_link_tt']:
                        # if any of these is true, dta.build_link_cost_map(True) is already invoked in compute_path_flow_grad_and_loss()
                        dta.build_link_cost_map(False)
                    
                    self.update_path_table(dta)
                    f_car, f_truck = self.update_path_flow(f_car, f_truck, car_init_scale, truck_init_scale)
                    dta = 0
            
                f_car = np.maximum(f_car, 1e-6)
                f_truck = np.maximum(f_truck, 1e-6)

                if column_generation[i]:
                    if normalized_by_scale:
                        f_car_tensor = torch.from_numpy(f_car / np.maximum(car_init_scale, 1e-6))
                        f_truck_tensor = torch.from_numpy(f_truck / np.maximum(truck_init_scale, 1e-6))
                    else:
                        f_car_tensor = torch.from_numpy(f_car)
                        f_truck_tensor = torch.from_numpy(f_truck)

                    f_car_tensor.requires_grad = True
                    f_truck_tensor.requires_grad = True

                    params = [
                        {'params': f_car_tensor, 'lr': car_step_size},
                        {'params': f_truck_tensor, 'lr': truck_step_size}
                    ]

                    optimizer = algo_dict[algo](params)

                loss += tmp_loss / float(self.num_data)
                for loss_type, loss_value in tmp_loss_dict.items():
                    loss_dict[loss_type] += loss_value / float(self.num_data)

            print("Epoch:", starting_epoch + i, "Loss:", loss, self.print_separate_accuracy(loss_dict))
            loss_list.append([loss, loss_dict])
                    
            if (best_epoch == 0) or (loss_list[best_epoch][0] > loss_list[-1][0]):
                best_epoch = starting_epoch + i
                best_f_car, best_f_truck = f_car, f_truck
                best_x_e_car, best_x_e_truck, best_tt_e_car, best_tt_e_truck, best_O_demand = x_e_car, x_e_truck, tt_e_car, tt_e_truck, O_demand

                if store_folder is not None:
                    self.save_simulation_input_files(os.path.join(store_folder, 'input_files_estimate_path_flow'), 
                                                     best_f_car, best_f_truck)
            
            # print(f_car, f_truck)
            # break
            if store_folder is not None:
                pickle.dump((loss, loss_dict, loss_list, best_epoch, f_car, f_truck,
                                x_e_car, x_e_truck, tt_e_car, tt_e_truck, O_demand), 
                                open(os.path.join(store_folder, str(starting_epoch + i) + '_iteration.pickle'), 'wb'))

        print("Best loss at Epoch:", best_epoch, "Loss:", loss_list[best_epoch][0], self.print_separate_accuracy(loss_list[best_epoch][1]))
        return best_f_car, best_f_truck, best_x_e_car, best_x_e_truck, best_tt_e_car, best_tt_e_truck, best_O_demand, loss_list

    def estimate_path_flow(self, car_step_size=0.1, truck_step_size=0.1, 
                            link_car_flow_weight=1, link_truck_flow_weight=1, 
                            link_car_tt_weight=1, link_truck_tt_weight=1, origin_vehicle_registration_weight=1e-6, od_demand_weight=1.0,
                            max_epoch=10, car_init_scale=10, truck_init_scale=1, store_folder=None, use_file_as_init=None, 
                            adagrad=False, starting_epoch=0):

        if np.isscalar(link_car_flow_weight):
            link_car_flow_weight = np.ones(max_epoch, dtype=bool) * link_car_flow_weight
        assert(len(link_car_flow_weight) == max_epoch)

        if np.isscalar(link_truck_flow_weight):
            link_truck_flow_weight = np.ones(max_epoch, dtype=bool) * link_truck_flow_weight
        assert(len(link_truck_flow_weight) == max_epoch)

        if np.isscalar(link_car_tt_weight):
            link_car_tt_weight = np.ones(max_epoch, dtype=bool) * link_car_tt_weight
        assert(len(link_car_tt_weight) == max_epoch)

        if np.isscalar(link_truck_tt_weight):
            link_truck_tt_weight = np.ones(max_epoch, dtype=bool) * link_truck_tt_weight
        assert(len(link_truck_tt_weight) == max_epoch)

        if np.isscalar(origin_vehicle_registration_weight):
            origin_vehicle_registration_weight = np.ones(max_epoch, dtype=bool) * origin_vehicle_registration_weight
        assert(len(origin_vehicle_registration_weight) == max_epoch)

        if np.isscalar(od_demand_weight):
            od_demand_weight = np.ones(max_epoch, dtype=bool) * od_demand_weight
        assert(len(od_demand_weight) == max_epoch)
    
        # here the basic variables to be estimated are path flows, not OD demand, so no route choice model, unlike in sDODE.py
        loss_list = list()
        best_epoch = starting_epoch
        best_f_car, best_f_truck = 0, 0
        best_x_e_car, best_x_e_truck, best_tt_e_car, best_tt_e_truck, best_O_demand = 0, 0, 0, 0, 0
        # here the basic variables to be estimated are path flows, not OD demand, so no route choice model, unlike in sDODE.py
        if use_file_as_init is None:
            (f_car, f_truck) = self.init_path_flow(car_scale=car_init_scale, truck_scale=truck_init_scale)
        elif use_file_as_init.split(os.sep)[-1] == 'init_path_flow.pkl':
            (f_car, f_truck) = pickle.load(open(use_file_as_init, 'rb'))
        else:
            # most recent
            _, _, loss_list, best_epoch, _, _, \
                _, _, _, _, _ = pickle.load(open(use_file_as_init, 'rb'))
            # best 
            use_file_as_init = os.path.join(store_folder, '{}_iteration.pickle'.format(best_epoch))
            _, _, _, _, best_f_car, best_f_truck, \
                best_x_e_car, best_x_e_truck, best_tt_e_car, best_tt_e_truck, best_O_demand = pickle.load(open(use_file_as_init, 'rb'))
            
            f_car, f_truck = best_f_car, best_f_truck

        assert((len(loss_list) >= best_epoch) and (starting_epoch >= best_epoch))

        for i in range(max_epoch):
            if adagrad:
                sum_g_square_car = 1e-6
                sum_g_square_truck = 1e-6
            seq = np.random.permutation(self.num_data)
            loss = float(0)
            # print("Start iteration", time.time())
            loss_dict = {'car_count_loss': 0.0, 'truck_count_loss': 0.0, 'car_tt_loss': 0.0, 'truck_tt_loss': 0.0, 
                         'origin_vehicle_registration_loss': 0.0, 'od_demand_loss': 0.0}

            self.config['link_car_flow_weight'] = link_car_flow_weight[i] * (self.config['use_car_link_flow'] or self.config['compute_car_link_flow_loss'])
            self.config['link_truck_flow_weight'] = link_truck_flow_weight[i] * (self.config['use_truck_link_flow'] or self.config['compute_truck_link_flow_loss'])
            
            self.config['link_car_tt_weight'] = link_car_tt_weight[i] * (self.config['use_car_link_tt'] or self.config['compute_car_link_tt_loss'])
            self.config['link_truck_tt_weight'] = link_truck_tt_weight[i] * (self.config['use_truck_link_tt'] or self.config['compute_truck_link_tt_loss'])

            self.config['origin_vehicle_registration_weight'] = origin_vehicle_registration_weight[i] * (self.config['use_origin_vehicle_registration_data'] or self.config['compute_origin_vehicle_registration_loss'])
            self.config['od_demand_weight'] = od_demand_weight[i] * (self.config['use_od_demand_data'] or self.config['compute_od_demand_loss'])
            for j in seq:
                one_data_dict = self._get_one_data(j)
                car_grad, truck_grad, tmp_loss, tmp_loss_dict, dta, x_e_car, x_e_truck, tt_e_car, tt_e_truck, O_demand = self.compute_path_flow_grad_and_loss(one_data_dict, f_car, f_truck)
                # print("gradient", car_grad, truck_grad)
                if adagrad:
                    sum_g_square_car = sum_g_square_car + np.power(car_grad, 2)
                    sum_g_square_truck = sum_g_square_truck + np.power(truck_grad, 2)
                    f_car = f_car - car_step_size * car_grad / np.sqrt(sum_g_square_car) 
                    f_truck = f_truck - truck_step_size * truck_grad / np.sqrt(sum_g_square_truck) 
                else:
                    f_car -= car_grad * car_step_size / np.sqrt(i+1)
                    f_truck -= truck_grad * truck_step_size / np.sqrt(i+1)
                f_car = np.maximum(f_car, 1e-3)
                f_truck = np.maximum(f_truck, 1e-3)
                # f_truck = np.minimum(f_truck, 30)
                loss += tmp_loss / float(self.num_data)
                for loss_type, loss_value in tmp_loss_dict.items():
                    loss_dict[loss_type] += loss_value / float(self.num_data)
            
            print("Epoch:", starting_epoch + i, "Loss:", loss, self.print_separate_accuracy(loss_dict))
            loss_list.append([loss, loss_dict])

            if (best_epoch == 0) or (loss_list[best_epoch][0] > loss_list[-1][0]):
                best_epoch = starting_epoch + i
                best_f_car, best_f_truck = f_car, f_truck
                best_x_e_car, best_x_e_truck, best_tt_e_car, best_tt_e_truck, best_O_demand = x_e_car, x_e_truck, tt_e_car, tt_e_truck, O_demand

                if store_folder is not None:
                    self.save_simulation_input_files(os.path.join(store_folder, 'input_files_estimate_path_flow'), 
                                                     best_f_car, best_f_truck)

            # print(f_car, f_truck)
            # break
            if store_folder is not None:
                pickle.dump((loss, loss_dict, loss_list, best_epoch, f_car, f_truck,
                             x_e_car, x_e_truck, tt_e_car, tt_e_truck, O_demand), open(os.path.join(store_folder, str(starting_epoch + i) + '_iteration.pickle'), 'wb'))

        print("Best loss at Epoch:", best_epoch, "Loss:", loss_list[best_epoch][0], self.print_separate_accuracy(loss_list[best_epoch][1]))
        return best_f_car, best_f_truck, best_x_e_car, best_x_e_truck, best_tt_e_car, best_tt_e_truck, best_O_demand, loss_list

    def estimate_path_flow_gd(self, car_step_size=0.1, truck_step_size=0.1, max_epoch=10, 
                                link_car_flow_weight=1, link_truck_flow_weight=1, 
                                link_car_tt_weight=1, link_truck_tt_weight=1,
                                car_init_scale=10, truck_init_scale=1, store_folder=None, use_file_as_init=None, adagrad=False, starting_epoch=0):
    
        if np.isscalar(link_car_flow_weight):
            link_car_flow_weight = np.ones(max_epoch, dtype=bool) * link_car_flow_weight
        assert(len(link_car_flow_weight) == max_epoch)

        if np.isscalar(link_truck_flow_weight):
            link_truck_flow_weight = np.ones(max_epoch, dtype=bool) * link_truck_flow_weight
        assert(len(link_truck_flow_weight) == max_epoch)

        if np.isscalar(link_car_tt_weight):
            link_car_tt_weight = np.ones(max_epoch, dtype=bool) * link_car_tt_weight
        assert(len(link_car_tt_weight) == max_epoch)

        if np.isscalar(link_truck_tt_weight):
            link_truck_tt_weight = np.ones(max_epoch, dtype=bool) * link_truck_tt_weight
        assert(len(link_truck_tt_weight) == max_epoch)
        
        # here the basic variables to be estimated are path flows, not OD demand, so no route choice model, unlike in sDODE.py
        loss_list = list()
        best_epoch = starting_epoch
        best_f_car, best_f_truck = 0, 0
        best_x_e_car, best_x_e_truck, best_tt_e_car, best_tt_e_truck = 0, 0, 0, 0
        # here the basic variables to be estimated are path flows, not OD demand, so no route choice model, unlike in sDODE.py
        if use_file_as_init is None:
            (f_car, f_truck) = self.init_path_flow(car_scale=car_init_scale, truck_scale=truck_init_scale)
        elif use_file_as_init.split(os.sep)[-1] == 'init_path_flow.pkl':
            (f_car, f_truck) = pickle.load(open(use_file_as_init, 'rb'))
        else:
            # most recent
            _, _, loss_list, best_epoch, _, _, \
                _, _, _, _ = pickle.load(open(use_file_as_init, 'rb'))
            # best 
            use_file_as_init = os.path.join(store_folder, '{}_iteration.pickle'.format(best_epoch))
            _, _, _, _, best_f_car, best_f_truck, \
                best_x_e_car, best_x_e_truck, best_tt_e_car, best_tt_e_truck = pickle.load(open(use_file_as_init, 'rb'))
            
            f_car, f_truck = best_f_car, best_f_truck

        assert((len(loss_list) >= best_epoch) and (starting_epoch >= best_epoch))

        start_time = time.time()
        for i in range(max_epoch):
            grad_car_sum = np.zeros(f_car.shape)
            grad_truck_sum = np.zeros(f_truck.shape)
            seq = np.random.permutation(self.num_data)
            loss = float(0)
            # print("Start iteration", time.time())
            loss_dict = {'car_count_loss': 0.0, 'truck_count_loss': 0.0, 'car_tt_loss': 0.0, 'truck_tt_loss': 0.0}

            self.config['link_car_flow_weight'] = link_car_flow_weight[i] * (self.config['use_car_link_flow'] or self.config['compute_car_link_flow_loss'])
            self.config['link_truck_flow_weight'] = link_truck_flow_weight[i] * (self.config['use_truck_link_flow'] or self.config['compute_truck_link_flow_loss'])
            
            self.config['link_car_tt_weight'] = link_car_tt_weight[i] * (self.config['use_car_link_tt'] or self.config['compute_car_link_tt_loss'])
            self.config['link_truck_tt_weight'] = link_truck_tt_weight[i] * (self.config['use_truck_link_tt'] or self.config['compute_truck_link_tt_loss'])

            for j in seq:
                one_data_dict = self._get_one_data(j)
                car_grad, truck_grad, tmp_loss, tmp_loss_dict, dta, x_e_car, x_e_truck, tt_e_car, tt_e_truck = self.compute_path_flow_grad_and_loss(one_data_dict, f_car, f_truck)
                # print("gradient", car_grad, truck_grad)
            
                grad_car_sum += car_grad
                grad_truck_sum += truck_grad
                #f_truck = np.minimum(f_truck, 30)

                loss += tmp_loss / float(self.num_data)
                for loss_type, loss_value in tmp_loss_dict.items():
                    loss_dict[loss_type] += loss_value / float(self.num_data)
        
            print("Epoch:", starting_epoch + i, "Loss:", loss, self.print_separate_accuracy(loss_dict))
            loss_list.append([loss, loss_dict])

            if (best_epoch == 0) or (loss_list[best_epoch][0] > loss_list[-1][0]):
                best_epoch = starting_epoch + i
                best_f_car, best_f_truck = f_car, f_truck
                best_x_e_car, best_x_e_truck, best_tt_e_car, best_tt_e_truck = x_e_car, x_e_truck, tt_e_car, tt_e_truck

                if store_folder is not None:
                    self.save_simulation_input_files(os.path.join(store_folder, 'input_files_estimate_path_flow'), 
                                                     best_f_car, best_f_truck)

            f_car -= grad_car_sum * car_step_size / np.sqrt(i+1) / float(self.num_data)
            f_truck -= grad_truck_sum * truck_step_size / np.sqrt(i+1) / float(self.num_data)
            f_car = np.maximum(f_car, 1e-3)
            f_truck = np.maximum(f_truck, 1e-3)

            # print(f_car, f_truck)
            # break
            if store_folder is not None:
                pickle.dump((loss, loss_dict, loss_list, best_epoch, f_car, f_truck,
                             x_e_car, x_e_truck, tt_e_car, tt_e_truck, time.time() - start_time), open(os.path.join(store_folder, str(starting_epoch + i) + '_iteration.pickle'), 'wb'))

        print("Best loss at Epoch:", best_epoch, "Loss:", loss_list[best_epoch][0], self.print_separate_accuracy(loss_list[best_epoch][1]))
        return best_f_car, best_f_truck, best_x_e_car, best_x_e_truck, best_tt_e_car, best_tt_e_truck, loss_list

    def compute_path_flow_grad_and_loss_mpwrapper(self, one_data_dict, f_car, f_truck, j, output):
        car_grad, truck_grad, tmp_loss, tmp_loss_dict, _, x_e_car, x_e_truck, tt_e_car, tt_e_truck = self.compute_path_flow_grad_and_loss(one_data_dict, f_car, f_truck, counter=j)
        # print("finished original grad loss")
        output.put([car_grad, truck_grad, tmp_loss, tmp_loss_dict, x_e_car, x_e_truck, tt_e_car, tt_e_truck])
        # output.put(grad)
        # print("finished put")
        return

    def estimate_path_flow_mp(self, car_step_size=0.1, truck_step_size=0.1, 
                                link_car_flow_weight=1, link_truck_flow_weight=1, 
                                link_car_tt_weight=1, link_truck_tt_weight=1,
                                max_epoch=10, car_init_scale=10,
                                truck_init_scale=1, store_folder=None, use_file_as_init=None,
                                adagrad=False, starting_epoch=0, n_process=4, record_time=False):

        if np.isscalar(link_car_flow_weight):
            link_car_flow_weight = np.ones(max_epoch, dtype=bool) * link_car_flow_weight
        assert(len(link_car_flow_weight) == max_epoch)

        if np.isscalar(link_truck_flow_weight):
            link_truck_flow_weight = np.ones(max_epoch, dtype=bool) * link_truck_flow_weight
        assert(len(link_truck_flow_weight) == max_epoch)

        if np.isscalar(link_car_tt_weight):
            link_car_tt_weight = np.ones(max_epoch, dtype=bool) * link_car_tt_weight
        assert(len(link_car_tt_weight) == max_epoch)

        if np.isscalar(link_truck_tt_weight):
            link_truck_tt_weight = np.ones(max_epoch, dtype=bool) * link_truck_tt_weight
        assert(len(link_truck_tt_weight) == max_epoch)

        # here the basic variables to be estimated are path flows, not OD demand, so no route choice model, unlike in sDODE.py
        loss_list = list()
        best_epoch = starting_epoch
        best_f_car, best_f_truck = 0, 0
        best_x_e_car, best_x_e_truck, best_tt_e_car, best_tt_e_truck = 0, 0, 0, 0
        # here the basic variables to be estimated are path flows, not OD demand, so no route choice model, unlike in sDODE.py
        if use_file_as_init is None:
            (f_car, f_truck) = self.init_path_flow(car_scale=car_init_scale, truck_scale=truck_init_scale)
        elif use_file_as_init.split(os.sep)[-1] == 'init_path_flow.pkl':
            (f_car, f_truck) = pickle.load(open(use_file_as_init, 'rb'))
        else:
            # most recent
            _, _, loss_list, best_epoch, _, _, \
                _, _, _, _ = pickle.load(open(use_file_as_init, 'rb'))
            # best 
            use_file_as_init = os.path.join(store_folder, '{}_iteration.pickle'.format(best_epoch))
            _, _, _, _, best_f_car, best_f_truck, \
                best_x_e_car, best_x_e_truck, best_tt_e_car, best_tt_e_truck = pickle.load(open(use_file_as_init, 'rb'))
            
            f_car, f_truck = best_f_car, best_f_truck

        assert((len(loss_list) >= best_epoch) and (starting_epoch >= best_epoch))
 
        # print("Start iteration", time.time())
        start_time = time.time()
        for i in range(max_epoch):
            if adagrad:
                sum_g_square_car = 1e-6
                sum_g_square_truck = 1e-6
            seq = np.random.permutation(self.num_data)
            split_seq = np.array_split(seq, np.maximum(1, int(self.num_data/n_process)))

            loss = float(0)
            loss_dict = {'car_count_loss': 0.0, 'truck_count_loss': 0.0, 'car_tt_loss': 0.0, 'truck_tt_loss': 0.0}

            self.config['link_car_flow_weight'] = link_car_flow_weight[i] * (self.config['use_car_link_flow'] or self.config['compute_car_link_flow_loss'])
            self.config['link_truck_flow_weight'] = link_truck_flow_weight[i] * (self.config['use_truck_link_flow'] or self.config['compute_truck_link_flow_loss'])
            
            self.config['link_car_tt_weight'] = link_car_tt_weight[i] * (self.config['use_car_link_tt'] or self.config['compute_car_link_tt_loss'])
            self.config['link_truck_tt_weight'] = link_truck_tt_weight[i] * (self.config['use_truck_link_tt'] or self.config['compute_truck_link_tt_loss'])

            for part_seq in split_seq:
                output = mp.Queue()
                processes = [mp.Process(target=self.compute_path_flow_grad_and_loss_mpwrapper, args=(self._get_one_data(j), f_car, f_truck, j, output)) for j in part_seq]
                for p in processes:
                    p.start()
                results = list()
                while 1:
                    running = any(p.is_alive() for p in processes)
                    while not output.empty():
                        results.append(output.get())
                    if not running:
                        break
                for p in processes:
                    p.join()
                # results = [output.get() for p in processes]
                for res in results:
                    [car_grad, truck_grad, tmp_loss, tmp_loss_dict, x_e_car, x_e_truck, tt_e_car, tt_e_truck] = res
                    loss += tmp_loss / float(self.num_data)
                    for loss_type, loss_value in tmp_loss_dict.items():
                        loss_dict[loss_type] += loss_value / float(self.num_data)
                    if adagrad:
                        sum_g_square_car = sum_g_square_car + np.power(car_grad, 2)
                        sum_g_square_truck = sum_g_square_truck + np.power(truck_grad, 2)
                        f_car = f_car - car_step_size * car_grad / np.sqrt(sum_g_square_car)
                        f_truck = f_truck - truck_step_size * truck_grad / np.sqrt(sum_g_square_truck)
                    else:
                        f_car -= car_grad * car_step_size / np.sqrt(i+1)
                        f_truck -= truck_grad * truck_step_size / np.sqrt(i+1)       
                    f_car = np.maximum(f_car, 1e-3)
                    f_truck = np.maximum(f_truck, 1e-3)
                    f_truck = np.minimum(f_truck, 30)

        print("Epoch:", starting_epoch + i, "Loss:", loss, self.print_separate_accuracy(loss_dict))

        if (best_epoch == 0) or (loss_list[best_epoch][0] > loss_list[-1][0]):
            best_epoch = starting_epoch + i
            best_f_car, best_f_truck = f_car, f_truck
            best_x_e_car, best_x_e_truck, best_tt_e_car, best_tt_e_truck = x_e_car, x_e_truck, tt_e_car, tt_e_truck

            if store_folder is not None:
                    self.save_simulation_input_files(os.path.join(store_folder, 'input_files_estimate_path_flow'), 
                                                     best_f_car, best_f_truck)

        if store_folder is not None:
            pickle.dump((loss, loss_dict, loss_list, best_epoch, f_car, f_truck,
                         x_e_car, x_e_truck, tt_e_car, tt_e_truck, time.time() - start_time), open(os.path.join(store_folder, str(starting_epoch + i) + '_iteration.pickle'), 'wb'))
        if record_time:
            loss_list.append([loss, loss_dict, time.time() - start_time])
        else:
            loss_list.append([loss, loss_dict])

        print("Best loss at Epoch:", best_epoch, "Loss:", loss_list[best_epoch][0], self.print_separate_accuracy(loss_list[best_epoch][1]))
        return best_f_car, best_f_truck, best_x_e_car, best_x_e_truck, best_tt_e_car, best_tt_e_truck, loss_list

    def generate_route_choice(self):
        pass

    def print_separate_accuracy(self, loss_dict):
        tmp_str = ""
        for loss_type, loss_value in loss_dict.items():
            tmp_str += loss_type + ": " + str(np.round(loss_value, 2)) + "|"
        return tmp_str

    def update_path_table(self, dta):
        # dta.build_link_cost_map() should be called before this method

        start_intervals = np.arange(0, self.num_loading_interval, self.ass_freq)
        self.nb.update_path_table(dta, start_intervals)

        self.num_path = self.nb.config.config_dict['FIXED']['num_path']

        # observed path IDs, np.array
        self.config['paths_list'] = np.array(list(self.nb.path_table.ID2path.keys()), dtype=int)
        assert(len(np.unique(self.config['paths_list'])) == len(self.config['paths_list']))
        self.paths_list = self.config['paths_list']
        assert (len(self.paths_list) == self.num_path)

    def update_path_flow(self, f_car, f_truck, car_init_scale=1, truck_init_scale=0.1):
        max_interval = self.nb.config.config_dict['DTA']['max_interval']
        # reshape path flow into ndarrays with dimensions of intervals x number of total paths
        f_car = f_car.reshape(max_interval, -1)
        f_truck = f_truck.reshape(max_interval, -1)

        if len(self.nb.path_table.ID2path) > f_car.shape[1]:
            _add_f = self.init_demand_vector(max_interval, len(self.nb.path_table.ID2path) - f_car.shape[1], car_init_scale)
            _add_f = _add_f.reshape(max_interval, -1)
            f_car = np.concatenate((f_car, _add_f), axis=1)
            assert(f_car.shape[1] == len(self.nb.path_table.ID2path))

            _add_f = self.init_demand_vector(max_interval, len(self.nb.path_table.ID2path) - f_truck.shape[1], truck_init_scale)
            _add_f = _add_f.reshape(max_interval, -1)
            f_truck = np.concatenate((f_truck, _add_f), axis=1)
            assert(f_truck.shape[1] == len(self.nb.path_table.ID2path))

        f_car = f_car.flatten(order='C')
        f_truck = f_truck.flatten(order='C')

        return f_car, f_truck


# Simultaneous Perturbation Stochastic Approximation
class mcSPSA():
  def __init__(self, nb, config):
    self.config = config
    self.nb = nb
    self.num_assign_interval = nb.config.config_dict['DTA']['max_interval']
    self.ass_freq = nb.config.config_dict['DTA']['assign_frq']
    self.num_link = nb.config.config_dict['DTA']['num_of_link']
    self.num_path = nb.config.config_dict['FIXED']['num_path']
    self.num_loading_interval = self.num_assign_interval * self.ass_freq
    self.data_dict = dict()
    self.num_data = self.config['num_data']
    self.observed_links = self.config['observed_links']
    self.paths_list = self.config['paths_list']
    self.car_count_agg_L_list = None
    self.truck_count_agg_L_list = None
    assert (len(self.paths_list) == self.num_path)

  def _add_car_link_flow_data(self, link_flow_df_list):
    # assert (self.config['use_car_link_flow'])
    assert (self.num_data == len(link_flow_df_list))
    self.data_dict['car_link_flow'] = link_flow_df_list

  def _add_truck_link_flow_data(self, link_flow_df_list):
    # assert (self.config['use_truck_link_flow'])
    assert (self.num_data == len(link_flow_df_list))
    self.data_dict['truck_link_flow'] = link_flow_df_list

  def _add_car_link_tt_data(self, link_spd_df_list):
    # assert (self.config['use_car_link_tt'])
    assert (self.num_data == len(link_spd_df_list))
    self.data_dict['car_link_tt'] = link_spd_df_list

  def _add_truck_link_tt_data(self, link_spd_df_list):
    # assert (self.config['use_truck_link_tt'])
    assert (self.num_data == len(link_spd_df_list))
    self.data_dict['truck_link_tt'] = link_spd_df_list

  def add_data(self, data_dict):
    if self.config['car_count_agg']:
      self.car_count_agg_L_list = data_dict['car_count_agg_L_list']
    if self.config['truck_count_agg']:
      self.truck_count_agg_L_list = data_dict['truck_count_agg_L_list']
    if self.config['use_car_link_flow'] or self.config['compute_car_link_flow_loss']:
      self._add_car_link_flow_data(data_dict['car_link_flow'])
    if self.config['use_truck_link_flow'] or self.config['compute_truck_link_flow_loss'] :
      self._add_truck_link_flow_data(data_dict['truck_link_flow'])
    if self.config['use_car_link_tt'] or self.config['compute_car_link_tt_loss']:
      self._add_car_link_tt_data(data_dict['car_link_tt'])
    if self.config['use_truck_link_tt']or self.config['compute_car_link_tt_loss']:
      self._add_truck_link_tt_data(data_dict['truck_link_tt'])

  def save_simulation_input_files(self, folder_path, f_car=None, f_truck=None):

        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        # update demand for each mode
        if (f_car is not None) and (f_truck is not None):
            self.nb.update_demand_path2(f_car, f_truck)
   
        # self.nb.config.config_dict['DTA']['flow_scalar'] = 3
        if self.config['use_car_link_tt'] or self.config['use_truck_link_tt']:
            self.nb.config.config_dict['DTA']['total_interval'] = self.num_loading_interval # * 2  # hopefully this is sufficient 
        else:
            self.nb.config.config_dict['DTA']['total_interval'] = self.num_loading_interval  # if only count data is used

        self.nb.config.config_dict['DTA']['routing_type'] = 'Biclass_Hybrid'

        # no output files saved from DNL
        self.nb.config.config_dict['STAT']['rec_volume'] = 1
        self.nb.config.config_dict['STAT']['volume_load_automatic_rec'] = 0
        self.nb.config.config_dict['STAT']['volume_record_automatic_rec'] = 0
        self.nb.config.config_dict['STAT']['rec_tt'] = 1
        self.nb.config.config_dict['STAT']['tt_load_automatic_rec'] = 0
        self.nb.config.config_dict['STAT']['tt_record_automatic_rec'] = 0

        self.nb.dump_to_folder(folder_path)

  def _run_simulation(self, f_car, f_truck, counter=0, show_loading=False):
    hash1 = hashlib.sha1()
    hash1.update(str(time.time()) + str(counter))
    new_folder = str(hash1.hexdigest())

    self.save_simulation_input_files(new_folder, f_car, f_truck)

    a = macposts.mcdta_api()
    a.initialize(new_folder)
    shutil.rmtree(new_folder)
    a.register_links(self.observed_links)
    a.register_paths(self.paths_list)
    a.install_cc()
    a.install_cc_tree()
    a.run_whole(show_loading)
    # print("Finish simulation", time.time())
    return a  

  def init_path_flow(self, car_scale=1, truck_scale=0.1):
    f_car = np.random.rand(self.num_assign_interval * self.num_path) * car_scale
    f_truck = np.random.rand(self.num_assign_interval * self.num_path) * truck_scale
    return f_car, f_truck

  def _get_one_data(self, j):
    assert (self.num_data > j)
    one_data_dict = dict()
    if self.config['use_car_link_flow'] or self.config['compute_car_link_flow_loss']:
      one_data_dict['car_link_flow'] = self.data_dict['car_link_flow'][j]
    if self.config['use_truck_link_flow']or self.config['compute_truck_link_flow_loss']:
      one_data_dict['truck_link_flow'] = self.data_dict['truck_link_flow'][j]
    if self.config['use_car_link_tt'] or self.config['compute_car_link_tt_loss']:
      one_data_dict['car_link_tt'] = self.data_dict['car_link_tt'][j]
    if self.config['use_truck_link_tt'] or self.config['compute_truck_link_tt_loss']:
      one_data_dict['truck_link_tt'] = self.data_dict['truck_link_tt'][j]
    if self.config['car_count_agg']:
      one_data_dict['car_count_agg_L'] = self.car_count_agg_L_list[j]
    if self.config['truck_count_agg']:
      one_data_dict['truck_count_agg_L'] = self.truck_count_agg_L_list[j]
    return one_data_dict

  def _get_loss(self, one_data_dict, dta):
    loss_dict = dict()
    if self.config['use_car_link_flow'] or self.config['compute_car_link_flow_loss']:
      x_e = dta.get_link_car_inflow(np.arange(0, self.num_loading_interval, self.ass_freq), 
                                    np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq).flatten(order='F')
      if self.config['car_count_agg']:
        x_e = one_data_dict['car_count_agg_L'].dot(x_e)
      loss = self.config['link_car_flow_weight'] * np.linalg.norm(np.nan_to_num(x_e - one_data_dict['car_link_flow']))
      loss_dict['car_count_loss'] = loss
    if self.config['use_truck_link_flow'] or self.config['compute_truck_link_flow_loss']:
      x_e = dta.get_link_truck_inflow(np.arange(0, self.num_loading_interval, self.ass_freq), 
                                      np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq).flatten(order='F')
      if self.config['truck_count_agg']:
        x_e = one_data_dict['truck_count_agg_L'].dot(x_e)
      loss = self.config['link_truck_flow_weight'] * np.linalg.norm(np.nan_to_num(x_e - one_data_dict['truck_link_flow']))
      loss_dict['truck_count_loss'] = loss
    if self.config['use_car_link_tt'] or self.config['compute_car_link_tt_loss']:
      x_tt_e = dta.get_car_link_tt(np.arange(0, self.num_loading_interval, self.ass_freq)).flatten(order='F')
      # x_tt_e = dta.get_car_link_tt_robust(np.arange(0, self.num_loading_interval, self.ass_freq),
      #                                     np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq).flatten(order = 'F')
      loss = self.config['link_car_tt_weight'] * np.linalg.norm(np.nan_to_num(x_tt_e - one_data_dict['car_link_tt']))
      loss_dict['car_tt_loss'] = loss
    if self.config['use_truck_link_tt'] or self.config['compute_truck_link_tt_loss']:
      x_tt_e = dta.get_truck_link_tt(np.arange(0, self.num_loading_interval, self.ass_freq)).flatten(order='F')
      loss = self.config['link_truck_tt_weight'] * np.linalg.norm(np.nan_to_num(x_tt_e - one_data_dict['truck_link_tt']))
      loss_dict['truck_tt_loss'] = loss

    total_loss = 0.0
    for loss_type, loss_value in loss_dict.items():
      total_loss += loss_value
    return total_loss, loss_dict

  def compute_path_flow_grad_and_loss(self, one_data_dict, f_car, f_truck, delta_car_scale=0.1, delta_truck_scale=0.01):
    delta_f_car = (np.random.rand(*f_car.shape) * 2 - 1) * delta_car_scale
    delta_f_truck = (np.random.rand(*f_truck.shape) * 2 - 1) * delta_truck_scale
    f_car_1 = np.maximum(f_car + delta_f_car, 1e-3)
    f_truck_1 = np.maximum(f_truck + delta_f_truck, 1e-3)
    f_car_2 = np.maximum(f_car - delta_f_car, 1e-3)
    f_truck_2 = np.maximum(f_truck - delta_f_truck, 1e-3)
    # print(f_car.sum(), f_truck.sum())
    # print(f_car_1.sum(), f_truck_1.sum())
    # print(f_car_2.sum(), f_truck_2.sum())
    dta_1 = self._run_simulation(f_car_1, f_truck_1, 11111)
    dta_2 = self._run_simulation(f_car_2, f_truck_2, 22222)
    loss_1 = self._get_loss(one_data_dict, dta_1)[0]
    loss_2 = self._get_loss(one_data_dict, dta_2)[0]
    grad_f_car = (loss_1 - loss_2) / (2 * delta_f_car)
    grad_f_truck = (loss_1 - loss_2) / (2 * delta_f_truck)
    dta = self._run_simulation(f_car, f_truck, 33333)
    total_loss, loss_dict = self._get_loss(one_data_dict, dta)
    return -grad_f_car, - grad_f_truck, total_loss, loss_dict

  def estimate_path_flow(self, car_step_size=0.1, truck_step_size=0.1, max_epoch=10, car_init_scale=10,
                         truck_init_scale=1, store_folder=None, use_file_as_init=None,
                         adagrad=False, delta_car_scale=0.1, delta_truck_scale=0.01):
    if use_file_as_init is None:
      (f_car, f_truck) = self.init_path_flow(car_scale=car_init_scale, truck_scale=truck_init_scale)
    elif use_file_as_init.split(os.sep)[-1] == 'init_path_flow.pkl':
      (f_car, f_truck) = pickle.load(open(use_file_as_init, 'rb'))
    else:
      (f_car, f_truck, _) = pickle.load(open(use_file_as_init, 'rb'))
    loss_list = list()
    for i in range(max_epoch):
      if adagrad:
        sum_g_square_car = 1e-6
        sum_g_square_truck = 1e-6
      seq = np.random.permutation(self.num_data)
      loss = float(0)
      # print("Start iteration", time.time())
      loss_dict = {'car_count_loss': 0.0, 'truck_count_loss': 0.0, 'car_tt_loss': 0.0, 'truck_tt_loss': 0.0}
      for j in seq:
        one_data_dict = self._get_one_data(j)
        car_grad, truck_grad, tmp_loss, tmp_loss_dict = self.compute_path_flow_grad_and_loss(one_data_dict, f_car, f_truck,
                                                                                             delta_car_scale=delta_car_scale,
                                                                                             delta_truck_scale=delta_truck_scale)
        # print("gradient", car_grad, truck_grad)
        if adagrad:
          sum_g_square_car = sum_g_square_car + np.power(car_grad, 2)
          sum_g_square_truck = sum_g_square_truck + np.power(truck_grad, 2)
          f_car = f_car - car_step_size * car_grad / np.sqrt(sum_g_square_car) 
          f_truck = f_truck - truck_step_size * truck_grad / np.sqrt(sum_g_square_truck) 
        else:
          f_car -= car_grad * car_step_size / np.sqrt(i+1)
          f_truck -= truck_grad * truck_step_size / np.sqrt(i+1)
        f_car = np.maximum(f_car, 1e-3)
        f_truck = np.maximum(f_truck, 1e-3)
        # f_truck = np.minimum(f_truck, 30)
        loss += tmp_loss
        for loss_type, loss_value in tmp_loss_dict.items():
          loss_dict[loss_type] += loss_value / float(self.num_data)
      print("Epoch:", i, "Loss:", np.round(loss / float(self.num_data),2), self.print_separate_accuracy(loss_dict))
      # print(f_car, f_truck)
      # break
      if store_folder is not None:
        pickle.dump((f_car, f_truck, loss), open(os.path.join(store_folder, str(i) + 'iteration.pickle'), 'wb'))
      loss_list.append([loss, loss_dict])
    return f_car, f_truck, loss_list

  def print_separate_accuracy(self, loss_dict):
    tmp_str = ""
    for loss_type, loss_value in loss_dict.items():
      tmp_str += loss_type + ": " + str(np.round(loss_value, 2)) + "|"
    return tmp_str


class PostProcessing:
    def __init__(self, dode, dta=None, f_car=None, f_truck=None,
                 estimated_car_count=None, estimated_truck_count=None,
                 estimated_car_cost=None, estimated_truck_cost=None,
                 link_length=None,
                 estimated_origin_demand=None,
                 estimated_od_demand=None,
                 result_folder=None):
        self.dode = dode
        self.dta = dta
        self.result_folder = result_folder
        self.one_data_dict = None

        self.observed_link_list = None

        self.f_car = f_car
        self.f_truck = f_truck

        self.color_list = ['teal', 'tomato', 'blue', 'sienna', 'plum', 'red', 'yellowgreen', 'khaki']
        self.marker_list = ["o", "v", "^", "<", ">", "p", "D", "*", "s", "D", "p"]

        self.r2_car_count, self.r2_truck_count = "NA", "NA"
        self.true_car_count, self.estimated_car_count = None, estimated_car_count
        self.true_truck_count, self.estimated_truck_count = None, estimated_truck_count

        self.r2_car_cost, self.r2_truck_cost = "NA", "NA"
        self.true_car_cost, self.estimated_car_cost = None, estimated_car_cost
        self.true_truck_cost, self.estimated_truck_cost = None, estimated_truck_cost

        self.r2_car_speed, self.r2_truck_speed = "NA", "NA"
        self.link_length = link_length
        self.true_car_speed, self.estimated_car_speed = None, None
        self.true_truck_speed, self.estimated_truck_speed = None, None

        self.r2_origin_vehicle_registration = "NA"
        self.true_origin_vehicle_registration, self.estimated_origin_vehicle_registration = None, None
        self.estimated_origin_demand = estimated_origin_demand

        self.r2_od_demand = "NA"
        self.true_od_demand, self.estimated_od_demand = None, estimated_od_demand
        
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

        plt.savefig(os.path.join(self.result_folder, fig_name), bbox_inches='tight')

        plt.show()

    def plot_breakdown_loss(self, loss_list, fig_name = 'breakdown_loss_pathflow.png'):

        if self.dode.config['use_car_link_flow'] + self.dode.config['use_truck_link_flow'] + \
            self.dode.config['use_car_link_tt'] + self.dode.config['use_truck_link_tt'] + self.dode.config['use_origin_vehicle_registration_data']:

            plt.figure(figsize = (16, 9), dpi=300)

            i = 0

            if self.dode.config['use_car_link_flow']:
                plt.plot(np.arange(len(loss_list))+1, list(map(lambda x: x[1]['car_count_loss']/loss_list[0][1]['car_count_loss'], loss_list)),
                        color = self.color_list[i],  marker = self.marker_list[i], linewidth = 3, label = "Car observed flow")
            i += self.dode.config['use_car_link_flow']

            if self.dode.config['use_truck_link_flow']:
                plt.plot(np.arange(len(loss_list))+1, list(map(lambda x: x[1]['truck_count_loss']/loss_list[0][1]['truck_count_loss'], loss_list)),
                        color = self.color_list[i],  marker = self.marker_list[i], linewidth = 3, label = "Truck observed flow")
            i += self.dode.config['use_truck_link_flow']

            if self.dode.config['use_car_link_tt']:
                plt.plot(np.arange(len(loss_list))+1, list(map(lambda x: x[1]['car_tt_loss']/loss_list[0][1]['car_tt_loss'], loss_list)),
                    color = self.color_list[i],  marker = self.marker_list[i], linewidth = 3, label = "Car observed travel cost")
            i += self.dode.config['use_car_link_tt']

            if self.dode.config['use_truck_link_tt']:
                plt.plot(np.arange(len(loss_list))+1, list(map(lambda x: x[1]['truck_tt_loss']/loss_list[0][1]['truck_tt_loss'], loss_list)), 
                        color = self.color_list[i],  marker = self.marker_list[i], linewidth = 3, label = "Truck observed travel cost")
            i += self.dode.config['use_truck_link_tt']

            if self.dode.config['use_origin_vehicle_registration_data']:
                plt.plot(np.arange(len(loss_list))+1, list(map(lambda x: x[1]['origin_vehicle_registration_loss']/loss_list[0][1]['origin_vehicle_registration_loss'], loss_list)), 
                        color = self.color_list[i],  marker = self.marker_list[i], linewidth = 3, label = "Origin vehicle registration data")

                # plt.plot(np.arange(len(loss_list))+1, list(map(lambda x: x[1]['origin_vehicle_registration_loss'], loss_list)), 
                #         color = self.color_list[i],  marker = self.marker_list[i], linewidth = 3, label = "Origin vehicle registration data")

            plt.ylabel('Loss')
            plt.xlabel('Iteration')
            plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 1))
            plt.ylim([0, 1.1])
            plt.xlim([1, len(loss_list)])

            plt.savefig(os.path.join(self.result_folder, fig_name), bbox_inches='tight')

            plt.show()


    def get_one_data(self, start_intervals, end_intervals, j=0):
        assert(len(start_intervals) == len(end_intervals))

        # assume only one observation exists
        self.one_data_dict = self.dode._get_one_data(j)  

        if 'mask_driving_link' not in self.one_data_dict:
            self.one_data_dict['mask_driving_link'] = np.ones(len(self.dode.observed_links) * len(start_intervals), dtype=bool)
        
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

        # travel cost
        if self.dode.config['use_car_link_tt']:
            tt_free = np.tile(list(map(lambda x: self.dode.nb.get_link(x).get_car_fft(), self.dode.observed_links)), (self.dode.num_assign_interval))
        
            self.true_car_cost = np.maximum(self.one_data_dict['car_link_tt'], tt_free)

            self.true_car_speed = self.link_length[np.newaxis, :] / self.true_car_cost.reshape(len(start_intervals), -1) * 3600 # mph
            self.true_car_speed = self.true_car_speed.flatten(order='C')
            
            if self.estimated_car_cost is None:
                # self.estimated_car_cost = self.dta.get_car_link_tt_robust(start_intervals, end_intervals, self.dode.ass_freq, True).flatten(order = 'F')
                self.estimated_car_cost = self.dta.get_car_link_tt(start_intervals, False).flatten(order = 'F')
            
            self.estimated_car_cost = np.maximum(self.estimated_car_cost, tt_free)

            if self.link_length is not None:
                self.estimated_car_speed = self.link_length[np.newaxis, :] / self.estimated_car_cost.reshape(len(start_intervals), -1) * 3600  # mph
                self.estimated_car_speed = self.estimated_car_speed.flatten(order='C')

            self.true_car_cost, self.estimated_car_cost = self.true_car_cost[self.one_data_dict['mask_driving_link']], self.estimated_car_cost[self.one_data_dict['mask_driving_link']]

            if self.link_length is not None:
                self.true_car_speed, self.estimated_car_speed = self.true_car_speed[self.one_data_dict['mask_driving_link']], self.estimated_car_speed[self.one_data_dict['mask_driving_link']]


        if self.dode.config['use_truck_link_tt']:
            tt_free = np.tile(list(map(lambda x: self.dode.nb.get_link(x).get_truck_fft(), self.dode.observed_links)), (self.dode.num_assign_interval))
        
            self.true_truck_cost = np.maximum(self.one_data_dict['truck_link_tt'], tt_free)

            self.true_truck_speed = self.link_length[np.newaxis, :] / self.true_truck_cost.reshape(len(start_intervals), -1) * 3600
            self.true_truck_speed = self.true_truck_speed.flatten(order='C')

            if self.estimated_truck_cost is None:
                # self.estimated_truck_cost = self.dta.get_truck_link_tt_robust(start_intervals, end_intervals, self.dode.ass_freq, True).flatten(order = 'F')
                self.estimated_truck_cost = self.dta.get_truck_link_tt(start_intervals, False).flatten(order = 'F')
            
            self.estimated_truck_cost = np.maximum(self.estimated_truck_cost, tt_free)

            if self.link_length is not None:
                self.estimated_truck_speed = self.link_length[np.newaxis, :] / self.estimated_truck_cost.reshape(len(start_intervals), -1) * 3600 
                self.estimated_truck_speed = self.estimated_truck_speed.flatten(order='C')
            
            self.true_truck_cost, self.estimated_truck_cost = self.true_truck_cost[self.one_data_dict['mask_driving_link']], self.estimated_truck_cost[self.one_data_dict['mask_driving_link']]

            if self.link_length is not None:
                self.true_truck_speed, self.estimated_truck_speed = self.true_truck_speed[self.one_data_dict['mask_driving_link']], self.estimated_truck_speed[self.one_data_dict['mask_driving_link']]

        # origin vehicle registration data
        if self.dode.config['use_origin_vehicle_registration_data']:
            assert(self.f_car is not None and self.f_truck is not None)
            # pandas DataFrame
            self.true_origin_vehicle_registration = self.one_data_dict['origin_vehicle_registration_data']

            if self.estimated_origin_demand is None:
                O_demand_est = self.dode.aggregate_f_to_O_demand(self.f_car, self.f_truck)
            else:
                O_demand_est = self.estimated_origin_demand
            # pandas DataFrame
            self.estimated_origin_vehicle_registration = self.true_origin_vehicle_registration.copy()
            self.estimated_origin_vehicle_registration['car'] = 0.
            self.estimated_origin_vehicle_registration['truck'] = 0.
            def process_one_row_car(row):
                return sum(O_demand_est[Origin_ID][0] for Origin_ID in row['origin_ID'])
            def process_one_row_truck(row):
                return sum(O_demand_est[Origin_ID][1] for Origin_ID in row['origin_ID'])
            self.estimated_origin_vehicle_registration['car'] = self.estimated_origin_vehicle_registration.apply(lambda row: process_one_row_car(row), axis=1)
            self.estimated_origin_vehicle_registration['truck'] = self.estimated_origin_vehicle_registration.apply(lambda row: process_one_row_truck(row), axis=1)
            
            # self.true_origin_vehicle_registration['car'] = self.true_origin_vehicle_registration['car'] * self.dode.config['origin_registration_data_car_weight']
            # self.true_origin_vehicle_registration['truck'] = self.true_origin_vehicle_registration['truck'] * self.dode.config['origin_registration_data_truck_weight']
            self.true_origin_vehicle_registration = self.true_origin_vehicle_registration.loc[:, ['car', 'truck']].sum(axis=1)

            # self.estimated_origin_vehicle_registration['car'] = self.estimated_origin_vehicle_registration['car'] * self.dode.config['origin_registration_data_car_weight']
            # self.estimated_origin_vehicle_registration['truck'] = self.estimated_origin_vehicle_registration['truck'] * self.dode.config['origin_registration_data_truck_weight']
            self.estimated_origin_vehicle_registration = self.estimated_origin_vehicle_registration.loc[:, ['car', 'truck']].sum(axis=1)
        
        if self.dode.config['use_od_demand_data']:
            assert(self.f_car is not None and self.f_truck is not None)
            # pandas DataFrame
            self.true_od_demand = self.one_data_dict['od_demand_data']

            if self.estimated_od_demand is None:
                # dictinory
                OD_demand_est = self.dode.aggregate_f_to_OD_demand(self.f_car, self.f_truck)
            else:
                OD_demand_est = self.estimated_od_demand.copy()

            # pandas DataFrame
            self.estimated_od_demand = self.true_od_demand.copy()
            def process_one_row_car(row):
                return OD_demand_est[(row['origin_ID'], row['destination_ID'])][0]
            def process_one_row_truck(row):
                return OD_demand_est[(row['origin_ID'], row['destination_ID'])][1]
            self.estimated_od_demand['car'] = self.estimated_od_demand.apply(lambda row: process_one_row_car(row), axis=1)
            self.estimated_od_demand['truck'] = self.estimated_od_demand.apply(lambda row: process_one_row_truck(row), axis=1)

            self.estimated_od_demand = self.estimated_od_demand['car'] + self.estimated_od_demand['truck']
            self.estimated_od_demand = np.hstack(self.estimated_od_demand.to_list()) + 1e-6

            # self.true_od_demand = self.true_od_demand['car'] * self.dode.config['od_demand_factor'] + self.true_od_demand['truck'] * self.dode.config['od_demand_factor']
            # self.true_od_demand = np.hstack(self.true_od_demand.to_list())

            self.true_od_demand = self.true_od_demand['car'] + self.true_od_demand['truck']
            self.true_od_demand = np.vstack(self.true_od_demand.to_list()) + 1e-6
            self.true_od_demand *= self.estimated_od_demand.reshape(self.true_od_demand.shape[0], -1) / self.true_od_demand
            self.true_od_demand = self.true_od_demand.flatten(order='C')

        self.observed_link_list = self.dode.observed_links[self.one_data_dict['mask_driving_link'][:len(self.dode.observed_links)]]

    def cal_r2_count(self):

        if self.dode.config['use_car_link_flow']:
            print('----- car count -----')
            print(self.true_car_count)
            print(self.estimated_car_count)
            print('----- car count -----')
            ind = ~(np.isinf(self.true_car_count) + np.isinf(self.estimated_car_count) + np.isnan(self.true_car_count) + np.isnan(self.estimated_car_count))
            self.r2_car_count = r2_score(self.true_car_count[ind], self.estimated_car_count[ind])

        if self.dode.config['use_truck_link_flow']:
            print('----- truck count -----')
            print(self.true_truck_count)
            print(self.estimated_truck_count)
            print('----- truck count -----')
            ind = ~(np.isinf(self.true_truck_count) + np.isinf(self.estimated_truck_count) + np.isnan(self.true_truck_count) + np.isnan(self.estimated_truck_count))
            self.r2_truck_count = r2_score(self.true_truck_count[ind], self.estimated_truck_count[ind])

        if self.dode.config['use_origin_vehicle_registration_data']:
            print('----- origin vehicle registration data count -----')
            print(self.true_origin_vehicle_registration)
            print(self.estimated_origin_vehicle_registration)
            print('----- origin vehicle registration data count -----')
            self.r2_origin_vehicle_registration = r2_score(self.true_origin_vehicle_registration, self.estimated_origin_vehicle_registration)

        if self.dode.config['use_od_demand_data']:
            print('----- od demand data count -----')
            print(self.true_od_demand)
            print(self.estimated_od_demand)
            print('----- od demand data count -----')
            self.r2_od_demand = r2_score(self.true_od_demand, self.estimated_od_demand)

        print("r2 count --- r2_car_count: {}, r2_truck_count: {}, r2_origin_vehicle_registration: {}, r2_od_demand: {}"
            .format(
                self.r2_car_count,
                self.r2_truck_count,
                self.r2_origin_vehicle_registration,
                self.r2_od_demand
                ))
        
        return self.r2_car_count, self.r2_truck_count, self.r2_origin_vehicle_registration

    def scatter_plot_count(self, fig_name='link_flow_scatterplot_pathflow.png'):
        if self.dode.config['use_car_link_flow'] + self.dode.config['use_truck_link_flow'] + self.dode.config['use_origin_vehicle_registration_data'] + self.dode.config['use_od_demand_data'] > 0:
            n = self.dode.config['use_car_link_flow'] + self.dode.config['use_truck_link_flow'] + self.dode.config['use_origin_vehicle_registration_data'] + self.dode.config['use_od_demand_data']
            fig, axes = plt.subplots(1, n, figsize=(9*n, 9), dpi=300, squeeze=False)

            i = 0

            if self.dode.config['use_car_link_flow']:
                ind = ~(np.isinf(self.true_car_count) + np.isinf(self.estimated_car_count) + np.isnan(self.true_car_count) + np.isnan(self.estimated_car_count))
                m_car_max = int(np.max((np.max(self.true_car_count[ind]), np.max(self.estimated_car_count[ind]))) + 1)
                axes[0, i].scatter(self.true_car_count[ind], self.estimated_car_count[ind], color = self.color_list[i], marker = self.marker_list[i], s = 100)
                axes[0, i].plot(range(m_car_max + 1), range(m_car_max + 1), color = 'gray')
                axes[0, i].set_ylabel('Estimated observed flow for car')
                axes[0, i].set_xlabel('True observed flow for car')
                axes[0, i].set_xlim([0, m_car_max])
                axes[0, i].set_ylim([0, m_car_max])
                axes[0, i].set_box_aspect(1)
                axes[0, i].text(0, 1, 'r2 = {:.3f}'.format(self.r2_car_count),
                            horizontalalignment='left',
                            verticalalignment='top',
                            transform=axes[0, i].transAxes)

            i += self.dode.config['use_car_link_flow']

            if self.dode.config['use_truck_link_flow']:
                ind = ~(np.isinf(self.true_truck_count) + np.isinf(self.estimated_truck_count) + np.isnan(self.true_truck_count) + np.isnan(self.estimated_truck_count))
                m_truck_max = int(np.max((np.max(self.true_truck_count[ind]), np.max(self.estimated_truck_count[ind]))) + 1)
                axes[0, i].scatter(self.true_truck_count[ind], self.estimated_truck_count[ind], color = self.color_list[i], marker = self.marker_list[i], s = 100)
                axes[0, i].plot(range(m_truck_max + 1), range(m_truck_max + 1), color = 'gray')
                axes[0, i].set_ylabel('Estimated observed flow for truck')
                axes[0, i].set_xlabel('True observed flow for truck')
                axes[0, i].set_xlim([0, m_truck_max])
                axes[0, i].set_ylim([0, m_truck_max])
                axes[0, i].set_box_aspect(1)
                axes[0, i].text(0, 1, 'r2 = {:.3f}'.format(self.r2_truck_count),
                            horizontalalignment='left',
                            verticalalignment='top',
                            transform=axes[0, i].transAxes)

            i += self.dode.config['use_truck_link_flow']

            if self.dode.config['use_origin_vehicle_registration_data']:
                m_max = int(np.max((np.max(self.true_origin_vehicle_registration), np.max(self.estimated_origin_vehicle_registration))) + 1)
                axes[0, i].scatter(self.true_origin_vehicle_registration, self.estimated_origin_vehicle_registration, color = self.color_list[i], marker = self.marker_list[i], s = 100)
                axes[0, i].plot(range(m_max + 1), range(m_max + 1), color = 'gray')
                axes[0, i].set_ylabel('Estimated origin vehicle registration count')
                axes[0, i].set_xlabel('True origin vehicle registration count')
                axes[0, i].set_xlim([0, m_max])
                axes[0, i].set_ylim([0, m_max])
                axes[0, i].set_box_aspect(1)
                axes[0, i].text(0, 1, 'r2 = {:.3f}'.format(self.r2_origin_vehicle_registration),
                            horizontalalignment='left',
                            verticalalignment='top',
                            transform=axes[0, i].transAxes)
                
            i += self.dode.config['use_origin_vehicle_registration_data']

            if self.dode.config['use_od_demand_data']:
                m_max = int(np.max((np.max(self.true_od_demand), np.max(self.estimated_od_demand))) + 1)
                axes[0, i].scatter(self.true_od_demand, self.estimated_od_demand, color = self.color_list[i], marker = self.marker_list[i], s = 100)
                axes[0, i].plot(range(m_max + 1), range(m_max + 1), color = 'gray')
                axes[0, i].set_ylabel('Estimated OD demand')
                axes[0, i].set_xlabel('True OD demand')
                axes[0, i].set_xlim([0, m_max])
                axes[0, i].set_ylim([0, m_max])
                axes[0, i].set_box_aspect(1)
                axes[0, i].text(0, 1, 'r2 = {:.3f}'.format(self.r2_od_demand),
                            horizontalalignment='left',
                            verticalalignment='top',
                            transform=axes[0, i].transAxes)

            plt.savefig(os.path.join(self.result_folder, fig_name), bbox_inches='tight')

            plt.show()
            plt.close()

    def scatter_plot_count_by_link(self, link_list=None, fig_name= 'link_flow_scatterplot_pathflow_by_link'):
        if self.dode.config['use_car_link_flow'] + self.dode.config['use_truck_link_flow']:
            if link_list is None:
                link_list = self.observed_link_list
            if self.dode.config['use_car_link_flow']:
                true_car_count = self.true_car_count.reshape(self.dode.num_assign_interval, -1) 
                estimated_car_count = self.estimated_car_count.reshape(self.dode.num_assign_interval, -1) 
            if self.dode.config['use_truck_link_flow']:
                true_truck_count = self.true_truck_count.reshape(self.dode.num_assign_interval, -1) 
                estimated_truck_count = self.estimated_truck_count.reshape(self.dode.num_assign_interval, -1) 

            n = self.dode.config['use_car_link_flow'] + self.dode.config['use_truck_link_flow']

            for li, link in enumerate(link_list):
                if np.sum(np.array(self.observed_link_list) == link) == 0:
                    print("Link {} is not in the observed link list.".format(link))
                    continue
                fig, axes = plt.subplots(1, n, figsize=(9*n, 9), dpi=300, squeeze=False)
                i = 0
                
                if self.dode.config['use_car_link_flow']:
                    link_true_car_count = true_car_count[:, np.where(np.array(self.observed_link_list) == link)[0][0]]
                    link_estimated_car_count = estimated_car_count[:, np.where(np.array(self.observed_link_list) == link)[0][0]]
                    ind = ~(np.isinf(link_true_car_count) + np.isinf(link_estimated_car_count) + np.isnan(link_true_car_count) + np.isnan(link_estimated_car_count))
                    m_car_max = int(np.max((np.max(link_true_car_count[ind]), np.max(link_estimated_car_count[ind]))) + 1) if any(ind) else 20
                    axes[0, i].scatter(link_true_car_count[ind], link_estimated_car_count[ind], color = self.color_list[i], marker = self.marker_list[i], s = 100)
                    axes[0, i].plot(range(m_car_max + 1), range(m_car_max + 1), color = 'gray')
                    axes[0, i].set_ylabel('Estimated observed flow for car')
                    axes[0, i].set_xlabel('True observed flow for car')
                    axes[0, i].set_xlim([0, m_car_max])
                    axes[0, i].set_ylim([0, m_car_max])
                    axes[0, i].set_box_aspect(1)
                    axes[0, i].text(0, 1, 'link {}, car count r2 = {:.3f}'.format(link, r2_score(link_true_car_count[ind], link_estimated_car_count[ind]) if any(ind) else 0.),
                                horizontalalignment='left',
                                verticalalignment='top',
                                transform=axes[0, i].transAxes)

                i += self.dode.config['use_car_link_flow']

                if self.dode.config['use_truck_link_flow']:
                    link_true_truck_count = true_truck_count[:, np.where(np.array(self.observed_link_list) == link)[0][0]]
                    link_estimated_truck_count = estimated_truck_count[:, np.where(np.array(self.observed_link_list) == link)[0][0]]
                    ind = ~(np.isinf(link_true_truck_count) + np.isinf(link_estimated_truck_count) + np.isnan(link_true_truck_count) + np.isnan(link_estimated_truck_count))
                    m_truck_max = int(np.max((np.max(link_true_truck_count[ind]), np.max(link_estimated_truck_count[ind]))) + 1) if any(ind) else 20
                    axes[0, i].scatter(link_true_truck_count[ind], link_estimated_truck_count[ind], color = self.color_list[i], marker = self.marker_list[i], s = 100)
                    axes[0, i].plot(range(m_truck_max + 1), range(m_truck_max + 1), color = 'gray')
                    axes[0, i].set_ylabel('Estimated observed flow for truck')
                    axes[0, i].set_xlabel('True observed flow for truck')
                    axes[0, i].set_xlim([0, m_truck_max])
                    axes[0, i].set_ylim([0, m_truck_max])
                    axes[0, i].set_box_aspect(1)
                    axes[0, i].text(0, 1, 'link {}, truck count r2 = {:.3f}'.format(link, r2_score(link_true_truck_count[ind], link_estimated_truck_count[ind]) if any(ind) else 0.),
                                horizontalalignment='left',
                                verticalalignment='top',
                                transform=axes[0, i].transAxes)

                plt.savefig(os.path.join(self.result_folder, fig_name + "_{}.png".format(link)), bbox_inches='tight')

                plt.show()
                plt.close()

    def cal_r2_cost(self):
        if self.dode.config['use_car_link_tt']:
            print('----- car cost -----')
            print(self.true_car_cost)
            print(self.estimated_car_cost)
            print('----- car cost -----')
            # ind = ~(np.isinf(self.true_car_cost) + np.isinf(self.estimated_car_cost) + np.isnan(self.true_car_cost) + np.isnan(self.estimated_car_cost) + (self.estimated_car_cost > 3 * self.true_car_cost))
            ind = ~(np.isinf(self.true_car_cost) + np.isinf(self.estimated_car_cost) + np.isnan(self.true_car_cost) + np.isnan(self.estimated_car_cost))
            self.r2_car_cost = r2_score(self.true_car_cost[ind], self.estimated_car_cost[ind])

        if self.dode.config['use_truck_link_tt']:
            print('----- truck cost -----')
            print(self.true_truck_cost)
            print(self.estimated_truck_cost)
            print('----- truck cost -----')
            # ind = ~(np.isinf(self.true_truck_cost) + np.isinf(self.estimated_truck_cost) + np.isnan(self.true_truck_cost) + np.isnan(self.estimated_truck_cost) + (self.estimated_truck_cost > 3 * self.true_truck_cost))
            ind = ~(np.isinf(self.true_truck_cost) + np.isinf(self.estimated_truck_cost) + np.isnan(self.true_truck_cost) + np.isnan(self.estimated_truck_cost))
            self.r2_truck_cost = r2_score(self.true_truck_cost[ind], self.estimated_truck_cost[ind])

        print("r2 cost --- r2_car_cost: {}, r2_truck_cost: {}"
            .format(
                self.r2_car_cost, 
                self.r2_truck_cost
                ))

        return self.r2_car_cost, self.r2_truck_cost

    def cal_r2_speed(self):
        if self.dode.config['use_car_link_tt']:
            if self.true_car_speed is not None and self.estimated_car_speed is not None:
                print('----- car speed -----')
                print(self.true_car_speed)
                print(self.estimated_car_speed)
                print('----- car speed -----')
                # ind = ~((self.true_car_speed > 90) + (self.estimated_car_speed > 90) + (self.true_car_speed < 10) + (self.estimated_car_speed < 10) + np.isnan(self.true_car_speed) + np.isnan(self.estimated_car_speed) + (self.estimated_car_speed > 3 * self.true_car_speed)
                #         + np.isinf(self.true_car_cost) + np.isinf(self.estimated_car_cost) + np.isnan(self.true_car_cost) + np.isnan(self.estimated_car_cost) + (self.estimated_car_cost > 3 * self.true_car_cost))
                ind = ~(np.isnan(self.true_car_speed) + np.isnan(self.estimated_car_speed) + np.isinf(self.true_car_cost) + np.isinf(self.estimated_car_cost) + np.isnan(self.true_car_cost) + np.isnan(self.estimated_car_cost))
                self.r2_car_speed = r2_score(self.true_car_speed[ind], self.estimated_car_speed[ind])
            else:
                print("Car speed not calculated.")

        if self.dode.config['use_truck_link_tt']:
            if self.true_truck_speed is not None and self.estimated_truck_speed is not None:
                print('----- truck speed -----')
                print(self.true_truck_speed)
                print(self.estimated_truck_speed)
                print('----- truck speed -----')
                # ind = ~((self.true_truck_speed > 90) + (self.estimated_truck_speed > 90) + (self.true_truck_speed < 10) + (self.estimated_truck_speed < 10) + np.isnan(self.true_truck_speed) + np.isnan(self.estimated_truck_speed) + (self.estimated_truck_speed > 3 * self.true_truck_speed)
                #         + np.isinf(self.true_truck_cost) + np.isinf(self.estimated_truck_cost) + np.isnan(self.true_truck_cost) + np.isnan(self.estimated_truck_cost) + (self.estimated_truck_cost > 3 * self.true_truck_cost))
                ind = ~(np.isnan(self.true_truck_speed) + np.isnan(self.estimated_truck_speed) + np.isinf(self.true_truck_cost) + np.isinf(self.estimated_truck_cost) + np.isnan(self.true_truck_cost) + np.isnan(self.estimated_truck_cost))
                self.r2_truck_speed = r2_score(self.true_truck_speed[ind], self.estimated_truck_speed[ind])
            else:
                print("Truck speed not calculated.")

        print("r2 speed --- r2_car_speed: {}, r2_truck_speed: {}"
            .format(
                self.r2_car_speed, 
                self.r2_truck_speed
                ))

        return self.r2_car_speed, self.r2_truck_speed

    def scatter_plot_cost(self, fig_name='link_cost_scatterplot_pathflow.png'):
        if self.dode.config['use_car_link_tt'] + self.dode.config['use_truck_link_tt']:
            n = self.dode.config['use_car_link_tt'] + self.dode.config['use_truck_link_tt']
            fig, axes = plt.subplots(1, n, figsize=(9*n, 9), dpi=300, squeeze=False)
            
            i = 0

            if self.dode.config['use_car_link_tt']:
                # ind = ~(np.isinf(self.true_car_cost) + np.isinf(self.estimated_car_cost) + np.isnan(self.true_car_cost) + np.isnan(self.estimated_car_cost) + (self.estimated_car_cost > 3 * self.true_car_cost))
                ind = ~(np.isinf(self.true_car_cost) + np.isinf(self.estimated_car_cost) + np.isnan(self.true_car_cost) + np.isnan(self.estimated_car_cost))
                car_tt_min = np.min((np.min(self.true_car_cost[ind]), np.min(self.estimated_car_cost[ind]))) - 1
                car_tt_max = np.max((np.max(self.true_car_cost[ind]), np.max(self.estimated_car_cost[ind]))) + 1
                axes[0, i].scatter(self.true_car_cost[ind], self.estimated_car_cost[ind], color = self.color_list[i], marker = self.marker_list[i], s = 100)
                axes[0, i].plot(np.linspace(car_tt_min, car_tt_max, 20), np.linspace(car_tt_min, car_tt_max, 20), color = 'gray')
                axes[0, i].set_ylabel('Estimated observed travel cost for car')
                axes[0, i].set_xlabel('True observed travel cost for car')
                axes[0, i].set_xlim([car_tt_min, car_tt_max])
                axes[0, i].set_ylim([car_tt_min, car_tt_max])
                axes[0, i].set_box_aspect(1)
                axes[0, i].text(0, 1, 'r2 = {:.3f}'.format(self.r2_car_cost),
                            horizontalalignment='left',
                            verticalalignment='top',
                            transform=axes[0, i].transAxes)

            i += self.dode.config['use_car_link_tt']

            if self.dode.config['use_truck_link_tt']:
                # ind = ~(np.isinf(self.true_truck_cost) + np.isinf(self.estimated_truck_cost) + np.isnan(self.true_truck_cost) + np.isnan(self.estimated_truck_cost) + (self.estimated_truck_cost > 3 * self.true_truck_cost))
                ind = ~(np.isinf(self.true_truck_cost) + np.isinf(self.estimated_truck_cost) + np.isnan(self.true_truck_cost) + np.isnan(self.estimated_truck_cost))
                truck_tt_min = np.min((np.min(self.true_truck_cost[ind]), np.min(self.estimated_truck_cost[ind]))) - 1
                truck_tt_max = np.max((np.max(self.true_truck_cost[ind]), np.max(self.estimated_truck_cost[ind]))) + 1
                axes[0, i].scatter(self.true_truck_cost[ind], self.estimated_truck_cost[ind], color = self.color_list[i], marker = self.marker_list[i], s = 100)
                axes[0, i].plot(np.linspace(truck_tt_min, truck_tt_max, 20), np.linspace(truck_tt_min, truck_tt_max, 20), color = 'gray')
                axes[0, i].set_ylabel('Estimated observed travel cost for truck')
                axes[0, i].set_xlabel('True observed travel cost for truck')
                axes[0, i].set_xlim([truck_tt_min, truck_tt_max])
                axes[0, i].set_ylim([truck_tt_min, truck_tt_max])
                axes[0, i].set_box_aspect(1)
                axes[0, i].text(0, 1, 'r2 = {:.3f}'.format(self.r2_truck_cost),
                            horizontalalignment='left',
                            verticalalignment='top',
                            transform=axes[0, i].transAxes)

            plt.savefig(os.path.join(self.result_folder, fig_name), bbox_inches='tight')

            plt.show()
            plt.close()

    def scatter_plot_cost_by_link(self, link_list=None, fig_name='link_cost_scatterplot_pathflow_by_link'):
        if self.dode.config['use_car_link_tt'] + self.dode.config['use_truck_link_tt']:
            if link_list is None:
                link_list = self.dode.observed_links
            if self.dode.config['use_car_link_tt']:
                true_car_cost = self.true_car_cost.reshape(self.dode.num_assign_interval, -1) 
                estimated_car_cost = self.estimated_car_cost.reshape(self.dode.num_assign_interval, -1) 
            if self.dode.config['use_truck_link_tt']:
                true_truck_cost = self.true_truck_cost.reshape(self.dode.num_assign_interval, -1) 
                estimated_truck_cost = self.estimated_truck_cost.reshape(self.dode.num_assign_interval, -1) 

            n = self.dode.config['use_car_link_tt'] + self.dode.config['use_truck_link_tt']
            
            for li, link in enumerate(link_list):
                if np.sum(np.array(self.observed_link_list) == link) == 0:
                    print("Link {} is not in the observed link list.".format(link))
                    continue
                fig, axes = plt.subplots(1, n, figsize=(9*n, 9), dpi=300, squeeze=False)
                i = 0

                if self.dode.config['use_car_link_tt']:
                    link_true_car_cost = true_car_cost[:, np.where(np.array(self.observed_link_list) == link)[0][0]]
                    link_estimated_car_cost = estimated_car_cost[:, np.where(np.array(self.observed_link_list) == link)[0][0]]
                    ind = ~(np.isinf(link_true_car_cost) + np.isinf(link_estimated_car_cost) + np.isnan(link_true_car_cost) + np.isnan(link_estimated_car_cost))
                    car_tt_min = np.min((np.min(link_true_car_cost[ind]), np.min(link_estimated_car_cost[ind]))) - 1 if any(ind) else 0
                    car_tt_max = np.max((np.max(link_true_car_cost[ind]), np.max(link_estimated_car_cost[ind]))) + 1 if any(ind) else 80
                    axes[0, i].scatter(link_true_car_cost[ind], link_estimated_car_cost[ind], color = self.color_list[i], marker = self.marker_list[i], s = 100)
                    axes[0, i].plot(np.linspace(car_tt_min, car_tt_max, 20), np.linspace(car_tt_min, car_tt_max, 20), color = 'gray')
                    axes[0, i].set_ylabel('Estimated observed travel cost for car')
                    axes[0, i].set_xlabel('True observed travel cost for car')
                    axes[0, i].set_xlim([car_tt_min, car_tt_max])
                    axes[0, i].set_ylim([car_tt_min, car_tt_max])
                    axes[0, i].set_box_aspect(1)
                    axes[0, i].text(0, 1, 'link {}, car cost r2 = {:.3f}'.format(link, r2_score(link_true_car_cost[ind], link_estimated_car_cost[ind]) if any(ind) else 0.),
                                horizontalalignment='left',
                                verticalalignment='top',
                                transform=axes[0, i].transAxes)

                i += self.dode.config['use_car_link_tt']

                if self.dode.config['use_truck_link_tt']:
                    link_true_truck_cost = true_truck_cost[:, np.where(np.array(self.observed_link_list) == link)[0][0]]
                    link_estimated_truck_cost = estimated_truck_cost[:, np.where(np.array(self.observed_link_list) == link)[0][0]]
                    ind = ~(np.isinf(link_true_truck_cost) + np.isinf(link_estimated_truck_cost) + np.isnan(link_true_truck_cost) + np.isnan(link_estimated_truck_cost))
                    truck_tt_min = np.min((np.min(link_true_truck_cost[ind]), np.min(link_estimated_truck_cost[ind]))) - 1 if any(ind) else 0
                    truck_tt_max = np.max((np.max(link_true_truck_cost[ind]), np.max(link_estimated_truck_cost[ind]))) + 1 if any(ind) else 80
                    axes[0, i].scatter(link_true_truck_cost[ind], link_estimated_truck_cost[ind], color = self.color_list[i], marker = self.marker_list[i], s = 100)
                    axes[0, i].plot(np.linspace(truck_tt_min, truck_tt_max, 20), np.linspace(truck_tt_min, truck_tt_max, 20), color = 'gray')
                    axes[0, i].set_ylabel('Estimated observed travel cost for truck')
                    axes[0, i].set_xlabel('True observed travel cost for truck')
                    axes[0, i].set_xlim([truck_tt_min, truck_tt_max])
                    axes[0, i].set_ylim([truck_tt_min, truck_tt_max])
                    axes[0, i].set_box_aspect(1)
                    axes[0, i].text(0, 1, 'link {}, truck cost r2 = {:.3f}'.format(link, r2_score(link_true_truck_cost[ind], link_estimated_truck_cost[ind]) if any(ind) else 0.),
                                horizontalalignment='left',
                                verticalalignment='top',
                                transform=axes[0, i].transAxes)

                plt.savefig(os.path.join(self.result_folder, fig_name + "_{}.png".format(link)), bbox_inches='tight')

                plt.show()
                plt.close()

    def scatter_plot_speed(self, fig_name='link_speed_scatterplot_pathflow.png'):

        if self.dode.config['use_car_link_tt'] + self.dode.config['use_truck_link_tt']:
            n = self.dode.config['use_car_link_tt'] + self.dode.config['use_truck_link_tt']
            fig, axes = plt.subplots(1, n, figsize=(9*n, 9), dpi=300, squeeze=False)
            
            i = 0

            if self.dode.config['use_car_link_tt'] and (self.true_car_speed is not None and self.estimated_car_speed is not None):
                # ind = ~((self.true_car_speed > 90) + (self.estimated_car_speed > 90) + (self.true_car_speed < 10) + (self.estimated_car_speed < 10) + np.isnan(self.true_car_speed) + np.isnan(self.estimated_car_speed) + (self.estimated_car_speed > 3 * self.true_car_speed)
                #          + np.isinf(self.true_car_cost) + np.isinf(self.estimated_car_cost) + np.isnan(self.true_car_cost) + np.isnan(self.estimated_car_cost) + (self.estimated_car_cost > 3 * self.true_car_cost))
                ind = ~(np.isnan(self.true_car_speed) + np.isnan(self.estimated_car_speed) + np.isinf(self.true_car_cost) + np.isinf(self.estimated_car_cost) + np.isnan(self.true_car_cost) + np.isnan(self.estimated_car_cost))
            
                car_speed_min = np.min((np.min(self.true_car_speed[ind]), np.min(self.estimated_car_speed[ind]))) - 1
                car_speed_max = np.max((np.max(self.true_car_speed[ind]), np.max(self.estimated_car_speed[ind]))) + 1
                axes[0, i].scatter(self.true_car_speed[ind], self.estimated_car_speed[ind], color = self.color_list[i], marker = self.marker_list[i], s = 100)
                axes[0, i].plot(np.linspace(car_speed_min, car_speed_max, 20), np.linspace(car_speed_min, car_speed_max, 20), color = 'gray')
                axes[0, i].set_ylabel('Estimated observed link speed for car')
                axes[0, i].set_xlabel('True observed link speed for car')
                axes[0, i].set_xlim([car_speed_min, car_speed_max])
                axes[0, i].set_ylim([car_speed_min, car_speed_max])
                axes[0, i].set_box_aspect(1)
                axes[0, i].text(0, 1, 'r2 = {:.3f}'.format(self.r2_car_speed),
                            horizontalalignment='left',
                            verticalalignment='top',
                            transform=axes[0, i].transAxes)

            i += self.dode.config['use_car_link_tt']

            if self.dode.config['use_truck_link_tt'] and (self.true_truck_speed is not None and self.estimated_truck_speed is not None):
                # ind = ~((self.true_truck_speed > 90) + (self.estimated_truck_speed > 90) + (self.true_truck_speed < 10) + (self.estimated_truck_speed < 10) + np.isnan(self.true_truck_speed) + np.isnan(self.estimated_truck_speed) + (self.estimated_truck_speed > 3 * self.true_truck_speed)
                #         + np.isinf(self.true_truck_cost) + np.isinf(self.estimated_truck_cost) + np.isnan(self.true_truck_cost) + np.isnan(self.estimated_truck_cost) + (self.estimated_truck_cost > 3 * self.true_truck_cost))
                ind = ~(np.isnan(self.true_truck_speed) + np.isnan(self.estimated_truck_speed) + np.isinf(self.true_truck_cost) + np.isinf(self.estimated_truck_cost) + np.isnan(self.true_truck_cost) + np.isnan(self.estimated_truck_cost))
            
                truck_speed_min = np.min((np.min(self.true_truck_speed[ind]), np.min(self.estimated_truck_speed[ind]))) - 1
                truck_speed_max = np.max((np.max(self.true_truck_speed[ind]), np.max(self.estimated_truck_speed[ind]))) + 1
                axes[0, i].scatter(self.true_truck_speed[ind], self.estimated_truck_speed[ind], color = self.color_list[i], marker = self.marker_list[i], s = 100)
                axes[0, i].plot(np.linspace(truck_speed_min, truck_speed_max, 20), np.linspace(truck_speed_min, truck_speed_max, 20), color = 'gray')
                axes[0, i].set_ylabel('Estimated observed link speed for truck')
                axes[0, i].set_xlabel('True observed link speed for truck')
                axes[0, i].set_xlim([truck_speed_min, truck_speed_max])
                axes[0, i].set_ylim([truck_speed_min, truck_speed_max])
                axes[0, i].set_box_aspect(1)
                axes[0, i].text(0, 1, 'r2 = {:.3f}'.format(self.r2_truck_speed),
                            horizontalalignment='left',
                            verticalalignment='top',
                            transform=axes[0, i].transAxes)

            plt.savefig(os.path.join(self.result_folder, fig_name), bbox_inches='tight')

            plt.show()
            plt.close()

    def scatter_plot_speed_by_link(self, link_list=None, fig_name='link_speed_scatterplot_pathflow_by_link'):
        if self.dode.config['use_car_link_tt'] + self.dode.config['use_truck_link_tt']:
            if link_list is None:
                link_list = self.dode.observed_links
            if self.dode.config['use_car_link_tt']:
                true_car_speed = self.true_car_speed.reshape(self.dode.num_assign_interval, -1) 
                estimated_car_speed = self.estimated_car_speed.reshape(self.dode.num_assign_interval, -1) 
            if self.dode.config['use_truck_link_tt']:
                true_truck_speed = self.true_truck_speed.reshape(self.dode.num_assign_interval, -1) 
                estimated_truck_speed = self.estimated_truck_speed.reshape(self.dode.num_assign_interval, -1) 

            n = self.dode.config['use_car_link_tt'] + self.dode.config['use_truck_link_tt']
            n_actual = 0
            for li, link in enumerate(link_list):
                if np.sum(np.array(self.observed_link_list) == link) == 0:
                    print("Link {} is not in the observed link list.".format(link))
                    continue
                fig, axes = plt.subplots(1, n, figsize=(9*n, 9), dpi=300, squeeze=False)
                i = 0

                if self.dode.config['use_car_link_tt']:
                    link_true_car_speed = true_car_speed[:, np.where(np.array(self.observed_link_list) == link)[0][0]]
                    link_estimated_car_speed = estimated_car_speed[:, np.where(np.array(self.observed_link_list) == link)[0][0]]
                    ind = ~(np.isinf(link_true_car_speed) + np.isinf(link_estimated_car_speed) + np.isnan(link_true_car_speed) + np.isnan(link_estimated_car_speed))
                    if sum(ind) > 0:
                        car_speed_min = np.min((np.min(link_true_car_speed[ind]), np.min(link_estimated_car_speed[ind]))) - 1
                        car_speed_max = np.max((np.max(link_true_car_speed[ind]), np.max(link_estimated_car_speed[ind]))) + 1
                        axes[0, i].scatter(link_true_car_speed[ind], link_estimated_car_speed[ind], color = self.color_list[i], marker = self.marker_list[i], s = 100)
                        axes[0, i].plot(np.linspace(car_speed_min, car_speed_max, 20), np.linspace(car_speed_min, car_speed_max, 20), color = 'gray')
                        axes[0, i].set_ylabel('Estimated observed link speed for car')
                        axes[0, i].set_xlabel('True observed link speed for car')
                        axes[0, i].set_xlim([car_speed_min, car_speed_max])
                        axes[0, i].set_ylim([car_speed_min, car_speed_max])
                        axes[0, i].set_box_aspect(1)
                        axes[0, i].text(0, 1, 'link {}, car speed r2 = {:.3f}'.format(link, r2_score(link_true_car_speed[ind], link_estimated_car_speed[ind]) if any(ind) else 0.),
                                    horizontalalignment='left',
                                    verticalalignment='top',
                                    transform=axes[0, i].transAxes)
                        n_actual += 1
                    else:
                        print(f"No valid data for link {link} for car speed.")

                i += self.dode.config['use_car_link_tt']

                if self.dode.config['use_truck_link_tt']:
                    link_true_truck_speed = true_truck_speed[:, np.where(np.array(self.observed_link_list) == link)[0][0]]
                    link_estimated_truck_speed = estimated_truck_speed[:, np.where(np.array(self.observed_link_list) == link)[0][0]]
                    ind = ~(np.isinf(link_true_truck_speed) + np.isinf(link_estimated_truck_speed) + np.isnan(link_true_truck_speed) + np.isnan(link_estimated_truck_speed))
                    if sum(ind) > 0:
                        truck_speed_min = np.min((np.min(link_true_truck_speed[ind]), np.min(link_estimated_truck_speed[ind]))) - 1 if any(ind) else 0
                        truck_speed_max = np.max((np.max(link_true_truck_speed[ind]), np.max(link_estimated_truck_speed[ind]))) + 1 if any(ind) else 80
                        axes[0, i].scatter(link_true_truck_speed[ind], link_estimated_truck_speed[ind], color = self.color_list[i], marker = self.marker_list[i], s = 100)
                        axes[0, i].plot(np.linspace(truck_speed_min, truck_speed_max, 20), np.linspace(truck_speed_min, truck_speed_max, 20), color = 'gray')
                        axes[0, i].set_ylabel('Estimated observed travel speed for truck')
                        axes[0, i].set_xlabel('True observed travel speed for truck')
                        axes[0, i].set_xlim([truck_speed_min, truck_speed_max])
                        axes[0, i].set_ylim([truck_speed_min, truck_speed_max])
                        axes[0, i].set_box_aspect(1)
                        axes[0, i].text(0, 1, 'link {}, truck speed r2 = {:.3f}'.format(link, r2_score(link_true_truck_speed[ind], link_estimated_truck_speed[ind]) if any(ind) else 0.),
                                    horizontalalignment='left',
                                    verticalalignment='top',
                                    transform=axes[0, i].transAxes)
                        n_actual += 1
                    else:
                        print(f"No valid data for link {link} for truck speed.")                       

                if n_actual > 0:
                    plt.savefig(os.path.join(self.result_folder, fig_name + "_{}.png".format(link)), bbox_inches='tight')

                    plt.show()
                    plt.close()


# def r2_score(y_true, y_pred):
#     # Check if inputs are numpy arrays, if not, convert them
#     y_true = np.asarray(y_true)
#     y_pred = np.asarray(y_pred)
    
#     # Check if y_true and y_pred are of the same length
#     if y_true.shape[0] != y_pred.shape[0]:
#         raise ValueError("y_true and y_pred must have the same length.")
    
#     # Check for empty input
#     if y_true.size == 0 or y_pred.size == 0:
#         raise ValueError("y_true and y_pred must not be empty.")
    
#     # Calculate the mean of y_true
#     y_mean = np.mean(y_true)
    
#     # Calculate the total sum of squares (variance of y_true)
#     total_sum_of_squares = np.sum((y_true - y_mean) ** 2)
    
#     # Handle case where y_true has no variance (all values are the same)
#     if total_sum_of_squares == 0:
#         return 1.0 if np.array_equal(y_true, y_pred) else 0.0
    
#     # Calculate the residual sum of squares
#     residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
    
#     # Calculate R^2
#     r2_score = 1 - (residual_sum_of_squares / total_sum_of_squares)
#     return r2_score
