import math
import random
import numpy as np
import torch
import sys
from surfree_utils.attack import get_init_with_noise

import surfree_utils.dct as torch_dct
from surfree_utils.utils import atleast_kdim


def distance(a, b):
    return (a - b).flatten(1).norm(dim=1)

class SurFree():
    def __init__(
        self, 
        steps: int = 100, 
        history_distance_with=None,
        save_adv_each=200000,
        max_queries: int = 5000,
        BS_gamma: float = 0.01, 
        BS_max_iteration: int = 7, 
        theta_max: float = 30, 
        n_ortho: int = 100, 
        rho: float = 0.95,  
        T: int = 1, 
        boundary: float = 1,
        last_boundary = None,
        increase: bool = False,
        with_alpha_line_search: bool = True, 
        with_distance_line_search: bool = False, 
        with_interpolation: bool = False,
        final_line_search: bool=True,
        quantification=True,
        clip=True):
        """
        Args:
            steps (int, optional): run steps. Defaults to 1000.
            max_queries (int, optional): stop running when each example require max_queries.
            BS_gamma ([type], optional): Binary Search Early Stop. Stop when precision is below BS_gamma. Defaults to 0.01.
            BS_max_iteration ([type], optional): Max iteration for . Defaults to 10.
            theta_max (int, optional): max theta watched to evaluate the direction. Defaults to 30.
            evolution (str, optional): Move in this direction. It can be linear or circular. Defaults to "circular".
            n_ortho (int, optional): Orthogonalize with the n last directions. Defaults to 100.
            rho (float, optional): Malus/Bonus factor given to the theta_max. Defaults to 0.98.
            T (int, optional): How many evaluation done to evaluated a direction. Defaults to 1.
            with_alpha_line_search (bool, optional): Activate Binary Search on Theta. Defaults to True.
            with_distance_line_search (bool, optional): Activate Binary Search between adversarial and x_o. Defaults to False.
            with_interpolation (bool, optional): Activate Interpolation. Defaults to False.
        """
        # Attack Parameters
        self._BS_gamma = BS_gamma
        self.history_distance_with = history_distance_with

        self._BS_max_iteration = BS_max_iteration

        self._steps = steps
        self._max_queries = max_queries
        self.best_advs = None
        self._theta_max = theta_max
        self.rho = rho
        self.T = T
        assert self.rho <= 1 and self.rho > 0
        self.clip = clip
        self.boundary = boundary
        self.last_boundary = last_boundary
        self.increase = increase

        # Add or remove some parts of the attack
        self.with_alpha_line_search = with_alpha_line_search
        self.with_distance_line_search = with_distance_line_search
        self.with_interpolation = with_interpolation
        self.final_line_search = final_line_search
        self.quantification = quantification
        if self.with_interpolation and not self.with_distance_line_search:
            Warning("It's higly recommended to use Interpolation with distance line search.")

        # Data saved during attack
        self.n_ortho = n_ortho
        self._directions_ortho = {}
        self._nqueries = {}
        self._history = {}
        self._extra_history = {}
        self._alpha_history = {}

        self.save_adv_each = save_adv_each
        self._advs_saved = {}
        self._images_finished = None

        if self.increase:
            self.max_MOS = 100
        else:
            self.max_MOS = 0

    def _quantify(self, x):
        return (x * 255).round() / 255

    def get_nqueries(self): 
        nqueries = {i:0 for i in range(self.original_len)}
        count = 0
        for i,stop_bool in enumerate(self.stop_attack):
            if not stop_bool:
                nqueries[i] = self._nqueries[count]
                count += 1
        return nqueries
    
    def remove_stoped(self, X, stop_attack):
        return X[~stop_attack]
    
    def recover_stoped(self, X, removed_X, stop_attack):
        best_advs = X.clone().cuda()
        best_advs[~stop_attack] = removed_X.clone()
        return best_advs
        
    def __call__(self, model, X, labels, starting_points=None, ref=None, X_JND=None, \
                X_edges=None, X_sals=None, X_ori=None, stop_attack=None, already_rand_t=None, **kwargs):

        self._nqueries = torch.zeros(len(X)).to(X.device)
        self._history  = {i: [] for i in range(len(X))}
        self.theta_max = torch.ones(len(X)).to(X.device) * self._theta_max

        self._model = model
        self.original_len = X.shape[0]

        # Get Starting Point
        self.best_advs, self.boundary, stop_attack, already_rand_t = get_init_with_noise(model, X, labels, \
            X_ori=X_ori, max_MOS=self.max_MOS, ref=ref, boundary=self.boundary, \
            last_boundary=self.last_boundary, X_JND=X_JND, X_edges=X_edges, \
            X_sals = X_sals, stop_attack=stop_attack, increase=self.increase, already_rand_t=already_rand_t) \
            if starting_points is None else starting_points

        self.X = self.remove_stoped(X,stop_attack)
        self.X_JND = self.remove_stoped(X_JND,stop_attack)
        self.X_ori = self.remove_stoped(X_ori,stop_attack)
        self.best_advs = self.remove_stoped(self.best_advs,stop_attack)
        self.labels = self.remove_stoped(labels,stop_attack)
        self.boundary = self.remove_stoped(self.boundary,stop_attack)
        self._nqueries = torch.zeros(len(self.X)).to(self.X.device)
        self._history  = {i: [] for i in range(len(self.X))}
        self.theta_max = torch.ones(len(self.X)).to(self.X.device) * self._theta_max
        self.stop_attack = stop_attack

        if stop_attack.all():
            self.best_advs = self.recover_stoped(X=X, removed_X=self.best_advs, stop_attack=stop_attack)
            self.boundary = self.recover_stoped(torch.zeros(self.original_len), self.boundary, stop_attack)
            return self.best_advs, self.boundary, stop_attack, already_rand_t
        bast_advs_norm = (self.best_advs-self.X).flatten(1).norm(dim=1)
        perturb_last = self.X - self.X_ori
        perturb_last_minus = torch.where(perturb_last<0,perturb_last,0)
        perturb_last_plus = torch.where(perturb_last>0,perturb_last,0)
        self.X_JND_minus_rest = -self.X_JND - perturb_last_minus
        self.X_JND_plus_rest = self.X_JND - perturb_last_plus

        self.meana = torch.ones_like(self.X).cuda()
        self.stda = torch.ones_like(self.X).cuda()
        self.meana[:,0,:,:]=0.485
        self.meana[:,1,:,:]=0.456
        self.meana[:,2,:,:]=0.406
        self.stda[:,0,:,:]=0.229
        self.stda[:,1,:,:]=0.224
        self.stda[:,2,:,:]=0.225

        # Check if X are already adversarials.
        if self.increase:
            self._images_finished = model(self.X) > self.labels + (abs(self.max_MOS-self.labels))*self.boundary ###
        else:
            self._images_finished = model(self.X) < self.labels - (abs(self.max_MOS-self.labels))*self.boundary
        

        # print("Already advs: ", self._images_finished.cpu().tolist()) # len:B
        self.best_advs = torch.where(atleast_kdim(self._images_finished, len(self.X.shape)), self.X, self.best_advs)
        self.best_advs = self._binary_search(self.best_advs, boost=True)
        # Initialize the direction orthogonalized with the first direction
        fd = self.best_advs - self.X
        self._directions_ortho = {}

        for i, v in enumerate(fd):
            if (v==0).all():
                self._directions_ortho[i] = torch.zeros_like(v.unsqueeze(0))
            else:
                self._directions_ortho[i] = v.unsqueeze(0) / v.norm()
        # Load Basis
        self._basis = Basis(self.X, **kwargs["basis_params"]) if "basis_params" in kwargs else Basis(self.X)
        for step_i in range(self._steps):
            candidates = self._get_candidates()
            if candidates is None:
                continue
            candidates = candidates.transpose(1, 0)
            
            best_candidates = torch.zeros_like(self.best_advs)
            for i, o in enumerate(self.X):
                o_repeated = torch.cat([o.unsqueeze(0)] * len(candidates[i]), dim=0)
                index = distance(o_repeated, candidates[i]).argmax()
                best_candidates[i] = candidates[i][index]

            is_success = distance(best_candidates, self.X) < distance(self.best_advs, self.X)
            self.best_advs = torch.where(
                atleast_kdim(is_success * self._images_finished.logical_not(), len(best_candidates.shape)), 
                best_candidates, 
                self.best_advs
                )

            # print("Best Advs distance:", distance(self.X, self.best_advs).cpu().numpy())
            if self._images_finished.all():
                # print("Max queries attained for all the images.")
                break
        if self.final_line_search:
            self.best_advs = self._binary_search(self.best_advs,  boost=True)
        
        if self.quantification:
            self.best_advs = self._quantify(self.best_advs)

        self.best_advs = torch.clamp(self.best_advs,min=self.X+self.X_JND_minus_rest,max=self.X+self.X_JND_plus_rest)

        self.best_advs = self.recover_stoped(X=X, removed_X=self.best_advs, stop_attack=stop_attack)
        self.boundary = self.recover_stoped(torch.zeros(self.original_len), self.boundary, stop_attack)

        return self.best_advs, self.boundary, stop_attack, already_rand_t

    def _is_adversarial(self, perturbed):
        # Faster than true_is_adversarial in batch  (time gain 20% )
        # Count the queries made for each image
        # Count if the vector is different from the null vector
        if self.quantification:
            perturbed = self._quantify(perturbed)
        if torch.isnan(perturbed).any():
            perturbed[torch.isnan(perturbed)] = 0
        perturbed_preds = []
        for i in range(perturbed.shape[0]):
            perturbed_i = perturbed[i]
            pred_i = self._model(perturbed_i.unsqueeze(0))
            perturbed_preds.append(pred_i.item())
        perturbed_preds = torch.tensor(perturbed_preds).cuda()
        
        if self.increase:
            is_advs = perturbed_preds > self.labels + (abs(self.max_MOS-self.labels))*self.boundary
        else:
            is_advs = perturbed_preds < self.labels - (abs(self.max_MOS-self.labels))*self.boundary
        ###
        zeros_per = []
        for i, p in enumerate(perturbed):
            if (p == 0).all():
                is_advs[i] = False
                zeros_per.append(True)
            else:
                zeros_per.append(False)

        indexes = []
        for i, p in enumerate(perturbed):
            if not (p == 0).all() and not self._images_finished[i]:
                self._nqueries[i] += 1
                indexes.append(i)

        self._images_finished = self._images_finished + (self._nqueries > self._max_queries)
        return is_advs

    def _get_candidates(self):
        """
        Find the lowest torchsilon to misclassified x following the direction: q of class 1 / q + torchs*direction of class 0
        """
        epsilon = torch.zeros(len(self.X)).to(self.X.device)
        direction_2 = torch.zeros_like(self.X)
        distances = (self.X - self.best_advs).flatten(1).norm(dim=1)

        queries_time = 0
        while (epsilon == 0).any():
            queries_time+=1
            if queries_time>80:
                print('80 queries_time in _get_candidates, break.')
                return None
            epsilon = torch.where(self._images_finished, torch.ones_like(epsilon), epsilon)

            new_directions = self._basis.get_vector(self._directions_ortho, indexes = [i for i, eps in enumerate(epsilon) if eps == 0]) #torch.Size([1, 3, 224, 224])
            if new_directions is None:
                continue
            direction_2 = torch.where(
                atleast_kdim(epsilon == 0, len(direction_2.shape)),
                new_directions,
                direction_2
            )
                              
            for i, eps_i in enumerate(epsilon):
                if i not in self._alpha_history:
                    self._alpha_history[i] = []
                if eps_i == 0:
                    # Concatenate the first directions and the last directions generated
                    self._directions_ortho[i] = torch.cat((
                        self._directions_ortho[i][:1],
                        self._directions_ortho[i][1 + len(self._directions_ortho[i]) - self.n_ortho:], 
                        direction_2[i].unsqueeze(0)), dim=0)

                    self._alpha_history[i].append([
                                len(self._history[i]),
                                float(self.theta_max[i].cpu()),
                                float(distances[i].cpu())
                        ])
            
            function_evolution = self._get_evolution_function(direction_2)
            
            new_epsilons = self._get_best_theta(function_evolution, epsilon == 0)

            for i, eps_i in enumerate(epsilon):
                if eps_i == 0:
                    if new_epsilons[i] == 0:
                        self._alpha_history[i][-1] += [False]
                    else:
                        self._alpha_history[i][-1] += [True]

            self.theta_max = torch.where(new_epsilons == 0, self.theta_max * self.rho, self.theta_max)
            self.theta_max = torch.where((new_epsilons != 0) * (epsilon == 0), self.theta_max / self.rho, self.theta_max)
            epsilon = torch.where((new_epsilons != 0) * (epsilon == 0), new_epsilons, epsilon)

        function_evolution = self._get_evolution_function(direction_2)
        if self.with_alpha_line_search:
            epsilon = self._alpha_binary_search(function_evolution, epsilon)

        epsilon = epsilon.unsqueeze(0)
        if self.with_interpolation:
            epsilon = torch.cat((epsilon, epsilon / 2), dim=0)

        candidates = torch.cat([function_evolution(eps).unsqueeze(0) for eps in epsilon], dim=0)
        
        if self.with_interpolation:
            d = (self.best_advs - self.X).flatten(1).norm(dim=1)
            delta = (self._binary_search(candidates[1],  boost=True) - self.X).flatten(1).norm(dim=1)
            theta_star = epsilon[0]

            num = theta_star * (4 * delta - d * (torch.cos(theta_star.raw) + 3))
            den = 4 * (2 * delta - d * (torch.cos(theta_star.raw) + 1))

            theta_hat = num / den
            q_interp = function_evolution(theta_hat)
            candidates = torch.cat((candidates, q_interp.unsqueeze(0)), dim=0)

        if self.with_distance_line_search:
            for i, candidate in enumerate(candidates):
                candidates[i] = self._binary_search(candidate,  boost=True)
        return candidates

    def _get_evolution_function(self, direction_2):
        distances = (self.best_advs - self.X).flatten(1).norm(dim=1, keepdim=True)
        direction_1 = (self.best_advs - self.X).flatten(1) / distances
        direction_1 = direction_1.reshape(self.X.shape)

        if self.clip:
            return lambda theta: (self.X + self._add_step_in_circular_direction(direction_1, direction_2, distances, theta)).clip(-self.meana/self.stda, (1-self.meana)/self.stda)
        else:
            return lambda theta: (self.X + self._add_step_in_circular_direction(direction_1, direction_2, distances, theta))

        
    def _add_step_in_circular_direction(self, direction1: torch.Tensor, direction2: torch.Tensor, r: torch.Tensor, degree: torch.Tensor) -> torch.Tensor:
        degree = atleast_kdim(degree, len(direction1.shape))
        r = atleast_kdim(r, len(direction1.shape))
        results = torch.cos(degree * np.pi/180) * direction1 + torch.sin(degree * np.pi/180) * direction2
        results = results * r * torch.cos(degree * np.pi / 180)
        return results

    def _get_best_theta(self, function_evolution, mask):
        coefficients = torch.zeros(2 * self.T).to(self.X.device)
        for i in range(0, self.T):
            coefficients[2 * i] = 1 - (i / self.T)
            coefficients[2 * i + 1] = - coefficients[2 * i]

        best_params = torch.zeros_like(self.theta_max)
        for i,  coeff in enumerate(coefficients):

            params = coeff * self.theta_max
            x_evol = function_evolution(params)
            x = torch.where(
                atleast_kdim((best_params == 0) * mask, len(self.X.shape)), 
                x_evol, 
                torch.zeros_like(self.X))

            is_advs = self._is_adversarial(x)
            # print('is_advs',is_advs)
            # for i,is_adv in enumerate(is_advs):
            #     if is_adv: #检查一下是否达到了最后的结果 # 这里应该删掉也行？
            #         img_i = self.X[i]
            #         adv_i = x[i]
            #         # JND_i = self.X_JND_rest[i] #X_JND
            #         JND_plus_i = self.X_JND_plus_rest[i]
            #         JND_minus_i = self.X_JND_minus_rest[i]
            #         adv_added = adv_i - img_i
            #         adv_added_plus = torch.where(adv_added>0,adv_added,0)
            #         adv_added_minus = torch.where(adv_added<0,adv_added,0)
            #         # print('adv_added',adv_added)
            #         # print('adv_added.mean()',adv_added.mean())
            #         # msked_advi = torch.where(JND_i>adv_added, adv_added, JND_i)
            #         msked_advi_plus = torch.where(JND_plus_i>adv_added_plus, adv_added_plus, JND_plus_i)
            #         msked_advi_minus = torch.where(JND_minus_i<adv_added_minus, adv_added_minus, JND_minus_i)
            #         msked_advi = msked_advi_plus + msked_advi_minus
            #         diffi = adv_added - msked_advi
            #         # print('diffi',diffi)
            #         print('diffi.mean()0',diffi.mean())
            #         if diffi.mean()!=0:
            #             x[i] = img_i + msked_advi #修正数值不稳定带来的问题 这里有没有写x[i]

                    
            best_params = torch.where(
                (best_params == 0) * mask * is_advs,
                params,
                best_params
            )
            if (best_params != 0).all():
                break
        
        return best_params

    def _alpha_binary_search(self, function_evolution, lower):
        # Upper --> not adversarial /  Lower --> adversarial
        mask = self._images_finished.logical_not()
        def get_alpha(theta: torch.Tensor) -> torch.Tensor:
            return 1 - torch.cos(theta * np.pi / 180)

        lower = torch.where(mask, lower, torch.zeros_like(lower))
        check_opposite = lower > 0 # if param < 0: abs(param) doesn't work
        
        # Get the upper range
        upper = torch.where(
            torch.logical_and(abs(lower) != self.theta_max, mask), 
            lower + torch.sign(lower) * self.theta_max / self.T,
            torch.zeros_like(lower)
            )

        mask_upper = torch.logical_and(upper == 0, mask)
        max_angle = torch.ones_like(lower) * 180
        while mask_upper.any():
            # Find the correct lower/upper range
            # if True in mask_upper, the range haven't been found
            new_upper = lower + torch.sign(lower) * self.theta_max / self.T
            new_upper = torch.where(new_upper < max_angle, new_upper, max_angle)
            new_upper_x = function_evolution(new_upper)
            x = torch.where(
                atleast_kdim(mask_upper, len(self.X.shape)),
                new_upper_x,
                torch.zeros_like(self.X)
            )

            is_advs =  self._is_adversarial(x)
            lower = torch.where(torch.logical_and(mask_upper, is_advs), new_upper, lower) 
            upper = torch.where(torch.logical_and(mask_upper, is_advs.logical_not()), new_upper, upper) 
            mask_upper = mask_upper * is_advs * torch.logical_not(self._images_finished)

        lower = torch.where(self._images_finished, torch.zeros_like(lower), lower)
        upper = torch.where(self._images_finished, torch.zeros_like(upper), upper)

        step = 0
        over_gamma = abs(get_alpha(upper) - get_alpha(lower)) > self._BS_gamma
        while step < self._BS_max_iteration and over_gamma.any(): 
            mid_bound = (upper + lower) / 2
            mid = torch.where(
                atleast_kdim(torch.logical_and(mid_bound != 0, over_gamma), len(self.X.shape)),
                function_evolution(mid_bound),
                torch.zeros_like(self.X)
            )
            is_adv = self._is_adversarial(mid)

            mid_opp = torch.where(
                atleast_kdim(torch.logical_and(check_opposite, over_gamma), len(mid.shape)),
                function_evolution(-mid_bound),
                torch.zeros_like(mid)
            )
            is_adv_opp = self._is_adversarial(mid_opp)

            lower = torch.where(mask * over_gamma * is_adv, mid_bound, lower)
            lower = torch.where(mask * over_gamma * is_adv.logical_not() * check_opposite * is_adv_opp, -mid_bound, lower)
            upper = torch.where(mask * over_gamma * is_adv.logical_not() * check_opposite * is_adv_opp, - upper, upper)
            upper = torch.where(mask * over_gamma * (abs(lower) != abs(mid_bound)), mid_bound, upper)

            check_opposite = mask * over_gamma * check_opposite * is_adv_opp * (lower > 0)
            over_gamma = abs(get_alpha(upper) - get_alpha(lower)) > self._BS_gamma

            step += 1
        
        return lower

    def _binary_search(self, perturbed: torch.Tensor, boost= False) -> torch.Tensor:
        # Choose upper thresholds in binary search based on constraint.
        highs = torch.ones(len(perturbed)).to(perturbed.device)
        d = np.prod(perturbed.shape[1:])
        thresholds = self._BS_gamma / (d * math.sqrt(d))
        lows = torch.zeros_like(highs)
        mask = atleast_kdim(self._images_finished, len(perturbed.shape))
        # Boost Binary search
        if boost:
            boost_vec = torch.where(mask, torch.zeros_like(perturbed), 0.2 * self.X + 0.8 * perturbed)
            is_advs = self._is_adversarial(boost_vec)
            is_advs = atleast_kdim(is_advs, len(self.X.shape))
            originals = torch.where(is_advs.logical_not(), boost_vec, self.X)
            perturbed = torch.where(is_advs, boost_vec, perturbed)
        else:
            originals = self.X

        # use this variable to check when mids stays constant and the BS has converged
        iteration = 0
        while torch.any(highs - lows > thresholds) and iteration < self._BS_max_iteration:
            iteration += 1
            mids = (lows + highs) / 2
            epsilon = atleast_kdim(mids, len(originals.shape))

            mids_perturbed = torch.where(
                mask, 
                torch.zeros_like(perturbed), 
                (1.0 - epsilon) * originals + epsilon * perturbed)

            is_adversarial_ = self._is_adversarial(mids_perturbed)

            highs = torch.where(is_adversarial_, mids, highs)
            lows = torch.where(is_adversarial_, lows, mids)

        epsilon = atleast_kdim(highs, len(originals.shape))
        return torch.where(mask, perturbed, (1.0 - epsilon) * originals + epsilon * perturbed) 


class Basis:
    def __init__(self, originals: torch.Tensor, random_noise: str = "uniform", basis_type: str = "dct", **kwargs):#random_noise: str = "normal"
        """
        Args:
            random_noise (str, optional): When basis is created, a noise will be added.This noise can be normal or 
                                          uniform. Defaults to "normal".
            basis_type (str, optional): Type of the basis: DCT, Random, Genetic,. Defaults to "random".
            device (int, optional): [description]. Defaults to -1.
            args, kwargs: In args and kwargs, there is the basis params:
                    * Random: No parameters                    
                    * DCT:
                            * function (tanh / constant / linear): function applied on the dct
                            * beta
                            * gamma
                            * frequence_range: tuple of 2 float
                            * dct_type: 8x8 or full
        """
        self.X = originals
        self._f_dct2 = lambda a: torch_dct.dct_2d(a)
        self._f_idct2 = lambda a: torch_dct.idct_2d(a)

        self.basis_type = basis_type
        self._function_generation = getattr(self, "_get_vector_" + self.basis_type)
        self._load_params(**kwargs)

        assert random_noise in ["normal", "uniform"]
        self.random_noise = random_noise

    def get_vector(self, ortho_with = None, indexes= None) -> torch.Tensor:
        if indexes is None:
            indexes = range(len(self.X))
        if ortho_with is None:
            ortho_with = {i: None for i in indexes}
            
        r: torch.Tensor = self._function_generation(indexes)

        if r is None:
            return None
        

        vectors = [
            self._gram_schmidt(r[i], ortho_with[i]) if ortho_with[i] is not None else r[i]
            for i in range(len(self.X))
        ]
        vectors = torch.cat([v.unsqueeze(0) for v in vectors], dim=0)
        norms = vectors.flatten(1).norm(dim=1)
        vectors /= atleast_kdim(norms, len(vectors.shape))
        return vectors

    def _gram_schmidt(self, v: torch.Tensor, ortho_with: torch.Tensor):
        v_repeated = torch.cat([v.unsqueeze(0)] * len(ortho_with), axis=0)
        
        #inner product
        gs_coeff = (ortho_with * v_repeated).flatten(1).sum(1)
        proj = atleast_kdim(gs_coeff, len(ortho_with.shape)) * ortho_with
        v = v - proj.sum(0)
        return v

    def _get_vector_dct(self, indexes) -> torch.Tensor:
        probs = self.X[indexes].uniform_(0, 3).long() - 1
        r_np = self.dcts[indexes] * probs
        if r_np.shape[0]==0:
            return None
        r_np = self._inverse_dct(r_np)
        new_v = torch.zeros_like(self.X)
        new_v[indexes] = (r_np + self.X[indexes].normal_(std=self._beta))
        return new_v

    def _get_vector_random(self, indexes) -> torch.Tensor:
        r = torch.zeros_like(self.X)
        r = getattr(r, self.random_noise + "_")(0, 1)
        new_v = torch.zeros_like(self.X)
        new_v[indexes] = r[indexes]
        return new_v

    def _load_params(
            self, 
            beta: float = 0,
            frequence_range= (0, 0.5),
            dct_type: str = "full",
            function: str = "tanh",
            tanh_gamma: float = 1
            ) -> None:
        if not hasattr(self, "_get_vector_" + self.basis_type):
            raise ValueError("Basis {} doesn't exist.".format(self.basis_type))

        if self.basis_type == "dct":
            self._beta = beta
            if dct_type == "8x8":
                mask_size = (8, 8) 
                dct_function = self.dct2_8_8
                self._inverse_dct = self.idct2_8_8
            elif dct_type == "full":
                mask_size = self.X.shape[-2:]
                dct_function = lambda x, mask: self._f_dct2(x)* mask
                self._inverse_dct =  self._f_idct2
            else:
                raise ValueError("DCT {} doesn't exist.".format(dct_type))
            
            dct_mask = self.get_zig_zag_mask(frequence_range, mask_size).to(self.X.device)
            self.dcts = dct_function(self.X, dct_mask)
        
            def get_function(function: str):
                if function == "tanh":
                    return lambda x: torch.tanh(tanh_gamma * x)
                elif function == "identity":
                    return lambda x: x
                elif function == "constant":
                    return lambda x: (abs(x) > 0).long()
                else:
                    raise ValueError("Function given for DCT is incorrect.")

            self.dcts = get_function(function)(self.dcts)

    def get_zig_zag_mask(self, frequence_range, mask_shape=(8, 8)):
        total_component = mask_shape[0] * mask_shape[1]
        n_coeff_kept = int(total_component * min(1, frequence_range[1]))
        n_coeff_to_start = int(total_component * max(0, frequence_range[0]))

        imsize = self.X.shape
        mask_shape = (imsize[0], imsize[1], mask_shape[0], mask_shape[1])
        mask = torch.zeros(mask_shape)
        s = 0
        
        while n_coeff_kept > 0:
            for i in range(min(s + 1, mask_shape[2])):
                for j in range(min(s + 1, mask_shape[3])):
                    if i + j == s:
                        if n_coeff_to_start > 0:
                            n_coeff_to_start -= 1
                            continue

                        if s % 2:
                            mask[:, :, i, j] = 1
                        else:
                            mask[:, :, j, i] = 1
                        n_coeff_kept -= 1
                        if n_coeff_kept == 0:
                            return mask
            s += 1
        return mask

    def dct2_8_8(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert mask.shape[-2:] == (8, 8)

        imsize = image.shape
        dct = torch.zeros_like(image)
        for i in np.r_[:imsize[2]:8]:
            for j in np.r_[:imsize[3]:8]:
                dct_i_j = self._f_dct2(image[:, :, i:(i+8),j:(j+8)]) 
                dct[:, :, i:(i+8),j:(j+8)] = dct_i_j * mask#[:dct_i_j.shape[0], :dct_i_j.shape[1]]
        return dct

    def idct2_8_8(self, dct: torch.Tensor) -> torch.Tensor:
        im_dct = torch.zeros_like(dct)
        for i in np.r_[:dct.shape[2]:8]:
            for j in np.r_[:dct.shape[3]:8]:
                im_dct[:, :, i:(i+8),j:(j+8)] = self._f_idct2(dct[:, :, i:(i+8),j:(j+8)])
        return im_dct