import numpy as np
import time
import logging

class GreyWolfOptimizer:
    def __init__(self, obj_func, lb, ub, dim, n_agents=20, max_iter=50):
        if len(lb) != dim or len(ub) != dim:
            raise ValueError("Lower and upper bounds must match the dimension.")
        if n_agents <= 0:
            raise ValueError("Number of agents must be a positive integer.")
        if max_iter <= 0:
            raise ValueError("Maximum iterations must be a positive integer.")

        self.obj_func = obj_func
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = dim
        self.n_agents = n_agents
        self.max_iter = max_iter

        self.positions = np.random.uniform(0, 1, (self.n_agents, self.dim)) * (self.ub - self.lb) + self.lb
        
        self.alpha_pos = np.zeros(self.dim)
        self.alpha_score = float("inf")
        self.beta_pos = np.zeros(self.dim)
        self.beta_score = float("inf")
        self.delta_pos = np.zeros(self.dim)
        self.delta_score = float("inf")
        
        self.convergence_curve = np.zeros(max_iter)
        self.improvement_history = []

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def optimize(self, verbose=True):
        start_time = time.time()
        self.logger.info("🐺 Starting Grey Wolf Optimization...")
        
        for iter in range(self.max_iter):
            a = 2 - iter * (2 / self.max_iter)  # Linear reduction
            
            for i in range(self.n_agents):
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
                fitness = self.obj_func(self.positions[i])
                
                if fitness < self.alpha_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    
                    self.beta_score = self.alpha_score
                    self.beta_pos = self.alpha_pos.copy()
                    
                    self.alpha_score = fitness
                    self.alpha_pos = self.positions[i].copy()
                
                elif fitness < self.beta_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    
                    self.beta_score = fitness
                    self.beta_pos = self.positions[i].copy()
                
                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.positions[i].copy()
            
            for i in range(self.n_agents):
                for j in range(self.dim):
                    r1, r2 = np.random.random(2)
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    
                    r1, r2 = np.random.random(2)
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    
                    r1, r2 = np.random.random(2)
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    
                    D_alpha = abs(C1 * self.alpha_pos[j] - self.positions[i, j])
                    D_beta = abs(C2 * self.beta_pos[j] - self.positions[i, j])
                    D_delta = abs(C3 * self.delta_pos[j] - self.positions[i, j])
                    
                    X1 = self.alpha_pos[j] - A1 * D_alpha
                    X2 = self.beta_pos[j] - A2 * D_beta
                    X3 = self.delta_pos[j] - A3 * D_delta
                    
                    self.positions[i, j] = np.clip((X1 + X2 + X3) / 3, self.lb[j], self.ub[j])
            
            self.convergence_curve[iter] = self.alpha_score
            self.improvement_history.append(self.alpha_score)
            
            if verbose and (iter + 1) % 5 == 0:
                self.logger.info(f"🐺 Iteration {iter + 1}/{self.max_iter}, Best MSE: {self.alpha_score:.6f}")
            
            if len(self.improvement_history) > 10:
                recent_scores = self.improvement_history[-10:]
                if np.std(recent_scores) < 1e-6:
                    self.logger.info("🐺 Early stopping: Convergence detected")
                    break
        
        end_time = time.time()
        
        if verbose:
            self.logger.info("\n🐺 Optimization Complete")
            self.logger.info(f"Best Parameters: {self.alpha_pos}")
            self.logger.info(f"Best Score: {self.alpha_score}")
            self.logger.info(f"Total Time: {end_time - start_time:.2f} seconds")
        
        return self.alpha_pos, self.alpha_score, self.convergence_curve