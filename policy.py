import numpy as np
import maxapproxi

class Policy:
    def __init__(self, n_action, rng=None, strategy="Random",scale = 1., eps_init=1.,eps_decay_rate=0.995):
        if rng is None:
            rng = np.random.RandomState(0)
        
        self.rng = rng
        self.n_action = n_action
        self.scale = scale
        self.eps = eps_init
        self.eps_decay_rate = eps_decay_rate
        self.eps_min = 0.00
        self.strategy = strategy
        self.explore = True
    
    def get_action(self, q_value):
        a_max = np.argmax(q_value)
        if self.explore:
            if self.strategy == "Random":
                policy = np.ones(self.n_action) / self.n_action
            elif self.strategy == "Epsilon":
                policy = np.ones(self.n_action) * self.eps / self.n_action
                policy[a_max] += 1. - self.eps
            elif self.strategy == "Softmax":
                softmax_policy = maxapproxi.softmax(q_value, scale=self.scale)
                policy = np.ones(self.n_action) * self.eps / self.n_action + softmax_policy * (1. - self.eps)
            elif self.strategy == "Sparsemax":
                sparsemax_policy = maxapproxi.sparsedist(q_value, scale=self.scale)
                sparsemax_policy = np.reshape(sparsemax_policy,(-1,))
                policy = np.ones(self.n_action) * self.eps / self.n_action + sparsemax_policy * (1. - self.eps)
            
            policy = policy/np.sum(policy)
            action = self.rng.choice(self.n_action, p=policy)
        else:
            if self.strategy == "Random":
                policy = np.zeros(self.n_action)
                policy[a_max] = 1.
            elif self.strategy == "Epsilon":
                policy = np.zeros(self.n_action)
                policy[a_max] = 1.
            elif self.strategy == "Softmax":
                policy = maxapproxi.softmax(q_value, scale=self.scale)
            elif self.strategy == "Sparsemax":
                policy = maxapproxi.sparsedist(q_value, scale=self.scale)
                policy = np.reshape(policy,(-1,))
                
            policy = policy/np.sum(policy)
            action = self.rng.choice(self.n_action, p=policy)
        
        return(action)
    
    def update_policy(self):
        self.eps = np.max((self.eps*self.eps_decay_rate,self.eps_min))