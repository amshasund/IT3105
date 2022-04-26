import numpy as np

class Coarse_Coder:
    
    def __init__(self, num_bins, num_grids):
        self.num_bins = num_bins
        self.num_grids = num_grids
        self.state_bins = [np.linspace(-np.pi, np.pi, num_bins + 1)[1:-1],
                           np.linspace(-np.pi, np.pi, num_bins + 1)[1:-1],
                           np.linspace(-5, 5, num_bins + 1)[1:-1],
                           np.linspace(-10, 10, num_bins + 1)[1:-1]]
        
        self.offset = np.array([np.linspace(-np.pi/8, np.pi/8, num_grids),
                                np.linspace(-np.pi/8, np.pi/8, num_grids),
                                np.linspace(-1, 1, num_grids),
                                np.linspace(-1, 1, num_grids)])
    
    
    def digitize(self, x):
        """Returns quadruple e.g. array([3, 2, 2, 3]) 
           number of bin to which given variable belongs"""
        return np.array([np.digitize(x[0], self.state_bins[0]),
                         np.digitize(x[1], self.state_bins[1]),
                         np.digitize(x[2], self.state_bins[2]),
                         np.digitize(x[3], self.state_bins[3])])
    
    
    def get_discrete_state(self, x):
        """Return a unique number for every possible state
           e.g. 3**0 + 2**1 + 2**2 + 3**3"""
        return np.dot(self.digitize(x), self.num_bins**np.arange(4))
    
    
    def get_state(self, x):
        """Returns zeros ones string"""
        ret = np.zeros(self.num_grids * self.num_bins**4, dtype=np.int32)
        
        for i in range(self.num_grids):
            discrete_state = self.get_discrete_state(x+self.offset[:, i])
            ret[i*self.num_grids + discrete_state] = 1

        return ret

    