class MonteCarlo:

    def __init__(self, model, start, T, n, num_sim):
        '''
        Parameters:
        model - The process to run a simulation on (class)
        start - The initial rate (float)
        T - The total time to run the simulation over
        n - The number of time steps to take for the whole simulation
        num_sim - The number of simulations to be run (integer)
        '''
        self.model = model
        self.current_rate = start
        self.T = T
        self.n = n
        self.num_sim = num_sim
        self.dt = T / n
        self.r0 = None

    def paths(self):
        '''
        Generating the paths of the Monte Carlo simulation
        '''

        import numpy as np

        dt = self.T / self.n        
        rt = []

        if self.r0 == None:
            self.r0 = self.current_rate

        r0 = self.r0

        for i in range(self.num_sim):
            rt.append([r0])
            for j in range(self.n):
                rate = self.model.next_rate(self.current_rate, self.dt)
                rt[i].append(rate)
                self.current_rate = rate

        return rt

    def plots(self):
        '''
        Plotting the paths generated from Monte Carlo Simulation
        '''
        
        import matplotlib.pyplot as plt
        import numpy as np

        paths = self.paths()
        
        for i in range(len(paths)):
            plt.plot(np.arange(0, len(paths[i])), paths[i])

    def price_estimates(self):
        '''
        Estimating the bond prices of the Monte Carlo simulations
        '''
        import numpy as np
        
        paths = self.paths()

        prices = []

        for i in range(len(paths)):
            temp = np.multiply(paths[i], self.dt)
            temp = -np.sum(temp)
            prices.append(np.exp(temp))

        avg_price = np.mean(prices)
        std_dev = np.std(prices)

        return avg_price, std_dev, prices

    def terminal_vals(self):
        '''
        Recording the terminal value for each rate path
        '''

        paths = self.paths()

        terminal = []
        for i in range(len(paths)):
            terminal.append(paths[i][-1])

        return terminal