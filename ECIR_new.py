import numpy as np
from scipy.stats import norm, poisson


class ECIRModel:
    """Extended Cox-Ingersoll-Ross (CIR) model with Negative Binomial jumps for interest rate dynamics."""
    
    def __init__(self, kappa: float, mu_r: float, sigma: float, mu: float, gamma: float):
        """
        Initializes the extended CIR model with parameters.
        :param kappa: Mean reversion speed.
        :param mu_r: Long-term mean interest rate.
        :param sigma: Volatility of the interest rate.
        :param mu: Mean of the normal distribution for jump sizes.
        :param gamma: Standard deviation of the normal distribution for jump sizes.
        """
        self.kappa = kappa
        self.mu_r = mu_r
        self.sigma = sigma
        self.mu = mu
        self.gamma = gamma


    def next_rate(self, current_rate: float, dt: float, with_jumps: bool = True) -> float:
        """
        Simulates the next interest rate using the Euler-Maruyama method, optionally including jumps.
        :param current_rate: Current interest rate.
        :param dt: Time increment.
        :param with_jumps: Boolean to include Negative Binomial jumps.
        :return: Next interest rate.
        """
        # Standard simulation using Euler-Maruyama method
        normal_shock = np.random.normal(0, 1)
        drift = self.kappa * (self.mu_r - current_rate) * dt
        diffusion = self.sigma * np.sqrt(max(current_rate, 0)) * np.sqrt(dt) * normal_shock
        new_rate = current_rate + drift + diffusion

        if with_jumps:
            num_jumps = np.random.poisson(self.mu)
            

            if num_jumps > 0:
                jump_sizes = np.random.normal(self.mu, self.gamma, num_jumps)
                signs = np.random.choice([-1, 1], num_jumps, p=[0.1, 0.9])
                jump_sizes *= signs

                total_jump = np.sum(jump_sizes)

                new_rate += total_jump

        '''
        if with_jumps:
            # Check for the occurrence of a jump if jumps are included
            num_jumps = np.random.negative_binomial(self.r, self.p)
            if num_jumps > 0:
                signs = np.random.choice([-1, 1], num_jumps, p= [0.1, 0.9])
                jump_sizes *= signs
                
                
                # jump_sizes = np.random.normal(self.mu, self.gamma, num_jumps)
                total_jump = np.sum(jump_sizes)
                new_rate += total_jump
        '''


    
        return max(new_rate, 0)

    def exact_solution(self, initial_rate: float, maturity: float) -> float:
        """
        Calculates the exact bond price under the CIR model for a zero-coupon bond.
        :param initial_rate: Initial interest rate.
        :param maturity: Time to maturity of the bond.
        :return: Bond price.
        """
        gamma = np.sqrt(self.kappa**2 + 2*self.sigma**2)
        B = (2 * (np.exp(gamma * maturity) - 1)) / ((gamma + self.kappa) * (np.exp(gamma * maturity) - 1) + 2 * gamma)
        A = ((2 * self.kappa * self.mu_r) / self.sigma**2) * np.log(
            2 * gamma * np.exp((gamma + self.kappa) * maturity / 2) / ((gamma + self.kappa) * (np.exp(gamma * maturity) - 1) + 2 * gamma))
        return np.exp(A - B * initial_rate)
    '''
    def transition_density(self, rt: float, rt_1: float, dt: float, terms_limit: int = 2) -> float:
        """
        Calculate the transition density for rt given rt_1 using an expansion.
        """
        density_sum = 0
        for n in range(terms_limit):
            rate_increment = self.mu * n + self.kappa * (self.mu_r - rt_1) * dt
            variance = n * (self.gamma**2) + dt * (rt_1 * (self.sigma**2))
            mean = rt_1 + rate_increment
            normal_density = norm.pdf(rt, loc=mean, scale=np.sqrt(variance))
            negative_binomial_density = nbinom.pmf(n, self.r, self.p)
            density_sum += normal_density * negative_binomial_density
        
        return density_sum
        '''

    def transition_density(self, rt: float, rt_1: float, dt: float, terms_limit: int = 10) -> float:
        """
        Calculate the transition density for rt given rt_1 using an expansion with Poisson jumps.
        """
        density_sum = 0
        for n in range(terms_limit):
            # Calculate the increment from the drift and the jumps
            rate_increment = self.mu * n + self.kappa * (self.mu_r - rt_1) * dt
            variance = n * (self.gamma**2) + dt * (rt_1 * (self.sigma**2))
            mean = rt_1 + rate_increment

            # Compute the density components
            normal_density = norm.pdf(rt, loc=mean, scale=np.sqrt(variance))
            poisson_density = poisson.pmf(n, self.mu * dt)  # Assume mu is rate parameter per unit time

            # Accumulate weighted density
            density_sum += normal_density * poisson_density
        
        return density_sum

