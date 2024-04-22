import numpy as np
from scipy.stats import truncnorm, nbinom, norm

class ECIRModel:
    """Cox-Ingersoll-Ross (CIR) model for interest rate dynamics with Negative Binomial jumps."""
    
    def __init__(self, kappa: float, mu_r: float, sigma: float, r: float = 1, p: float = 1, mu: float = 0, gamma: float = 0.01):
        """
        Initializes the CIR model with parameters.
        :param kappa: Mean reversion speed.
        :param mu_r: Long-term mean interest rate.
        :param sigma: Volatility of the interest rate.
        :param r: Number of successes for the Negative Binomial distribution (related to jump intensity).
        :param p: Success probability for each trial in the Negative Binomial distribution.
        :param mu: Mean of the jump size distribution.
        :param gamma: Standard deviation of the jump size distribution. Must be greater than 0.
        """
        if gamma <= 0:
            raise ValueError("Gamma must be greater than 0 to avoid division by zero.")
        
        # Ensure r is a positive integer
        if not (r > 0 and isinstance(r, int)):
            raise ValueError("Parameter 'r' must be a positive integer for the Negative Binomial distribution.")
        
        self.kappa = kappa
        self.mu_r = mu_r
        self.sigma = sigma
        self.r = int(r)
        self.p = p
        self.mu = mu
        self.gamma = gamma

    def next_rate(self, current_rate: float, dt: float) -> float:
        """
        Simulates the next interest rate using the Euler-Maruyama method with Negative Binomial jumps.
        """
        normal_shock = np.random.normal(0, dt)
        # make sure self.r > 0 and an integer
        if not (self.r > 0 and isinstance(self.r, int)):
            raise ValueError("Parameter 'r' must be a positive integer for the Negative Binomial distribution.")
   
        p_success = 1 - np.exp(-self.p * dt)
        if not (0 < p_success < 1):
            raise ValueError("Calculated success probability is out of bounds. It must be between 0 and 1.")
        
        nb_jump = nbinom.rvs(self.r, p_success)
        
        drift = self.kappa * (self.mu_r - current_rate) * dt
        diffusion = self.sigma * np.sqrt(max(current_rate, 0)) * normal_shock
        jump = 0
        if nb_jump > 0:
            jump = self.calculate_jump(nb_jump, current_rate, dt, drift, diffusion)
        
        return max(current_rate + drift + diffusion + jump, 0)
        
    def calculate_jump(self, nb_jump: int, current_rate: float, dt: float, drift: float, diffusion: float) -> float:
        """
        Calculates the jump size using a truncated normal distribution to avoid negative rates.
        """
        total_jump = 0
        if nb_jump > 0:
            lower_bound = - (current_rate + drift + diffusion) / nb_jump
            if lower_bound < 0:
                a, b = lower_bound / self.gamma, -lower_bound / self.gamma
                jump_distribution = truncnorm(a, b, loc=self.mu, scale=self.gamma)
                total_jump = jump_distribution.rvs() * nb_jump
        return total_jump

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

    def exact_solution(self, initial_rate: float, maturity: float) -> float:
        """
        Calculates the exact bond price under the CIR model for a zero-coupon bond.
        """
        gamma = np.sqrt(self.kappa**2 + 2*self.sigma**2)
        B = (2 * (np.exp(gamma * maturity) - 1)) / ((gamma + self.kappa) * (np.exp(gamma * maturity) - 1) + 2 * gamma)
        A = ((2 * self.kappa * self.mu_r) / self.sigma**2) * np.log(
            2 * gamma * np.exp((gamma + self.kappa) * maturity / 2) / ((gamma + self.kappa) * (np.exp(gamma * maturity) - 1) + 2 * gamma))
        return np.exp(A - B * initial_rate)
