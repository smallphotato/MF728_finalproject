import numpy as np
from scipy.stats import truncnorm, norm, poisson

class JumpCIR:
    """CIR Model with stochastic jumps for interest rate modeling."""
    
    def __init__(self, kappa: float, mu_r: float, sigma: float, h: float, mu: float, gamma: float):
        """
        Initialize the Jump CIR model with necessary parameters.
        
        Parameters:
            kappa (float): Speed of mean reversion.
            mu_r (float): Long-term mean level of interest rate.
            sigma (float): Volatility of interest rate.
            h (float): Jump intensity (rate of Poisson process).
            mu (float): Mean of jump size distribution.
            gamma (float): Standard deviation of jump size distribution.
        """
        self.kappa = kappa
        self.mu_r = mu_r
        self.sigma = sigma
        self.h = h
        self.mu = mu
        self.gamma = gamma

    def next_interest_rate(self, current_rate: float, dt: float) -> float:
        """
        Simulate the next interest rate including jumps.
        
        Parameters:
            current_rate (float): Current interest rate.
            dt (float): Time increment.
        
        Returns:
            float: Simulated next interest rate.
        """
        normal_shock = np.random.normal()
        poisson_jumps = np.random.poisson(self.h * dt)
        time_step = self.kappa * (self.mu_r - current_rate) * dt
        stochastic_step = self.sigma * np.sqrt(max(current_rate, 0)) * np.sqrt(dt) * normal_shock
        jump_step = self.calculate_jump(current_rate, dt, poisson_jumps, time_step, stochastic_step)
        
        return current_rate + time_step + stochastic_step + jump_step

    def calculate_jump(self, rj: float, dt: float, pj: int, time_step: float, stochastic_step: float) -> float:
        """
        Calculate the total jump effect using a truncated normal distribution.
        
        Parameters:
            rj (float): Current rate just before jump.
            dt (float): Time increment.
            pj (int): Number of jumps (Poisson outcome).
            time_step (float): Deterministic part of the rate change.
            stochastic_step (float): Stochastic part of the rate change.
            
        Returns:
            float: Total jump contribution to rate change.
        """
        if pj == 0:
            return 0
        
        lower_bound = (-rj - time_step - stochastic_step) / pj
        upper_bound = np.inf  # Symmetric bounds for simplicity
        
        if lower_bound < 0:
            trunc_distribution = truncnorm(a=lower_bound / self.gamma, b=upper_bound / self.gamma, loc=self.mu, scale=self.gamma)
            return trunc_distribution.rvs(pj).sum()
        return 0

    def transition_density(self, rt: float, rt_1: float, dt: float, terms_limit: int = 2) -> float:
        """
        Calculate the transition density for rt given rt_1 using an expansion.
        
        Parameters:
            rt (float): Observed rate at t.
            rt_1 (float): Observed rate at t-dt.
            dt (float): Time step.
            terms_limit (int): Number of terms in the series expansion to calculate.
        
        Returns:
            float: Transition density value.
        """
        density_sum = 0
        for n in range(terms_limit):
            rate_increment = self.mu * n + self.kappa * (self.mu_r - rt_1) * dt
            variance = n * (self.gamma**2) + dt * (rt_1 * (self.sigma**2))
            mean = rt_1 + rate_increment
            normal_density = norm.pdf(rt, loc=mean, scale=np.sqrt(variance))
            exponential_density = poisson.pmf(n, mu=self.h * dt)
            density_sum += normal_density * exponential_density
        
        return density_sum
