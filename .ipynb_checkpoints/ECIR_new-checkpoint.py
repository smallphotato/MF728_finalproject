import numpy as np

class ECIRModel:
    """Extended Cox-Ingersoll-Ross (CIR) model with Negative Binomial jumps for interest rate dynamics."""
    
    def __init__(self, kappa: float, mu_r: float, sigma: float, p: float, r: int, mu: float, gamma: float):
        """
        Initializes the extended CIR model with parameters.
        :param kappa: Mean reversion speed.
        :param mu_r: Long-term mean interest rate.
        :param sigma: Volatility of the interest rate.
        :param p: Probability of success in each Bernoulli trial for the Negative Binomial distribution.
        :param r: Number of successes until the process is stopped (Negative Binomial parameter).
        :param mu: Mean of the normal distribution for jump sizes.
        :param gamma: Standard deviation of the normal distribution for jump sizes.
        """
        self.kappa = kappa
        self.mu_r = mu_r
        self.sigma = sigma
        self.p = p
        self.r = r
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
            # Check for the occurrence of a jump if jumps are included
            num_jumps = np.random.negative_binomial(self.r, self.p)
            if num_jumps > 0:
                jump_sizes = np.random.normal(self.mu, self.gamma, num_jumps)
                total_jump = np.sum(jump_sizes)
                new_rate += total_jump
        
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


# Example usage
if __name__ == "__main__":
    cir_extended = ECIRModel(kappa=0.3, mu_r=0.04, sigma=0.02, p=0.7, r=5, mu=0.02, gamma=0.01)
    initial_rate = 0.05
    dt = 0.01
    print("Next interest rate without jumps:", cir_extended.next_rate(initial_rate, dt))
    print("Next interest rate with jumps:", cir_extended.next_rate_with_jumps(initial_rate, dt))
    print("Exact bond price:", cir_extended.exact_solution(initial_rate, 1))  # Bond maturity in 1 year
