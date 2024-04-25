import numpy as np

class BasicCIRModel:
    """Cox-Ingersoll-Ross (CIR) model for interest rate dynamics."""
    
    def __init__(self, kappa: float, mu_r: float, sigma: float):
        """
        Initializes the CIR model with parameters.
        :param kappa: Mean reversion speed.
        :param mu_r: Long-term mean interest rate.
        :param sigma: Volatility of the interest rate.
        """
        self.kappa = kappa
        self.mu_r = mu_r
        self.sigma = sigma
        self.gamma = np.sqrt(self.kappa**2 + 2*self.sigma**2)  # Initialize gamma here


    def next_rate(self, current_rate: float, dt: float) -> float:
        """
        Simulates the next interest rate using the Euler-Maruyama method.
        :param current_rate: Current interest rate.
        :param dt: Time increment.
        :return: Next interest rate.
        """
        normal_shock = np.random.normal(0, dt)
        drift = self.kappa * (self.mu_r - current_rate) * dt
        diffusion = self.sigma * np.sqrt(max(current_rate, 0)) * normal_shock
        new_rate = current_rate + drift + diffusion
        return max(new_rate, 0)
    
    def bond_price(self, rt, T, t=0):
        """Calculates the zero-coupon bond price.
        :param rt: Current interest rate.
        :param T: Maturity of the bond.
        :param t: Current time. Default is 0.
        :return: Price of the bond.
        """
        gamma = self.gamma
        kappa, sigma, mu_r = self.kappa, self.sigma, self.mu_r
        B = (2 * (np.exp(gamma * (T - t)) - 1)) / \
            ((gamma + kappa) * (np.exp(gamma * (T - t)) - 1) + 2 * gamma)
        A = ((2 * kappa * mu_r) / sigma**2) * np.log(
            2 * gamma * np.exp((gamma + kappa) * (T - t) / 2) / \
            ((gamma + kappa) * (np.exp(gamma * (T - t)) - 1) + 2 * gamma))
        price = np.exp(A - B * rt)
        return price

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
