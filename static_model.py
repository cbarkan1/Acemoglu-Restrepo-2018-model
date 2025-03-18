import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root_scalar
from abc import ABC, abstractmethod

class StaticModel(ABC):
    """
    Acemoglu & Restropo's Static Model from "The Race Against Man and Machine" (2018)

    This class makes the following assumptions:
    - Assumption 1 (gamma(i) is increasing)
    - Assumption 2 (eta=0 and Bbar=1)
    - Btilde=1
    """
    def __init__(self, I, K, sigma, N):
        assert 0<N-I<1, "N-I must be between 0 and 1"
        self.I = I
        self.K = K
        self.sigma = sigma
        self.N = N
        self.equilibrium_Istar = None
        self.equilibrium_omega = None
        self.equilibrium_L = None
        self.equilibrium_Y = None
        self.equilibrium_W = None
        self.equilibrium_R = None

    @abstractmethod
    def gamma(self,i):
        """Productivity of Labor"""
        ...

    @abstractmethod
    def gamma_integral(self,Istar)->float:
        """
        ∫(1/gamma(i))^(1-sigma)di integrating from Istar to N
        """
        ...
    
    @abstractmethod
    def Ls(self, omega)->float:
        """Labor supply as a function of omega"""
        ...

    def find_equilibrium(self):
        Istar, omega = self.find_Istar_and_omega()
        L = self.Ls(omega)
        Y = self.find_Y(Istar, L)
        R = (self.K/(Y * (Istar-self.N+1)))**(-1/self.sigma)
        W = omega * R * self.K
        self.equilibrium_Istar = Istar
        self.equilibrium_omega = omega
        self.equilibrium_L = L
        self.equilibrium_Y = Y
        self.equilibrium_W = W
        self.equilibrium_R = R
        return Istar, omega, L, Y, R, W

    def eq9_over_eq8(self,Istar,omega):
        A = self.gamma_integral(Istar)
        return A/(Istar-self.N+1)*omega**(-self.sigma) - self.Ls(omega)/self.K**(1-self.sigma)

    def find_Istar_and_omega(self)->tuple[float, float]:
        F1 = lambda Itilde: self.eq9_over_eq8(Itilde, self.gamma(Itilde)/self.K)
        Itilde = root_scalar(F1, bracket=[1e-5, 10]).root
        omega_at_Itilde = self.gamma(Itilde)/self.K
        if Itilde <= self.I:
            Istar = Itilde
            omega = omega_at_Itilde
        else:
            Istar = self.I
            F2 = lambda omega: self.eq9_over_eq8(self.I, omega)
            omega = root_scalar(F2, bracket=[1e-5, 10]).root
        return Istar, omega

    def find_Y(self, Istar, L):
        A = self.gamma_integral(Istar)
        s = self.sigma
        return ( (Istar-self.N+1)**(1/s)*self.K**(1-1/s) + A**(1/s)*L**(1-1/s) ) ** (s/(s-1)) 

    def figure3(self):
        # Suppress divide by zero warnings
        with np.errstate(divide='ignore', invalid='ignore'):
            omega_range = np.linspace(0, 3, 100)
            Istar_range = np.linspace(self.N-1, self.N, 100)
            Istar_mesh, omega_mesh = np.meshgrid(Istar_range, omega_range)
            F1 = self.eq9_over_eq8(Istar_mesh, omega_mesh)
        
        Istar_below = np.linspace(self.N-1, self.I, 100)
        Istar_above = np.linspace(self.I, self.N, 100)
        gamma_over_K_below = self.gamma(Istar_below)/self.K
        gamma_over_K_above = self.gamma(Istar_above)/self.K
        plt.figure()
        plt.contour(Istar_mesh, omega_mesh, F1, levels=[0], colors='red')
        plt.plot(Istar_below, gamma_over_K_below, label='gamma(Istar)/K', color="blue")
        plt.plot(Istar_above, gamma_over_K_above, ":",color="blue")
        plt.arrow(self.I, gamma_over_K_above[0], 0, omega_range[-1]-gamma_over_K_above[0], head_width=0.02, head_length=0.1, color="blue")
        plt.plot(self.equilibrium_Istar, self.equilibrium_omega, 'ko', label='Equilibrium')
        plt.xlabel('Istar')
        plt.ylabel('omega')
        plt.text(Istar_range[-1]-0.2, omega_range[-1]-0.5, f"Equilibrium:\nW={self.equilibrium_W:.2f}\nR={self.equilibrium_R:.2f}\nL={self.equilibrium_L:.2f}\nY={self.equilibrium_Y:.2f}")
        plt.show()
        return
