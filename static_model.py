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
    - Btilde=1 (just a choice of units)

    TODO: Calculate K->0 limit, and program it in.
    """
    def __init__(self, I, K, sigma, N):
        assert N-1<I<N, "I must be between N-1 and N"
        assert sigma>0
        assert K>=0
        # NOTE: Everything works fine if I>N, but there are issues if I is too low. I think this is
        # because assumption 3 will be violated if I is too low. As I goes down, wage goes up relative
        # to rent, so assumption 3 will eventually be broken.
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
        #TODO: revisit assumption 3. I think my code can be modified to remove this assumption.
        Istar, omega = self.find_Istar_and_omega()
        L = self.Ls(omega)
        Y = self.find_Y(Istar, L)
        R = (self.K/(Y * (Istar-self.N+1)))**(-1/self.sigma)
        W = omega * R * self.K

        # Check assumption 3:
        assert R>W/self.gamma(self.N), "K violates assumption 3. Try decreasing K."

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
        # TODO: I should improve the brackets (i.e. bounds of search) in the root_scalar()
        F1 = lambda Itilde: self.eq9_over_eq8(Itilde, self.gamma(Itilde)/self.K)
        Itilde = root_scalar(F1, bracket=[self.N-1+1e-5, 10]).root
        omega_at_Itilde = self.gamma(Itilde)/self.K
        if Itilde <= self.I:
            Istar = Itilde
            omega = omega_at_Itilde
        else:
            Istar = self.I
            F2 = lambda omega: self.eq9_over_eq8(self.I, omega)
            print("here")
            omega = root_scalar(F2, bracket=[1e-5, 200]).root
        return Istar, omega

    def find_Y(self, Istar, L):
        A = self.gamma_integral(Istar)
        s = self.sigma
        return ( (Istar-self.N+1)**(1/s)*self.K**(1-1/s) + A**(1/s)*L**(1-1/s) ) ** (s/(s-1)) 

    def figure3(self,save=False):
        omega_range = np.linspace(0,self.gamma(self.N)/self.K * 1.2)
        Istar_range = np.linspace(self.N-1, self.N, 100)
        # Suppress divide by zero warnings:
        with np.errstate(divide='ignore', invalid='ignore'):
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
        plt.xlabel(r'Task index $i$')
        plt.ylabel(r'$\omega=W/RK$')
        
        # Plotting Text of Equilibrium Quantities:
        xlim = plt.xlim()
        ylim = plt.ylim()
        text_x = xlim[0] + 0.95 * (xlim[1] - xlim[0])
        text_y = ylim[0] + 0.95 * (ylim[1] - ylim[0])
        plt.text(text_x, text_y, 
                f"Equilibrium:\nW={self.equilibrium_W:.2f}\nR={self.equilibrium_R:.2f}\nL={self.equilibrium_L:.2f}\nY={self.equilibrium_Y:.2f}",
                horizontalalignment='right',
                verticalalignment='top')
        
        if save:
            plt.savefig("figure3.svg")
        plt.show()
        return
