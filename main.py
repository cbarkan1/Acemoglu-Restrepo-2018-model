from static_model import StaticModel

class Model1(StaticModel):
    """
    version of Acemoglu & Restropo's Static Model with the following choices:
    1) gamma(i) = a + b*i
    2) nu(L) = 1/2 * L^2

    The gamma() and Ls() methods can be modified, so long as gamma_integral() is updated to be consistent with gamma(i).
    """

    def __init__(self, I, K, sigma, N, a, b):
        super().__init__(I, K, sigma, N)
        self.a = a
        self.b = b

    def gamma(self,i):
        """Productivity of Labor"""
        return self.a + self.b*i

    def gamma_integral(self,Istar):
        """
        ∫(1/gamma(i))^(1-sigma)di integrating from Istar to N
        """
        return 1/(self.b*self.sigma) * ( (self.a+self.b*self.N)**self.sigma - (self.a+self.b*Istar)**self.sigma )

    def Ls(self, omega):
        """
        Labor supply with nu(L)= 1/2 * L^2. This is the solution to nu'(L)=W/(RK+WL).
        omega = W/(R*K)
        """
        L = (-1/omega + (1/(omega**2) + 4)**0.5)/2
        return L


if __name__ == "__main__":
    I = .7
    K = .6
    sigma = 0.5
    N = 1.2
    a = .5
    b = .8

    model = Model1(I, K, sigma, N, a, b)
    model.find_equilibrium()
    model.figure3(save=True)
