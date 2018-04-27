from scipy import integrate
import numpy as np


class PythPendelSim(object):

    def __init__(self, Anzahl, sFphi0, sFw0, matrix, time_number, time_steps, Reibung=0):
        """Initializer

        :param Anzahl: der Pendel
        :param sFphi0: liefert Anfangsauslenkung von phi
        :param sFw0: liefert Anfangsgeschwindigkeit von phi
        :param matrix: Matrix aus der DGL
        :param time_number: Anzahl der Zeitschritte
        :param time_steps: diskretes Zeitintervall
        :param Reibung: Reibungskoeffizient aus der DGL
        """
        self.n = Anzahl

        # Anfangsauslenkung
        self.phi0 = sFphi0
        # Anfangsgeschwindigkeit
        self.phi0 = np.append(self.phi0, sFw0)

        self.etha = matrix
        self.Lambda = Reibung


        # Intervall fuer die numerische Berechnung
        self.time = np.linspace(0, time_steps*time_number, time_number+1)


    def dgl(self, phi_array, dphi_array):
        """Berechnet die Differentialgleichung

        :param phi_array: Winkel der n Pendel
        :param dphi_array: Winkelgeschwindigkeit der n Pendel
        :return: d^2/dt^2 phi_n fuer n = 1,...,N
        """
        res = - np.sin(phi_array) \
            + np.dot(self.etha, phi_array)-self.Lambda*dphi_array
        return res


    def deriv(self, y, t):
        """Differentialgleichungssystem 1. Ordnung
        von d/dt phi_n fuer n = 1,..,2N

        :param y: Variablen des Differentialgleichungssystems
        :param t: Zeit, wird automatisch mit uebergeben
        :return: erste Ableitung von y
        """
        end = int(self.n)
        # d/dt phi_n fuer n = N+1,...,2N
        phis = y[0:end]
        # d/dt phi_n fuer n = 1,...,N
        dphis = y[end:]
        return np.append(dphis, self.dgl(phis, dphis))


    def solver(self):
        """ Loest das DGL System mit LSODA aus FORTRAN

        :return: Matrix, Spalten entsprechen den Pendeln
                Reihen der Zeitentwicklung
        """
        sol = integrate.odeint(self.deriv, self.phi0, self.time)
        return sol






