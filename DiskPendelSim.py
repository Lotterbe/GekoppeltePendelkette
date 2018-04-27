import numpy as np


class DiskPendelSim(object):
    def __init__(self, Anzahl, phi_null, w_null, matrix, time_number, time_steps, Reibung=0):
        """Initializer

        :param Anzahl: der Pendel
        :param phi_null: liefert Anfangsauslenkung von phi
        :param w_null: liefert Anfangsgeschwindigkeit von phi
        :param matrix: aus der DGL
        :param time_number: Anzahl der Zeitschritte
        :param time_steps: diskretes Zeitintervall
        :param Reibung: Reibungskoeffizient aus der DGL
        """

        self.n = Anzahl
        self.phi0 = phi_null
        self.w0 = w_null

        self.etha = matrix
        self.R = Reibung
        self.Lambda = Reibung
        self.dtau = time_steps
        self.factor = self.dtau*self.Lambda/2

        self.NT = time_number


    def expDiskSim(self):
        """Explizite Berechnung von phi mit diskretisierter DGL
        nur fuer Stokesche Reibung

        :return: phi
        """

        phi = np.zeros((self.n, self.NT+1))
        phi[:, 0] = self.phi0
        phi[:, 1] = self.phi0 + self.dtau * self.w0 + self.dtau ** 2 / 2 \
                    * ( - np.sin(self.phi0) + np.dot(self.etha, self.phi0)
                        - self.Lambda * self.w0**self.R)
        for i in range(1, self.NT):
            phi[:, i + 1] = 1/(1 + self.factor) * \
                            (2 * phi[:, i] - (1-self.factor) * phi[:, i - 1]
                             + self.dtau ** 2 * (- np.sin(phi[:, i])
                                                 + np.dot(self.etha, phi[:, i])))
        return phi


    def startComp(self):
        """Startet die Berechnung von phi

        :return: phi, wobei in den Reihen die Pendel und in den Spalten die Zeiten stehen
        """
        solphi = self.expDiskSim()

        return solphi