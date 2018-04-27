import numpy as np


class IterativePendelSim(object):
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
        self.etha = - matrix
        self.R = Reibung
        self.Lambda = Reibung
        self.dtau = time_steps

        self.NT = time_number
        self.time = np.linspace(0, time_steps * time_number, time_number + 1)


        self.EW, self.AT = np.linalg.eig(self.etha)
        self.A = np.transpose(self.AT)
        # Da ein Eigenwert 0 ist, liefert die Numerik
        # teilweise sehr kleine aber negative Werte,
        # daher wird der Betrag genommen
        self.W = np.sqrt(np.abs(self.EW))
        self.WTau = self.W[:, np.newaxis] * self.time[np.newaxis, :]
        self.S = np.sin(self.WTau)
        self.C = np.cos(self.WTau)

        self.alpha_init = np.dot(self.A, self.phi0)
        self.beta_init = np.dot(self.A, self.w0)


    def nonLin(self, phi, dphi):
        """Berrechnet den nichtlinearen Anteil

        :param phi: Werte phi
        :param dphi: Ableitung von phi
        :return: nichtlinearer Anteil der DGL
        """

        return np.dot(self.A, (np.sin(phi) + self.Lambda*dphi))

    def initGuess(self):
        """Berrechnet die Startewerte der Iteration

        :return: Startwerte fuer die Iteration
        """
        Phi_init = self.alpha_init[:, np.newaxis]*self.C \
                   + self.beta_init[:, np.newaxis]*self.S/self.W[:, np.newaxis]
        dphi_init = - self.phi0[:, np.newaxis]*self.S*self.W[:, np.newaxis] \
                    + self.w0[:, np.newaxis]*self.C
        phi_init = np.dot(self.AT, Phi_init)
        return phi_init, dphi_init

    def diskInt(self, f):
        """Berrechnet numerisch die diskreten Integrale

        :param f: Teil der zu iterierenden Funktion
        :return: alpha und beta
        """

        int1 = self.S * f
        int2 = -self.C * f
        alpha = np.zeros((self.n, self.NT + 1))
        beta = np.zeros((self.n, self.NT + 1))

        for zeit in range(1, self.NT + 1):
            alpha[:, zeit] = alpha[:, zeit - 1] + int1[:, zeit] + int1[:, zeit-1]
            beta[:, zeit] = beta[:, zeit - 1] + int2[:, zeit] + int2[:, zeit-1]

        alpha = alpha*self.dtau/2.0
        beta = beta * self.dtau / 2.0 + self.beta_init[:, np.newaxis]

        alpha = alpha / self.W[:, np.newaxis]
        beta = beta / self.W[:, np.newaxis]
        alpha = alpha + self.alpha_init[:, np.newaxis]

        return alpha, beta


    def mpeinsIteration(self, phim, dphim):
        """Fuehrt den naechsten Iterationsschritt aus

        :param phim: phi-Werte aus der vorherigen Iteration
        :param dphim: Ableitung der phi-Werte
        :return: phi und dphi aus dem aktuellen Iterationsschritt
        """

        AN = self.nonLin(phim, dphim)
        alpha, beta = self.diskInt(AN)
        Phimpluseins = self.C * alpha + self.S*beta
        dPhimpluseins = self.W[:, np.newaxis]*(self.C*beta - self.S*alpha)
        phimpluseins = np.dot(self.AT, Phimpluseins)
        dphimpluseins = np.dot(self.AT, dPhimpluseins)

        return phimpluseins, dphimpluseins

    def startIter(self, itersteps=300):
        """Startet die Iteration

        :param itersteps: Anzahl der maximalen Iterationsschritte
        :return: phi-Werte aus dem letzten Iterationsschritt
        """

        print('Initialisierung...')
        phim, dphim = self.initGuess()

        for iter in range(0, itersteps):
            print('Starte Iteration...')
            oldphim = np.copy(phim)
            phim, dphim = self.mpeinsIteration(phim, dphim)

            abs = np.max(np.absolute(oldphim - phim))
            if(np.max(abs) < 10**(-5)):
                print('Iteration konvergierte nach ', iter, ' Schritten')
                self.iternumber = iter

                return phim

            if(iter == itersteps-1):
                print('Iteration ist nicht konvergiert.')
                print('Zuletzt berechnete Werte werden zurueckgegeben '
                      'und sind mit Vorsicht zu geniessen!')

                return phim
