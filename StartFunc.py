import numpy as np

class StartFunc(object):
    def __init__(self, Faktor, cond):
        """Initialisierung der Klasse zur Berechnung der
        Anfangsauslenkung und -geschwindigkeit von phi

        :param Faktor: Parameter aus der DGL
        :param cond: Anfangsbedingung
        """
        self.Condition = {
            'Soliton': cond,
            'NewKink-Kink': cond,
            'Antisoliton': cond,
            'Kink-Kink': cond,
            'Breather': cond,
            'EinPendel': cond,
            'HalfPendel': cond
        }
        self.Condition = self.Condition.get(cond, False)
        if(self.Condition==False):
            raise NameError(cond + ' ist keine gueltige Anfangsbedingung.')
        self.factor = Faktor
        self.c = 0.85*np.sqrt(self.factor)
        self.om = 0.95

    def phi_null(self, x):
        """Berechnet fuer die gegebenen x Werte die
        Anfangsauslenkung phi zum Zeitpunkt t = 0

        :param x: x Werte
        :return: array mit phi(0)
        """
        # Anfangsbed. ein Pendel ausgelenkt
        ret = np.zeros(len(x))
        ret[0] = np.pi / 4


        pen = np.zeros(len(x))
        for i in range(0, round(len(x)/2)):
            pen[i] = 3*np.pi/5


        result = {
            'Soliton': (4 * np.arctan(np.exp((x-0)*len(x)/2
                                             / (np.sqrt(self.factor - self.c**2))))),
            'Antisoliton': (- 4 * np.arctan(np.exp((x-x[len(x)-1])*len(x)/2
                                                   / (np.sqrt(self.factor - self.c**2))))),
            'Kink-Kink': 4*np.arctan(self.c*np.sinh((x-(2*x[-1]-x[-2])/2)*len(x)/2
                                                    /np.sqrt(self.factor-self.c**2))),
            'Breather': np.zeros(len(x)),
            'EinPendel': ret,
            'HalfPendel': pen
        }
        return result.get(self.Condition, 'Error')

    def w(self, x):
        """Berechnet fuer die gegebenen x Werte die Anfangsgeschwindigkeit
        fuer t = 0

        :param x: x Werte
        :return: array mit d/dt phi(0)
        """
        result = {
            'Soliton': (-2 * self.c / (np.sqrt(self.factor - self.c ** 2))
                        / (np.cosh((x-0)*len(x)/2 / (np.sqrt(self.factor - self.c ** 2))))),
            'Antisoliton': (+2 * self.c / (np.sqrt(self.factor - self.c ** 2))
                            / (np.cosh((x-x[len(x)-1])*len(x)/2
                                       / (np.sqrt(self.factor - self.c ** 2))))),
            'Kink-Kink': np.zeros(len(x)),
            'Breather': 4*np.sqrt(1-self.om**2)
                        / (np.cosh(np.sqrt(1-self.om**2)*(x-(2*x[-1]-x[-2])/2)*len(x)/2)),
            'EinPendel': np.zeros(len(x)),
            'HalfPendel': np.zeros(len(x))
        }
        return result.get(self.Condition, 'Error')



