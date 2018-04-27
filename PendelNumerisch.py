import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import threading

# eigene Klassen
from StartFunc import StartFunc
from PythPendelSim import PythPendelSim
from DiskPendelSim import DiskPendelSim
from IterativePendelSim import IterativePendelSim


class PendelNumerisch:

    def __init__(self, Anzahl, dtau, Zeitschritte, Plotanteil=1,
                 Startbedingung = 'Soliton', Methode = 'Python', Form='Linear',
                 Kettenlaenge=1, Federkonst=5, Pendellaenge=0.5, Masse=0.0001,
                 Ortsfaktor=9.81, Reibungsfaktor=0
                 ):
        """Initialisiert die Parameter fuer die Simulierung


        :param Anzahl: der dargestellten Pendel
        :param dtau: diskretisierte Zeit dt
        :param Zeitschritte: Anzahl der Zeitschritte
        :param Kettenlaenge: Laenge der Pendelkette
        :param Federkonst: der Feder
        :param Pendellange: Laenge des Pendels
        :param Masse: der Pendel
        :param Ortsfaktor:  Fallbeschleunigung g
        :param Kopplung: der Feder mit den Pendeln
        :param Startbedingung: Wahl der Anfangsauslenkung und -geschwindigkeit
        :param Methode: zur Loesung der DGL
        :param Form: Kette oder Kreis
        :rtype: object
        Werte die oben in __init__ schon zugewiesen
        werden, sind Default Werte
        """



        # bisherige Grenze liegt bei knapp 150 Pendeln
        # bei mehr Pendeln ist die Animation sehr langsam

        self.dt = dtau
        self.Zeit = Zeitschritte
        self.Plotanteil = Plotanteil
        self.d = Kettenlaenge

        if(Anzahl>0):
            self.n = Anzahl
        else:
            raise ValueError('Anzahl der Pendel muss groesser als 0 sein.')

        self.g = Ortsfaktor
        self.l = Pendellaenge
        self.alpha = 0.05*self.l
        self.sigma = self.alpha/self.l
        self.k = Federkonst
        self.m = Masse
        self.Gamma = Reibungsfaktor

        self.kappa = self.k * self.d
        self.rho = self.m / self.d

        self.Form = Form

        '''neu eingefuehrte Parameter'''

        # Werte fuer die Differentialgleichung
        self.Omega_g_quad = self.g / self.l / (self.sigma**2 + 1)
        self.Omega_k_quad = self.k / self.m * self.sigma**2 / (self.sigma**2 + 1)
        self.dgl_param = self.Omega_k_quad/self.Omega_g_quad

        # Matrix etha_nl definieren
        self.etha = np.zeros((self.n, self.n))
        if(self.n > 1):
            self.etha[self.n - 1][self.n - 2] = 1
            self.etha[0][1] = 1

        for i in range(0, self.n):
            if (not (i == 0 or i == self.n - 1)):
                self.etha[i][i] = -2
                self.etha[i][i - 1] = 1
                self.etha[i][i + 1] = 1
            else:
                self.etha[i][i] = -1

        if(self.Form=='Kreis'):
            K = np.zeros((self.n, self.n))
            K[0][0] = -1
            if (self.n > 1):
                K[self.n-1][self.n-1] = -1
                K[0][self.n - 1] = 1
                K[self.n - 1][0] = 1
            self.etha = self.etha + K
        self.etha = self.etha * self.dgl_param

        self.AUSWAHL = Startbedingung
        self.meth = Methode


        # Klasse die Anfangsfunktionen phi_null und w liefert
        self.sF = StartFunc(self.dgl_param, self.AUSWAHL)

        self.mylines = np.array([])
        self.mypoints = np.array([])

        self.pause = False
        self.p = 0
        self.ani = None



        "Parametrisierung der Pendel zur Simulierung der DGL fuer " \
        "endlich viele Pendel"

        if (self.Form == 'Linear'):
            self.Kette = np.array(range(0, self.n)) * self.d / self.n
            self.phi0_array = self.sF.phi_null(self.Kette)
            self.w0_array = self.sF.w(self.Kette)
            self.XS = self.Kette + self.alpha*self.phi0_array
            self.ZS = np.array(-self.l * np.cos(self.phi0_array))
            self.YS = np.array(self.l * np.sin(self.phi0_array))
            self.X = np.copy(self.XS)
            self.Y = np.zeros(self.n)
            self.Z = np.zeros(self.n)

        elif(self.Form=='Kreis'):
            self.beta_null = self.beta_init(self.n)
            self.Kette = self.beta_null/(self.d/2*np.pi)
            self.phi0_array = self.sF.phi_null(self.Kette)
            self.w0_array = self.sF.w(self.Kette)
            self.XS = (self.d/(2*np.pi) + self.l*np.sin(self.phi0_array)) \
                      * np.sin(self.beta_null)
            self.YS = -(self.d/(2*np.pi) + self.l*np.sin(self.phi0_array)) \
                      * np.cos(self.beta_null)
            self.ZS = - self.l * np.cos(self.phi0_array)
            self.X = self.d/(2*np.pi) * np.sin(self.beta_null)
            self.Y = -self.d/(2*np.pi) * np.cos(self.beta_null)
            self.Z = np.zeros(self.n)

        else:
            raise NameError(self.Form + ' ist keine gueltige Form der Pendelkette')

        if (self.meth == 'Python'):
            # loest die DGL fuer endlich viele Pendel
            self.solution = PythPendelSim(
                self.n, self.phi0_array, self.w0_array, self.etha, self.Zeit,
                self.dt, self.Gamma
            ).solver()
            self.T = len(self.solution)

        elif (self.meth == 'Diskret'):
            # loest die DGL iterativ fuer endlich viele Pendel
            self.solution = np.transpose(DiskPendelSim(
                self.n, self.phi0_array, self.w0_array, self.etha, self.Zeit,
                self.dt, self.Gamma
            ).startComp())
            print("computing was finished")
            self.T = len(self.solution)

        elif (self.meth == 'Iterative'):
            self.solution = np.transpose(IterativePendelSim(
                self.n, self.phi0_array, self.w0_array, self.etha, self.Zeit,
                self.dt, self.Gamma
            ).startIter())
            self.T = len(self.solution)

        else:
            raise NameError(self.meth + ' ist keine gueltige Methode.')



        ''' Einstellungen des Plots, Achsen usw '''

        self.fig = plt.figure()
        self.ax = Axes3D(self.fig)
        #self.ax.axes.set_pane_color(255,255,255,0)
        self.ax.w_xaxis.set_pane_color((1, 1, 1, 0))
        self.ax.w_yaxis.set_pane_color((1, 1, 1, 0))
        self.ax.w_zaxis.set_pane_color((1, 1, 1, 0))

        self.ax.grid(False)

        self.ax.set_xlabel('x-Achse')
        self.ax.set_ylabel('y-Achse')
        self.ax.set_zlabel('z-Achse')

        #self.ax.set_axis_off()

        if(self.Form=='Linear'):
            self.ax.set_xlim(-0.1, self.d+0.1)
            self.ax.set_ylim(-self.l - 0.2, self.l + 0.2)
            self.ax.set_zlim(-self.l - 0.2, self.l + 0.2)
            self.ax.plot(
                [0, self.Kette[-1]], [0, 0], [0, 0], c='sienna', ls='-', lw=3
            )

        elif(self.Form=='Kreis'):
            r = self.d/(2*np.pi)
            self.ax.set_xlim(-r - 0.2, r + 0.2)
            self.ax.set_ylim(-r - 0.2, r + 0.1)
            self.ax.set_zlim(-r - 0.2, r + 0.2)
            p = matplotlib.patches.Circle(
                (0, 0), r, fill=False, color='sienna', lw=3
            )
            self.ax.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")
            self.ax.plot(
                [self.X[-1], self.X[0]], [self.Y[-1], self.Y[0]], [0, 0],
                c='sienna', ls='-', lw=3
            )


    def beta_init(self, number):
        """Berrechnet die Anfangswinkel beta
        der Kreispendelkette

        :param number: Anzahl der Pendel
        :return: beta
        """
        vec_beta_null = np.array([])
        for i in range(1, number + 1):
            com_beta_null = (number - i) * 2 * np.pi / number
            vec_beta_null = np.append(vec_beta_null, [com_beta_null])

        return vec_beta_null

    def initPlot(self):
        """Plottet die Pendel fuer t = 0, initialisiert
        mylines und mypoints: LineObjects deren Werte einzelnd
        in loop() durch die Animation animate() neu gesetzt werden

        """

        if(len(self.mylines)==0):
            for i in range(0, self.n):
                myline = self.ax.plot(
                    [self.X[i], self.XS[i]], [self.Y[i], self.YS[i]],
                    [self.Z[i], self.ZS[i]], c='indigo', marker='o'
                )
                self.mylines = np.append(self.mylines, [myline])
        else:
            self.loop(self.XS, self.YS, self.ZS)



    def onClick(self, event):
        """Funktion die es erlaubt, die Animation
        per Klick zu pausieren

        :param event: hier: Klicken
        """
        if self.pause:
            self.ani.event_source.stop()
            self.pause = False
        else:
            self.ani.event_source.start()
            self.pause = True

    def loop(self, newxs, newys, newzs, newx=None, newy=None):
        """Ueberschreibt die Werte von mylines und mypoints
        mit den neuen phi-Werten

        :param newx: neue X Werte
        :param newy: neue Y Werte
        :param newz: neue Z Werte
        """
        if newy is None:
            Y = self.Y
            X = newxs
        else:
            X = newx
            Y = newy

        for k in range(0, self.n):
            self.mylines[k].set_xdata([X[k], newxs[k]])
            self.mylines[k].set_ydata([Y[k], newys[k]])
            self.mylines[k].set_3d_properties([self.Z[k], newzs[k]])


    def animate(self, phi):
        """aktualisiert die Winkel und plottet diese

        :param phi: Winkel zur jeweiligen zu plottenden Zeit
        """

        vec_phi_tpluseins = phi[:self.n]

        if(self.Form=='Linear'):
            newx = self.X + self.alpha * vec_phi_tpluseins
            newys = np.array(self.l * np.sin(vec_phi_tpluseins))
            newzs = np.array(-self.l * np.cos(vec_phi_tpluseins))
            threading.Thread(
                target=self.loop, args=(newx, newys, newzs)
            ).start()

        elif(self.Form=='Kreis'):
            betanew = self.beta_null \
                      + self.alpha/(self.d/(2*np.pi)) * vec_phi_tpluseins
            newxs = +(self.d / (2 * np.pi) + self.l * np.sin(vec_phi_tpluseins))\
                    * np.sin(betanew)
            newys = -(self.d / (2 * np.pi) + self.l * np.sin(vec_phi_tpluseins))\
                    * np.cos(betanew)
            newzs = - self.l * np.cos(vec_phi_tpluseins)
            newx = self.d / (2 * np.pi) * np.sin(betanew)
            newy = -self.d / (2 * np.pi) * np.cos(betanew)
            threading.Thread(
                target=self.loop, args=(newxs, newys, newzs, newx, newy)
            ).start()

    def show(self):
        """startet die Animation

        """

        self.fig.canvas.mpl_connect('button_press_event', self.onClick)

        try:
            self.ani = animation.FuncAnimation(
                self.fig, self.animate, init_func=self.initPlot,
                frames=self.solution[::self.Plotanteil], interval=1, repeat=False,
                save_count=len(self.solution[::self.Plotanteil]), blit=False
            )
            plt.show()

        except AttributeError:
            print('Simulation wurde beendet')



    def save(self, name='pendelMovie', dtype='ffmpeg'):
        """Ermoeglicht das Speichern der zuvor gezeigten
        Simulation

        :param name: Name der Datei
        :param dtype: Dateityp zum Speichern
        """
        Writer = animation.writers['ffmpeg']
        writer = Writer(metadata=dict(artist='Me'), bitrate=1800, fps=30)
        self.ani.save(name + '.mp4', writer=writer)

        if(dtype=='html'):
            file = open(name + '.html', "w")
            file.write(
                '<!DOCTYPE html> <html> <head> <meta charset="utf-8"/>'
                '<title>Bachelor-Thesis Lea Otterbeck</title>'
                '</head> <body> <div align="center"> <h1>Simulation Pendelkette</h1>'
                '<video width="50%" controls> <source src="' + name + '.mp4' +
                '" type="video/mp4"/> Your browser does not support the video tag.'
                '</video> </div> </body> </html> '
            )
            file.close()


if "__main__" == __name__:

    pendel = PendelNumerisch(
        Anzahl=50, dtau=0.01, Zeitschritte=10000, Plotanteil=1,
        Startbedingung='Breather', Methode='Python', Form='Kreis',
        Kettenlaenge=5, Federkonst=5, Pendellaenge=0.5, Masse=0.0001,
        Ortsfaktor=9.81, Reibungsfaktor=0
    )

    pendel.show()
    #pendel.save(name='Soliton', dtype='mp4')

