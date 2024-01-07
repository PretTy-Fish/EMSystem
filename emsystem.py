import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from IPython.display import display, HTML, Image

def q2s(X, Y, Z, U, V, W, length=1, arrow_length_ratio=0.3):
    """Quiver data to segments
    
    Args:
        X (numpy.ndarray): X component of mesh grid.
        Y (numpy.ndarray): Y component of mesh grid.
        Z (numpy.ndarray): Z component of mesh grid.
        U (numpy.ndarray): X component of quiver.
        V (numpy.ndarray): Y component of quiver.
        W (numpy.ndarray): Z component of quiver.
        length (float): Scaling factor of the quiver.
        arrow_length_ratio (float): The ratio between the arrow length and the quiver.
    """
    root = (X, Y, Z)
    head = (X+U*length, Y+V*length, Z+W*length)
    segments = np.array(root + head).reshape(6,-1)
    return [[[x, y, z], [u, v, w]] for x, y, z, u, v, w in zip(*list(segments))]

class EMModel:
    """A model in the electromagnetic system.

    Args:
        name (str, optional): The name of the model.
        modeller (function, optional): A function that assign the desired values to the desired points to build the model.
            Takes (x, y, z, epsilon, mu, sigma) as arguments.
        visual (function, optional): The function for visualising the model in Matplotlib.

    Attributes:
        name (str): The name of the model.
        modeller (function): A function that assign the desired values to the desired points to build the model.
            Takes (x, y, z, epsilon, mu, sigma) as arguments.
        visual (function): The function for visualising the model in Matplotlib, takes (fig, ax) as arguments.
    """
    def __init__(self, name='', modeller=lambda x, y, z, epsilon_x, epsilon_y, epsilon_z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z: None, visual=lambda fig, ax: None):
        self.name = name
        self.modeller = modeller
        self.visual = visual


class IEMModel:
    """An interface of the model in the electromagnetic system.

    Args:
        name (str, optional): The name of the model.

    Attributes:
        name (str): The name of the model.

    Notes:
        Does not quite work with GPU version of the module, as it is difficult to apply JIT to class methods.
    """
    def __init__(self, name=''):
        self.name = name

    def modeller(self, x, y, z, epsilon_x, epsilon_y, epsilon_z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z):
        """The function that assign the desired values to the desired points to build the model.

        Args:
            x (:obj:`list` of :obj:`float`): List of x values.
            y (:obj:`list` of :obj:`float`): List of y values.
            z (:obj:`list` of :obj:`float`): List of z values.
            epsilon_x, epsilon_y, epsilon_z (numpy.ndarray): Permittivity in each direction.
            mu_x, mu_y, mu_z (numpy.ndarray): Permeability in each direction.
            sigma_x, sigma_y, sigma_z (numpy.ndarray): Conductivity in each direction.
        """
        raise NotImplementedError

    def visual(self, fig, ax):
        """The for visualising the model in Matplotlib.

        Args:
            fig (matplotlib.figure.Figure): Matplotlib figure.
            ax (matplotlib.axes.Axes): Matplotlib axes.
        """
        raise NotImplementedError


class EMState:
    """A state of the electromagnetic system.

    Args:
        Ex, Ey, Ez (numpy.ndarray): X, Y, Z component of electric field.
        Hx, Hy, Hz (numpy.ndarray): X, Y, Z component of magnetic field strength.

    Attributes:
        Ex, Ey, Ez (numpy.ndarray): X, Y, Z component of electric field.
        Hx, Hy, Hz (numpy.ndarray): X, Y, Z component of magnetic field strength.
    """
    def __init__(self, Ex, Ey, Ez, Hx, Hy, Hz):
        self.Ex = Ex
        self.Ey = Ey
        self.Ez = Ez
        self.Hx = Hx
        self.Hy = Hy
        self.Hz = Hz

    def copy(self):
        """Create a copy of the state. This is to mimic the behaviour of a np.ndarray.

        Returns:
            state (EMState): a copy of the state.
        """
        return EMState(self.Ex.copy(), self.Ey.copy(), self.Ez.copy(),
                       self.Hx.copy(), self.Hy.copy(), self.Hz.copy())


class EMMaterialState:
    """A state of the electromagnetic system, taken the material into account.

    Args:
        Dx, Dy, Dz (numpy.ndarray): X, Y, Z component of displacement field.
        Bx, By, Bz (numpy.ndarray): X, Y, Z component of magnetic field.
        Jx, Jy, Jz (numpy.ndarray): X, Y, Z component of current density.

    Attributes:
        Dx, Dy, Dz (numpy.ndarray): X, Y, Z component of displacement field.
        Bx, By, Bz (numpy.ndarray): X, Y, Z component of magnetic field.
        Jx, Jy, Jz (numpy.ndarray): X, Y, Z component of current density.
    """
    def __init__(self, Dx, Dy, Dz, Bx, By, Bz, Jx, Jy, Jz):
        self.Dx = Dx
        self.Dy = Dy
        self.Dz = Dz
        self.Bx = Bx
        self.By = By
        self.Bz = Bz
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz

    def copy(self):
        """Create a copy of the material state. This is to mimic the behaviour of a np.ndarray.

        Returns:
            state (EMMaterialState): a copy of the state.
        """
        return EMMaterialState(self.Dx.copy(), self.Dy.copy(), self.Dz.copy(),
                               self.Bx.copy(), self.By.copy(), self.Bz.copy(),
                               self.Jx.copy(), self.Jy.copy(), self.Jz.copy())


def Updater(dt, dx, dy, dz, shape, Ex, Ey, Ez, Hx, Hy, Hz,
            epsilon_x, epsilon_y, epsilon_z, mu_x, mu_y, mu_z, Jx, Jy, Jz):
    """Update the state of the electromagnetic system.

    Args:
        dt (float): Time step.
        dx, dy, dz (float): Increment in X, Y, Z coordinates.
        shape (numpy.ndarray): Shape of the state.
        Ex, Ey, Ez (numpy.ndarray): X, Y, Z components of electric field.
        Hx, Hy, Hz (numpy.ndarray): X, Y, Z components of magnetic field strength.
        epsilon_x, epsilon_y, epsilon_z (numpy.ndarray): Permittivity in each direction.
        mu_x, mu_y, mu_z (numpy.ndarray): Permeability in each direction.
        Jx, Jy, Jz (numpy.ndarray): X, Y, Z components of current density.

    Notes:
        The output fields should be copies of the input fields to start with.
    """
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if i < shape[0]-1 and i > 0 and j < shape[1]-1 and j > 0 and k < shape[2]-1 and k > 0:
                    # middle
                    # update H field
                    Hx[i,j,k] += ((Ey[i,j,k+1]-Ey[i,j,k]) / dz
                                -(Ez[i,j+1,k]-Ez[i,j,k]) / dy) / mu_x[i,j,k] * dt
                    Hy[i,j,k] += ((Ez[i+1,j,k]-Ez[i,j,k]) / dx
                                -(Ex[i,j,k+1]-Ex[i,j,k]) / dz) / mu_y[i,j,k] * dt
                    Hz[i,j,k] += ((Ex[i,j+1,k]-Ex[i,j,k]) / dy
                                -(Ey[i+1,j,k]-Ey[i,j,k]) / dx) / mu_z[i,j,k] * dt
                    # update E field
                    Ex[i,j,k] += ( ((Hz[i,j,k]-Hz[i,j-1,k]) / dy
                                -(Hy[i,j,k]-Hy[i,j,k-1]) / dz)
                                - Jx[i,j,k]) / epsilon_x[i,j,k] * dt
                    Ey[i,j,k] += ( ((Hx[i,j,k]-Hx[i,j,k-1]) / dz
                                -(Hz[i,j,k]-Hz[i-1,j,k]) / dx)
                                - Jy[i,j,k]) / epsilon_y[i,j,k] * dt
                    Ez[i,j,k] += ( ((Hy[i,j,k]-Hy[i-1,j,k]) / dx
                                -(Hx[i,j,k]-Hx[i,j-1,k]) / dy)
                                - Jz[i,j,k]) / epsilon_z[i,j,k] * dt
                else:
                    # boundary conditions: reflective
                    # generate boundary indices
                    im = 0 if i == 0 else i - 1
                    jm = 0 if j == 0 else j - 1
                    km = 0 if k == 0 else k - 1
                    ip = shape[0] - 1 if i == shape[0] - 1 else i + 1
                    jp = shape[1] - 1 if j == shape[1] - 1 else j + 1
                    kp = shape[2] - 1 if k == shape[2] - 1 else k + 1
                    # update H field
                    Hx[i,j,k] += ((Ey[i,j,kp]-Ey[i,j,k]) / dz
                                -(Ez[i,jp,k]-Ez[i,j,k]) / dy) / mu_x[i,j,k] * dt
                    Hy[i,j,k] += ((Ez[ip,j,k]-Ez[i,j,k]) / dx
                                -(Ex[i,j,kp]-Ex[i,j,k]) / dz) / mu_y[i,j,k] * dt
                    Hz[i,j,k] += ((Ex[i,jp,k]-Ex[i,j,k]) / dy
                                -(Ey[ip,j,k]-Ey[i,j,k]) / dx) / mu_z[i,j,k] * dt
                    # update E field
                    Ex[i,j,k] += ( ((Hz[i,j,k]-Hz[i,jm,k]) / dy
                                -(Hy[i,j,k]-Hy[i,j,km]) / dz)
                                - Jx[i,j,k]) / epsilon_x[i,j,k] * dt
                    Ey[i,j,k] += ( ((Hx[i,j,k]-Hx[i,j,km]) / dz
                                -(Hz[i,j,k]-Hz[im,j,k]) / dx)
                                - Jy[i,j,k]) / epsilon_y[i,j,k] * dt
                    Ez[i,j,k] += ( ((Hy[i,j,k]-Hy[im,j,k]) / dx
                                -(Hx[i,j,k]-Hx[i,jm,k]) / dy)
                                - Jz[i,j,k]) / epsilon_z[i,j,k] * dt


def MUpdater(shape, Ex, Ey, Ez, Hx, Hy, Hz,
             epsilon_x, epsilon_y, epsilon_z, mu_x, mu_y, mu_z,
             sigma_x, sigma_y, sigma_z, Dx, Dy, Dz, Bx, By, Bz, Jx, Jy, Jz):
    """Update the material state of the electromagnetic system.

    Args:
        shape (numpy.ndarray): Shape of the state.
        Ex, Ey, Ez (numpy.ndarray): X, Y, Z components of electric field.
        Hx, Hy, Hz (numpy.ndarray): X, Y, Z components of magnetic field strength.
        epsilon_x, epsilon_y, epsilon_z (numpy.ndarray): Permittivity in each direction.
        mu_x, mu_y, mu_z (numpy.ndarray): Permeability in each direction.
        sigma_x, sigma_y, sigma_z (numpy.ndarray): Conductivity in each direction.
        Dx, Dy, Dz (numpy.ndarray): The output X, Y, Z components of displacement field.
        Bx, By, Bz (numpy.ndarray): The output X, Y, Z components of magnetic field.
        Jx, Jy, Jz (numpy.ndarray): The output X, Y, Z components of current density.

    Notes:
        The output fields should be a copy of the input field to start with.
    """
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                Dx[i,j,k] = Ex[i,j,k] * epsilon_x[i,j,k]
                Dy[i,j,k] = Ey[i,j,k] * epsilon_y[i,j,k]
                Dz[i,j,k] = Ez[i,j,k] * epsilon_z[i,j,k]
                Bx[i,j,k] = Hx[i,j,k] * mu_x[i,j,k]
                By[i,j,k] = Hy[i,j,k] * mu_y[i,j,k]
                Bz[i,j,k] = Hz[i,j,k] * mu_z[i,j,k]
                Jx[i,j,k] = Ex[i,j,k] * sigma_x[i,j,k]
                Jy[i,j,k] = Ey[i,j,k] * sigma_y[i,j,k]
                Jz[i,j,k] = Ez[i,j,k] * sigma_z[i,j,k]


class EMSystem:
    """An electromagnetic system, modelling the time evolution of electric and magnetic fields
        using Maxwell's Equations (specifically, Faraday's Law of Induction and Ampere's Law
        with Maxwell's addition).

    Args:
        dx, dy, dz (float): Resolution in x, y, z directions
        xbounds, ybounds, zbounds (:obj:`list` of :obj:`float`): The boundaries in x, y, z direction.
        models (:obj:`list` of :obj:`EMModel`, optional): List of models, used for system setup and visualisation.
        init (function, optional): Initializer of the state.

    Attributes:
        dx, dy, dz (float): Resolution in x, y, z directions
        x, y, z (:obj:`list` of :obj:`float`): List of x, y, z values.
        mx, my, mz, (numpy.ndarray): Mesh grid of x, y, z values.
        shape (numpy.ndarray): Number of points in each direction.
        epsilon_x, epsilon_y, epsilon_z (numpy.ndarray): Permittivity in each direction, on GPU side to speed up computation.
        mu_x, mu_y, mu_z (numpy.ndarray): Permeability in each direction, on GPU side to speed up computation.
        sigma_x, sigma_y, sigma_z (numpy.ndarray): Conductivity in each direction, on GPU side to speed up computation.
        state (EMState): The current state of the system, all field components on GPU side to speed up computation.
        mstate (EMMaterialState): The current material state of the system, all field components on GPU side to speed up computation.
        models (:obj:`list` of :obj:`EMModel`): List of models, used for system setup and visualisation.
        time (:obj:`list` of :obj:`float`): Record of time values.
        states (:obj:`list` of :obj:`EMState`): Record of states.
        init (function): Initializer of the state.

    Notes:
        Assume isotropic material, and the material at each cell is the same.
    """
    def __init__(self, dx, dy, dz, xbounds, ybounds, zbounds, models=None, init=lambda state, system: None):
        # Discretisation conditions
        self.dx = float(dx)
        self.dy = float(dy)
        self.dz = float(dz)
        # create the spatial grid
        self.x, self.dx = np.linspace(xbounds[0], xbounds[1], int((xbounds[1]-xbounds[0])/dx)+1, retstep=True)
        self.y, self.dy = np.linspace(ybounds[0], ybounds[1], int((ybounds[1]-ybounds[0])/dy)+1, retstep=True)
        self.z, self.dz = np.linspace(zbounds[0], zbounds[1], int((zbounds[1]-zbounds[0])/dz)+1, retstep=True)
        self.mx, self.my, self.mz = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        self.shape = np.array([len(self.x), len(self.y), len(self.z)])
        # system state - will be initialized in Reset()
        self.state = None
        self.mstate = None
        self.epsilon_x = None
        self.epsilon_y = None
        self.epsilon_z = None
        self.mu_x = None
        self.mu_y = None
        self.mu_z = None
        self.sigma_x = None
        self.sigma_y = None
        self.sigma_z = None
        # model information
        self.models = [] if models is None else models
        # record - will be initialized in Reset()
        self.time = []
        self.states = []
        # initialize
        self.init = init
        self.Initialize()

    def AddModel(self, model):
        """Add a model to the system.

        Args:
            model (EMModel): The model to be added.
        """
        self.models.append(model)
        model.modeller(self.x, self.y, self.z, self.epsilon_x, self.epsilon_y, self.epsilon_z,
                       self.mu_x, self.mu_y, self.mu_z, self.sigma_x, self.sigma_y, self.sigma_z)

    def Initialize(self):
        """Initialize the system.
        """
        self.state = EMState(np.zeros(self.shape),
                             np.zeros(self.shape),
                             np.zeros(self.shape),
                             np.zeros(self.shape),
                             np.zeros(self.shape),
                             np.zeros(self.shape))
        self.epsilon_x = np.full(self.shape, 1/(4e-7*np.pi*299792458**2))
        self.epsilon_y = np.full(self.shape, 1/(4e-7*np.pi*299792458**2))
        self.epsilon_z = np.full(self.shape, 1/(4e-7*np.pi*299792458**2))
        self.mu_x = np.full(self.shape, 4e-7*np.pi)
        self.mu_y = np.full(self.shape, 4e-7*np.pi)
        self.mu_z = np.full(self.shape, 4e-7*np.pi)
        self.sigma_x = np.zeros(self.shape)
        self.sigma_y = np.zeros(self.shape)
        self.sigma_z = np.zeros(self.shape)
        self.time = [0.]
        models = self.models
        self.models = []
        for model in models:
            self.AddModel(model)
        self.init(self.state, self)
        self.states = [self.state.copy()]
        self.mstate = EMMaterialState(np.zeros(self.shape),
                                      np.zeros(self.shape),
                                      np.zeros(self.shape),
                                      np.zeros(self.shape),
                                      np.zeros(self.shape),
                                      np.zeros(self.shape),
                                      np.zeros(self.shape),
                                      np.zeros(self.shape),
                                      np.zeros(self.shape))
        MUpdater(self.shape, self.state.Ex, self.state.Ey, self.state.Ez,
                 self.state.Hx, self.state.Hy, self.state.Hz,
                 self.epsilon_x, self.epsilon_y, self.epsilon_z,
                 self.mu_x, self.mu_y, self.mu_z,
                 self.sigma_x, self.sigma_y, self.sigma_z,
                 self.mstate.Dx, self.mstate.Dy, self.mstate.Dz,
                 self.mstate.Bx, self.mstate.By, self.mstate.Bz,
                 self.mstate.Jx, self.mstate.Jy, self.mstate.Jz)

    def Reset(self, t=None):
        """Keep only the first state of the system, with an optional new time value.

        Args:
            t (float, optional): The new starting time of the system
        """
        if t is None:
            self.time = self.time[:1]
        else:
            self.time = [t]
        self.states = self.states[:1]
        self.state = EMState(self.states[0].Ex,
                             self.states[0].Ey,
                             self.states[0].Ez,
                             self.states[0].Hx,
                             self.states[0].Hy,
                             self.states[0].Hz)

    def ClearHistory(self, t=None):
        """Keep only the last state of the system, with an optional new time value.

        Args:
            t (float, optional): The new starting time of the system
        """
        if t is None:
            self.time = self.time[-1:]
        else:
            self.time = [t]
        self.states = self.states[-1:]

    def UpdateState(self, dt, f=lambda state, *fargs: None, *fargs):
        """Update the state for a given timestep only, according to Maxwell's Equations.

        Args:
            dt (float): Time step.
            f (function, optional): The function to apply at the end of each update, usually for driving function.
                Has argument list of (state, *fargs) and return the modified state.
            *fargs: Argument to be passed to function f.
        """
        Updater(dt, self.dx, self.dy, self.dz, self.shape,
                self.state.Ex, self.state.Ey, self.state.Ez,
                self.state.Hx, self.state.Hy, self.state.Hz,
                self.epsilon_x, self.epsilon_y, self.epsilon_z,
                self.mu_x, self.mu_y, self.mu_z,
                self.mstate.Jx, self.mstate.Jy, self.mstate.Jz)
        f(self.state, *fargs)
        MUpdater(self.shape, self.state.Ex, self.state.Ey, self.state.Ez,
                 self.state.Hx, self.state.Hy, self.state.Hz,
                 self.epsilon_x, self.epsilon_y, self.epsilon_z,
                 self.mu_x, self.mu_y, self.mu_z,
                 self.sigma_x, self.sigma_y, self.sigma_z,
                 self.mstate.Dx, self.mstate.Dy, self.mstate.Dz,
                 self.mstate.Bx, self.mstate.By, self.mstate.Bz,
                 self.mstate.Jx, self.mstate.Jy, self.mstate.Jz)

    def Update(self, dt, t=None, count=None, f=lambda state, *fargs: None, *fargs):
        """Update the system according to Maxwell's Equations.

        Args:
            dt (float): Time step.
            t (float, optional): Period of time.
            count (int, optional): Number of states to save.
            f (function, optional): The function to apply at the end of each update, usually for driving function.
                Has argument list of (state, *fargs) and return the modified state.
            *fargs: Argument to be passed to function f.
        """
        n = 1
        step = 1
        if t is not None:
            n = int(t/dt)
            count = n if count is None else count
            step = max(int(n/count), 1)
        else:
            count = 1 if count is None else count
            t = dt * count
            n = count
        tstep = dt*step
        for i in range(n):
            if i == n - 1:
                tstep = t - count*step*dt
                dt = t - (n-1)*dt
            self.UpdateState(dt, f, *fargs)
            if (i+1) % step == 0 or i == n - 1:
                self.time.append(self.time[-1] + tstep)
                self.states.append(EMState(self.state.Ex.copy_to_host(),
                                           self.state.Ey.copy_to_host(),
                                           self.state.Ez.copy_to_host(),
                                           self.state.Hx.copy_to_host(),
                                           self.state.Hy.copy_to_host(),
                                           self.state.Hz.copy_to_host()))

    def PlotState(self, state=None, models=None, fields={}, elev=None, azim=None, resolution=[20, 20, 20], title='State', figsize=[15, 15]):
        """Plot a quiver plot of the fields in a state, the current state of the system if not specified.
        
        Args:
            state (:obj:`numpy.ndarray` of :obj:`EMPoint`, optional): The state to plot.
            models (:obj:`list` of :obj:`EMModel`, optional): The models to visualise.
            fields (dict, optional): Keys are the fields to be plot ('E', 'B', 'D', 'H', 'J'), values are the scale of the quivers.
            elev (float, optional): Elevation angle of camera.
            azim (float, optional): Azimuth angle of camera.
            resolution (list, optional): Number of points to plot in each direction
            title (str, optional): The title of the plot.
            figsize (list, optional): The size of the figure.

        Notes:
            Only the state with the same shape as the system could be correctly plotted.
        """
        if state is None:
            state = self.states[-1]
        if models is None:
            models = self.models
        
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection='3d')
        ax.view_init(elev=elev, azim=azim)
        for model in models:
            model.visual(fig, ax)

        i = max(int(self.shape[0]/resolution[0]), 1)
        j = max(int(self.shape[1]/resolution[1]), 1)
        k = max(int(self.shape[2]/resolution[2]), 1)
        if 'E' in fields:
            ax.quiver(self.mx[::i,::j,::k], self.my[::i,::j,::k], self.mz[::i,::j,::k],
                      state.Ex[::i,::j,::k], state.Ey[::i,::j,::k], state.Ez[::i,::j,::k],
                      length=fields['E'], color='red')
        if 'B' in fields:
            Bx = state.Hx[::i,::j,::k] * self.mu_x[::i,::j,::k]
            By = state.Hy[::i,::j,::k] * self.mu_y[::i,::j,::k]
            Bz = state.Hz[::i,::j,::k] * self.mu_z[::i,::j,::k]
            ax.quiver(self.mx[::i,::j,::k] + self.dx/2, self.my[::i,::j,::k] + self.dy/2, self.mz[::i,::j,::k] + self.dz/2,
                      Bx, By, Bz, length=fields['B'], color='navy')
        if 'D' in fields:
            Dx = state.Ex[::i,::j,::k] * self.epsilon_x[::i,::j,::k]
            Dy = state.Ey[::i,::j,::k] * self.epsilon_y[::i,::j,::k]
            Dz = state.Ez[::i,::j,::k] * self.epsilon_z[::i,::j,::k]
            ax.quiver(self.mx[::i,::j,::k], self.my[::i,::j,::k], self.mz[::i,::j,::k],
                      Dx, Dy, Dz, length=fields['D'], color='maroon')
        if 'H' in fields:
            ax.quiver(self.mx[::i,::j,::k] + self.dx/2, self.my[::i,::j,::k] + self.dy/2, self.mz[::i,::j,::k] + self.dz/2,
                      state.Hx[::i,::j,::k], state.Hy[::i,::j,::k], state.Hz[::i,::j,::k],
                      length=fields['H'], color='blue')
        if 'J' in fields:
            Jx = state.Ex[::i,::j,::k] * self.sigma_x[::i,::j,::k]
            Jy = state.Ey[::i,::j,::k] * self.sigma_y[::i,::j,::k]
            Jz = state.Ez[::i,::j,::k] * self.sigma_z[::i,::j,::k]
            ax.quiver(self.mx[::i,::j,::k], self.my[::i,::j,::k], self.mz[::i,::j,::k],
                      Jx, Jy, Jz, length=fields['J'], color='green')

        ax.set_title(title, fontsize='xx-large')
        ax.set_xlabel('x position')
        ax.set_ylabel('y position')
        ax.set_zlabel('z position')
        ax.set_xlim(self.x[0],self.x[-1])
        ax.set_ylim(self.y[0],self.y[-1])
        ax.set_zlim(self.z[0],self.z[-1])
        # scaling axes
        range_x = self.x[-1] - self.x[0]
        range_y = self.y[-1] - self.y[0]
        range_z = self.z[-1] - self.z[0]
        scale_x = range_x / max(range_x, range_y, range_z)
        scale_y = range_y / max(range_x, range_y, range_z)
        scale_z = range_z / max(range_x, range_y, range_z)
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1]))
        return fig, ax

    def Animator(self, index, Q, timelabel, fields, i, j, k):
        """Animator to be used in conjunction with matplotlib.animation.FuncAnimation.
        
        Args:
            index (int): Index passed in from matplotlib.animation.FuncAnimation.
            Q (:obj:`dict` of :obj:`mpl_toolkits.mplot3d.axes3d.Axes3D.quiver`): The dictionary of quiver plots.
            timelabel (:obj:`mpl_toolkits.mplot3d.axes3d.Axes3D.text`): The time label.
            fields (dict): Keys are the fields to be plot ('E', 'B', 'D', 'H', 'J'), values are the scale of the quivers.
            i (int): Step in x values.
            j (int): Step in y values.
            k (int): Step in z values.
        """
        timelabel.set_text(f't={self.time[index]:.3}s')
        if 'E' in Q:
            Q['E'].set_segments(q2s(self.mx[::i,::j,::k], self.my[::i,::j,::k], self.mz[::i,::j,::k],
                                    self.states[index].Ex[::i,::j,::k], self.states[index].Ey[::i,::j,::k], self.states[index].Ez[::i,::j,::k],
                                    length=fields['E']))
        if 'B' in Q:
            Bx = self.states[index].Hx[::i,::j,::k] * self.mu_x[::i,::j,::k]
            By = self.states[index].Hy[::i,::j,::k] * self.mu_y[::i,::j,::k]
            Bz = self.states[index].Hz[::i,::j,::k] * self.mu_z[::i,::j,::k]
            Q['B'].set_segments(q2s(self.mx[::i,::j,::k] + self.dx/2, self.my[::i,::j,::k] + self.dy/2, self.mz[::i,::j,::k] + self.dz/2,
                                    Bx, By, Bz, length=fields['B']))
        if 'D' in Q:
            Dx = self.states[index].Ex[::i,::j,::k] * self.epsilon_x[::i,::j,::k]
            Dy = self.states[index].Ey[::i,::j,::k] * self.epsilon_y[::i,::j,::k]
            Dz = self.states[index].Ez[::i,::j,::k] * self.epsilon_z[::i,::j,::k]
            Q['D'].set_segments(q2s(self.mx[::i,::j,::k], self.my[::i,::j,::k], self.mz[::i,::j,::k],
                                    Dx, Dy, Dz, length=fields['D']))
        if 'H' in Q:
            Q['H'].set_segments(q2s(self.mx[::i,::j,::k] + self.dx/2, self.my[::i,::j,::k] + self.dy/2, self.mz[::i,::j,::k] + self.dz/2,
                                    self.states[index].Hx[::i,::j,::k], self.states[index].Hy[::i,::j,::k], self.states[index].Hz[::i,::j,::k],
                                    length=fields['H']))
        if 'J' in Q:
            Jx = self.states[index].Ex[::i,::j,::k] * self.sigma_x[::i,::j,::k]
            Jy = self.states[index].Ey[::i,::j,::k] * self.sigma_y[::i,::j,::k]
            Jz = self.states[index].Ez[::i,::j,::k] * self.sigma_z[::i,::j,::k]
            Q['J'].set_segments(q2s(self.mx[::i,::j,::k], self.my[::i,::j,::k], self.mz[::i,::j,::k],
                                    Jx, Jy, Jz, length=fields['J']))
        return Q.values()

    def AnimateEvolution(self, models=None, fields={}, frames=None, elev=None, azim=None, resolution=[20, 20, 20], fps=60, title='Time Evolution', figsize=[15, 15], animname='Time Evolution'):
        """Create an animation of the time evolution of the system.
        
        Args:
            models (:obj:`list` of :obj:`EMModel`, optional): The models to visualise.
            fields (dict, optional): Keys are the fields to be plot ('E', 'B', 'D', 'H', 'J'), values are the scale of the quivers.
            frames (iterable, int, generator function, or None, optional): The frames argument to be passed to matplotlib.animation.FuncAnimation.
            elev (float, optional): Elevation angle of camera.
            azim (float, optional): Azimuth angle of camera.
            resolution (list, optional): Number of points to plot in each direction
            fps (float, optional): Frame rate.
            title (str, optional): The title of the plot.
            figsize (list, optional): The size of the figure.
            animname (str, optional): File name of the saved animation.

        Note:
            Quivers will be represented by line segments.
        """
        plt.rcParams['animation.html'] = 'html5'
        interval = 1000. / fps
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection='3d')
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize='xx-large')
        ax.set_xlabel('x position')
        ax.set_ylabel('y position')
        ax.set_zlabel('z position')
        ax.set_xlim(self.x[0],self.x[-1])
        ax.set_ylim(self.y[0],self.y[-1])
        ax.set_zlim(self.z[0],self.z[-1])
        # scaling axes
        range_x = self.x[-1] - self.x[0]
        range_y = self.y[-1] - self.y[0]
        range_z = self.z[-1] - self.z[0]
        scale_x = range_x / max(range_x, range_y, range_z)
        scale_y = range_y / max(range_x, range_y, range_z)
        scale_z = range_z / max(range_x, range_y, range_z)
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1]))

        if models is None:
            models = self.models
        for model in models:
            model.visual(fig, ax)
        i = max(int(self.shape[0]/resolution[0]), 1)
        j = max(int(self.shape[1]/resolution[1]), 1)
        k = max(int(self.shape[2]/resolution[2]), 1)
        Q = {}
        timelabel = ax.text(0.9, 0.9, 1.1, f't = {self.time[0]:.3} s',
                            transform=ax.transAxes, ha='left',
                            bbox={'boxstyle': 'round', 'facecolor': 'wheat'})
        if 'E' in fields:
            Q['E'] = ax.quiver(self.mx[::i,::j,::k], self.my[::i,::j,::k], self.mz[::i,::j,::k],
                               self.states[0].Ex[::i,::j,::k], self.states[0].Ey[::i,::j,::k], self.states[0].Ez[::i,::j,::k],
                               length=fields['E'], color='red')
        if 'B' in fields:
            Bx = self.states[0].Hx[::i,::j,::k] * self.mu_x[::i,::j,::k]
            By = self.states[0].Hy[::i,::j,::k] * self.mu_y[::i,::j,::k]
            Bz = self.states[0].Hz[::i,::j,::k] * self.mu_z[::i,::j,::k]
            Q['B'] = ax.quiver(self.mx[::i,::j,::k] + self.dx/2, self.my[::i,::j,::k] + self.dy/2, self.mz[::i,::j,::k] + self.dz/2,
                               Bx, By, Bz, length=fields['B'], color='navy')
        if 'D' in fields:
            Dx = self.states[0].Ex[::i,::j,::k] * self.epsilon_x[::i,::j,::k]
            Dy = self.states[0].Ey[::i,::j,::k] * self.epsilon_y[::i,::j,::k]
            Dz = self.states[0].Ez[::i,::j,::k] * self.epsilon_z[::i,::j,::k]
            Q['D'] = ax.quiver(self.mx[::i,::j,::k], self.my[::i,::j,::k], self.mz[::i,::j,::k],
                               Dx, Dy, Dz, length=fields['D'], color='maroon')
        if 'H' in fields:
            Q['H'] = ax.quiver(self.mx[::i,::j,::k] + self.dx/2, self.my[::i,::j,::k] + self.dy/2, self.mz[::i,::j,::k] + self.dz/2,
                               self.states[0].Hx[::i,::j,::k], self.states[0].Hy[::i,::j,::k], self.states[0].Hz[::i,::j,::k],
                               length=fields['H'], color='blue')
        if 'J' in fields:
            Jx = self.states[0].Ex[::i,::j,::k] * self.sigma_x[::i,::j,::k]
            Jy = self.states[0].Ey[::i,::j,::k] * self.sigma_y[::i,::j,::k]
            Jz = self.states[0].Ez[::i,::j,::k] * self.sigma_z[::i,::j,::k]
            Q['J'] = ax.quiver(self.mx[::i,::j,::k], self.my[::i,::j,::k], self.mz[::i,::j,::k],
                               Jx, Jy, Jz, length=fields['J'], color='green')

        if frames is None:
            frames = range(len(self.time))
        anim = animation.FuncAnimation(fig, self.Animator, frames=frames,
                                       fargs=(Q, timelabel, fields, i, j, k),
                                       interval=interval, blit=True)
        anim.save(f'{animname}.gif', writer='imagemagick', fps=fps)