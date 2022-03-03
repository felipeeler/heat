from sympy import Symbol, Eq
import numpy as np
from scipy.special import erf
import tensorflow as tf

from modulus.solver import Solver
from modulus.dataset import TrainDomain, ValidationDomain, MonitorDomain
from modulus.data import Validation, Monitor
from modulus.sympy_utils.geometry_1d import Line1D
from modulus.controller import ModulusController

from heat_equation import HeatEquation1D
from rnn import GRUArch

# domain parameters
L = 0.1			# m
Ti = 100.0      # C
Ts = 25.0       # C

# gru parameters
seq_length = 10
time_length = 1

# physical parameters
alpha = float(1.34e-5)         # m2/s


# define geometry
geo = Line1D(0, L)

# define sympy varaibles to parametize domain curves
t_symbol = Symbol('t')
time_range = {t_symbol: (0, time_length)}

class HeatTrain(TrainDomain):
  def __init__(self, **config):
    super(HeatTrain, self).__init__()
    # sympy variables
    x = Symbol('x')

    #initial conditions
    IC = geo.interior_bc(outvar_sympy={'T': Ti},
                         bounds={x: (0, L)},
                         batch_size_per_area=50,
                         param_ranges={t_symbol: 0.0})
    self.add(IC, name="IC")

    #boundary conditions
    BC1 = geo.boundary_bc(outvar_sympy={'T': Ts},
                         batch_size_per_area=50,
                         param_ranges=time_range,
                         criteria=Eq(x, 0.0))			 
    self.add(BC1, name="BC1")
    
    BC2 = geo.boundary_bc(outvar_sympy={'T': Ti},
                         batch_size_per_area=500,
                         param_ranges=time_range,
                         criteria=Eq(x, L))
    self.add(BC2, name="BC2")

    # interior
    interior = geo.interior_bc(outvar_sympy={'heat_equation': 0},
                               bounds={x: (0, L)},
                               batch_size_per_area=100,
                               lambda_sympy={'lambda_heat_equation': 1.0},
                               param_ranges=time_range)
    self.add(interior, name="Interior")

class HeatVal(ValidationDomain):
  def __init__(self, **config):
    super(HeatVal, self).__init__()
    # make validation data
    deltaTT = 0.005
    deltaX = 0.0005
    x = np.arange(0, L, deltaX)
    t = np.arange(0, time_length, deltaTT)
    X, TT = np.meshgrid(x, t)
    X = np.expand_dims(X.flatten(), axis=-1)
    TT = np.expand_dims(TT.flatten(), axis=-1)
       
    # Analytical Solution using 25 terms of Eq. 5.42
    T = erf(X/(2*(alpha*TT)**0.5))*(Ti-Ts)+Ts
    T = np.nan_to_num(T,nan=Ts)
    
    invar_numpy = {'x': X, 't': TT}
    outvar_numpy = {'T': T}
    val = Validation.from_numpy(invar_numpy, outvar_numpy)
    self.add(val, name='Val')


class HeatMonitor(MonitorDomain):
  def __init__(self, **config):
    super(HeatMonitor, self).__init__()
    x = Symbol('x')
    
    # metric for peak temp
    temp_monitor = Monitor(geo.sample_interior(100, bounds={x: (0, L)}, param_ranges={t_symbol: (0,1)}),
                         {'peak_temp': lambda var: tf.reduce_max(var['T'])})
    self.add(temp_monitor, 'PeakTempMonitor')


# Define neural network
class HeatSolver(Solver):
  train_domain = HeatTrain
  val_domain = HeatVal
  monitor_domain = HeatMonitor
  arch = GRUArch


  def __init__(self, **config):
    super(HeatSolver, self).__init__(**config)
    
    self.arch.set_time_steps(0, time_length, int(1000/100))
    self.equations = HeatEquation1D(alpha=1.34e-5).make_node()
    heat_net = self.arch.make_node(name='heat_net',
                                   inputs=['x', 't'],
                                   outputs=['T'])
    self.nets = [heat_net]

  @classmethod # Explain This
  def update_defaults(cls, defaults):
    defaults.update({
        'network_dir': './network_checkpoint_heat',
        'max_steps': 150000,
        'decay_steps': 1000,
        'nr_layers':3,
        'layer_size':256
        })

if __name__ == '__main__':
  ctr = ModulusController(HeatSolver)
  ctr.run()
  