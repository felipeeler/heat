from sympy import Symbol, Eq, Abs
import numpy as np

from modulus.solver import Solver
from modulus.dataset import TrainDomain, ValidationDomain
from modulus.data import Validation
from modulus.sympy_utils.geometry_2d import Rectangle
# from modulus.csv_utils.csv_rw import csv_to_dict
# from modulus.PDES import NavierStokes
from heat_equation import HeatEquation2D
from modulus.controller import ModulusController

# fixing the random seed for the weights generation:
np.random.seed(1)

# params for domain
height = 1.0
width = 1.0
T2 = 150.0
T1 = 50.0

# define geometry
rec = Rectangle((-width / 2, -height / 2), (width / 2, height / 2))
geo = rec

# define sympy varaibles to parametize domain curves
x, y = Symbol('x'), Symbol('y')

class Heat2DTrain(TrainDomain):
  def __init__(self, **config):
    super(Heat2DTrain, self).__init__()

    #top wall
    topWall = geo.boundary_bc(outvar_sympy={'T': T2},
                              batch_size_per_area=100,
                              lambda_sympy={'lambda_T': 1.0 - 2 * Abs(x)},
                              criteria=Eq(y, height / 2))
    self.add(topWall, name="TopWall")

    # bottom wall
    bottomWall = geo.boundary_bc(outvar_sympy={'T': T1},
                                 batch_size_per_area=100,
                                 criteria = y < height / 2)
    self.add(bottomWall, name="bottomWall")

    # left wall
#    leftWall = geo.boundary_bc(outvar_sympy={'T': T1},
#                                 batch_size_per_area=100,
#                                 criteria=Eq(x, - width / 2))
#    self.add(leftWall, name="leftWall")

    
    # right wall
#    rightWall = geo.boundary_bc(outvar_sympy={'T': T1},
#                                 batch_size_per_area=100,
#                                 criteria=Eq(x,  width / 2))
#    self.add(rightWall, name="rightWall")

    # interior
    interior = geo.interior_bc(outvar_sympy={'heat_equation': 0},
                               bounds={x: (-width / 2, width / 2),
                                       y: (-height / 2, height / 2)},
                               batch_size_per_area=4000)
    self.add(interior, name="Interior")


# validation data

class Heat2DVal(ValidationDomain):
  def __init__(self, **config):
    super(Heat2DVal, self).__init__()
    # make validation data
    deltaX = 0.01
    deltaY = 0.01
    x = np.arange(0, width, deltaX)
    y = np.arange(0, height, deltaY)
    X, Y = np.meshgrid(x, y)
    X = np.expand_dims(X.flatten(), axis=-1)
    Y = np.expand_dims(Y.flatten(), axis=-1)
    n = 1
    teta =  ((-1)**(n+1)+1)/n*np.sin(n*np.pi*X/width)*np.sinh(n*np.pi*Y/width)/np.sinh(n*np.pi*height/width)
    for n in range(2,100):
        teta += ((-1)**(n+1)+1)/n*np.sin(n*np.pi*X/width)*np.sinh(n*np.pi*Y/width)/np.sinh(n*np.pi*height/width)

    teta=2/np.pi*teta
    T = teta*(T2-T1)+T1
    x = np.arange(-width/2, width/2, deltaX)
    y = np.arange(-height/2, height/2, deltaY)
    X, Y = np.meshgrid(x, y)
    X = np.expand_dims(X.flatten(), axis=-1)
    Y = np.expand_dims(Y.flatten(), axis=-1)
    invar_numpy = {'x': X, 'y': Y}
    outvar_numpy = {'T': T}
    val = Validation.from_numpy(invar_numpy, outvar_numpy)
    self.add(val, name='Val')


class Heat2DSolver(Solver):
  train_domain = Heat2DTrain
  val_domain = Heat2DVal

  def __init__(self, **config):
    super(Heat2DSolver, self).__init__(**config)
    self.equations = HeatEquation2D().make_node()
    heat_net = self.arch.make_node(name='heat_net',
                                   inputs=['x', 'y'],
                                   outputs=['T'])
    self.nets = [heat_net]

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'network_dir': './network_checkpoint_heat_2d',
        'decay_steps': 8000,
        'max_steps': 300000,
        'nr_layers':6,
        'layer_size':64
    })


if __name__ == '__main__':
  ctr = ModulusController(Heat2DSolver)
  ctr.run()
