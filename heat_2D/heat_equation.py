"""Heat equation
Reference: Fundamentals of Heat and Mass Transfer (Bergman & Lavine)
"""

from sympy import Symbol, Function

from modulus.pdes import PDES
from modulus.variables import Variables

class HeatEquation2D(PDES):
  """
  Heat equation 2D
  
  """

  name = 'HeatEquation2D'

  def __init__(self):
    # coordinates
    x = Symbol('x')
    y = Symbol('y')

    # make input variables
    input_variables = {'x':x,'y':y}

    # make u function
    T = Function('T')(*input_variables)

    # set equations
    self.equations = Variables()
    self.equations['heat_equation'] = T.diff(x, 2) + T.diff(y,2)
