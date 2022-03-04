"""Heat equation
Reference: Fundamentals of Heat and Mass Transfer (Bergman & Lavine)
"""

from sympy import Symbol, Function, Number

from modulus.pdes import PDES
from modulus.variables import Variables

class HeatEquation1D(PDES):
  """
  Heat equation 1D
  

  Parameters
  ==========
  alpha : float, string
      constant thermal conductivity
  """

  name = 'HeatEquation1D'

  def __init__(self, alpha=1.34e-5):
    # coordinates
    x = Symbol('x')

    # time
    t = Symbol('t')

    # make input variables
    input_variables = {'x':x,'t':t}

    # make u function
    T = Function('T')(*input_variables)

    # wave speed coefficient
    if type(alpha) is str:
      alpha = Function(alpha)(*input_variables)
    elif type(alpha) in [float, int]:
      alpha = Number(alpha)

    # set equations
    self.equations = Variables()
    self.equations['heat_equation'] = T.diff(t, 1) - alpha * T.diff(x,2)
    