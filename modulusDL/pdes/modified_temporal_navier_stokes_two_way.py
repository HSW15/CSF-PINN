"""Equations related to Navier Stokes Equations
"""

from sympy import Symbol, Function, Number

from modulus.eq.pde import PDE
from modulus.node import Node


class TemporalNavierStokes_two_way(PDE):
    """
    Compressible Navier Stokes equations with third-order derivatives to be used for gradient-enhanced training.

    Parameters
    ==========
    nu : float, Sympy Symbol/Expr, str
        The kinematic viscosity. If `nu` is a str then it is
        converted to Sympy Function of form `nu(x,y,z,t)`.
        If `nu` is a Sympy Symbol or Expression then this
        is substituted into the equation. This allows for
        variable viscosity.
    rho : float, Sympy Symbol/Expr, str
        The density of the fluid. If `rho` is a str then it is
        converted to Sympy Function of form 'rho(x,y,z,t)'.
        If 'rho' is a Sympy Symbol or Expression then this
        is substituted into the equation to allow for
        compressible Navier Stokes. Default is 1.
    dim : int
        Dimension of the Navier Stokes (2 or 3). Default is 3.
    time : bool
        If time-dependent equations or not. Default is True.

    """

    name = "TemporalNavierStokes"

    def __init__(self, nu, rho=1, dim=3, time=True):
        # set params
        self.dim = dim
        self.time = time

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        u_1,v_1,w_1 = Symbol("u"), Symbol("v"), Symbol("w")

        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        # velocity componets
        u = Function("u_0")(*input_variables)
        v = Function("v_0")(*input_variables)
        if self.dim == 3:
            w = Function("w_0")(*input_variables)
        else:
            w = Number(0)
            w_0 = Number(0)

        # pressure
        p = Function("p_0")(*input_variables)

        # kinematic viscosity
        if isinstance(nu, str):
            nu = Function(nu)(*input_variables)
        elif isinstance(nu, (float, int)):
            nu = Number(nu)

        # density
        if isinstance(rho, str):
            rho = Function(rho)(*input_variables)
        elif isinstance(rho, (float, int)):
            rho = Number(rho)

        # dynamic viscosity
        mu = rho * nu
        vel_scale = 0.1
        length_scale = 0.005
        time_scale = 1./(vel_scale/length_scale)
        time_step = 0.011669993/time_scale

        # curl
        curl = Number(0) if rho.diff() == 0 else u.diff(x) + v.diff(y) + w.diff(z)

        # set equations
        self.equations = {}
        self.equations["continuity_1st"] = (
            rho.diff(t) + (rho * u).diff(x) + (rho * v).diff(y) + (rho * w).diff(z)
        )
        self.equations["continuity_dx_1st"] = (
            rho.diff(t) + (rho * u).diff(x) + (rho * v).diff(y) + (rho * w).diff(z)
        ).diff(x)
        self.equations["continuity_dy_1st"] = (
            rho.diff(t) + (rho * u).diff(x) + (rho * v).diff(y) + (rho * w).diff(z)
        ).diff(y)
        self.equations["continuity_dz_1st"] = (
            rho.diff(t) + (rho * u).diff(x) + (rho * v).diff(y) + (rho * w).diff(z)
        ).diff(z)
        self.equations["momentum_x_1st"] = (
            (rho * u).diff(t)
            + rho * ((u_1 - u) / (time_step))
            + (
                u * ((rho * u).diff(x))
                + v * ((rho * u).diff(y))
                + w * ((rho * u).diff(z))
                + rho * u * (curl)
            )
            + p.diff(x)
            - (-2 / 3 * mu * (curl)).diff(x)
            - (mu * u.diff(x)).diff(x)
            - (mu * u.diff(y)).diff(y)
            - (mu * u.diff(z)).diff(z)
            - (mu * (curl).diff(x))
        )
        self.equations["momentum_y_1st"] = (
            (rho * v).diff(t)
            + rho * ((v_1 - v) / (time_step))
            + (
                u * ((rho * v).diff(x))
                + v * ((rho * v).diff(y))
                + w * ((rho * v).diff(z))
                + rho * v * (curl)
            )
            + p.diff(y)
            - (-2 / 3 * mu * (curl)).diff(y)
            - (mu * v.diff(x)).diff(x)
            - (mu * v.diff(y)).diff(y)
            - (mu * v.diff(z)).diff(z)
            - (mu * (curl).diff(y))
        )
        self.equations["momentum_z_1st"] = (
            (rho * w).diff(t)
            + rho * ((w_1 - w) / (time_step))
            + (
                u * ((rho * w).diff(x))
                + v * ((rho * w).diff(y))
                + w * ((rho * w).diff(z))
                + rho * w * (curl)
            )
            + p.diff(z)
            - (-2 / 3 * mu * (curl)).diff(z)
            - (mu * w.diff(x)).diff(x)
            - (mu * w.diff(y)).diff(y)
            - (mu * w.diff(z)).diff(z)
            - (mu * (curl).diff(z))
        )
        self.equations["momentum_x_dx_1st"] = (
            (
                (rho * u).diff(t)
                + rho * ((u_1 - u) / (time_step))
                + (
                    u * ((rho * u).diff(x))
                    + v * ((rho * u).diff(y))
                    + w * ((rho * u).diff(z))
                    + rho * u * (curl)
                )
                + p.diff(x)
                - (-2 / 3 * mu * (curl)).diff(x)
                - (mu * u.diff(x)).diff(x)
                - (mu * u.diff(y)).diff(y)
                - (mu * u.diff(z)).diff(z)
                - (mu * (curl).diff(x))
            )
        ).diff(x)
        self.equations["momentum_x_dy_1st"] = (
            (
                (rho * u).diff(t)
                + rho * ((u_1 - u) / (time_step))
                + (
                    u * ((rho * u).diff(x))
                    + v * ((rho * u).diff(y))
                    + w * ((rho * u).diff(z))
                    + rho * u * (curl)
                )
                + p.diff(x)
                - (-2 / 3 * mu * (curl)).diff(x)
                - (mu * u.diff(x)).diff(x)
                - (mu * u.diff(y)).diff(y)
                - (mu * u.diff(z)).diff(z)
                - (mu * (curl).diff(x))
            )
        ).diff(y)
        self.equations["momentum_x_dz_1st"] = (
            (
                (rho * u).diff(t)
                + rho * ((u_1 - u) / (time_step))
                + (
                    u * ((rho * u).diff(x))
                    + v * ((rho * u).diff(y))
                    + w * ((rho * u).diff(z))
                    + rho * u * (curl)
                )
                + p.diff(x)
                - (-2 / 3 * mu * (curl)).diff(x)
                - (mu * u.diff(x)).diff(x)
                - (mu * u.diff(y)).diff(y)
                - (mu * u.diff(z)).diff(z)
                - (mu * (curl).diff(x))
            )
        ).diff(z)
        self.equations["momentum_y_dx_1st"] = (
            (
                (rho * v).diff(t)
                + rho * ((v_1 - v) / (time_step))
                + (
                    u * ((rho * v).diff(x))
                    + v * ((rho * v).diff(y))
                    + w * ((rho * v).diff(z))
                    + rho * v * (curl)
                )
                + p.diff(y)
                - (-2 / 3 * mu * (curl)).diff(y)
                - (mu * v.diff(x)).diff(x)
                - (mu * v.diff(y)).diff(y)
                - (mu * v.diff(z)).diff(z)
                - (mu * (curl).diff(y))
            )
        ).diff(x)
        self.equations["momentum_y_dy_1st"] = (
            (
                (rho * v).diff(t)
                + rho * ((v_1 - v) / (time_step))
                + (
                    u * ((rho * v).diff(x))
                    + v * ((rho * v).diff(y))
                    + w * ((rho * v).diff(z))
                    + rho * v * (curl)
                )
                + p.diff(y)
                - (-2 / 3 * mu * (curl)).diff(y)
                - (mu * v.diff(x)).diff(x)
                - (mu * v.diff(y)).diff(y)
                - (mu * v.diff(z)).diff(z)
                - (mu * (curl).diff(y))
            )
        ).diff(y)
        self.equations["momentum_y_dz_1st"] = (
            (
                (rho * v).diff(t)
                + rho * ((v_1 - v) / (time_step))
                + (
                    u * ((rho * v).diff(x))
                    + v * ((rho * v).diff(y))
                    + w * ((rho * v).diff(z))
                    + rho * v * (curl)
                )
                + p.diff(y)
                - (-2 / 3 * mu * (curl)).diff(y)
                - (mu * v.diff(x)).diff(x)
                - (mu * v.diff(y)).diff(y)
                - (mu * v.diff(z)).diff(z)
                - (mu * (curl).diff(y))
            )
        ).diff(z)
        self.equations["momentum_z_dx_1st"] = (
            (
                (rho * w).diff(t)
                + rho * ((w_1 - w) / (time_step))
                + (
                    u * ((rho * w).diff(x))
                    + v * ((rho * w).diff(y))
                    + w * ((rho * w).diff(z))
                    + rho * w * (curl)
                )
                + p.diff(z)
                - (-2 / 3 * mu * (curl)).diff(z)
                - (mu * w.diff(x)).diff(x)
                - (mu * w.diff(y)).diff(y)
                - (mu * w.diff(z)).diff(z)
                - (mu * (curl).diff(z))
            )
        ).diff(x)
        self.equations["momentum_z_dy_1st"] = (
            (
                (rho * w).diff(t)
                + rho * ((w_1 - w) / (time_step))
                + (
                    u * ((rho * w).diff(x))
                    + v * ((rho * w).diff(y))
                    + w * ((rho * w).diff(z))
                    + rho * w * (curl)
                )
                + p.diff(z)
                - (-2 / 3 * mu * (curl)).diff(z)
                - (mu * w.diff(x)).diff(x)
                - (mu * w.diff(y)).diff(y)
                - (mu * w.diff(z)).diff(z)
                - (mu * (curl).diff(z))
            )
        ).diff(y)
        self.equations["momentum_z_dz_1st"] = (
            (
                (rho * w).diff(t)
                + rho * ((w_1 - w) / (time_step))
                + (
                    u * ((rho * w).diff(x))
                    + v * ((rho * w).diff(y))
                    + w * ((rho * w).diff(z))
                    + rho * w * (curl)
                )
                + p.diff(z)
                - (-2 / 3 * mu * (curl)).diff(z)
                - (mu * w.diff(x)).diff(x)
                - (mu * w.diff(y)).diff(y)
                - (mu * w.diff(z)).diff(z)
                - (mu * (curl).diff(z))
            )
        ).diff(z)

        if self.dim == 2:
            self.equations.pop("momentum_z_1st")
            self.equations.pop("continuity_dz_1st")
            self.equations.pop("momentum_x_dz_1st")
            self.equations.pop("momentum_y_dz_1st")
            self.equations.pop("momentum_z_dx_1st")
            self.equations.pop("momentum_z_dy_1st")
            self.equations.pop("momentum_z_dz_1st")
