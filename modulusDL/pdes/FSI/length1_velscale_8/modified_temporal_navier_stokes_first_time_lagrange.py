"""Equations related to Navier Stokes Equations
"""

from sympy import Symbol, Function, Number

from modulus.eq.pde import PDE
from modulus.node import Node


class TemporalNavierStokes_1st_time_lagrange(PDE):
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
        u_1,v_1,w_1 = Symbol("u_1"), Symbol("v_1"), Symbol("w_1")
        u_2,v_2,w_2 = Symbol("u"), Symbol("v"), Symbol("w")

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
        vel_scale = 0.8
        length_scale = 0.1
        time_scale = 1./(vel_scale/length_scale)
        time_step = 0.02/time_scale

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
            + rho * ((-3.*u + 4.*u_1 - u_2) / (2.*time_step))
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
            + rho * ((-3.*v + 4.*v_1 - v_2) / (2.*time_step))
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
            + rho * ((-3.*w + 4.*w_1 - w_2) / (2.*time_step))
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
                + rho * ((-3.*u + 4.*u_1 - u_2) / (2.*time_step))
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
                + rho * ((-3.*u + 4.*u_1 - u_2) / (2.*time_step))
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
                + rho * ((-3.*u + 4.*u_1 - u_2) / (2.*time_step))
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
                + rho * ((-3.*v + 4.*v_1 - v_2) / (2.*time_step))
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
                + rho * ((-3.*v + 4.*v_1 - v_2) / (2.*time_step))
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
                + rho * ((-3.*v + 4.*v_1 - v_2) / (2.*time_step))
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
                + rho * ((-3.*w + 4.*w_1 - w_2) / (2.*time_step))
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
                + rho * ((-3.*w + 4.*w_1 - w_2) / (2.*time_step))
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
                + rho * ((-3.*w + 4.*w_1 - w_2) / (2.*time_step))
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


class wall_shear_stress(PDE):
    """
    Wall shear stress
    add wall shear stress equations
    $\tau_{xx}=$

    """
    name = "wall_shear_stress"

    def __init__(self, nu,case_coord_strList=None,case_param_strList=None, rho=1, dim=3, mixed_form=False):
        # set params
        self.dim = dim
        self.mixed_form = mixed_form
        if case_param_strList is None:
            case_param_strList = {}
        if case_coord_strList is None:
            case_coord_strList = ["x", "y", "z"]
        # if (case_coord_strList) == 1:
            # case_coord_strList = case_coord_strList+["y", "z"]
        # elif (case_coord_strList) == 2:
            # case_coord_strList = case_coord_strList+["z"]
        # coordinates
        
        print("hey")
		
        x = Symbol(case_coord_strList[0])
        y = Symbol(case_coord_strList[1])
        z = Symbol(case_coord_strList[2])
        input_variables = {case_coord_strList[0]: x, case_coord_strList[1]: y, case_coord_strList[2]: z}
        for key in case_param_strList:
            input_variables[key] = case_param_strList[key]
        if self.dim == 2:
            input_variables.pop("z")


        # velocity componets
        u_0 = Function("u_0")(*input_variables)
        v_0 = Function("v_0")(*input_variables)
        u_1 = Function("u_1")(*input_variables)
        v_1 = Function("v_1")(*input_variables)
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        if self.dim == 3:
            w_0 = Function("w_0")(*input_variables)
            w_1 = Function("w_1")(*input_variables)
            w = Function("w")(*input_variables)
        else:
            w_0 = Number(0)
            w = Number(0)
            w_1 = Number(0)

        # pressure
        p = Function("p")(*input_variables)

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

        # set equations
        self.equations = {}
        
        self.equations["partial_ux"] = u.diff(x)
        self.equations['partial_uy'] = u.diff(y)
        self.equations['partial_uz'] = u.diff(z)
        self.equations["partial_vx"] = v.diff(x)
        self.equations['partial_vy'] = v.diff(y)
        self.equations['partial_vz'] = v.diff(z)
        self.equations["partial_wx"] = w.diff(x)
        self.equations['partial_wy'] = w.diff(y)
        self.equations['partial_wz'] = w.diff(z)
        
        self.equations["partial_ux_0"] = u_0.diff(x)
        self.equations['partial_uy_0'] = u_0.diff(y)
        self.equations['partial_uz_0'] = u_0.diff(z)
        self.equations["partial_vx_0"] = v_0.diff(x)
        self.equations['partial_vy_0'] = v_0.diff(y)
        self.equations['partial_vz_0'] = v_0.diff(z)
        self.equations["partial_wx_0"] = w_0.diff(x)
        self.equations['partial_wy_0'] = w_0.diff(y)
        self.equations['partial_wz_0'] = w_0.diff(z)
        
        self.equations["partial_ux_1"] = u_1.diff(x)
        self.equations['partial_uy_1'] = u_1.diff(y)
        self.equations['partial_uz_1'] = u_1.diff(z)
        self.equations["partial_vx_1"] = v_1.diff(x)
        self.equations['partial_vy_1'] = v_1.diff(y)
        self.equations['partial_vz_1'] = v_1.diff(z)
        self.equations["partial_wx_1"] = w_1.diff(x)
        self.equations['partial_wy_1'] = w_1.diff(y)
        self.equations['partial_wz_1'] = w_1.diff(z)

        # I think there is no relaxed version for non-analytical geometries
        # The scale of stress is ML^-1T^-2
        if self.dim == 2:
            
            self.equations.pop("partial_uz")
            self.equations.pop("partial_vz")
            self.equations.pop("partial_wx")
            self.equations.pop("partial_wy")
            self.equations.pop("partial_wz")
            