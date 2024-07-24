import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
sys.path.insert(0,"/examples")
from modulusDL.models.arch import CustomModuleArch,CustomDualInputModuleArch, InheritedFullyConnectedFlexiLayerSizeArch, FullyConnectedFlexiLayerSizeArch, SingleInputInheritedFullyConnectedFlexiLayerSizeArch, CaseIDtoFeatureArch
from modulusDL.eq.pde import NavierStokes_CoordTransformed
from modulusDL.solver.solver import Solver_ReduceLROnPlateauLoss
from modulusDL.models.arch import AdditionArch
import shutil
from modulus.utils.io.vtk import var_to_polyvtk 
            
import numpy as np
from sympy import (
    Symbol,
    Function,
    Eq,
    Number,
    Abs,
    Max,
    Min,
    sqrt,
    pi,
    sin,
    cos,
    atan,
    atan2,
    acos,
    asin,
    sign,
    exp,
)

import modulus
from modulus.hydra import to_absolute_path, instantiate_arch, ModulusConfig, to_yaml
from modulus.utils.io import csv_to_dict
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.geometry import Bounds
from modulus.geometry.geometry import Geometry, csg_curve_naming
from modulus.geometry.curve import SympyCurve
from modulus.models.dgm import DGMArch
from modulus.geometry.helper import _sympy_sdf_to_sdf
from modulus.geometry.parameterization import Parameterization, Parameter, Bounds
from modulus.models.fully_connected import FullyConnectedArch
from modulus.geometry.primitives_2d import Line, Circle, Channel2D
from modulusDL.pdes.two_way_couple.modified_temporal_navier_stokes_first_time_lagrange import TemporalNavierStokes_1st_time_lagrange
from modulusDL.pdes.two_way_couple.modified_temporal_navier_stokes_central import TemporalNavierStokes_2nd_time_point
from modulusDL.pdes.two_way_couple.modified_temporal_navier_stokes_2nd_time_lagrange import TemporalNavierStokes_3rd_time_point_lagrange
from modulus.eq.pdes.basic import NormalDotVec, Curl
from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
    PointwiseConstraint,
)
#from modulus.sym.geometry.primitives_3d import Box
#from modulus.sym.utils.io.vtk import var_to_polyvtk

from modulus.domain.inferencer import PointwiseInferencer
from modulus.utils.io.plotter import ValidatorPlotter, InferencerPlotter
from modulus.domain.validator import PointwiseValidator
from modulus.key import Key
from modulus import quantity
from modulus.eq.non_dim import NonDimensionalizer, Scaler
from modulus.eq.pde import PDE
from modulus.geometry.tessellation import Tessellation

from modulus.domain.monitor import PointwiseMonitor
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import stl
from stl import mesh
import pandas as pd

        
class transducer_probe(torch.nn.Module):
    def __init__(self, transducer_center):
        super().__init__()
        transducer_center=torch.tensor(transducer_center)
        self.register_buffer("transducer_center", transducer_center, persistent=False)
        
    def forward(self,x):#1:x,2:y,3:z
        
        dxdp = (-x[...,0:1] + self.transducer_center[0])
        dydp = (-x[...,1:2] + self.transducer_center[1])
        dzdp = (-x[...,2:3] + self.transducer_center[2])
        
        mag=torch.sqrt(dxdp**2.+dydp**2.+dzdp**2.)
        vec_x=dxdp/mag
        vec_y=dydp/mag
        vec_z=dzdp/mag
        
        return torch.cat((vec_x,vec_y,vec_z),-1)
        
class US_doppler_conversion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,x):#1:r_x,2:r_y,3:r_z 4:u, 5:v, 6:w
        
        us_mag = (x[...,0:1])*(x[...,3:4]) + (x[...,1:2])*(x[...,4:5]) + (x[...,2:3])*(x[...,5:6])  # u.r
        
        
        return us_mag

@modulus.main(config_path="conf", config_name="conf_lr")
def run(cfg: ModulusConfig) -> None:
    
    cfg.optimizer.lr=1e-03
    nu = quantity(0.0035, "kg/(m*s)")
    rho = quantity(1060, "kg/m^3")
    
    center_hardcode = (0.0171423, 0.0205446, 0.0170172)
    
    length_scale_amp = 0.005  # turn off scaling for now
    vel_scale_amp = 0.1
    
    distance_to_transducer = 0.1 #this is in meters
    distance_to_transducer = (0., 0.1, 0.)
    transducer_center_hardcode = (0.0171423, 0.0205446, 0.0170172)
    transducer_point = tuple(map(sum,zip(distance_to_transducer,transducer_center_hardcode)))
    
    temp_center_hardcode = tuple(ti*-1 for ti in center_hardcode)
    transducer_point = tuple(map(sum,zip(transducer_point,temp_center_hardcode)))
    transducer_point = tuple(ti/length_scale_amp for ti in transducer_point)
    
    velocity_scale = quantity(vel_scale_amp, "m/s")
    density_scale = rho
    length_scale = quantity(length_scale_amp, "m")
    nd = NonDimensionalizer(
        length_scale=length_scale,
        time_scale=length_scale / velocity_scale,
        mass_scale=density_scale * (length_scale ** 3),
    )
    
    point_path = to_absolute_path("./temporal_stls")
    
    interior_mesh_pre = Tessellation.from_stl(
        point_path + "/Interior/P2_wk31_interior_01072_interior.stl", airtight=True
    )
    
    interior_mesh_2nd_time = Tessellation.from_stl(
        point_path + "/Interior/P2_wk31_interior_01082_interior.stl", airtight=True
    )
    
    interior_mesh_3rd_time = Tessellation.from_stl(
        point_path + "/Interior/P2_wk31_interior_01092_interior.stl", airtight=True
    )
    
    # normalize meshes
    def normalize_mesh(mesh, center, scale):
        mesh = mesh.translate([-c for c in center])
        mesh = mesh.scale(scale)
        return mesh

    # normalize invars
    def normalize_invar(invar, center, scale, dims=2):
        invar["x"] -= center[0]
        invar["y"] -= center[1]
        invar["z"] -= center[2]
        invar["x"] *= scale
        invar["y"] *= scale
        invar["z"] *= scale
        if "area" in invar.keys():
            invar["area"] *= scale**dims
        return invar
    
    def normalize_invar_vel_pre(invar, center, scale, vel_scale, pres_scale, dims=2):
        
        invar["p_0"] *= pres_scale
        if "area" in invar.keys():
            invar["area"] *= scale**dims
        return invar
    
    def normalize_invar_vel_2nd_time(invar, center, scale, vel_scale, pres_scale, dims=2):
        
        if "area" in invar.keys():
            invar["area"] *= scale**dims
        return invar
    
    def normalize_invar_vel(invar, center, scale, vel_scale, pres_scale, dims=2):
        
        invar["p"] *= pres_scale
        if "area" in invar.keys():
            invar["area"] *= scale**dims
        return invar
    
    center = center_hardcode
    print('Overall geometry center: ', center)

    interior_mesh_pre = normalize_mesh(interior_mesh_pre, center, 1./length_scale_amp)
    
    interior_mesh_2nd_time = normalize_mesh(interior_mesh_2nd_time, center, 1./length_scale_amp)
    
    interior_mesh_3rd_time = normalize_mesh(interior_mesh_3rd_time, center, 1./length_scale_amp)
    
    domain = Domain()
    
    ns_1st = TemporalNavierStokes_1st_time_lagrange(nu=nd.ndim(nu), rho=nd.ndim(rho), dim=3, time=False)
    ns_2nd = TemporalNavierStokes_2nd_time_point(nu=nd.ndim(nu), rho=nd.ndim(rho), dim=3, time=False)
    ns_3rd = TemporalNavierStokes_3rd_time_point_lagrange(nu=nd.ndim(nu), rho=nd.ndim(rho), dim=3, time=False)
    
   
    flow_net_pre = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("u_0"), Key("v_0"), Key("w_0"), Key("p_0")],
        cfg=cfg.arch.fully_connected,
        layer_size=200,
        nr_layers=10,
        # adaptive_activations=cfg.custom.adaptive_activations,
    )
    vec_ref_mod_pre=CustomModuleArch(
        input_keys=[Key("x"),Key("y"),Key("z")],
        output_keys=[Key("r_x_0"), Key("r_y_0"), Key("r_z_0")],
        module=transducer_probe(transducer_point)
    )
    
    US_doppler_pre = CustomModuleArch(
        input_keys=[Key("r_x_0"), Key("r_y_0"), Key("r_z_0"),Key("u_0"), Key("v_0"), Key("w_0")],
        output_keys=[Key("us_mag_0")],
        module=US_doppler_conversion(),
    )
    
    flow_net_2nd_time = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("u_1"), Key("v_1"), Key("w_1"), Key("p_1")],
        cfg=cfg.arch.fully_connected,
        layer_size=200,
        nr_layers=10,
        # adaptive_activations=cfg.custom.adaptive_activations,
    )
    vec_ref_mod_2nd_time=CustomModuleArch(
        input_keys=[Key("x"),Key("y"),Key("z")],
        output_keys=[Key("r_x_1"), Key("r_y_1"), Key("r_z_1")],
        module=transducer_probe(transducer_point)
    )
    
    US_doppler_2nd_time = CustomModuleArch(
        input_keys=[Key("r_x_1"), Key("r_y_1"), Key("r_z_1"),Key("u_1"), Key("v_1"), Key("w_1")],
        output_keys=[Key("us_mag_1")],
        module=US_doppler_conversion(),
    )
    
    flow_net_3rd_time = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        cfg=cfg.arch.fully_connected,
        layer_size=200,
        nr_layers=10,
        # adaptive_activations=cfg.custom.adaptive_activations,
    )
    vec_ref_mod_3rd_time=CustomModuleArch(
        input_keys=[Key("x"),Key("y"),Key("z")],
        output_keys=[Key("r_x_2"), Key("r_y_2"), Key("r_z_2")],
        module=transducer_probe(transducer_point)
    )
    
    US_doppler_3rd_time = CustomModuleArch(
        input_keys=[Key("r_x_2"), Key("r_y_2"), Key("r_z_2"),Key("u"), Key("v"), Key("w")],
        output_keys=[Key("us_mag")],
        module=US_doppler_conversion(),
    )
    
    normal_dot_vel = NormalDotVec(["u", "v", "w"])
    Curl(["u", "v", "w"], curl_name=["u", "v", "w"])
    nodes = (
            ns_1st.make_nodes()
            + ns_2nd.make_nodes()
            + ns_3rd.make_nodes()
            + normal_dot_vel.make_nodes()
            + [flow_net_pre.make_node(name="flow_network_pre")]
            + [vec_ref_mod_pre.make_node(name='vec_ref_mod_pre')]
            + [US_doppler_pre.make_node(name="US_doppler_pre")]
            + [flow_net_2nd_time.make_node(name="flow_net_2nd_time")]
            + [vec_ref_mod_2nd_time.make_node(name='vec_ref_mod_2nd_time')]
            + [US_doppler_2nd_time.make_node(name="US_doppler_2nd_time")]
            + [flow_net_3rd_time.make_node(name="flow_net_3rd_time")]
            + [vec_ref_mod_3rd_time.make_node(name='vec_ref_mod_3rd_time')]
            + [US_doppler_3rd_time.make_node(name="US_doppler_3rd_time")]
            + Scaler(
                ["u_0","v_0", "w_0", "p_0", "u_1","v_1", "w_1", "p_1", "u", "v","w","p"],
                ["u_0_scaled", "v_0_scaled", "w_0_scaled", "p_0_scaled", "u_1_scaled", "v_1_scaled", "w_1_scaled", "p_1_scaled", "u_scaled", "v_scaled", "w_scaled", "p_scaled"],
                ["m/s", "m/s","m/s", "m^2/s^2","m/s", "m/s","m/s", "m^2/s^2", "m/s", "m/s","m/s", "m^2/s^2"],
                nd,
            ).make_node()
    )
    
    domain = Domain()
    batchsizefactor=1
    
    inflow_var = csv_to_dict(
       to_absolute_path("./temporal_stls/3_time_point_temporal/1092/inlet.csv"))
    inflow_invar = {
       key: value for key, value in inflow_var.items() if key in ["x", "y", "z"]
    }
    inflow_invar = normalize_invar(inflow_invar, center, 1./length_scale_amp, dims=3)
    inflow_outvar = {
       key: value for key, value in inflow_var.items() if key in ["p"]
    }
    inflow_outvar = normalize_invar_vel(inflow_outvar, center, 1./length_scale_amp, (1./vel_scale_amp), (1./((vel_scale_amp**2.)*1060.)), dims=3)
    
    inlet_numpy = PointwiseConstraint.from_numpy(
       nodes,
       inflow_invar,
       inflow_outvar,
       batch_size=10000,
    )
    domain.add_constraint(inlet_numpy, "inlet_numpy_post")
    
    inflow_var = csv_to_dict(
       to_absolute_path("./temporal_stls/3_time_point_temporal/1082/inlet.csv"))
    inflow_invar = {
       key: value for key, value in inflow_var.items() if key in ["x", "y", "z"]
    }
    inflow_invar = normalize_invar(inflow_invar, center, 1./length_scale_amp, dims=3)
    inflow_outvar = {
       key: value for key, value in inflow_var.items() if key in ["p_1"]
    }
    inflow_outvar = normalize_invar_vel_2nd_time(inflow_outvar, center, 1./length_scale_amp, (1./vel_scale_amp), (1./((vel_scale_amp**2.)*1060.)), dims=3)
    
    inlet_numpy = PointwiseConstraint.from_numpy(
       nodes,
       inflow_invar,
       inflow_outvar,
       batch_size=10000,
    )
    domain.add_constraint(inlet_numpy, "inlet_numpy_post")
    
    
    inflow_var = csv_to_dict(
       to_absolute_path("./temporal_stls/3_time_point_temporal/1072/1072_inlet.csv"))
    inflow_invar = {
       key: value for key, value in inflow_var.items() if key in ["x", "y", "z"]
    }
    inflow_invar = normalize_invar(inflow_invar, center, 1./length_scale_amp, dims=3)
    inflow_outvar = {
       key: value for key, value in inflow_var.items() if key in ["p_0"]
    }
    inflow_outvar = normalize_invar_vel_pre(inflow_outvar, center, 1./length_scale_amp, (1./vel_scale_amp), (1./((vel_scale_amp**2.)*1060.)), dims=3)
    
    inlet_numpy = PointwiseConstraint.from_numpy(
       nodes,
       inflow_invar,
       inflow_outvar,
       batch_size=10000,
    )
    domain.add_constraint(inlet_numpy, "inlet_numpy_pre")
    
    
    inflow_var = csv_to_dict(
       to_absolute_path("./temporal_stls/3_time_point_temporal/1072/1072_wall.csv"))
    inflow_invar = {
       key: value for key, value in inflow_var.items() if key in ["x", "y", "z"]
    }
    inflow_invar = normalize_invar(inflow_invar, center, 1./length_scale_amp, dims=3)
    inflow_outvar = {
       key: value * (1./vel_scale_amp) for key, value in inflow_var.items() if key in ["u_0", "v_0", "w_0"]
    }
    
    wall = PointwiseConstraint.from_numpy(
        nodes,
        inflow_invar,
        inflow_outvar,
        batch_size=10000,
    )
    domain.add_constraint(wall, "no_slip_1072")
    
    inflow_var = csv_to_dict(
       to_absolute_path("./temporal_stls/3_time_point_temporal/1082/wall.csv"))
    inflow_invar = {
       key: value for key, value in inflow_var.items() if key in ["x", "y", "z"]
    }
    inflow_invar = normalize_invar(inflow_invar, center, 1./length_scale_amp, dims=3)
    inflow_outvar = {
       key: value * (1./vel_scale_amp) for key, value in inflow_var.items() if key in ["u_1", "v_1", "w_1"]
    }
    
    wall = PointwiseConstraint.from_numpy(
        nodes,
        inflow_invar,
        inflow_outvar,
        batch_size=10000,
    )
    domain.add_constraint(wall, "no_slip_1082")
    
    
    inflow_var = csv_to_dict(
       to_absolute_path("./temporal_stls/3_time_point_temporal/1092/wall.csv"))
    inflow_invar = {
       key: value for key, value in inflow_var.items() if key in ["x", "y", "z"]
    }
    inflow_invar = normalize_invar(inflow_invar, center, 1./length_scale_amp, dims=3)
    inflow_outvar = {
       key: value * (1./vel_scale_amp) for key, value in inflow_var.items() if key in ["u", "v", "w"]
    }
    
    wall = PointwiseConstraint.from_numpy(
        nodes,
        inflow_invar,
        inflow_outvar,
        batch_size=10000,
    )
    domain.add_constraint(wall, "no_slip_1092")
    
    
    interior = PointwiseInteriorConstraint(
        nodes,
        geometry=interior_mesh_3rd_time,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        batch_size=4000,
        
    )
    domain.add_constraint(interior, "interior_1092")
    
    interior = PointwiseInteriorConstraint(
        nodes,
        geometry=interior_mesh_2nd_time,
        outvar={"continuity_2nd": 0, "momentum_x_2nd": 0, "momentum_y_2nd": 0, "momentum_z_2nd": 0},
        batch_size=4000,
    )
    domain.add_constraint(interior, "interior_1082")
    
    interior = PointwiseInteriorConstraint(
        nodes,
        geometry=interior_mesh_pre,
        outvar={"continuity_1st": 0, "momentum_x_1st": 0, "momentum_y_1st": 0, "momentum_z_1st": 0},
        batch_size=4000,
    )
    domain.add_constraint(interior, "interior_1072")
    
    
    inflow_var = csv_to_dict(
       to_absolute_path("./temporal_stls/3_time_point_temporal/1072/US_threshold_no_noise_new_model.csv"))
    inflow_invar = {
       key: value for key, value in inflow_var.items() if key in ["x", "y", "z"]
    }
    inflow_invar = normalize_invar(inflow_invar, center, 1./length_scale_amp, dims=3)
    inflow_outvar = {
       key: value*(1./vel_scale_amp) for key, value in inflow_var.items() if key in ["us_mag_0"]
    }
    doppler_numpy = PointwiseConstraint.from_numpy(
       nodes,
       inflow_invar,
       inflow_outvar,
       batch_size=10000,
    )
    domain.add_constraint(doppler_numpy, "color_doppler_mag_1072")
    
    
    inflow_var = csv_to_dict(
       to_absolute_path("./temporal_stls/3_time_point_temporal/1092/US_threshold_1092_no_noise.csv"))
    inflow_invar = {
       key: value for key, value in inflow_var.items() if key in ["x", "y", "z"]
    }
    inflow_invar = normalize_invar(inflow_invar, center, 1./length_scale_amp, dims=3)
    inflow_outvar = {
       key: value*(1./vel_scale_amp) for key, value in inflow_var.items() if key in ["us_mag"]
    }
    doppler_numpy = PointwiseConstraint.from_numpy(
       nodes,
       inflow_invar,
       inflow_outvar,
       batch_size=10000,
    )
    domain.add_constraint(doppler_numpy, "color_doppler_mag_1092")
    
    interior_pts=interior_mesh_3rd_time.sample_interior(100000)
   
    openfoam_invar_numpy={"x":interior_pts["x"],
                          "y":interior_pts["y"],
                          "z":interior_pts["z"],
                          }
    openfoam_inferencer=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy, output_names=["u_scaled", "v_scaled", "w_scaled","p_scaled"]
    )
    domain.add_inferencer(openfoam_inferencer, "inf_data_1092")
    
    interior_pts=interior_mesh_2nd_time.sample_interior(100000)
    
    openfoam_invar_numpy={"x":interior_pts["x"],
                          "y":interior_pts["y"],
                          "z":interior_pts["z"],
                          }
    openfoam_inferencer=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy, output_names=["u_1_scaled", "v_1_scaled", "w_1_scaled","p_1_scaled"]
    )
    domain.add_inferencer(openfoam_inferencer, "inf_data_1082")
    
    interior_pts=interior_mesh_pre.sample_interior(100000)
    
    openfoam_invar_numpy={"x":interior_pts["x"],
                          "y":interior_pts["y"],
                          "z":interior_pts["z"],
                          }
    openfoam_inferencer=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy, output_names=["u_0_scaled", "v_0_scaled", "w_0_scaled","p_0_scaled"]
    )
    domain.add_inferencer(openfoam_inferencer, "inf_data_1072")
    
    # make solver
    slv = Solver(cfg, domain)# start solver
    slv.solve()


if __name__ == "__main__":
    run()
