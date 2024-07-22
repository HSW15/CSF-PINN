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


class hardbound(torch.nn.Module):
    def __init__(self,power=None):
        super().__init__()
        if power is not None:
            power=torch.tensor(power)
            self.register_buffer("power", power, persistent=False)
        else:
            self.power=None
        self.curved = nn.Tanh()
    def forward(self,x):#Key("pre_u"), Key("pre_v"), Key("pre_w"), Key("dist_to_wall"),Key("u0"), Key("v0"), Key("w0")
        
        distowall = abs(x[...,3:4])
        
        distowall = (self.curved(distowall*3))
        self.softplus=torch.nn.Softplus(beta=50., threshold=10.)
        if self.power is None:
            distowall=torch.nn.functional.relu(1.-self.softplus(1.-distowall))
        else:
            distowall=torch.nn.functional.relu(1.-self.softplus(1.-distowall)**self.power)
        u=distowall*x[...,0:1] + (1-distowall)*x[...,4:5]  
        v=distowall*x[...,1:2] + (1-distowall)*x[...,5:6] 
        w=distowall*x[...,2:3] + (1-distowall)*x[...,6:7] 
        
        return torch.cat((u,v,w,distowall),-1)


class incrorder(torch.nn.Module):
    '''
    cross-multiplication of variables
    '''
    def __init__(self,maxdisttowall,maxdisttoinlet):
        super().__init__()
        maxdisttowall=torch.tensor(maxdisttowall)
        self.register_buffer("maxdisttowall", maxdisttowall, persistent=False)
        maxdisttoinlet=torch.tensor(maxdisttoinlet)
        self.register_buffer("maxdisttoinlet", maxdisttoinlet, persistent=False)
        self.softplus=torch.nn.Softplus(beta=50., threshold=10.)
    def forward(self,x):#towall,toinlet,tooutlet
        towall=torch.nn.functional.relu(1.-self.softplus(1.-x[...,0:1]))
        toinlet=torch.nn.functional.relu(1.-self.softplus(1.-x[...,1:2]))
        tooutlet=torch.nn.functional.relu(1.-self.softplus(1.-x[...,2:3]))
        towallinlet=torch.nn.functional.relu(1.-self.softplus(1.-x[...,3:4]))
        towallsq=(1.-towall)**2.
        toinletsq=(1.-toinlet)**2.
        tooutletsq=(1.-tooutlet)**2.
        towallinletsq=(1.-towallinlet)**2.
        invalltosq=torch.cat((towallsq,toinletsq,tooutletsq,towallinletsq),-1)
        return torch.cat((1.-invalltosq,1.-towallsq*invalltosq,1.-toinletsq*invalltosq,1.-tooutletsq*invalltosq,1.-towallinletsq*invalltosq,towall,toinlet,tooutlet,towallinlet),-1)

class crossvel(torch.nn.Module):
    '''
    cross-multiplication of variables
    '''
    def __init__(self):
        super().__init__()
    def forward(self,x,o):
        return torch.cross(x,o)

class asymcompressibility(torch.nn.Module):
    '''
    cross-multiplication of variables
    '''
    def __init__(self,factor):
        super().__init__()
        factor=torch.tensor(factor)
        self.register_buffer("factor", factor, persistent=False)
        #curved = nn.Tanh()
    def forward(self,x):#Key("continuity"), Key("asym_momentum_x"), Key("asym_momentum_y"), Key("asym_momentum_z"), Key("dist_to_wall")
        #self.softplus=torch.nn.Softplus(beta=50., threshold=10.)
        #fac=(torch.nn.functional.relu(1.-self.softplus(1.-x[...,4:5]))).detach() 
        #fac=(curved((x[...,4:5])*6)).detach() #needs detach()
        
        return torch.cat((x[...,0:1],x[...,1:4]),-1)
        
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

class meanvel(torch.nn.Module):
    '''
    cross-multiplication of variables
    '''
    def __init__(self):
        super().__init__()
    def forward(self,x):#veldotnormal
        inlet_ID = (x[...,1:2] > 0.92).detach()  # if inlet is the smaller one
        # print(x)
        #
        outlet_ID = (x[...,1:2]< 0.92).detach()  #if outlet is larger than the other one
        # print(outlet_ID)
        e_area_inlet = (2.63446e-5)/(0.005*0.005)
        e_area_outlet = (2.82085e-5)/(0.005*0.005)
        
        y = (torch.sum(x[...,0:1]*inlet_ID*e_area_inlet)/torch.sum(inlet_ID,dim=0).detach()+torch.sum(x[...,0:1]*outlet_ID*e_area_outlet)/torch.sum(outlet_ID,dim=0).detach())
        inlet_mass_flow = torch.sum(x[...,0:1]*inlet_ID*e_area_inlet)/torch.sum(inlet_ID,dim=0).detach()
        outlet_mass_flow = torch.sum(x[...,0:1]*outlet_ID*e_area_outlet)/torch.sum(outlet_ID,dim=0).detach()
        y=y.expand(x[...,0:1].size())
        inlet_mass_flow=inlet_mass_flow.expand(x[...,0:1].size())
        outlet_mass_flow=outlet_mass_flow.expand(x[...,0:1].size())
        
        return torch.cat((y, inlet_mass_flow,outlet_mass_flow),-1)

@modulus.main(config_path="conf", config_name="conf_lr")
def run(cfg: ModulusConfig) -> None:
    # os.makedirs("/examples/Sean_LV/outputs/"+sys.argv[0][:-3], exist_ok = True) 
    # if not(os.path.isfile("/examples/Sean_LV/outputs/"+sys.argv[0][:-3]+"/distance_net.0.pth")):
        # print("Copy file ","/examples/Sean_LV/outputs/"+sys.argv[0][:-3]+"/distance_net.0.pth")
        # shutil.copy("/examples/Sean_LV/outputs/"+sys.argv[0][:-18]+"/distance_net.0.pth","/examples/Sean_LV/outputs/"+sys.argv[0][:-3]+"/distance_net.0.pth")
    
    cfg.optimizer.lr=1e-03
    #nu = quantity(0.0038, "kg/(m*s)")
    #rho = quantity(1060, "kg/m^3")
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
    #transducer_point = tuple(map(sum,zip(transducer_point,temp_center_hardcode)))
    
    velocity_scale = quantity(vel_scale_amp, "m/s")
    density_scale = rho
    length_scale = quantity(length_scale_amp, "m")
    nd = NonDimensionalizer(
        length_scale=length_scale,
        time_scale=length_scale / velocity_scale,
        mass_scale=density_scale * (length_scale ** 3),
    )
    maxdisttowall=1.
    maxdisttoinlet=3.
     # geometry scaling
    # geometry scaling
    
    
    point_path = to_absolute_path("./temporal_stls")
    # inlet_mesh_pre = Tessellation.from_stl(
        # point_path + "/Inlet/P2_wk31_01072_inlet0.stl", airtight=False
    # )
    # outlet_mesh_pre = Tessellation.from_stl(
        # point_path + "/Outlet/P2_wk31_01072_outlet0.stl", airtight=False
    # )
    # noslip_mesh_pre = Tessellation.from_stl(
        # point_path + "/Wall/P2_wk31_01072_wall0.stl", airtight=False
    # )
    interior_mesh_pre = Tessellation.from_stl(
        point_path + "/Interior/P2_wk31_interior_01072_interior.stl", airtight=True
    )
    
    # inlet_mesh_2nd_time = Tessellation.from_stl(
        # point_path + "/Inlet/P2_wk31_01082_inlet0.stl", airtight=False
    # )
    # outlet_mesh_2nd_time = Tessellation.from_stl(
        # point_path + "/Outlet/P2_wk31_01082_outlet0.stl", airtight=False
    # )
    # noslip_mesh_2nd_time = Tessellation.from_stl(
        # point_path + "/Wall/P2_wk31_01082_wall0.stl", airtight=False
    # )
    interior_mesh_2nd_time = Tessellation.from_stl(
        point_path + "/Interior/P2_wk31_interior_01082_interior.stl", airtight=True
    )
    
    # inlet_mesh_3rd_time = Tessellation.from_stl(
        # point_path + "/Inlet/P2_wk31_01092_inlet0.stl", airtight=False
    # )
    # outlet_mesh_3rd_time = Tessellation.from_stl(
        # point_path + "/Outlet/P2_wk31_01092_outlet0.stl", airtight=False
    # )
    # noslip_mesh_3rd_time = Tessellation.from_stl(
        # point_path + "/Wall/P2_wk31_01092_wall0.stl", airtight=False
    # )
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
        
    def normalize_invar_pre(invar, center, scale, dims=2):
        invar["x_0"] -= center[0]
        invar["y_0"] -= center[1]
        invar["z_0"] -= center[2]
        invar["x_0"] *= scale
        invar["y_0"] *= scale
        invar["z_0"] *= scale
        if "area" in invar.keys():
            invar["area"] *= scale**dims
        return invar
        
    def normalize_invar_combo(invar, center, scale, dims=2):
        invar["x"] -= center[0]
        invar["y"] -= center[1]
        invar["z"] -= center[2]
        invar["x"] *= scale
        invar["y"] *= scale
        invar["z"] *= scale
        invar["x_0"] -= center[0]
        invar["y_0"] -= center[1]
        invar["z_0"] -= center[2]
        invar["x_0"] *= scale
        invar["y_0"] *= scale
        invar["z_0"] *= scale
        if "area" in invar.keys():
            invar["area"] *= scale**dims
        return invar
    
    def normalize_invar_vel_pre(invar, center, scale, vel_scale, pres_scale, dims=2):
        
        invar["u_0"] *= vel_scale
        invar["v_0"] *= vel_scale
        invar["w_0"] *= vel_scale
        invar["p_0"] *= pres_scale
        if "area" in invar.keys():
            invar["area"] *= scale**dims
        return invar
    
    def normalize_invar_vel_2nd_time(invar, center, scale, vel_scale, pres_scale, dims=2):
        
        invar["u_1"] *= vel_scale
        invar["v_1"] *= vel_scale
        invar["w_1"] *= vel_scale
        invar["p_1"] *= pres_scale
        if "area" in invar.keys():
            invar["area"] *= scale**dims
        return invar
    
    def normalize_invar_vel(invar, center, scale, vel_scale, pres_scale, dims=2):
        
        invar["u"] *= vel_scale
        invar["v"] *= vel_scale
        invar["w"] *= vel_scale
        invar["p"] *= pres_scale
        if "area" in invar.keys():
            invar["area"] *= scale**dims
        return invar
    
    # normalize invars
    def normalize_velvar(invar, dims=2):
        invar["lambda_u"] *= 2.
        invar["lambda_v"] *= 1.
        invar["lambda_w"] *= 6.
        
        return invar
    
    # normalize invars
    def normalize_velvar_plane(invar, dims=2):
        invar["lambda_u"] *= 1.0    
        invar["lambda_v"] *= 1.0    
        invar["lambda_w"] *= 1.0
        
        return invar
    
    
    # center of overall geometry
    # center of overall geometry
    center = center_hardcode
    print('Overall geometry center: ', center)

    # scale and center the geometry files
    # inlet_mesh_pre = normalize_mesh(inlet_mesh_pre, center, 1./length_scale_amp)
    # outlet_mesh_pre = normalize_mesh(outlet_mesh_pre, center, 1./length_scale_amp)
    # noslip_mesh_pre = normalize_mesh(noslip_mesh_pre, center, 1./length_scale_amp)
    interior_mesh_pre = normalize_mesh(interior_mesh_pre, center, 1./length_scale_amp)
    
    # inlet_mesh_2nd_time = normalize_mesh(inlet_mesh_2nd_time, center, 1./length_scale_amp)
    # outlet_mesh_2nd_time = normalize_mesh(outlet_mesh_2nd_time, center, 1./length_scale_amp)
    # noslip_mesh_2nd_time = normalize_mesh(noslip_mesh_2nd_time, center, 1./length_scale_amp)
    interior_mesh_2nd_time = normalize_mesh(interior_mesh_2nd_time, center, 1./length_scale_amp)
    
    # inlet_mesh_3rd_time = normalize_mesh(inlet_mesh_3rd_time, center, 1./length_scale_amp)
    # outlet_mesh_3rd_time = normalize_mesh(outlet_mesh_3rd_time, center, 1./length_scale_amp)
    # noslip_mesh_3rd_time = normalize_mesh(noslip_mesh_3rd_time, center, 1./length_scale_amp)
    interior_mesh_3rd_time = normalize_mesh(interior_mesh_3rd_time, center, 1./length_scale_amp)
    # transducer_point = normalize_mesh(transducer_point, center, 1./length_scale_amp)
    # interior_intersection_2nd_time = interior_mesh_pre & interior_mesh_2nd_time
    # interior_intersection_3rd_time = interior_mesh_3rd_time & interior_intersection_2nd_time
    
    # find center of inlet in original coordinate system

    # scale end center the inlet center

    # find inlet normal vector; should point towards aneurysm, not outwards

    # make aneurysm domain
    domain = Domain()
    
    # params
    
    ns_1st = TemporalNavierStokes_1st_time_lagrange(nu=nd.ndim(nu), rho=nd.ndim(rho), dim=3, time=False)
    ns_2nd = TemporalNavierStokes_2nd_time_point(nu=nd.ndim(nu), rho=nd.ndim(rho), dim=3, time=False)
    ns_3rd = TemporalNavierStokes_3rd_time_point_lagrange(nu=nd.ndim(nu), rho=nd.ndim(rho), dim=3, time=False)
    ######################     DISTANCE NET ##################
    
    ###############################################################
   
   
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
    
    # distance_net.load_state_dict(
        # torch.load(
            # "/examples/Sean_LV/outputs/"+sys.argv[0][:-3]+"/distance_net.0.pth"
        # ))
    # for param in distance_net.parameters():
        # param.requires_grad = False
    # add constraints to solver
    # make geometry
    domain = Domain()
    batchsizefactor=1
    
    #########################           PLANE PROFILE MATCHING       ################################
    # color_doppler_threshold
    
    inflow_var = csv_to_dict(
       to_absolute_path("./temporal_stls/3_time_point_temporal/1092/inlet.csv"))
    inflow_invar = {
       key: value for key, value in inflow_var.items() if key in ["x", "y", "z"]
    }
    inflow_invar = normalize_invar(inflow_invar, center, 1./length_scale_amp, dims=3)
    inflow_outvar = {
       key: value for key, value in inflow_var.items() if key in ["u", "v", "w", "p"]
    }
    inflow_outvar = normalize_invar_vel(inflow_outvar, center, 1./length_scale_amp, (1./vel_scale_amp), (1./((vel_scale_amp**2.)*1060.)), dims=3)
    #inflow_invar = normalize_velvar(inflow_invar, velocity_scale, dims=3)
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
       key: value for key, value in inflow_var.items() if key in ["u_1", "v_1", "w_1", "p_1"]
    }
    inflow_outvar = normalize_invar_vel_2nd_time(inflow_outvar, center, 1./length_scale_amp, (1./vel_scale_amp), (1./((vel_scale_amp**2.)*1060.)), dims=3)
    #inflow_invar = normalize_velvar(inflow_invar, velocity_scale, dims=3)
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
       key: value for key, value in inflow_var.items() if key in ["u_0", "v_0", "w_0", "p_0"]
    }
    inflow_outvar = normalize_invar_vel_pre(inflow_outvar, center, 1./length_scale_amp, (1./vel_scale_amp), (1./((vel_scale_amp**2.)*1060.)), dims=3)
    #inflow_invar = normalize_velvar(inflow_invar, velocity_scale, dims=3)
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
    
    # inflow_var = csv_to_dict(
       # to_absolute_path("./stl_files_temporal/1082_P2_wk31_continuity.csv"))
    # inflow_invar = {
       # key: value for key, value in inflow_var.items() if key in ["x", "y", "z"]
    # }
    # inflow_invar = normalize_invar(inflow_invar, center, 1./length_scale_amp, dims=3)
    # inflow_outvar = {
       # key: value for key, value in inflow_var.items() if key in ["continuity", "momentum_x", "momentum_y", "momentum_z"]
    # }
    
    # previous_velocity = PointwiseConstraint.from_numpy(
        # nodes,
        # inflow_invar,
        # inflow_outvar,
        # batch_size=10000,
    # )
    # domain.add_constraint(previous_velocity, "NS_eq")
    
    # interior
    
    interior = PointwiseInteriorConstraint(
        nodes,
        geometry=interior_mesh_3rd_time,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        batch_size=4000,
        # lambda_weighting={
            # "continuity": Symbol("sdf"),
            # "momentum_x": Symbol("sdf"),
            # "momentum_y": Symbol("sdf"),
            # "momentum_z": Symbol("sdf"),
            # },
    )
    domain.add_constraint(interior, "interior_1092")
    
    interior = PointwiseInteriorConstraint(
        nodes,
        geometry=interior_mesh_2nd_time,
        outvar={"continuity_2nd": 0, "momentum_x_2nd": 0, "momentum_y_2nd": 0, "momentum_z_2nd": 0},
        batch_size=4000,
        # lambda_weighting={
            # "continuity_1st": Symbol("sdf"),
            # "momentum_x_1st": Symbol("sdf"),
            # "momentum_y_1st": Symbol("sdf"),
            # "momentum_z_1st": Symbol("sdf"),
            # },
    )
    domain.add_constraint(interior, "interior_1082")
    
    interior = PointwiseInteriorConstraint(
        nodes,
        geometry=interior_mesh_pre,
        outvar={"continuity_1st": 0, "momentum_x_1st": 0, "momentum_y_1st": 0, "momentum_z_1st": 0},
        batch_size=4000,
        # lambda_weighting={
            # "continuity_1st": Symbol("sdf"),
            # "momentum_x_1st": Symbol("sdf"),
            # "momentum_y_1st": Symbol("sdf"),
            # "momentum_z_1st": Symbol("sdf"),
            # },
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
    
    # inflow_var = csv_to_dict(
       # to_absolute_path("./temporal_stls/3_time_point_temporal/1082/US_threshold_1082_no_noise.csv"))
    # inflow_invar = {
       # key: value for key, value in inflow_var.items() if key in ["x", "y", "z"]
    # }
    # inflow_invar = normalize_invar(inflow_invar, center, 1./length_scale_amp, dims=3)
    # inflow_outvar = {
       # key: value*(1./vel_scale_amp) for key, value in inflow_var.items() if key in ["us_mag_1"]
    # }
    # doppler_numpy = PointwiseConstraint.from_numpy(
       # nodes,
       # inflow_invar,
       # inflow_outvar,
       # batch_size=10000,
    # )
    # domain.add_constraint(doppler_numpy, "color_doppler_mag_1082")
    
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
   
    # boundary_pts=noslip_mesh_post.sample_boundary(10000)
    # inlet_pts=inlet_mesh_post.sample_boundary(10000)
    # outlet_pts=outlet_mesh_post.sample_boundary(10000)
    
    
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
    
    # openfoam_invar_numpy={"x":boundary_pts["x"],
                          # "y":boundary_pts["y"],
                          # "z":boundary_pts["z"],
                          # }
    # openfoam_inferencer=PointwiseInferencer(
    # nodes=nodes, invar=openfoam_invar_numpy, output_names=["u_scaled", "v_scaled", "w_scaled","p_scaled"]
    # )
    # domain.add_inferencer(openfoam_inferencer, "wall_data")
    
    
    # openfoam_invar_numpy={"x":inlet_pts["x"],
                          # "y":inlet_pts["y"],
                          # "z":inlet_pts["z"],
                          # }
    # openfoam_inferencer=PointwiseInferencer(
    # nodes=nodes, invar=openfoam_invar_numpy, output_names=["u_scaled", "v_scaled", "w_scaled","p_scaled"]
    # )
    # domain.add_inferencer(openfoam_inferencer, "inlet_data")
    
    # openfoam_invar_numpy={"x":outlet_pts["x"],
                          # "y":outlet_pts["y"],
                          # "z":outlet_pts["z"],
                          # }
    # openfoam_inferencer=PointwiseInferencer(
    # nodes=nodes, invar=openfoam_invar_numpy, output_names=["u_scaled", "v_scaled", "w_scaled","p_scaled"]
    # )
    # domain.add_inferencer(openfoam_inferencer, "outlet_data")
    
    # make solver
    slv = Solver(cfg, domain)# start solver
    slv.solve()


if __name__ == "__main__":
    run()
