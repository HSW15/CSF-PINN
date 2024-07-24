# -*- coding: utf-8 -*-
"""
################################################
MIT License
Copyright (c) 2021 L. C. Lee
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
################################################
File: ModulusModel_GeometricalModes.py
Description: Geometrical-Modes Model Architecture For Nvidia Modulus

History:
    Date    Programmer SAR# - Description
    ---------- ---------- ----------------------------
  Author: w.x.chan@gmail.com         10Mar2023           - Created
"""
_version='1.0.0'
from typing import Optional, Dict, Tuple, Union, List
from modulus.key import Key

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from modulus.models.layers import Activation, FCLayer, Conv1dFCLayer
from modulus.models.arch import Arch
from .layers import inheritedFCLayer, singleInputInheritedFCLayer

class AdditionArchCore(nn.Module):
    def __init__(
        self
    ) -> None:
        super().__init__()
    def forward(self, x: Tensor,o: Tensor) -> Tensor:
        return x+o
class SubtractionArchCore(nn.Module):
    def __init__(
        self
    ) -> None:
        super().__init__()
    def forward(self, x: Tensor,o: Tensor) -> Tensor:
        return x-o
class MultiplicationArchCore(nn.Module):
    def __init__(
        self
    ) -> None:
        super().__init__()
    def forward(self, x: Tensor,o: Tensor) -> Tensor:
        return x*o
class SumMultiplicationArchCore(nn.Module):
    def __init__(
        self
    ) -> None:
        super().__init__()
    def forward(self, x: Tensor,o: Tensor) -> Tensor:
        return torch.sum(x*o,-1,keepdim=True)
class ParametricInsert(nn.Module):
    def __init__(
        self,
        out_features: int = 512
    ) -> None:
        super().__init__()
        self.out_features=out_features
        self.weight_g = nn.Parameter(torch.empty((1,out_features)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.constant_(self.weight_g, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        return self.weight_g.expand(x.size()[:-1]+(-1,))

    def extra_repr(self) -> str:
        return "out_features={}".format(
            self.out_features
        )
class FullyConnectedFlexiLayerSizeArchCore(nn.Module):
    def __init__(
        self,
        in_features: int = 512,
        layer_sizeList: Union[int,List[int]] = 512,
        skip_connections: bool = False,
        activation_fn: Activation = Activation.SILU,
        adaptive_activations: bool = False,
        weight_norm: bool = True,
        conv_layers: bool = False,
    ) -> None:
        super().__init__()

        self.skip_connections = skip_connections
        if isinstance(layer_sizeList,int):
            layer_sizeList=[layer_sizeList]
        nr_layers=len(layer_sizeList)-1
        # Allows for regular linear layers to be swapped for 1D Convs
        # Useful for channel operations in FNO/Transformers
        if conv_layers:
            fc_layer = Conv1dFCLayer
        else:
            fc_layer = FCLayer

        if adaptive_activations:
            activation_par = nn.Parameter(torch.ones(1))
        else:
            activation_par = None

        if not isinstance(activation_fn, list):
            activation_fn = [activation_fn] * nr_layers
        if len(activation_fn) < nr_layers:
            activation_fn = activation_fn + [activation_fn[-1]] * (
                nr_layers - len(activation_fn)
            )

        self.layers = nn.ModuleList()

        layer_in_features = in_features
        for i in range(nr_layers):
            self.layers.append(
                fc_layer(
                    layer_in_features,
                    layer_sizeList[i],
                    activation_fn[i],
                    weight_norm,
                    activation_par,
                )
            )
            layer_in_features = layer_sizeList[i]

        self.final_layer = fc_layer(
            in_features=layer_in_features,
            out_features=layer_sizeList[-1],
            activation_fn=Activation.IDENTITY,
            weight_norm=False,
            activation_par=None,
        )

    def forward(self, x: Tensor) -> Tensor:
        x_skip: Optional[Tensor] = None
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.skip_connections and i % 2 == 0:
                if x_skip is not None:
                    x, x_skip = x + x_skip, x
                else:
                    x_skip = x

        x = self.final_layer(x)
        return x

    def get_weight_list(self):
        weights = [layer.conv.weight for layer in self.layers] + [
            self.final_layer.conv.weight
        ]
        biases = [layer.conv.bias for layer in self.layers] + [
            self.final_layer.conv.bias
        ]
        return weights, biases
class InheritedFullyConnectedFlexiLayerSizeArchCore(nn.Module):
    def __init__(
        self,
        in_features: int = 512,
        layer_sizeList: Union[int,List[int]] = 512,
        skip_connections: bool = False,
        activation_fn: Activation = Activation.SILU,
        caseNN_in_features: int = 512,
        caseNN_layer_sizeList: Union[int,List[int],None] = None,
        caseNN_skip_connections: bool = False,
        caseNN_activation_fn: Activation = Activation.SILU,
        caseNN_adaptive_activations: bool = False,
        caseNN_weight_norm: bool = True,
    ) -> None:
        super().__init__()
        self.in_features=in_features
        self.caseNN_in_features=caseNN_in_features
        self.skip_connections = skip_connections
        self.caseNN_skip_connections = caseNN_skip_connections
        if isinstance(layer_sizeList,int):
            layer_sizeList=[layer_sizeList]
        nr_layers=len(layer_sizeList)-1
        domainNN_size=[0,(in_features+1)*layer_sizeList[0]]
        domainNN_weight_size=[in_features*layer_sizeList[0]]
        domainNN_bias_size=[layer_sizeList[0]]
        for n in range(1,len(layer_sizeList)):
            domainNN_weight_size.append(layer_sizeList[n-1]*layer_sizeList[n])
            domainNN_bias_size.append(layer_sizeList[n])
            domainNN_size.append((layer_sizeList[n-1]+1)*layer_sizeList[n])
        self.domainNN_cumulative_size=list(np.cumsum(domainNN_size))
        #domainNN_size_torch = torch.tensor(domainNN_size,dtype=torch.long)
        #self.register_buffer("domainNN_size", domainNN_size_torch, persistent=False)
        
        if caseNN_layer_sizeList is None:
            caseNN_layer_sizeList=[]
        elif isinstance(caseNN_layer_sizeList,int):
            caseNN_layer_sizeList=[caseNN_layer_sizeList]
        caseNN_layer_sizeList=caseNN_layer_sizeList+[sum(domainNN_size)]
        caseNN_nr_layers=len(caseNN_layer_sizeList)-1
        # Allows for regular linear layers to be swapped for 1D Convs
        # Useful for channel operations in FNO/Transformers

        if caseNN_activation_fn:
            caseNN_activation_par = nn.Parameter(torch.ones(1))
        else:
            caseNN_activation_par = None

        if not isinstance(caseNN_activation_fn, list):
            caseNN_activation_fn = [caseNN_activation_fn] * caseNN_nr_layers
        if len(caseNN_activation_fn) < caseNN_nr_layers:
            caseNN_activation_fn = caseNN_activation_fn + [caseNN_activation_fn[-1]] * (
                caseNN_nr_layers - len(caseNN_activation_fn)
            )

        self.caseNN_layers = nn.ModuleList()

        layer_in_features = caseNN_in_features
        for i in range(caseNN_nr_layers):
            self.caseNN_layers.append(
                FCLayer(
                    layer_in_features,
                    caseNN_layer_sizeList[i],
                    caseNN_activation_fn[i],
                    caseNN_weight_norm,
                    caseNN_activation_par,
                )
            )
            layer_in_features = caseNN_layer_sizeList[i]
            
        self.caseNN_final_layer_weight = nn.ModuleList()
        self.caseNN_final_layer_bias = nn.ModuleList()
        for i in range(len(layer_sizeList)):
            self.caseNN_final_layer_weight.append( FCLayer(
                in_features=layer_in_features,
                out_features=domainNN_weight_size[i],
                activation_fn=Activation.IDENTITY,
                weight_norm=False,
                activation_par=None,
            ))
            self.caseNN_final_layer_bias.append( FCLayer(
                in_features=layer_in_features,
                out_features=domainNN_bias_size[i],
                activation_fn=Activation.IDENTITY,
                weight_norm=False,
                activation_par=None,
            ))
        if not isinstance(activation_fn, list):
            activation_fn = [activation_fn] * nr_layers
        if len(activation_fn) < nr_layers:
            activation_fn = activation_fn + [activation_fn[-1]] * (
                nr_layers - len(activation_fn)
            )
        self.layers = nn.ModuleList()
        layer_in_features = in_features
        for i in range(nr_layers):
            self.layers.append(
                inheritedFCLayer(
                    layer_in_features,
                    layer_sizeList[i],
                    activation_fn[i],
                )
            )
            layer_in_features = layer_sizeList[i]
        self.final_layer = inheritedFCLayer(
            in_features=layer_in_features,
            out_features=layer_sizeList[-1],
            activation_fn=Activation.IDENTITY,
        )
        
    def forward(self, x: Tensor,o: Tensor) -> Tensor:#wrong!!!
        x_skip: Optional[Tensor] = None
        for i, layer in enumerate(self.caseNN_layers):
            o = layer(o)
            if self.caseNN_skip_connections and i % 2 == 0:
                if x_skip is not None:
                    o, x_skip = o + x_skip, o
                else:
                    x_skip = o

        o = self.caseNN_final_layer(o)
        for i, layer in enumerate(self.layers):
            x = layer(x.unsqueeze(-3),o[...,self.domainNN_cumulative_size[i]:self.domainNN_cumulative_size[i+1]])
            if self.skip_connections and i % 2 == 0:
                if x_skip is not None:
                    x, x_skip = x + x_skip, x
                else:
                    x_skip = x

        x = self.final_layer(x,o[...,self.domainNN_cumulative_size[-2]:self.domainNN_cumulative_size[-1]])
        return x.view((-1,)+x.size()[2:x.dim()])
class SingleInputInheritedFullyConnectedFlexiLayerSizeArchCore(InheritedFullyConnectedFlexiLayerSizeArchCore):
    def __init__(
        self,
        in_features: int = 512,
        layer_sizeList: Union[int,List[int]] = 512,
        skip_connections: bool = False,
        activation_fn: Activation = Activation.SILU,
        caseNN_in_features: int = 512,
        caseNN_layer_sizeList: Union[int,List[int],None] = None,
        caseNN_skip_connections: bool = False,
        caseNN_activation_fn: Activation = Activation.SILU,
        caseNN_adaptive_activations: bool = False,
        caseNN_weight_norm: bool = True,
        with_caseID: int = 0,
    ) -> None:
        super().__init__(in_features = in_features,
                         layer_sizeList=layer_sizeList,
                         skip_connections=skip_connections,
                         activation_fn=activation_fn,
                         caseNN_in_features=caseNN_in_features,
                         caseNN_layer_sizeList=caseNN_layer_sizeList,
                         caseNN_skip_connections=caseNN_skip_connections,
                         caseNN_activation_fn=caseNN_activation_fn,
                         caseNN_adaptive_activations=caseNN_adaptive_activations,
                         caseNN_weight_norm=caseNN_weight_norm,)
        if with_caseID:
            caseID_range = torch.tensor(np.arange(with_caseID).reshape((-1,1)),dtype=torch.long)
            self.register_buffer("caseID_range", caseID_range, persistent=False)
            self.unit_function=Unit_function.apply
            self.consolidative_matmul=Consolidative_matmul.apply
            self.with_caseID=True
        else:
            self.with_caseID=False
    def forward(self, x: Tensor) -> Tensor:
        x_skip: Optional[Tensor] = None
        if self.with_caseID:
            caseID=x[...,(self.in_features+self.caseNN_in_features):(self.in_features+self.caseNN_in_features+1)].detach()
            caseID_unit=self.unit_function(torch.transpose(caseID.expand((-1,self.caseID_range.size(-2))),-1,-2)-self.caseID_range)#case_num,training_points
            case_inputs=self.consolidative_matmul(caseID_unit,x[...,self.in_features:(self.in_features+self.caseNN_in_features)])
        else:
            case_inputs=x[...,self.in_features:(self.in_features+self.caseNN_in_features)]
        
        for i, layer in enumerate(self.caseNN_layers):
            case_inputs = layer(case_inputs)
            if self.caseNN_skip_connections and i % 2 == 0:
                if x_skip is not None:
                    case_inputs, x_skip = case_inputs + x_skip, case_inputs
                else:
                    x_skip = case_inputs

        domain_inputs=x[...,:self.in_features].unsqueeze(-2)
        domain_inputs_size=domain_inputs.size(0)
        for i, layer in enumerate(self.layers):
            weights_case=self.caseNN_final_layer_weight[i](case_inputs)
            bias_case=self.caseNN_final_layer_bias[i](case_inputs)
            if self.with_caseID:
                weights_case=torch.matmul(torch.transpose(caseID_unit,-1,-2),weights_case)
                bias_case=torch.matmul(torch.transpose(caseID_unit,-1,-2),bias_case)
            weights_case=weights_case.view((domain_inputs_size,layer.in_features,layer.out_features))
            bias_case=bias_case.unsqueeze(-2)
            domain_inputs = layer(domain_inputs,weights_case,bias_case)
            if self.skip_connections and i % 2 == 0:
                if x_skip is not None:
                    domain_inputs, x_skip = domain_inputs + x_skip, domain_inputs
                else:
                    x_skip = domain_inputs
        weights_case=self.caseNN_final_layer_weight[-1](case_inputs)
        bias_case=self.caseNN_final_layer_bias[-1](case_inputs)
        if self.with_caseID:
            weights_case=torch.matmul(torch.transpose(caseID_unit,-1,-2),weights_case)
            bias_case=torch.matmul(torch.transpose(caseID_unit,-1,-2),bias_case)
        weights_case=weights_case.view((domain_inputs_size,self.final_layer.in_features,self.final_layer.out_features))
        bias_case=bias_case.unsqueeze(-2)
        domain_inputs = self.final_layer(domain_inputs,weights_case,bias_case)
        return domain_inputs.squeeze(-2)

class HypernetFullyConnectedFlexiLayerSizeArchCore(nn.Module):
    def __init__(
        self,
        in_features: int = 512,
        layer_sizeList: Union[int,List[int],None] = None,
        skip_connections: bool = False,
        activation_fn: Activation = Activation.SILU,
        adaptive_activations: bool = False,
        weight_norm: bool = True,
        with_caseID: int = 0,
    ) -> None:
        super().__init__()
        self.in_features=in_features
        self.skip_connections = skip_connections
        if isinstance(layer_sizeList,int):
            layer_sizeList=[layer_sizeList]
        nr_layers=len(layer_sizeList)-1
        # Allows for regular linear layers to be swapped for 1D Convs
        # Useful for channel operations in FNO/Transformers

        if adaptive_activations:
            activation_par = nn.Parameter(torch.ones(1))
        else:
            activation_par = None

        if not isinstance(activation_fn, list):
            activation_fn = [activation_fn] * nr_layers
        if len(activation_fn) < nr_layers:
            activation_fn = activation_fn + [activation_fn[-1]] * (
                nr_layers - len(activation_fn)
            )

        self.layers = nn.ModuleList()

        layer_in_features = in_features
        for i in range(nr_layers):
            self.layers.append(
                FCLayer(
                    layer_in_features,
                    layer_sizeList[i],
                    activation_fn[i],
                    weight_norm,
                    activation_par,
                )
            )
            layer_in_features = layer_sizeList[i]

        self.final_layer = FCLayer(
            in_features=layer_in_features,
            out_features=layer_sizeList[-1],
            activation_fn=Activation.IDENTITY,
            weight_norm=False,
            activation_par=None,
        )
        if with_caseID:
            caseID_range = torch.tensor(np.arange(with_caseID).reshape((-1,1)),dtype=torch.long)
            self.register_buffer("caseID_range", caseID_range, persistent=False)
            self.unit_function=Unit_function.apply
            self.consolidative_matmul=Consolidative_matmul.apply
            self.with_caseID=True
        else:
            self.with_caseID=False
    def forward(self, x: Tensor) -> Tensor:
        x_skip: Optional[Tensor] = None
        if self.with_caseID:
            caseID=x[...,self.in_features:(self.in_features+1)].detach()
            caseID_unit=self.unit_function(torch.transpose(caseID.expand(caseID.size()[:-1]+(self.caseID_range.size(-2),)),-1,-2)-self.caseID_range)#case_num,training_points
            case_inputs=self.consolidative_matmul(caseID_unit,x[...,0:self.in_features])
        else:
            case_inputs=x[...,0:self.in_features]
        
        for i, layer in enumerate(self.layers):
            case_inputs = layer(case_inputs)
            if self.skip_connections and i % 2 == 0:
                if x_skip is not None:
                    case_inputs, x_skip = case_inputs + x_skip, case_inputs
                else:
                    x_skip = case_inputs
        case_inputs=self.final_layer(case_inputs)
        if self.with_caseID:
            case_inputs=torch.matmul(torch.transpose(caseID_unit,-1,-2),case_inputs)
        
        return case_inputs

class CaseIDtoFeatureArchCore(nn.Module):
    def __init__(
        self,
        feature_array
    ) -> None:
        super().__init__()
        self.out_features=feature_array.shape[1]
        self.total_case=feature_array.shape[0]
        feature_array = torch.tensor(feature_array,dtype=torch.float)##!!!QUICK FIXED
        self.register_buffer("feature_array", feature_array, persistent=False)
        self.unit_function=Unit_function.apply
        case_range = torch.tensor(np.arange(self.total_case).reshape((1,-1)))
        self.register_buffer("case_range", case_range, persistent=False)
    def forward(self, x: Tensor) -> Tensor:
        caseMatrix=self.unit_function(x-self.case_range)
        return torch.matmul(caseMatrix,self.feature_array)

    def extra_repr(self) -> str:
        return "out_features={}".format(
            self.out_features
        )
class FixedFeatureArchCore(nn.Module):
    def __init__(
        self,
        feature_array
    ) -> None:
        super().__init__()
        self.out_features=feature_array.shape[1]
        self.total_case=feature_array.shape[0]
        feature_array = torch.tensor(feature_array,dtype=torch.float)##!!!QUICK FIXED
        self.register_buffer("feature_array", feature_array, persistent=False)
    def forward(self) -> Tensor:
        return self.feature_array

    def extra_repr(self) -> str:
        return "out_features={}".format(
            self.out_features
        )
class Consolidative_matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx,select_mat, x):
        ctx.save_for_backward(select_mat)
        select_sum=torch.sum(select_mat, -1, keepdim=True)
        select_sum=(torch.nn.functional.relu(select_sum-1.)+1.).detach()
        return torch.matmul(select_mat,x)/select_sum
    @staticmethod
    def backward(ctx, grad_out):
        select_mat, = ctx.saved_tensors
        return grad_out*0. , torch.matmul(torch.transpose(select_mat,-1,-2),grad_out)
class Unit_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y=torch.ones_like(x)
        y[x<=-0.5]=0.
        y[x>=0.5]=0.
        return y.detach()
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out*0.
def Heaviside_function(input ,values):
    y=input*0.
    y[input>0.]=1.
    y[input==0.]=values
    return y
class BsplineBasis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        B0 = Heaviside_function(x-1., 0.)*Heaviside_function(-x+2., 0.)*(2.-x)**3./6.
        B1 = Heaviside_function(x, 1.)*Heaviside_function(-x+1., 1.)*(3.*x**3.-6.*x**2.+4.)/6. 
        B2 = Heaviside_function(x+1., 0.)*Heaviside_function(-x, 1.)*(-3.*x**3.-6.*x**2.+4.)/6. 
        B3 = Heaviside_function(x+2., 0.)*Heaviside_function(-x-1., 0.)*(x+2.)**3./6.
        ctx.save_for_backward(x)
        return B0+B1+B2+B3
    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        B0=Heaviside_function(x-1., 0.)*Heaviside_function(-x+2., 0.)*(2.-x)**2./-2.
        B1=Heaviside_function(x, 1.)*Heaviside_function(-x+1., 1.)*(3.*x**2.-4.*x**2.)/2. 
        B2=Heaviside_function(x+1., 0.)*Heaviside_function(-x, 1.)*(-3.*x**2.-4.*x**2.)/2. 
        B3=Heaviside_function(x+2., 0.)*Heaviside_function(-x-1., 0.)*(x+2.)**2./2.
        return grad_out*(B0+B1+B2+B3)
Bspline1D=BsplineBasis.apply
class Bspline2D(nn.Module):
    def __init__(
        self,
        origin:List[float],
        spacing:List[float],
        nodes_shape:List[int],
        fix_coef=None
    ) -> None:
        super().__init__()
        origin=torch.tensor(np.array(origin).reshape((1,-1)))
        self.register_buffer("origin", origin, persistent=False)
        spacing=torch.tensor(np.array(spacing).reshape((1,-1)))
        self.register_buffer("spacing", spacing, persistent=False)
        self.nodes_shape=nodes_shape
        nodes_coord=np.mgrid[0:self.nodes_shape[0],0:self.nodes_shape[1]]
        shift_coord_x=torch.tensor(nodes_coord[0].reshape((1,-1)))
        self.register_buffer("shift_coord_x", shift_coord_x, persistent=False)
        shift_coord_y=torch.tensor(nodes_coord[1].reshape((1,-1)))
        self.register_buffer("shift_coord_y", shift_coord_y, persistent=False)
        if fix_coef is None:
            self.weight_u = nn.Parameter(torch.empty((1,np.prod(nodes_shape))))
            self.weight_v = nn.Parameter(torch.empty((1,np.prod(nodes_shape))))
            self.reset_parameters()
        elif len(fix_coef)==len(nodes_shape):
            weight_u=torch.tensor(np.array(fix_coef[0]).reshape((1,-1)))
            self.register_buffer("weight_u", weight_u, persistent=False)
            weight_v=torch.tensor(np.array(fix_coef[1]).reshape((1,-1)))
            self.register_buffer("weight_v", weight_v, persistent=False)
        else:
            weight_u=torch.tensor(np.array(fix_coef[...,0]).reshape((1,-1)))
            self.register_buffer("weight_u", weight_u, persistent=False)
            weight_v=torch.tensor(np.array(fix_coef[...,1]).reshape((1,-1)))
            self.register_buffer("weight_v", weight_v, persistent=False)
            
    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.weight_u)
        torch.nn.init.xavier_uniform_(self.weight_v)

    def forward(self, x: Tensor) -> Tensor:
        BsplineCoord=(x-self.origin)/self.spacing
        shifted_weight=Bspline1D(BsplineCoord[...,0:1]-self.shift_coord_x)*Bspline1D(BsplineCoord[...,1:2]-self.shift_coord_y)
        u=torch.sum(shifted_weight*self.weight_u,-1, keepdim=True)
        v=torch.sum(shifted_weight*self.weight_v,-1, keepdim=True)
        return torch.cat((u,v),-1)
class Bspline3D(nn.Module):
    def __init__(
        self,
        origin:List[float],
        spacing:List[float],
        nodes_shape:List[int],
        fix_coef=None
    ) -> None:
        super().__init__()
        origin=torch.tensor(np.array(origin).reshape((1,-1)))
        self.register_buffer("origin", origin, persistent=False)
        spacing=torch.tensor(np.array(spacing).reshape((1,-1)))
        self.register_buffer("spacing", spacing, persistent=False)
        self.nodes_shape=nodes_shape
        nodes_coord=np.mgrid[0:self.nodes_shape[0],0:self.nodes_shape[1],0:self.nodes_shape[2]]
        shift_coord_x=torch.tensor(nodes_coord[0].reshape((1,-1)))
        self.register_buffer("shift_coord_x", shift_coord_x, persistent=False)
        shift_coord_y=torch.tensor(nodes_coord[1].reshape((1,-1)))
        self.register_buffer("shift_coord_y", shift_coord_y, persistent=False)
        shift_coord_z=torch.tensor(nodes_coord[2].reshape((1,-1)))
        self.register_buffer("shift_coord_z", shift_coord_z, persistent=False)
        if fix_coef is None:
            self.weight_u = nn.Parameter(torch.empty((1,np.prod(nodes_shape))))
            self.weight_v = nn.Parameter(torch.empty((1,np.prod(nodes_shape))))
            self.weight_w = nn.Parameter(torch.empty((1,np.prod(nodes_shape))))
            self.reset_parameters()
        elif len(fix_coef)==len(nodes_shape):
            weight_u=torch.tensor(np.array(fix_coef[0]).reshape((1,-1)))
            self.register_buffer("weight_u", weight_u, persistent=False)
            weight_v=torch.tensor(np.array(fix_coef[1]).reshape((1,-1)))
            self.register_buffer("weight_v", weight_v, persistent=False)
            weight_w=torch.tensor(np.array(fix_coef[2]).reshape((1,-1)))
            self.register_buffer("weight_w", weight_w, persistent=False)
        else:
            weight_u=torch.tensor(np.array(fix_coef[...,0]).reshape((1,-1)))
            self.register_buffer("weight_u", weight_u, persistent=False)
            weight_v=torch.tensor(np.array(fix_coef[...,1]).reshape((1,-1)))
            self.register_buffer("weight_v", weight_v, persistent=False)
            weight_w=torch.tensor(np.array(fix_coef[...,2]).reshape((1,-1)))
            self.register_buffer("weight_w", weight_w, persistent=False)
            
    def reset_parameters(self) -> None:
        torch.nn.init.constant_(self.weight_u, 1.0)
        torch.nn.init.constant_(self.weight_v, 1.0)
        torch.nn.init.constant_(self.weight_w, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        BsplineCoord=(x-self.origin)/self.spacing
        shifted_weight=Bspline1D(BsplineCoord[...,0:1]-self.shift_coord_x)*Bspline1D(BsplineCoord[...,1:2]-self.shift_coord_y)*Bspline1D(BsplineCoord[...,2:3]-self.shift_coord_z)
        u=torch.sum(shifted_weight*self.weight_u,-1, keepdim=True)
        v=torch.sum(shifted_weight*self.weight_v,-1, keepdim=True)
        w=torch.sum(shifted_weight*self.weight_w,-1, keepdim=True)
        return torch.cat((u,v,w)-1)
class CustomModuleArch(Arch):
    """Fully Connected Neural Network.

    Parameters
    ----------
    input_keys : List[Key]
        Input key list.
    output_keys : List[Key]
        Output key list.
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    domain_layer_sizeList : List[None,List[int]], optional
        List of Layer size for every hidden layer of the model including the output, by default 512,512,512,512,512,512,len(output_keys)
    domain_inputKeyList : List[boolList[Key]], optional
        Key to send to each NN of the domain cluster, by default [True]*len(domain_layer_sizeList)
    modes_decoding : torch.nn.Module, optional
        module to run before output of architecture, number of output must match number of output_keys, by default torch.nn.Identity()
    gEncoding_layer_sizeList: List[None,List[int]], optional
        List of Layer size for every hidden layer of the geometry encoding including the output, by default None
    gEncoding_inputKeyList: List[boolList[Key]], optional
        Key to send to each NN of the geometry encoding, by default [True]*len(gEncoding_layer_sizeList)
    gfEncoding_outputKey: Union[None,List[Key]] =None,
        Key to output from geometry and functional encoding, by default None
    functionalEncoding : torch.nn.Module, optional
        module to run between geometry encoding and domain cluster,  number of output must match number of gfEncoding_outputKey, by default None
    activation_fn : Activation, optional
        Activation function used by network, by default :obj:`Activation.SILU`
    periodicity : Union[Dict[str, Tuple[float, float]], None], optional
        Dictionary of tuples that allows making model give periodic predictions on
        the given bounds in tuple.
    skip_connections : bool, optional
        Apply skip connections every 2 hidden layers, by default False
    weight_norm : bool, optional
        Use weight norm on fully connected layers, by default True
    adaptive_activations : bool, optional
        Use an adaptive activation functions, by default False

    Variable Shape
    --------------
    - Input variable tensor shape: :math:`[N, size]`
    - Output variable tensor shape: :math:`[N, size]`

    Example
    -------
    Fully-connected model (2 -> 64 -> 64 -> 2)

    >>> arch = .geometrical_modes.GeometricalModesArch(
    >>>    [Key("x", size=2)],
    >>>    [Key("y", size=2)],
    >>>    domain_layer_sizeList = [64,64,2])
    >>> model = arch.make_node()
    >>> input = {"x": torch.randn(64, 2)}
    >>> output = model.evaluate(input)
    Note
    ----
    For information regarding adaptive activations please refer to
    https://arxiv.org/abs/1906.01170.
    """
    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        module=None,
    ) -> None:
        super().__init__(
                input_keys=input_keys,
                output_keys=output_keys,
                detach_keys=detach_keys,
                periodicity=None,
            )
        self.module=nn.ModuleList()
        if module is None:
            self.module.append(torch.nn.Identity())
        else:
            self.module.append(module)
        
    def _tensor_forward(self, x: Tensor) -> Tensor:
        x = self.process_input(
            x,
            self.input_scales_tensor,
            periodicity=self.periodicity,
            input_dict=self.input_key_dict,
            dim=-1,
        )
        x = self.module[0](x)
        x = self.process_output(x, self.output_scales_tensor)
        return x

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.concat_input(
            in_vars,
            self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
        )
        y = self._tensor_forward(x)
        return self.split_output(y, self.output_key_dict, dim=-1)

    def _dict_forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        This is the original forward function, left here for the correctness test.
        """
        x = self.prepare_input(
            in_vars,
            self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
            input_scales=self.input_scales,
            periodicity=self.periodicity,
        )
        x = self.module[0](x)
        x = self.process_output(x, self.output_scales_tensor)
        return self.prepare_output(
            x, self.output_key_dict, dim=-1, output_scales=self.output_scales
        )
class CustomDualInputModuleArch(Arch):
    """Fully Connected Neural Network.

    Parameters
    ----------
    input_keys : List[Key]
        Input key list.
    output_keys : List[Key]
        Output key list.
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    domain_layer_sizeList : List[None,List[int]], optional
        List of Layer size for every hidden layer of the model including the output, by default 512,512,512,512,512,512,len(output_keys)
    domain_inputKeyList : List[boolList[Key]], optional
        Key to send to each NN of the domain cluster, by default [True]*len(domain_layer_sizeList)
    modes_decoding : torch.nn.Module, optional
        module to run before output of architecture, number of output must match number of output_keys, by default torch.nn.Identity()
    gEncoding_layer_sizeList: List[None,List[int]], optional
        List of Layer size for every hidden layer of the geometry encoding including the output, by default None
    gEncoding_inputKeyList: List[boolList[Key]], optional
        Key to send to each NN of the geometry encoding, by default [True]*len(gEncoding_layer_sizeList)
    gfEncoding_outputKey: Union[None,List[Key]] =None,
        Key to output from geometry and functional encoding, by default None
    functionalEncoding : torch.nn.Module, optional
        module to run between geometry encoding and domain cluster,  number of output must match number of gfEncoding_outputKey, by default None
    activation_fn : Activation, optional
        Activation function used by network, by default :obj:`Activation.SILU`
    periodicity : Union[Dict[str, Tuple[float, float]], None], optional
        Dictionary of tuples that allows making model give periodic predictions on
        the given bounds in tuple.
    skip_connections : bool, optional
        Apply skip connections every 2 hidden layers, by default False
    weight_norm : bool, optional
        Use weight norm on fully connected layers, by default True
    adaptive_activations : bool, optional
        Use an adaptive activation functions, by default False

    Variable Shape
    --------------
    - Input variable tensor shape: :math:`[N, size]`
    - Output variable tensor shape: :math:`[N, size]`

    Example
    -------
    Fully-connected model (2 -> 64 -> 64 -> 2)

    >>> arch = .geometrical_modes.GeometricalModesArch(
    >>>    [Key("x", size=2)],
    >>>    [Key("y", size=2)],
    >>>    domain_layer_sizeList = [64,64,2])
    >>> model = arch.make_node()
    >>> input = {"x": torch.randn(64, 2)}
    >>> output = model.evaluate(input)
    Note
    ----
    For information regarding adaptive activations please refer to
    https://arxiv.org/abs/1906.01170.
    """
    def __init__(
        self,
        input1_keys: List[Key],
        input2_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        module=None,
    ) -> None:
        super().__init__(
                input_keys=input1_keys+input2_keys,
                output_keys=output_keys,
                detach_keys=detach_keys,
                periodicity=None,
            )
        self.input1_key_dict = {str(var): var.size for var in input1_keys}
        self.total_input1_size=sum(self.input1_key_dict.values())
        self.total_input_size=sum(self.input_key_dict.values())
        self.input2_key_dict = {str(var): var.size for var in input2_keys}
        self.module=nn.ModuleList()
        if module is None:
            raise Exception("module cannot be None.")
        else:
            self.module.append(module)
        
    def _tensor_forward(self, x: Tensor,o: Tensor) -> Tensor:
        if self.input_scales_tensor is None:
            input_scales_tensor1=None
            input_scales_tensor2=None
        else:
            input_scales_tensor1=self.input_scales_tensor[...,0:self.total_input1_size]
            input_scales_tensor2=self.input_scales_tensor[...,self.total_input1_size:self.total_input_size]
        x = self.process_input(
            x,
            input_scales_tensor1,
            periodicity=self.periodicity,
            input_dict=self.input1_key_dict,
            dim=-1,
        )
        o = self.process_input(
            o,
            input_scales_tensor2,
            periodicity=self.periodicity,
            input_dict=self.input2_key_dict,
            dim=-1,
        )
        x = self.module[0](x,o)
        x = self.process_output(x, self.output_scales_tensor)
        return x

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.concat_input(
            in_vars,
            self.input1_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
        )
        o = self.concat_input(
            in_vars,
            self.input2_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
        )
        y = self._tensor_forward(x,o)
        return self.split_output(y, self.output_key_dict, dim=-1)

    def _dict_forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        This is the original forward function, left here for the correctness test.
        """
        x = self.prepare_input(
            in_vars,
            self.input1_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
            input_scales=self.input_scales,
            periodicity=self.periodicity,
        )
        o = self.prepare_input(
            in_vars,
            self.input2_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
            input_scales=self.input_scales,
            periodicity=self.periodicity,
        )
        x = self.module[0](x,o)
        x = self.process_output(x, self.output_scales_tensor)
        return self.prepare_output(
            x, self.output_key_dict, dim=-1, output_scales=self.output_scales
        )
class CustomZeroInputModuleArch(Arch):
    """Fully Connected Neural Network.

    Parameters
    ----------
    input_keys : List[Key]
        Input key list.
    output_keys : List[Key]
        Output key list.
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    domain_layer_sizeList : List[None,List[int]], optional
        List of Layer size for every hidden layer of the model including the output, by default 512,512,512,512,512,512,len(output_keys)
    domain_inputKeyList : List[boolList[Key]], optional
        Key to send to each NN of the domain cluster, by default [True]*len(domain_layer_sizeList)
    modes_decoding : torch.nn.Module, optional
        module to run before output of architecture, number of output must match number of output_keys, by default torch.nn.Identity()
    gEncoding_layer_sizeList: List[None,List[int]], optional
        List of Layer size for every hidden layer of the geometry encoding including the output, by default None
    gEncoding_inputKeyList: List[boolList[Key]], optional
        Key to send to each NN of the geometry encoding, by default [True]*len(gEncoding_layer_sizeList)
    gfEncoding_outputKey: Union[None,List[Key]] =None,
        Key to output from geometry and functional encoding, by default None
    functionalEncoding : torch.nn.Module, optional
        module to run between geometry encoding and domain cluster,  number of output must match number of gfEncoding_outputKey, by default None
    activation_fn : Activation, optional
        Activation function used by network, by default :obj:`Activation.SILU`
    periodicity : Union[Dict[str, Tuple[float, float]], None], optional
        Dictionary of tuples that allows making model give periodic predictions on
        the given bounds in tuple.
    skip_connections : bool, optional
        Apply skip connections every 2 hidden layers, by default False
    weight_norm : bool, optional
        Use weight norm on fully connected layers, by default True
    adaptive_activations : bool, optional
        Use an adaptive activation functions, by default False

    Variable Shape
    --------------
    - Input variable tensor shape: :math:`[N, size]`
    - Output variable tensor shape: :math:`[N, size]`

    Example
    -------
    Fully-connected model (2 -> 64 -> 64 -> 2)

    >>> arch = .geometrical_modes.GeometricalModesArch(
    >>>    [Key("x", size=2)],
    >>>    [Key("y", size=2)],
    >>>    domain_layer_sizeList = [64,64,2])
    >>> model = arch.make_node()
    >>> input = {"x": torch.randn(64, 2)}
    >>> output = model.evaluate(input)
    Note
    ----
    For information regarding adaptive activations please refer to
    https://arxiv.org/abs/1906.01170.
    """
    def __init__(
        self,
        output_keys: List[Key],
        module=None,
    ) -> None:
        super().__init__(
                input_keys=[],
                output_keys=output_keys,
                detach_keys=[],
                periodicity=None,
            )
        self.module=nn.ModuleList()
        if module is None:
            raise Exception("module cannot be None.")
        else:
            self.module.append(module)
        
    def _tensor_forward(self) -> Tensor:
        x = self.module[0]()
        x = self.process_output(x, self.output_scales_tensor)
        return x

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        y = self._tensor_forward()
        return self.split_output(y, self.output_key_dict, dim=-1)

    def _dict_forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        This is the original forward function, left here for the correctness test.
        """
        x = self.module[0]()
        x = self.process_output(x, self.output_scales_tensor)
        return self.prepare_output(
            x, self.output_key_dict, dim=-1, output_scales=self.output_scales
        )
def createKey(keystr,startnumber,endnumber):
    result=[]
    for n in range(startnumber,endnumber):
        result.append(Key(keystr+str(n)))
    return result

def FullyConnectedFlexiLayerSizeArch(
        input_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        layer_sizeList: Union[int,List[int]] = 512,
        skip_connections: bool = False,
        activation_fn: Activation = Activation.SILU,
        adaptive_activations: bool = False,
        weight_norm: bool = True,
        conv_layers: bool = False,
    ):
    if isinstance(layer_sizeList,int):
        if layer_sizeList<=0:
            layer_sizeList=[]
        else:
            layer_sizeList=[layer_sizeList]
    layer_sizeList=layer_sizeList+[sum([x.size for x in output_keys])]
    return CustomModuleArch(input_keys,
                            output_keys,
                            detach_keys,
                            module=FullyConnectedFlexiLayerSizeArchCore(in_features=sum([x.size for x in input_keys]),
                                                                        layer_sizeList=layer_sizeList,
                                                                        skip_connections=skip_connections,
                                                                        activation_fn=activation_fn,
                                                                        adaptive_activations=adaptive_activations,
                                                                        weight_norm=weight_norm,
                                                                        conv_layers=conv_layers
                                                                        )
                            )
def InheritedFullyConnectedFlexiLayerSizeArch(
        input_keys: List[Key],
        output_keys: List[Key],
        case_input_keys: List[Key],
        detach_keys: List[Key] = [],
        layer_sizeList: Union[int,List[int]] = 512,
        skip_connections: bool = False,
        activation_fn: Activation = Activation.SILU,
        caseNN_layer_sizeList: Union[int,List[int],None] = None,
        caseNN_skip_connections: bool = False,
        caseNN_activation_fn: Activation = Activation.SILU,
        caseNN_adaptive_activations: bool = False,
        caseNN_weight_norm: bool = True,
        with_caseID: bool = True,
    ):
    if isinstance(layer_sizeList,int):
        if layer_sizeList<=0:
            layer_sizeList=[]
        else:
            layer_sizeList=[layer_sizeList]
    layer_sizeList=layer_sizeList+[sum([x.size for x in output_keys])]
    return CustomDualInputModuleArch(input_keys,
                            case_input_keys,
                            output_keys,
                            detach_keys,
                            module=InheritedFullyConnectedFlexiLayerSizeArchCore(in_features=sum([x.size for x in input_keys]),
                                                                        layer_sizeList=layer_sizeList,
                                                                        skip_connections=skip_connections,
                                                                        activation_fn=activation_fn,
                                                                        caseNN_in_features=sum([x.size for x in case_input_keys]),
                                                                        caseNN_layer_sizeList=caseNN_layer_sizeList,
                                                                        caseNN_skip_connections=caseNN_skip_connections,
                                                                        caseNN_activation_fn=caseNN_activation_fn,
                                                                        caseNN_adaptive_activations=caseNN_adaptive_activations,
                                                                        caseNN_weight_norm=caseNN_weight_norm,
                                                                        )
                            )
def AdditionArch(
        input1_keys: List[Key],
        input2_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
    ):
    if sum([x.size for x in input1_keys])!=sum([x.size for x in input2_keys]):
        raise Exception("input1_keys and input2_keys in AdditionArch has to be the same total size")
    return CustomDualInputModuleArch(input1_keys,
                            input2_keys,
                            output_keys,
                            detach_keys,
                            module=AdditionArchCore()
                            )
def SubtractionArch(
        input1_keys: List[Key],
        input2_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
    ):
    if sum([x.size for x in input1_keys])!=sum([x.size for x in input2_keys]):
        raise Exception("input1_keys and input2_keys in AdditionArch has to be the same total size")
    return CustomDualInputModuleArch(input1_keys,
                            input2_keys,
                            output_keys,
                            detach_keys,
                            module=SubtractionArchCore()
                            )
def MultiplicationArch(
        input1_keys: List[Key],
        input2_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
    ):
    size1=sum([x.size for x in input1_keys])
    size2=sum([x.size for x in input2_keys])
    if size1!=size2 and size1!=1 and size2!=1:
        raise Exception("input1_keys and input2_keys in AdditionArch has to be the same total size")
    return CustomDualInputModuleArch(input1_keys,
                            input2_keys,
                            output_keys,
                            detach_keys,
                            module=MultiplicationArchCore()
                            )
def SumMultiplicationArch(
        input1_keys: List[Key],
        input2_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
    ):
    size1=sum([x.size for x in input1_keys])
    size2=sum([x.size for x in input2_keys])
    if size1!=size2 and size1!=1 and size2!=1:
        raise Exception("input1_keys and input2_keys in AdditionArch has to be the same total size")
    return CustomDualInputModuleArch(input1_keys,
                            input2_keys,
                            output_keys,
                            detach_keys,
                            module=SumMultiplicationArchCore()
                            )
def SingleInputInheritedFullyConnectedFlexiLayerSizeArch(
        input_keys: List[Key],
        output_keys: List[Key],
        case_input_keys: List[Key],
        detach_keys: List[Key] = [],
        layer_sizeList: Union[int,List[int]] = 512,
        skip_connections: bool = False,
        activation_fn: Activation = Activation.SILU,
        caseNN_layer_sizeList: Union[int,List[int],None] = None,
        caseNN_skip_connections: bool = False,
        caseNN_activation_fn: Activation = Activation.SILU,
        caseNN_adaptive_activations: bool = False,
        caseNN_weight_norm: bool = True,
        with_caseID: int = 0,
    ):
    adjust_caseNN_in_features=0
    if with_caseID:
        adjust_caseNN_in_features=-1
    if isinstance(layer_sizeList,int):
        if layer_sizeList<=0:
            layer_sizeList=[]
        else:
            layer_sizeList=[layer_sizeList]
    layer_sizeList=layer_sizeList+[sum([x.size for x in output_keys])]
    return CustomModuleArch(input_keys+case_input_keys,
                            output_keys,
                            detach_keys,
                            module=SingleInputInheritedFullyConnectedFlexiLayerSizeArchCore(in_features=sum([x.size for x in input_keys]),
                                                                        layer_sizeList=layer_sizeList,
                                                                        skip_connections=skip_connections,
                                                                        activation_fn=activation_fn,
                                                                        caseNN_in_features=sum([x.size for x in case_input_keys])+adjust_caseNN_in_features,
                                                                        caseNN_layer_sizeList=caseNN_layer_sizeList,
                                                                        caseNN_skip_connections=caseNN_skip_connections,
                                                                        caseNN_activation_fn=caseNN_activation_fn,
                                                                        caseNN_adaptive_activations=caseNN_adaptive_activations,
                                                                        caseNN_weight_norm=caseNN_weight_norm,
                                                                        with_caseID=with_caseID
                                                                        )
                            )
def HypernetFullyConnectedFlexiLayerSizeArch(
        input_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        layer_sizeList: Union[int,List[int],None] = None,
        skip_connections: bool = False,
        activation_fn: Activation = Activation.SILU,
        adaptive_activations: bool = False,
        weight_norm: bool = True,
        with_caseID: int = 0,
    ):
    adjust_caseNN_in_features=0
    if with_caseID:
        adjust_caseNN_in_features=-1
    if isinstance(layer_sizeList,int):
        if layer_sizeList<=0:
            layer_sizeList=[]
        else:
            layer_sizeList=[layer_sizeList]
    layer_sizeList=layer_sizeList+[sum([x.size for x in output_keys])]
    return CustomModuleArch(input_keys,
                            output_keys,
                            detach_keys,
                            module=HypernetFullyConnectedFlexiLayerSizeArchCore(in_features=sum([x.size for x in input_keys])+adjust_caseNN_in_features,
                                                                        layer_sizeList=layer_sizeList,
                                                                        skip_connections=skip_connections,
                                                                        activation_fn=activation_fn,
                                                                        adaptive_activations=adaptive_activations,
                                                                        weight_norm=weight_norm,
                                                                        with_caseID=with_caseID
                                                                        )
                            )
def CaseIDtoFeatureArch(
        input_key:Key,
        output_keys: List[Key],
        feature_array,
    ):
    return CustomModuleArch([input_key],
                            output_keys,
                            [],
                            module=CaseIDtoFeatureArchCore(feature_array,
                                                    )
                            )
def FixedFeatureArch(
        output_keys: List[Key],
        feature_array,
    ):
    return CustomZeroInputModuleArch(output_keys,
                            module=FixedFeatureArchCore(feature_array,
                                                    )
                            )
    
def ParametricInsertArch(
        output_keys: List[Key],
        input_keys=None
    ):
    if input_keys is None:
        input_keys=[Key('x')]
    return CustomModuleArch(input_keys,
                            output_keys,
                            [],
                            module=ParametricInsert(out_features=sum([x.size for x in output_keys]),
                                                    )
                            )
def BsplineArch(
        input_keys: List[Key],
        output_keys: List[Key],
        origin:List[float],
        spacing:List[float],
        nodes_shape:List[int],
        fix_coef=None,
        detach_keys: List[Key] = [],
    ):
    if len(origin)==2:
        return CustomModuleArch(input_keys,
                                output_keys,
                                detach_keys,
                                module=Bspline2D(origin,
                                                 spacing,
                                                 nodes_shape,
                                                 fix_coef
                                                 )
                                )
    elif len(origin)==3:
        return CustomModuleArch(input_keys,
                                output_keys,
                                detach_keys,
                                module=Bspline3D(origin,
                                                 spacing,
                                                 nodes_shape,
                                                 fix_coef
                                                 )
                                )
    else:
        raise Exception("Dimension "+str(len(origin))+" not implemented")