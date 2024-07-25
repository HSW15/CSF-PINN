# CSF-PINN: Coupled Sequential Frames - Physics Informed Neural Network

The repository contains sample codes for flow reconstruction of left ventricle using from echocardiography, guided by scarce samples of colour doppler flow imaging measurements. The PINN framework was demonstrated on two patient-specific foetal left ventricles and one adult left ventricle. Simulated ground truth was obtained from computational fluid dynamics and fluid structure interaction models. 
Ground truth model:

https://pubmed.ncbi.nlm.nih.gov/35731342/
https://pubmed.ncbi.nlm.nih.gov/28886196/

Codes extentions from NVIDIA Modulus can be found in folder "ModulusDL"

Foetal Left Ventricle geometry, boundary conditions obtained from motion-tracking algorithm, and synthetically generated color Doppler data is located in folder "temporal_stls". For the adult model, it is located in folder "FSI_model"

Ground truth model results are available in the folder "Simulation Results"

Various python sample codes are available, ready for use.

Description of CSF-PINN architecture
"This framework utilize a time-marching approach to simulate the temporal evolution of intracardiac flows, where the flow field is resolved at 2-4 discrete, consecutive time instances within a window that match the time instances of Doppler imaging. The framework can be subsequently repeated for the next time window to resolve flow fields in later time instances, using the existing PINN solution as the initial condition for this subsequent time window."

PINN based on Nvidia Modulus v20.09 framework. For Installation of NVIDIA Modulus, please refer to 

https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/getting_started/installation.html 


