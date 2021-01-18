# TODO 
#    verification of case 1 and 2 [Kim09] 
#    roller BC leftright
#    quadliteral mesh
#    monolithic formulation
"""
2D Minimal model (1D Terzaghi), hydromechanical (HM), staggered scheme (stress-split)
spatial FE discretization for both H and M
temporal Euler-backward discretization
incompressible fluid, incompressible solid (but linear elastic bulk)
"""
from __future__ import print_function
import fenics as fe
import numpy as np

fe.set_log_level(30)  # control info/warning/error messages

# discretization
Nx=1        # mesh divisions x-direction
Ny=1       # mesh divisions y-direction
dt=0.1
Nt=3   # number of time steps
Nci=3   # number of coupling iterations, fixed number is uncool, TODO convergence check
# physical parameters
p_ref = 1.0    # reference pressure
E = 10.0  # Young's modulus (bulk, drained)
nu = 0.0;     # Poisson ratio (bulk)
k = 0.1   # permeability
mu = 1.0    # viscosity

overburden = -2.0*p_ref    # load on top (y-direction)
ZeroVector = fe.Constant((0,0))
ZeroScalar = fe.Constant((0))	
# dependent parameters
Lame1 = E*nu/(1+nu)/(1-2*nu)  
Lame2 = E/2/(1+nu) 

def max_norm_delta(f1, f2):
    vertex_values_f1 = f1.compute_vertex_values(mesh)
    vertex_values_f2 = f2.compute_vertex_values(mesh)
    return np.max( np.abs(vertex_values_f1 - vertex_values_f2) )

# Define strain and stress
def epsilon(u):
    return fe.sym(fe.grad(u))
def sigma_eff(u):
    return Lame1*fe.div(u)*fe.Identity(2) + 2*Lame2*epsilon(u)

## Create mesh (simplex elements in 2D=triangles) and define function spaces
mesh = fe.UniSquareMesh(Nx, Ny)
VH = fe.FunctionSpace(mesh, 'P', 1)
VM = fe.VectorFunctionSpace(mesh, 'P', 2)
Vsig = fe.TensorFunctionSpace(mesh, "P", 2)

# IC undeformed, at rest, constant pressure everywhere
p_0 = fe.Constant(p_ref)
p_n = fe.interpolate(p_0, VH)
u_0 = ZeroVector
u_n = fe.interpolate(u_0, VM)
sv_0 = fe.Constant(ZeroScalar)
sv_n = fe.interpolate(sv_0, VH) # hydrostatic total stress
sv = fe.interpolate(sv_0, VH) # same as sv_n ("at rest")

# DirichletBC, assign geometry via functions
tol = 1E-14
def top(x, on_boundary):    
    return on_boundary and fe.near(x[1], 1.0, tol) 
bcH = fe.DirichletBC(VH, p_ref, topbottom)  # drainage on top and bottom

def bottom(x, on_boundary):    
    return on_boundary and fe.near(x[1], 0.0, tol)   
bcM = fe.DirichletBC(VM, ZeroVector, bottom)   # fixed bottom

# Neumann BC, assign geometry via subdomains to have ds accessible in variational problem
boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
topSD = fe.AutoSubDomain(lambda x: fe.near(x[1], Length, tol))
topSD.mark(boundaries, 1)   # access via ds(1)
ds = fe.ds(subdomain_data=boundaries)
traction_topM = fe.Constant((0, overburden))		

# Define variational H problem a(v,p)=L(v)
pH = fe.TrialFunction(VH)
vH = fe.TestFunction(VH)

aH = 

LH = 
# no non-zero Neumann BC, i.e. no prescribed in-outflows (only flow via DirichletBC possible)
# no sources

# Define variational M problem a(v,p)=L(v)
uM = fe.TrialFunction(VM) 
vM = fe.TestFunction(VM)

aM = 

LM = 

# no body forces

# Time-stepping
p = fe.Function(VH, name="pressure")    # function solved for and written to file
#sv = fe.Function(VH, name="hydrostatic_stress")    # function solved for and written to file
u = fe.Function(VM, name="displacement") # function to solve for and written to file
s_eff = fe.Function(Vsig, name="effective_stress")
vtkfile_p = fe.File('pressure.pvd')
vtkfile_u = fe.File('displacement.pvd')
vtkfile_s = fe.File('stress.pvd')
#vtkfile_sv = fe.File('hydrostatic_stress.pvd')

t = 0.0
p.assign(p_n)
vtkfile_p << (p, t)
u.assign(u_n)
vtkfile_u << (u, t)
#sv = fe.assign(sv_n)   # hydrostatic total stress
# TODO sigma and sv
for n in range(Nt):
    t += dt
    print(n+1,".step   t=",t)
    for nn in range(Nci):
        
        fe.solve(aH == LH, p, bcH)
        delta_p=max_norm_delta(p_n, p)
        p_n.assign(p)
        
        fe.solve(aM == LM, u, bcM)
        s_eff.assign(fe.project(sigma_eff(u), Vsig))
        sigma_v=fe.project( (1.0/3.0)*fe.tr(sigma_eff(u)) - alpha_B*p, VH)
        sv.assign(sigma_v)
        delta_sv=max_norm_delta(sv_n, sv)
        
        print(delta_p, delta_sv)
        
    sv_n.assign(sigma_v)
    vtkfile_p << (p,t)
    vtkfile_u << (u,t)
    #vtkfile_sv << (sv,t)
    vtkfile_s << (s_eff,t)

        
        
        