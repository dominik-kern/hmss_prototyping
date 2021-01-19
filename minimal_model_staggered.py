# TODO 
#    verification  vs analytical solution 
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
dt=1.0
Nt=5   # number of time steps
Nci=30   # number of coupling iterations, fixed number is uncool, TODO convergence check
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
K=E/(3.0*(1-2*nu))
Lame1 = E*nu/(1+nu)/(1-2*nu)  
Lame2 = E/2/(1+nu) 
k_mu=k/mu

def max_norm_delta(f1, f2):
    vertex_values_f1 = f1.compute_vertex_values(mesh)
    vertex_values_f2 = f2.compute_vertex_values(mesh)
    return np.max( np.abs(vertex_values_f1 - vertex_values_f2) )

# Define strain and stress (2D plain strain)
def epsilon(u):
    return fe.sym(fe.grad(u))
def sigma_eff(u):
    return Lame1*fe.div(u)*fe.Identity(2) + 2*Lame2*epsilon(u)

## Create mesh (simplex elements in 2D=triangles) and define function spaces
mesh = fe.UnitSquareMesh(Nx, Ny)
VH = fe.FunctionSpace(mesh, 'P', 1)
VM = fe.VectorFunctionSpace(mesh, 'P', 2)
Vsig = fe.TensorFunctionSpace(mesh, "P", 2)

p = fe.Function(VH, name="fluidpressure")    # function solved for and written to file
u = fe.Function(VM, name="displacement") # function to solve for and written to file
s_eff = fe.Function(Vsig, name="effective_stress")
sv = fe.Function(VH, name="hydrostatic_totalstress")    # function solved for and written to file
sv0 = fe.Function(VH, name="initial hydrostatic_totalstress")    # function solved for and written to file

# IC undeformed, at rest, constant pressure everywhere
p_0 = fe.Constant(p_ref)
p_n = fe.interpolate(p_0, VH)   # fluid pressure at t=t_n
p_ = fe.interpolate(p_0, VH)   # fluid pressure at t=t_(n+1) at coupling iterations
u_0 =fe.Expression( ('0','0'), degree=2)    # undeformed
u_n = fe.interpolate(u_0, VM)   # displacement at t=t_n
sv0.assign(fe.project(fe.div(u_n), VH))     # derive from initial displacement
sv_n = fe.interpolate(sv0, VH) # hydrostatic total stress at t_n 
sv_ = fe.interpolate(sv0, VH) # same as sv_n at t_(n+1) at coupling iterations (at first coupling iteration sv_=sv_n)

# DirichletBC, assign geometry via functions
tol = 1E-14
def top(x, on_boundary):    
    return on_boundary and fe.near(x[1], 1.0, tol) 
bcH = fe.DirichletBC(VH, p_ref, top)  # drainage on top 

def bottom(x, on_boundary):    
    return on_boundary and fe.near(x[1], 0.0, tol)   
bcMfixed = fe.DirichletBC(VM, ZeroVector, bottom)   # fixed bottom

def leftright(x, on_boundary):    
    return on_boundary and (fe.near(x[0], 0.0, tol) or fe.near(x[0], 1.0, tol))
bcMroller = fe.DirichletBC(VM.sub(0), ZeroScalar, leftright)   # rollers on side
bcM = [bcMfixed, bcMroller]

# Neumann BC, assign geometry via subdomains to have ds accessible in variational problem
boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
topSD = fe.AutoSubDomain(lambda x: fe.near(x[1], 1, tol))
topSD.mark(boundaries, 1)   # access via ds(1)
ds = fe.ds(subdomain_data=boundaries)
traction_topM = fe.Constant((0, overburden))		

# Define variational H problem a(v,p)=L(v)
pH = fe.TrialFunction(VH)
vH = fe.TestFunction(VH)

aH = ( vH*pH/K + k_mu*fe.dot(fe.grad(vH), fe.grad(pH))*dt )*fe.dx

LH = (vH*(p_n+sv_n-sv_)/K)*fe.dx 
# no non-zero Neumann BC, i.e. no prescribed in-outflows (only flow via DirichletBC possible)
# no sources

# Define variational M problem a(v,p)=L(v)
uM = fe.TrialFunction(VM) 
vM = fe.TestFunction(VM)

aM = fe.inner(sigma_eff(vM), epsilon(uM))*fe.dx

LM = p_*fe.div(vM)*fe.dx + fe.dot(vM, traction_topM)*ds(1)

# no body forces

# Time-stepping
vtkfile_p = fe.File('fluidpressure.pvd')
vtkfile_u = fe.File('displacement.pvd')
vtkfile_s_eff = fe.File('effective_stress.pvd')
vtkfile_sv = fe.File('hydrostatic_totalstress.pvd')

t = 0.0
p.assign(p_n)
u.assign(u_n)
sv.assign(sv_n)   # hydrostatic total stress
s_eff.assign(fe.project(sigma_eff(u), Vsig))
vtkfile_p << (p, t)
vtkfile_u << (u, t)
vtkfile_sv << (sv,t)
vtkfile_s_eff << (s_eff,t)
for n in range(Nt):
    t += dt
    print(n+1,".step   t=",t)
    for nn in range(Nci):
        
        fe.solve(aH == LH, p, bcH)
        delta_p=max_norm_delta(p_, p)
        p_.assign(p)
        
        fe.solve(aM == LM, u, bcM)
        s_eff.assign(fe.project(sigma_eff(u), Vsig))
        sv.assign( fe.project( (1.0/3.0)*fe.tr(s_eff) - p, VH) )
        delta_sv=max_norm_delta(sv_, sv)
        sv_.assign(sv)
        
        print(delta_p, delta_sv)
        
    sv_n.assign(sv)
    p_n.assign(p)
    print()
    vtkfile_p << (p,t)
    vtkfile_u << (u,t)
    vtkfile_sv << (sv,t)
    vtkfile_s_eff << (s_eff,t)

