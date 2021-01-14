# TODO 
#    monitor convergence
#    verification of case 1 and 2 [Kim09] (then involve Tengfei)
#    quadliteral mesh
#    monolithic formulation
"""
Kim case 1 (Terzaghi), hydromechanical (HM), staggered scheme (stress-split)
spatial FE discretization for both H and M
temporal Euler-backward discretization
"""
from __future__ import print_function
import fenics as fe

## input variables
# geometry
Length = 15;    # 0...yMAX 
Width = 1;      # 0...xMAX
# discretization
Nx=1        # mesh divisions x-direction
Ny=15       # mesh divisions y-direction
dt=86400.0/5.0   # time step
Nt=10   # number of time steps
Nci=3   # number of coupling iterations, fixed number is uncool, TODO convergence check
# physical parameters
rho = 2400 # bulk density
rho_FR = 1000.0    # real fluid density 
p_ref = 2.125*10**6    # reference pressure
alpha_B = 1.0 # Biot coefficient
phi = 0.3 # porosity
K_S=10.0**9  # drained bulk modulus
nu = 0;     # Poisson ratio
beta_p = 4.0*10**(-8) # fluid compressibility
K = 50.0*10**(-15)   # permeability
mu = 1.0*10**(-3)    # viscosity

overburden = 2.0*2.125*10**6    # load on top
acceleration_vector = fe.Constant((0, 0))   # causing body forces
ZeroVector = fe.Constant((0,0))
ZeroScalar = fe.Constant((0))	
# dependent parameters
E = K_S/(3*(1-2*nu));    # Young's modulus (bulk, drained)
Lame1 = E*nu/(1+nu)/(1-2*nu)  
Lame2 = E/2/(1+nu) 
M = 1/( (alpha_B-phi)/K_S+phi*beta_p )# 1/M M...Biot modulus

# Define strain and stress
def epsilon(u):
    return fe.sym(fe.grad(u))
def sigma_eff(u):
    return Lame1*fe.div(u)*fe.Identity(2) + 2*Lame2*epsilon(u)

## Create mesh (simplex elements in 2D=triangles) and define function spaces
mesh = fe.RectangleMesh(fe.Point(0, 0), fe.Point(Width, Length), Nx, Ny)
VH = fe.FunctionSpace(mesh, 'P', 1)
VM = fe.VectorFunctionSpace(mesh, 'P', degree=2)

# IC undeformed, at rest, constant pressure everywhere
p_0 = fe.Constant(p_ref)
p_n = fe.interpolate(p_0, VH)
u_0 = ZeroVector
u_n = fe.interpolate(u_0, VM)
sv_0 = fe.Constant(-p_ref)
sv = fe.interpolate(sv_0, VH)   # hydrostatic total stress
sv_n = fe.interpolate(sv_0, VH) # hydrostatic total stress


# DirichletBC, assign geometry via functions
tol = 1E-14
def topbottom(x, on_boundary):    
    return on_boundary and ( fe.near(x[1], Length, tol) or fe.near(x[1], 0.0, tol) )   
bcH = fe.DirichletBC(VH, 2*p_ref, topbottom)  # drainage on top and bottom

def bottom(x, on_boundary):    
    return on_boundary and (fe.near(x[1], 0.0, tol))   
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
bH = rho_FR*acceleration_vector     # body force

aH = vH*rho_FR*((alpha_B**2)/K_S+1/M)*pH*fe.dx     # storage with p_(n+1)
+ dt*rho_FR*(K/mu)*fe.dot(fe.grad(vH), fe.grad(pH))*fe.dx   # pressure-driven Darcy

LH = vH*rho_FR*((alpha_B**2)/K_S+1/M)*p_n*fe.dx    # storage with p_n
+ dt*(rho_FR**2)*(K/mu)*fe.dot(fe.grad(vH), bH)*fe.dx    # body-force-driven Darcy
- vH*rho_FR*(alpha_B/K_S)*(sv-sv_n)*fe.dx # M coupling (fixed-stress)
# no Neumann BC (inflows)
# no sources

# Define variational M problem a(v,p)=L(v)
uM = fe.TrialFunction(VM) 
vM = fe.TestFunction(VM)
bM = rho*acceleration_vector	# body force

aM = fe.inner(epsilon(vM), sigma_eff(uM))*fe.dx

LM = fe.dot(vM, bM)*fe.dx # body force
+ fe.inner(fe.grad(vM), alpha_B*p_n*fe.Identity(2)) *fe.dx # fluid pressure 
+ fe.dot(vM, traction_topM)*ds(1)   # boundary tractions

# Time-stepping
p = fe.Function(VH, name="pressure")    # function solved for and written to file
vtkfileH = fe.File('pressure.pvd')
u = fe.Function(VM, name="displacement") # function to solve for and written to file
vtkfileM = fe.File('displacement.pvd')

t = 0.0
p.assign(p_n)
vtkfileH << (p, t)
u.assign(u_n)
vtkfileM << (u, t)
for n in range(Nt):
    t += dt
    for nn in range(Nci):
        fe.solve(aH == LH, p, bcH)
        p_n.assign(p)
        fe.solve(aM == LM, u, bcM)
        sigma_v=fe.project( (1.0/3.0)*fe.tr(sigma_eff(u)) - alpha_B*p, VH)
        sv.assign(sigma_v)
    sv_n.assign(sigma_v)
    vtkfileH << (p,t)
    vtkfileM << (u,t)
        
        
        