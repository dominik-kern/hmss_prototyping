"""
Kim case 1 (Terzaghi), hydromechanical (HM), staggered scheme (stress-split)
spatial FE discretization for both H and M
temporal Euler-backward discretization
"""
from __future__ import print_function
import fenics as fe

## input variables
# geometry
Length = 15;     # height
Width = 1;      # width
# discretization
Nx=1        # mesh divisions x-direction
Ny=15       # mesh divisoins y-direction
dt=86400.0/5   # time step
Nt=10   # number of time steps
Nci=3   # number of coupling iterations, fixed number is uncool, TODO convergence check
# physical parameters
rho = 2400 # bulk density
rho_FR = 1000.0    # real fluid density 
p_ref = 2.125*10**6    # load on top
alpha_B = 1.0 # Biot coefficient
phi = 0.3 # porosity
K_SR=10.0**9  # drained bulk modulus
beta_p = 4.0*10**(-8) # fluid compressibility
K = 50.0*10**(-15)   # permeability
mu = 1.0*10**(-3)    # viscosity
E = 100*10**6;    # Young's modulus TODO derive from bulk modulus
nu = 0;     # Poisson ratio
overburden = 2.125*10**6    # load on top
acceleration_vector = fe.Constant((0, 0))   # causing body forces
ZeroVector = fe.Constant((0,0))
ZeroScalar = fe.Constant((0))	
# dependent parameters
Lame1 = E*nu/(1+nu)/(1-2*nu)  
Lame2 = E/2/(1+nu) 
M = 1/( (alpha_B-phi)/K_SR+phi*beta_p )# 1/M M...Biot modulus

# Define strain and stress
def epsilon(u):
    return fe.sym(fe.grad(u))
def sigma(u):
    return Lame1*fe.div(u)*fe.Identity(2) + 2*Lame2*epsilon(u)

## Create mesh (simplex elements=triangles in 2D) and define function space
mesh = fe.RectangleMesh(fe.Point(0, 0), fe.Point(Width, Length), Nx, Ny)
VH = fe.FunctionSpace(mesh, 'P', 1)
VM = fe.VectorFunctionSpace(mesh, 'P', degree=2)
p = fe.Function(VH, name="pressure")
vtkfileH = fe.File('pressure.pvd')
u = fe.Function(VM, name="displacement")
vtkfileM = fe.File('displacement.pvd')

# IC
p_0 = fe.Constant(p_ref)
p_n = fe.interpolate(p_0, VH)
u_0 = ZeroVector
u_n = fe.interpolate(u_0, VM)
sv = fe.interpolate(ZeroScalar, VH)
sv_n = fe.interpolate(ZeroScalar, VH)


# DirichletBC
tol = 1E-14
def top(x, on_boundary):    
    return on_boundary and (fe.near(x[1], Length, tol))   
bcH = fe.DirichletBC(VH, 2*p_ref, top)

def bottom(x, on_boundary):    
    return on_boundary and (fe.near(x[1], 0, tol))   
bcM = fe.DirichletBC(VM, fe.Constant((0, 0)), bottom)

# Neumann BC
boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
topSD = fe.AutoSubDomain(lambda x: fe.near(x[1], Length, tol))
topSD.mark(boundaries, 1)
ds = fe.ds(subdomain_data=boundaries)
traction_topM = fe.Constant((0, overburden))		


# Define variational H problem a(v,p)=L(v)
pH = fe.TrialFunction(VH)
vH = fe.TestFunction(VH)
sH = ZeroScalar     # source terms (NOT IN INTEGRAL AS ZERO MAKES TROUBLE)
bH = rho_FR*acceleration_vector     # body force
aH = vH*(rho_FR*pH/M)*fe.dx     # storage with p_(n+1)
+ dt*rho_FR*(K/mu)*fe.dot(fe.grad(vH), fe.grad(pH))*fe.dx   # pressure-driven Darcy
+ vH*(rho_FR*(alpha_B**2)/K_SR)*pH*fe.dx    # M coupling with p_(n+1)
LH = vH*(rho_FR*p_n/M)*fe.dx    # storage with p_n
+ dt*rho_FR*(K/mu)*fe.dot(fe.grad(vH), bH)*fe.dx    # body-force-driven Darcy
+ vH*(rho_FR*(alpha_B/K_SR)*(sv-sv_n-alpha_B*p_n))*fe.dx # M coupling (fixed-stress)
# Define variational M problem a(v,p)=L(v)
uM = fe.TrialFunction(VM)
vM = fe.TestFunction(VM)
bM = rho*acceleration_vector	# body force
aM = fe.inner(sigma(uM), epsilon(vM))*fe.dx
LM = fe.dot(bM, vM)*fe.dx # body force
+ fe.inner(fe.grad(vM), p_n*fe.Identity(2)) *fe.dx # fluid pressure 
+ fe.dot(traction_topM, vM)*ds(1)   # boundary tractions

# Time-stepping
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
        sigma_v=fe.project( (1.0/3.0)*fe.tr(sigma(u)), VH)
        sv.assign(sigma_v)
    sv_n.assign(sigma_v)
    vtkfileH << (p,t)
    vtkfileM << (u,t)
        
        
        