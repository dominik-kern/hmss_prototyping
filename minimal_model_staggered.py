# TODO 
#    verification  vs analytical and monolithic solution 
#    quadliteral mesh
#    monolithic formulation
"""
2D Minimal model (of 1D Terzaghi), hydromechanical (HM), staggered scheme (stress-split)
spatial FE discretization for both H and M (Taylor Hood)
temporal Euler-backward discretization
incompressible fluid, incompressible solid (but linear elastic bulk)
"""
from __future__ import print_function
import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import terzaghi_analytical_solution # TODO class

fe.set_log_level(30)  # control info/warning/error messages
vtkfile_p = fe.File('fluidpressure.pvd')
vtkfile_u = fe.File('displacement.pvd')
vtkfile_s_eff = fe.File('effective_stress.pvd')
vtkfile_sv = fe.File('hydrostatic_totalstress.pvd')
vtkfile_s_total = fe.File('totalstress.pvd')

# discretization
Nx=10        # mesh divisions x-direction
Ny=10       # mesh divisions y-direction
dt=0.01 # initial time step
dt_prog=1.15 # time step progression
Nt=20   # number of time steps
Nci_max=100   # maximal number of coupling iterations
N_Fourier=50 # representation of analytical solution as Fourier series for comparison
RelTol_ci=1.0e-10   # relative tolerance of coupling iterations
# physical parameters
p_ref = 1.0    # reference pressure
E = 10.0  # Young's modulus (bulk, drained)
nu = 0.0;     # Poisson ratio (bulk)
k = 0.1   # permeability
mu = 1.0    # viscosity

overburden = -p_ref    # load on top (y-direction)
ZeroVector = fe.Constant((0,0))
ZeroScalar = fe.Constant((0))	
# dependent parameters
Length = 1 # unit square!
Width = 1 # unit square!
K=E/(3.0*(1-2*nu))
Lame1 = E*nu/((1+nu)*(1-2*nu))
Lame2 = E/(2*(1+nu)) 
k_mu = k/mu
cc = E*k_mu # consolidation coefficient
dT=fe.Constant(dt) #  make time step mutable

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
#mesh = fe.RectangleMesh.create([fe.Point(0, 0), fe.Point(Width, Length)], [Nx,Ny], fe.CellType.Type.quadrilateral)

VH = fe.FunctionSpace(mesh, 'P', 1)
VM = fe.VectorFunctionSpace(mesh, 'P', 2)   # TODO tests
Vsig = fe.TensorFunctionSpace(mesh, "P", 1)     # TODO tests

p = fe.Function(VH, name="fluidpressure")    # function solved for and written to file
u = fe.Function(VM, name="displacement") # function to solve for and written to file
s_eff = fe.Function(Vsig, name="effective_stress")
s_total = fe.Function(Vsig, name="total_stress")
sv = fe.Function(VH, name="hydrostatic_totalstress")    # function solved for and written to file
sv0 = fe.Function(VH, name="initial hydrostatic_totalstress")    # function solved for and written to file

# IC undeformed, at rest, constant pressure everywhere
p_0 = fe.Constant(0*p_ref)
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
    return on_boundary and fe.near(x[1], Length, tol) 
bcH = fe.DirichletBC(VH, 0*p_ref, top)  # drainage on top 

def bottom(x, on_boundary):    
    return on_boundary and fe.near(x[1], 0.0, tol)   
bcMfixed = fe.DirichletBC(VM, ZeroVector, bottom)   # fixed bottom

def leftright(x, on_boundary):    
    return on_boundary and (fe.near(x[0], 0.0, tol) or fe.near(x[0], Width, tol))
bcMroller = fe.DirichletBC(VM.sub(0), ZeroScalar, leftright)   # rollers on side, i.e. fix only in x-direction
bcM = [bcMfixed, bcMroller]

# Neumann BC, assign geometry via subdomains to have ds accessible in variational problem
boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
topSD = fe.AutoSubDomain(lambda x: fe.near(x[1], Length, tol))
topSD.mark(boundaries, 1)   # accessable via ds(1)
ds = fe.ds(subdomain_data=boundaries)
traction_topM = fe.Constant((0, overburden))		

# Define variational H problem a(v,p)=L(v)
pH = fe.TrialFunction(VH)
vH = fe.TestFunction(VH)

aH = ( vH*pH/K + dT*k_mu*fe.dot(fe.grad(vH), fe.grad(pH)) )*fe.dx

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
t = 0.0
p.assign(p_n)
u.assign(u_n)
sv.assign(sv_n)   # hydrostatic total stress
s_eff.assign(fe.project(sigma_eff(u), Vsig))    # effective stress
s_total.assign(fe.project(sigma_eff(u)-p*fe.Identity(2), Vsig))
vtkfile_p << (p, t)
vtkfile_u << (u, t)
vtkfile_sv << (sv, t)
vtkfile_s_eff << (s_eff, t)
vtkfile_s_total << (s_total, t)

y_fem=np.linspace(tol, Length-tol, Ny+1)
y_ana=np.linspace(tol, Length-tol, 101)
points=[ ((1.0/2.0)*Width, y_) for y_ in y_fem ]
conv_monitor=np.zeros((Nt, Nci_max))
for n in range(Nt):     # time steps
    t += dt
    print(n+1,".step   t=",t)
    for nn in range(Nci_max):   # couplint iterations
        
        fe.solve(aH == LH, p, bcH)
        delta_p=max_norm_delta(p_, p)
        p_.assign(p)    # shift forward coupling iteration
        
        fe.solve(aM == LM, u, bcM)
        s_eff.assign(fe.project(sigma_eff(u), Vsig))    # effective stress
        sv.assign( fe.project( (1.0/3.0)*fe.tr(s_eff) - p, VH) )    # hydrostatic total stress
        delta_sv=max_norm_delta(sv_, sv)
        sv_.assign(sv)   # shift forward coupling iteration
        
        print(nn+1,'. ', delta_p, delta_sv)
        
        conv_criterium=np.abs(delta_p/p_ref) + np.abs(delta_sv/p_ref)  # take both to exclude random hit of one variable at initial state
        conv_monitor[n,nn]=conv_criterium
        if conv_criterium < RelTol_ci:
            break
        
    if nn+1==Nci_max:
        print("Solution not converged to RelTol=", RelTol_ci)
    p_fem = np.array([ p(point) for point in points])
    p_ana = terzaghi_analytical_solution.dim_sol(y_ana, t, N_Fourier, Length, cc, p_ref)
    color_code=[0.9*(1-(n+1)/Nt)]*3
    plt.plot(y_fem, p_fem,  color=color_code, linestyle='none', marker='x', markersize=6)    
    plt.plot(y_ana, p_ana,  color=color_code)    
    #plt.plot([ np.log10(R) for R in conv_monitor[n,:] if R>0 ])
    sv_n.assign(sv)     # shift forward time step
    p_n.assign(p)       # shift forward time step
    s_total.assign(fe.project(sigma_eff(u)-p*fe.Identity(2), Vsig))
    dt*=dt_prog
    dT.assign(dt)
    print()
    vtkfile_p << (p,t)
    vtkfile_u << (u,t)
    vtkfile_sv << (sv,t)
    vtkfile_s_eff << (s_eff,t)
    vtkfile_s_total << (s_total, t)
    # TODO xdmf for multiple fields

plt.show()

