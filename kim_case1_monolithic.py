# TODO 
#    elasticity 3D
#    quadliteral mesh
"""
2D model of [Kim2009] case 1 (of 1D Terzaghi), hydromechanical (HM)
spatial FE discretization for both H and M (Taylor Hood)
temporal mid-point discretization
"""
from __future__ import print_function
import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import kim_case1_parameters as kc1
import stress_and_strain as sas


# DECLARATION
model=kc1.KC1()
ZeroScalar = fe.Constant((0))	
ZeroVector = fe.Constant((0,0))
p_ref,  E,  nu,  k,  mu,  rho_f,  tau,  beta_s,  phi=model.get_physical_parameters()
Nx,  Ny,  dt,  dt_prog,  Nt,  _,  _, _=model.get_fem_parameters()
Length,  Width,  K,  Lame1,  Lame2,  k_mu,  cc,  alpha,  S, tc_DK, tc_TD=model.get_dependent_parameters()
p_ic,  p_bc,  p_load=model.get_icbc()
material=sas.SAS(Lame1, Lame2, K)  # stress and strain
dT=fe.Constant(dt) #  make time step mutable

fe.set_log_level(30)  # control info/warning/error messages
vtkfile_pu = fe.File('kim1_mono_pressure_displacement.pvd')
vtkfile_s_total = fe.File('kim1_mono_totalstress.pvd')    # xdmf for multiple fields


## MESH (simplex elements in 2D=triangles) 
mesh = fe.RectangleMesh(fe.Point(0, 0), fe.Point(Width, Length), Nx,Ny)   # , fe.CellType.Type.quadrilateral

Pp = fe.FiniteElement('P', fe.triangle, 1)
Pu = fe.VectorElement('P', fe.triangle, 2)
element = fe.MixedElement([Pp, Pu])
V = fe.FunctionSpace(mesh, element)

Vsigma = fe.TensorFunctionSpace(mesh, "P", 1)     

pu_ = fe.Function(V, name="pressure_displacement")    # function solved for and written to file
sigma_ = fe.Function(Vsigma, name="total_stress")    # function solved for and written to file


# INITIAL CONDITIONS: undeformed, at rest, zero pressure everywhere
pu_ic = fe.Expression(
        (
            p_ic,       		# p    
            "0.0","0.0", 	# (ux, uy)
        ), degree = 2)
pu_n = fe.interpolate(pu_ic, V)     # current value in time-stepping
pu_.assign(fe.interpolate(pu_ic, V))      # previous value in time-stepping


# BOUNDARY CONDITIONS
# DirichletBC, assign geometry via functions
tol = 1E-14
def top(x, on_boundary):    
    return on_boundary and fe.near(x[1], Length, tol) 
bc_top = fe.DirichletBC(V.sub(0), p_bc, top)  # drainage on top 

def bottom(x, on_boundary):    
    return on_boundary and fe.near(x[1], 0.0, tol)   
bc_bottom = fe.DirichletBC(V, (p_bc, 0,0), bottom)   # fixed bottom and drainage

def leftright(x, on_boundary):    
    #return True
    return on_boundary and (fe.near(x[0], 0.0, tol) or fe.near(x[0], Width, tol))
bc_leftright = fe.DirichletBC(V.sub(1).sub(0), ZeroScalar, leftright)   # rollers on side, i.e. fix only in x-direction

bc = [bc_top, bc_bottom, bc_leftright]

# Neumann BC, assign geometry via subdomains to have ds accessible in variational problem
boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
topSD = fe.AutoSubDomain(lambda x: fe.near(x[1], Length, tol))
topSD.mark(boundaries, 1)   # accessable via ds(1)
ds = fe.ds(subdomain_data=boundaries)
traction_topM = fe.Constant((0, -p_load))		


# FEM SYSTEM
# Define variational HM problem a(v,p)=L(v)
vp, vu = fe.TestFunctions(V)
pu = fe.TrialFunction(V)
p, u = fe.split(pu)
#vp, vu =fe.split(vpvu)
p_n, u_n = fe.split(pu_n)
sigma_.assign(fe.project(material.sigma(p_n, u_n), Vsigma))  # for vtkfile

# CONTINUE
Fdx = ( alpha*(fe.div(u)-fe.div(u_n))*vp
     + S*(p-p_n)*vp
     + dT*k_mu*fe.dot(fe.grad(p/2+p_n/2), fe.grad(vp))   # midpoint
     + fe.inner(material.sigma(p,u), material.epsilon(vu)) )*fe.dx 
Fds = - fe.dot(vu, traction_topM)*ds(1) 
F=Fdx+Fds

a, L=fe.lhs(F), fe.rhs(F)
# no non-zero Neumann BC for H, i.e. no prescribed in-outflows (only flow via DirichletBC possible)
# no sources (H)
# no body forces (M)


# TIME-STEPPING
t = 0.0

vtkfile_pu << (pu_, t)
vtkfile_s_total << (sigma_, t)
#y_ana=np.linspace(tol, Length-tol, 10*Ny+1)
y_mono=np.linspace(tol, Length-tol, Ny+1)
p_mono=np.zeros((Nt, Ny+1))
points=[ ((1.0/2.0)*Width, y_) for y_ in y_mono ]
t_grid=[]
x_obs=0.5*Width
y_obs=(10.5/15.0)*Length
p_obs=[]
for n in range(Nt):     # time steps
    t += dt
    print(n+1,".step   t=",t)     
    fe.solve(a == L, pu_, bc)
    pu_n.assign(pu_)
    p_n, u_n = fe.split(pu_n)
    sigma_.assign(fe.project(material.sigma(p_n, u_n), Vsigma))

    p_mono[n, :] = np.array([ p_n(point) for point in points])
#    plt.plot(y_mono, p_mono[n,:])   
    vtkfile_pu << (pu_, t)
    vtkfile_s_total << (sigma_, t)    
    dt*=dt_prog
    dT.assign(dt)
    t_grid.append(t/(tc_DK))
    if n==0:
        dp_ini=pu_n(x_obs, y_obs)[0]-p_ref
    p_obs.append( (pu_n(x_obs, y_obs)[0]-p_ref)/dp_ini )

plt.plot(t_grid, p_obs)
plt.ylim(0.2, 1.2)
plt.show()

np.savetxt("kim1_y_mono.txt", y_mono)
np.savetxt("kim1_p_mono.txt", p_mono)
