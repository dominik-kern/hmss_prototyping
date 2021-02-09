"""
Convergence of the HM formulation at a minimum example
running and comparing several cases with numerical reference example

TODO redesign in an all-in-one  class (material, model, discretization) 
"""

from __future__ import print_function
import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import minimal_model_parameters as mmp
import stress_and_strain as sas


# DECLARATION
model=mmp.MMP()
ZeroScalar = fe.Constant((0))	
ZeroVector = fe.Constant((0,0))
p_ref, _, _, k, mu = model.get_physical_parameters()
Nx, Ny, dt, dt_prog, Nt, _, _, _ = model.get_fem_parameters()
Length, Width, K, Lame1, Lame2, k_mu, cc = model.get_dependent_parameters()
p_ic, p_bc, p_load = model.get_icbc() 
material=sas.SAS(Lame1, Lame2, K)  # stress and strain
dT=fe.Constant(dt) #  make time step mutable

fe.set_log_level(30)  # control info/warning/error messages
vtkfile_pu = fe.File('results/mini_mono_pressure_displacement.pvd')
#vtkfile_s_total = fe.File('results/mini_mono_totalstress.pvd')    # xdmf for multiple fields


## MESH (simplex elements in 2D=triangles) 
mesh = fe.UnitSquareMesh(Nx, Ny)
#mesh = fe.RectangleMesh.create([fe.Point(0, 0), fe.Point(Width, Length)], [Nx,Ny], fe.CellType.Type.quadrilateral)

Pp = fe.FiniteElement('P', fe.triangle, 1)
Pu = fe.VectorElement('P', fe.triangle, 2)
element = fe.MixedElement([Pp, Pu])
V = fe.FunctionSpace(mesh, element)

Vsigma = fe.TensorFunctionSpace(mesh, "P", 1)     

pu_ = fe.Function(V, name="pressure_displacement")    # function solved for and written to file
#sigma_ = fe.Function(Vsigma, name="total_stress")    # function solved for and written to file


# INITIAL CONDITIONS: undeformed, at rest, same pressure everywhere
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
bc_bottom = fe.DirichletBC(V.sub(1), ZeroVector, bottom)   # fixed bottom

def side(x, on_boundary):    
    #return True
    return on_boundary and (fe.near(x[0], 0.0, tol) or fe.near(x[0], Width, tol))
bc_side = fe.DirichletBC(V.sub(1).sub(0), ZeroScalar, side)   # rollers on side, i.e. fix only in x-direction

bc = [bc_top, bc_bottom, bc_side]

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

Fdx = ( (fe.div(u)-fe.div(u_n))*vp
     + dT*k_mu*fe.dot(fe.grad(p/2+p_n/2), fe.grad(vp))   # midpoint
     + fe.inner(material.sigma(p,u), material.epsilon(vu)) )*fe.dx 
Fds = - fe.dot(vu, traction_topM)*ds(1) 
F=Fdx+Fds

a, L=fe.lhs(F), fe.rhs(F)
# no non-zero Neumann BC for H, i.e. no prescribed in-outflows (only flow via DirichletBC possible)
# no sources (H)
# no body forces (M)


# TIME-STEPPING TODO first reference solution then cases
t = 0.0

vtkfile_pu << (pu_, t)
#vtkfile_s_total << (sigma_, t)
#y_ana=np.linspace(tol, Length-tol, 10*Ny+1)
#y_mono=np.linspace(tol, Length-tol, Ny+1)
#p_mono=np.zeros((Nt, Ny+1))
#points=[ ((1.0/2.0)*Width, y_) for y_ in y_mono ]

p_history = []
u_history = []
for n in range(Nt):     # time steps
    t += dt
    #print("reference solution:", n+1,".step   t=",t)     
    fe.solve(a == L, pu_, bc)
    pu_n.assign(pu_)
    p_n, u_n = pu_n.split() #fe.split(pu_n)
    #sigma_.assign(fe.project(material.sigma(p_n, u_n), Vsigma))
    print(p_n(0.5, 0.5))
    #p_mono[n, :] = np.array([ p_n(point) for point in points])
   
    vtkfile_pu << (pu_, t)
    #vtkfile_s_total << (sigma_, t)    
    dt*=dt_prog
    dT.assign(dt)
    p_history.append(p_n.compute_vertex_values())
    u_history.append(u_n.compute_vertex_values())

##############################################################################
#run cases and setup convergence plot
cases=[]

deg_p=1
deg_u=2
Nx=2
Ny=2
cases.append((deg_p, deg_u, Nx, Ny, dt, Nt))

Nx=4
Ny=4
cases.append((deg_p, deg_u, Nx, Ny, dt, Nt))

Nx=8
Ny=8
cases.append((deg_p, deg_u, Nx, Ny, dt, Nt))

Nx=16
Ny=16
cases.append((deg_p, deg_u, Nx, Ny, dt, Nt))


print("case")

for case in cases:
    (deg_p, deg_u, Nx, Ny, dt, Nt)=case
    
    meshc = fe.UnitSquareMesh(Nx, Ny)    
    Ppc = fe.FiniteElement('P', fe.triangle, deg_p)
    Puc = fe.VectorElement('P', fe.triangle, deg_u)
    elementc = fe.MixedElement([Ppc, Puc])
    Vc = fe.FunctionSpace(meshc, elementc)
     
    puc_ = fe.Function(Vc, name="pressure_displacement")    # case
    
    
    # INITIAL CONDITIONS: undeformed, at rest, same pressure everywhere
    puc_n = fe.interpolate(pu_ic, Vc)     # current value in time-stepping
    puc_.assign(fe.interpolate(pu_ic, Vc))      # previous value in time-stepping
    
    
    # BOUNDARY CONDITIONS
    # DirichletBC, assign geometry via functions
    bcc_top = fe.DirichletBC(Vc.sub(0), p_bc, top)  # drainage on top  
    bcc_bottom = fe.DirichletBC(Vc.sub(1), ZeroVector, bottom)   # fixed bottom
    bcc_side = fe.DirichletBC(Vc.sub(1).sub(0), ZeroScalar, side)   # rollers on side, i.e. fix only in x-direction  
    bcc = [bcc_top, bcc_bottom, bcc_side]
    
    # Neumann BC, assign geometry via subdomains to have ds accessible in variational problem
    boundariesc = fe.MeshFunction("size_t", meshc, meshc.topology().dim() - 1)
    boundariesc.set_all(0)
    topSDc = fe.AutoSubDomain(lambda x: fe.near(x[1], Length, tol))
    topSDc.mark(boundariesc, 1)   # accessable via ds(1)
    dsc = fe.ds(subdomain_data=boundariesc)
    
    
    
    # FEM SYSTEM
    # Define variational HM problem a(v,p)=L(v)
    vpc, vuc = fe.TestFunctions(Vc)
    puc = fe.TrialFunction(Vc)
    pc, uc = fe.split(puc)
    #vp, vu =fe.split(vpvu)
    pc_n, uc_n = fe.split(puc_n)
    
    Fcdx = ( (fe.div(uc) - fe.div(uc_n))*vpc 
         + dT*k_mu*fe.dot(fe.grad(pc/2+pc_n/2), fe.grad(vpc))   # midpoint
         + fe.inner(material.sigma(pc,uc), material.epsilon(vuc)) )*fe.dx 
    Fcds = - fe.dot(vuc, traction_topM)*dsc(1) 
    Fc=Fcdx+Fcds
    
    ac, Lc=fe.lhs(Fc), fe.rhs(Fc)
    # no non-zero Neumann BC for H, i.e. no prescribed in-outflows (only flow via DirichletBC possible)
    # no sources (H)
    # no body forces (M)
    
    
    # TIME-STEPPING case solution
    t = 0.0
    pc_history = []
    uc_history = []
    for n in range(Nt):     # time steps
        t += dt
        #print("case solution:", n+1,".step   t=",t)     
        fe.solve(ac == Lc, puc_, bcc)
        puc_n.assign(puc_)
        pc_n, uc_n = puc_n.split()   # fe.split(puc_n)
        print(pc_n(0.5, 0.5))
        dt *= dt_prog
        dT.assign(dt)
        pucV = fe.project(puc_n, V)   # for error computation on same mesh
        pcV, ucV = pucV.split()
        pc_history.append(pcV.compute_vertex_values())
        uc_history.append(ucV.compute_vertex_values())
    
    # postprocessing
    for n in range(Nt):
        color_code=[0.9*(1-(n+1)/Nt)]*3
        h=2/(Nx+Ny)
        
        PC = pc_history[n]       
        PREF = p_history[n]
        p_error = np.max(np.abs(PC-PREF))
        plt.figure(0)
        plt.plot(np.log(h), np.log(p_error), 'o', color=color_code) 
        plt.xlabel("log(h)") 
        plt.ylabel("log(p_error)")
        
        UC = uc_history[n]       
        UREF = u_history[n]
        u_error = np.max(np.abs(UC-UREF))
        plt.figure(1)
        plt.plot(np.log(h), np.log(u_error), 'o', color=color_code) 
        plt.xlabel("log(h)") 
        plt.ylabel("log(u_error)")
        



