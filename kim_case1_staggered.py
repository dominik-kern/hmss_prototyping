# TODO 
#    compare with undrained
#    nonzero initial guess
#    quadliteral mesh
"""
2D model of Kim case 1 (1D Terzaghi), hydromechanical (HM), staggered scheme (stress-split)
spatial FE discretization (plane strain) for both H and M (Taylor Hood)
temporal midpoint discretization
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
Nx,  Ny,  dt,  dt_prog,  Nt,  Nci_max,  RelTol_ci, betaFS=model.get_fem_parameters()
Length,  Width,  K,  Lame1,  Lame2,  k_mu,  cc,  alpha,  S, tc_DK, tc_TD=model.get_dependent_parameters()
p_ic,  p_bc,  p_load=model.get_icbc()
material=sas.SAS(Lame1, Lame2, K)  # stress and strain
dT=fe.Constant(dt) #  make time step mutable

fe.set_log_level(30)  # control info/warning/error messages

# vtu files for paraview (alternative: xdmf for multiple fields in one file)
vtkfile_p = fe.File('kim1_staggered_fluidpressure.pvd')
vtkfile_u = fe.File('kim1_staggered_displacement.pvd')
vtkfile_s_total = fe.File('kim1_staggered_totalstress.pvd')

def max_norm_delta(f1, f2):
    vertex_values_f1 = f1.compute_vertex_values(mesh)
    vertex_values_f2 = f2.compute_vertex_values(mesh)
    return np.max( np.abs(vertex_values_f1 - vertex_values_f2) )


# MESH (simplex elements in 2D=triangles)
mesh = fe.RectangleMesh(fe.Point(0, 0), fe.Point(Width, Length), Nx,Ny)   # , fe.CellType.Type.quadrilateral

VH = fe.FunctionSpace(mesh, 'P', 1)
VM = fe.VectorFunctionSpace(mesh, 'P', 2)   
Vsigma = fe.TensorFunctionSpace(mesh, "P", 1)     

p = fe.Function(VH, name="fluidpressure")    # function solved for and written to file
u = fe.Function(VM, name="displacement") # function to solve for and written to file
s_eff = fe.Function(Vsigma, name="effective_stress")
s_total = fe.Function(Vsigma, name="total_stress")
sv = fe.Function(VH, name="hydrostatic_totalstress")    # function solved for and written to file
ev = fe.Function(VH, name="volumetric strain")    # function solved for and written to file

# INITIAL CONDITIONS undeformed, at rest, constant pressure everywhere
p_0 = fe.Constant(p_ic)
p_n = fe.interpolate(p_0, VH)   # fluid pressure at t=t_n
p_ = fe.interpolate(p_0, VH)   # fluid pressure at t=t_(n+1) at coupling iterations
u_0 =fe.Expression( ('0','0'), degree=2)    # undeformed
u_n = fe.interpolate(u_0, VM)   # displacement at t=t_n
sv_0 = material.sv(p_n, u_n)    # derive from initial state
sv_n = fe.project(sv_0, VH)  # hydrostatic total stress at t_n 
sv_ = fe.project(sv_0, VH) # same as sv_n at t_(n+1) at coupling iterations (at first coupling iteration sv_=sv_n)
ev_0 = fe.div(u_n)
ev_n = fe.project(ev_0, VH)
ev_ = fe.project(ev_0, VH) 

# BOUNDARY CONDITIONS
# DirichletBC, assign geometry via functions
tol = 1E-14
def top(x, on_boundary):    
    return on_boundary and fe.near(x[1], Length, tol) 
bcHdraintop = fe.DirichletBC(VH, p_bc, top)  # drainage on top 

def bottom(x, on_boundary):    
    return on_boundary and fe.near(x[1], 0.0, tol)   
bcMfixed = fe.DirichletBC(VM, ZeroVector, bottom)   # fixed bottom
bcHdrainbottom = fe.DirichletBC(VH, p_bc, bottom)  # drainage on bottom

def leftright(x, on_boundary):    
    #return True
    return on_boundary and (fe.near(x[0], 0.0, tol) or fe.near(x[0], Width, tol))
bcMroller = fe.DirichletBC(VM.sub(0), ZeroScalar, leftright)   # rollers on side, i.e. fix only in x-direction
bcM = [bcMfixed, bcMroller]
bcH = [bcHdraintop, bcHdrainbottom]

# Neumann BC, assign geometry via subdomains to have ds accessible in variational problem
boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
topSD = fe.AutoSubDomain(lambda x: fe.near(x[1], Length, tol))
topSD.mark(boundaries, 1)   # accessable via ds(1)
ds = fe.ds(subdomain_data=boundaries)
traction_topM = fe.Constant((0, -p_load))		


# FEM SYSTEM
# Define variational H problem a(v,p)=L(v)


pH = fe.TrialFunction(VH)
vH = fe.TestFunction(VH)

aH = ( vH*( betaFS + S )*pH + dT*k_mu*fe.dot(fe.grad(vH), fe.grad(pH)) )*fe.dx

LH = vH*( betaFS*p_ + S*p_n - alpha*(ev_-ev_n) )*fe.dx 
# no non-zero Neumann BC, i.e. no prescribed in-outflows (only flow via DirichletBC possible)
# no sources

# Define variational M problem a(v,p)=L(v)
uM = fe.TrialFunction(VM) 
vM = fe.TestFunction(VM)

aM = fe.inner(material.sigma_eff(uM), material.epsilon(vM))*fe.dx

LM = p_*fe.div(vM)*fe.dx + fe.dot(vM, traction_topM)*ds(1)
# no body forces


# TIME-STEPPING
t = 0.0
p.assign(p_n)
u.assign(u_n)
#sv.assign(sv_n)   # hydrostatic total stress
#s_eff.assign(fe.project(material.sigma_eff(u), Vsigma))    # effective stress
s_total.assign(fe.project(material.sigma(p, u), Vsigma))
vtkfile_p << (p, t)
vtkfile_u << (u, t)
#vtkfile_sv << (sv, t)
#vtkfile_s_eff << (s_eff, t)
vtkfile_s_total << (s_total, t)

y_staggered=np.linspace(tol, Length-tol, Ny+1)
p_staggered=np.zeros((Nt, Ny+1))
points=[ ((1.0/2.0)*Width, y_) for y_ in y_staggered ]
conv_monitor=np.zeros((Nt, Nci_max))
for n in range(Nt):     # time steps
    t += dt
    for nn in range(Nci_max):   # couplint iterations
        
        fe.solve(aH == LH, p, bcH)
        delta_p=max_norm_delta(p_, p)
        p_.assign(p)    # shift forward coupling iteration
        
        fe.solve(aM == LM, u, bcM)
        ev = fe.project( fe.div(u), VH )
        #s_eff.assign(fe.project(material.sigma_eff(u), Vsigma))    # effective stress
        sv.assign( fe.project( material.sv(p, u), VH) )    # hydrostatic total stress
        delta_sv=max_norm_delta(sv_, sv)
        sv_.assign(sv)   # shift forward coupling iteration
        ev_.assign(ev) 
        
        #print(nn+1,'. ', delta_p, delta_sv)
        conv_criterium=np.abs(delta_p/p_ref) + np.abs(delta_sv/p_ref)  # take both to exclude random hit of one variable at initial state
        conv_monitor[n,nn]=conv_criterium
        if conv_criterium < RelTol_ci:
            break

    sv_.assign(sv + dt_prog*(sv-sv_n))               # linear predictor
    sv_n.assign(sv)     # shift forward time step
    ev_n.assign(ev)
    p_n.assign(p)       # shift forward time step
    s_total.assign(fe.project(material.sigma(p, u), Vsigma))
    dt*=dt_prog
    dT.assign(dt)
    if nn+1==Nci_max:
        print(str(n+1),". time step   t=",str(t),"   Solution not converged to ",str(RelTol_ci)," in ",str(nn+1)," coupling iterations!")
    else:
        print(str(n+1),". time step   t=",str(t),"   ",str(nn+1)," coupling iterations")
    
    p_staggered[n,:] = np.array([ p(point) for point in points])

    color_code=[0.9*(1-(n+1)/Nt)]*3
    plt.plot([ np.log10(R) for R in conv_monitor[n,:] if R>0 ], color=color_code)
    
    #print()
    vtkfile_p << (p,t)
    vtkfile_u << (u,t)
    #vtkfile_sv << (sv,t)
    #vtkfile_s_eff << (s_eff,t)
    vtkfile_s_total << (s_total, t)

plt.show()
np.savetxt("kim1_y_staggered.txt", y_staggered)
np.savetxt("kim1_p_staggered.txt", p_staggered)