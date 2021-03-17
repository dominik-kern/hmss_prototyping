# TODO 
#    nonzero initial guess (for iterative schemes)
#    quadliteral/hexaeder mesh
"""
2D Minimal model (of 1D Terzaghi), hydromechanical (HM), staggered scheme (stress-split)
spatial FE discretization (plane strain) for both H and M (Taylor Hood)
temporal Euler-backward discretization
incompressible fluid, incompressible solid (but linear elastic bulk)
"""
from __future__ import print_function
import fenics as fe
import numpy as np
import time
import matplotlib.pyplot as plt
import minimal_model_parameters as mmp
import stress_and_strain as sas
model=mmp.MMP()


# DECLARATION
model=mmp.MMP()
ZeroScalar = fe.Constant((0))	
ZeroVector = fe.Constant((0,0,0))
p_ref, _, _, k, mu = model.get_physical_parameters()
Nx, dt, dt_prog, Nt, Nci_max, RelTol_ci, betaFS = model.get_fem_parameters()
Length, Width, Height, K, Lame1, Lame2, k_mu, cc = model.get_dependent_parameters()
p_ic, p_bc, p_load = model.get_icbc() 
material=sas.SAS(Lame1, Lame2, K)  # stress and strain
dT=fe.Constant(dt) #  make time step mutable

fe.set_log_level(30)  # control info/warning/error messages

# vtu files for paraview (alternative: xdmf for multiple fields in one file)
vtkfile_p = fe.File('mini_staggered_fluidpressure.pvd')
vtkfile_u = fe.File('mini_staggered_displacement.pvd')
#vtkfile_s_eff = fe.File('mini_staggered_effective_stress.pvd')
#vtkfile_sv = fe.File('mini_staggered_hydrostatic_totalstress.pvd')
vtkfile_s_total = fe.File('mini_staggered_totalstress.pvd')

def max_norm_delta(f1, f2):   # TODO RelTol in dimensional equations
    vertex_values_f1 = f1.compute_vertex_values(mesh)
    vertex_values_f2 = f2.compute_vertex_values(mesh)
    return np.max( np.abs(vertex_values_f1 - vertex_values_f2) )

Nrefine=1
timingN=np.zeros(Nrefine)
timingT=np.zeros(Nrefine)
for nx in range(Nrefine):
    # MESH (simplex elements in 2D=triangles)
    mesh = fe.UnitCubeMesh(Nx, Nx, Nx)
    #mesh = fe.RectangleMesh.create([fe.Point(0, 0), fe.Point(Width, Length)], [Nx,Ny], fe.CellType.Type.quadrilateral)
    
    VH = fe.FunctionSpace(mesh, 'P', 1)
    VM = fe.VectorFunctionSpace(mesh, 'P', 1)   
    Vsigma = fe.TensorFunctionSpace(mesh, "P", 1)     
    
    p = fe.Function(VH, name="fluidpressure")    # function solved for and written to file
    pp = fe.Function(VH, name="predicted_fluidpressure")    # function solved for and written to file
    u = fe.Function(VM, name="displacement") # function to solve for and written to file
    s_eff = fe.Function(Vsigma, name="effective_stress")
    s_total = fe.Function(Vsigma, name="total_stress")
    sv = fe.Function(VH, name="hydrostatic_totalstress")    # function solved for and written to file
    ev = fe.Function(VH, name="volumetric strain")    # function solved for and written to file
    pp = fe.Function(VH, name="predicted pressure")    # function solved for and written to file
    
    
    # INITIAL CONDITIONS undeformed, at rest, constant pressure everywhere
    p_0 = fe.Constant(p_ic)
    p_n = fe.interpolate(p_0, VH)   # fluid pressure at t=t_n
    p_ = fe.interpolate(p_0, VH)   # fluid pressure at t=t_(n+1) at coupling iterations
    u_0 =fe.Expression( ('0','0','0'), degree=2)    # undeformed
    u_n = fe.interpolate(u_0, VM)   # displacement at t=t_n
    
    #sv_0 = material.sv(p_n, u_n)    # derive from initial state
    #sv_n = fe.project(sv_0, VH)  # hydrostatic total stress at t_n 
    #sv_ = fe.project(sv_0, VH) # same as sv_n at t_(n+1) at coupling iterations (at first coupling iteration sv_=sv_n)
    ev_0 = fe.div(u_n)
    ev_n = fe.project(ev_0, VH)
    ev_ = fe.project(ev_0, VH)
    
    # BOUNDARY CONDITIONS
    # DirichletBC, assign geometry via functions
    tol = 1E-14
    def top(x, on_boundary):    
        return on_boundary and fe.near(x[2], Height, tol) 
    bcH = fe.DirichletBC(VH, p_bc, top)  # drainage on top 
    
    def bottom(x, on_boundary):    
        return on_boundary and fe.near(x[2], 0.0, tol)   
    bcMfixed = fe.DirichletBC(VM, ZeroVector, bottom)   # fixed bottom
    
    def sides(x, on_boundary):    
        #return True
        return on_boundary and ( fe.near(x[0], 0.0, tol) or fe.near(x[0], Length, tol) or fe.near(x[1], 0.0, tol) or fe.near(x[1], Width, tol) )
    bcMroller_x = fe.DirichletBC(VM.sub(0), ZeroScalar, sides)   # rollers on side, i.e. fix only in x-direction
    bcMroller_y = fe.DirichletBC(VM.sub(1), ZeroScalar, sides)   # rollers on side, i.e. fix only in y-direction
    bcM = [bcMfixed, bcMroller_x, bcMroller_y]
    
    # Neumann BC, assign geometry via subdomains to have ds accessible in variational problem
    boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    topSD = fe.AutoSubDomain(lambda x: fe.near(x[2], Height, tol))
    topSD.mark(boundaries, 1)   # accessable via ds(1)
    ds = fe.ds(subdomain_data=boundaries)
    traction_topM = fe.Constant((0, 0, -p_load))		
    
    
    # FEM SYSTEM
    # Define variational H problem a(v,p)=L(v)
    pH = fe.TrialFunction(VH)
    vH = fe.TestFunction(VH)
    
    aH = ( vH*pH*betaFS + dT*k_mu*fe.dot(fe.grad(vH), fe.grad(pH/2)) )*fe.dx
    
    LH = ( vH*p_*betaFS - vH*(ev_-ev_n) - dT*k_mu*fe.dot(fe.grad(vH), fe.grad(p_n/2)) )*fe.dx # 
    # no non-zero Neumann BC, i.e. no prescribed in-outflows (only flow via DirichletBC possible)
    # no sources
    
    # Define variational M problem a(v,p)=L(v)
    uM = fe.TrialFunction(VM) 
    vM = fe.TestFunction(VM)
    
    aM = fe.inner(material.sigma_eff(uM), material.epsilon(vM))*fe.dx
    
    LM = p_*fe.div(vM)*fe.dx + fe.dot(vM, traction_topM)*ds(1)
    # no body forces
    
    # predictor
    pP = fe.TrialFunction(VH)
    vP = fe.TestFunction(VH)
    
    aP = vP*pP*fe.dx
    
    LP = ( vP*p_ - dT*K*k_mu*fe.dot(fe.grad(vP), fe.grad(p_)) )*fe.dx 
    
    # TIME-STEPPING
    tic = time.perf_counter()
    t = 0.0
    p.assign(p_n)
    u.assign(u_n)
    #s_total.assign(fe.project(material.sigma(p, u), Vsigma))
    #vtkfile_p << (p, t)
    #vtkfile_u << (u, t)
    #vtkfile_s_total << (s_total, t)
    
    z_staggered=np.linspace(tol, Height-tol, Nx+1)
    p_staggered=np.zeros((Nt, Nx+1))
    points=[ ( 0.5*Length, 0.5*Width, z_) for z_ in z_staggered ]
    conv_monitor=np.zeros((Nt, Nci_max))
    for n in range(Nt):     # time steps
        t += dt
        for nn in range(Nci_max):   # couplint iterations
            
            fe.solve(aH == LH, p, bcH, solver_parameters={'linear_solver' : 'mumps'})   
            delta_p=max_norm_delta(p_, p)
            p_.assign(p)    # shift forward coupling iteration
            
            fe.solve(aM == LM, u, bcM, solver_parameters={'linear_solver' : 'mumps'})
            ev = fe.project( fe.div(u), VH )
            delta_ev=max_norm_delta(ev_, ev)
            ev_.assign(ev)   # shift forward coupling iteration
            
            #print(nn+1,'. ', delta_p, delta_sv)
            conv_criterium=np.abs(delta_p/p_ref) + np.abs(delta_ev/p_ref)  # take both to exclude random hit of one variable at initial state
            conv_monitor[n,nn]=conv_criterium
            if conv_criterium < RelTol_ci:
                break
    
        # postprocess
        #s_total.assign( fe.project(material.sigma(p, u), Vsigma))
        p_staggered[n,:] = np.array([ p(point) for point in points])
        #color_code=[0.9*(1-(n+1)/Nt)]*3
        #plt.plot([ np.log10(R) for R in conv_monitor[n,:] if R>0 ], color=color_code)
        
        # predictor
        #p_.assign( fe.project(p + 0.0*dt_prog*(p-p_n), VH))    # linear extrapolation for p
        #fe.solve(aP == LP, pp, bcH) # explicit..
        #p_.assign(pp)               # ..predictor
        
        # shift forward time step
        p_n.assign(p)
        ev_n.assign(ev)     
    
        dt*=dt_prog # used in predictor system
        dT.assign(dt)
        if nn+1==Nci_max:
            print(str(n+1),". time step   t=",str(t),"   Solution not converged to ",str(RelTol_ci)," in ",str(nn+1)," coupling iterations!")
        else:
            print(str(n+1),". time step   t=",str(t),"   ",str(nn+1)," coupling iterations")
    
        
        #print()
        #vtkfile_p << (p,t)
        #vtkfile_u << (u,t)
        #vtkfile_s_total << (s_total, t)
    
    toc = time.perf_counter()
    run_time=toc-tic
    timingT[nx]=run_time
    timingN[nx]=Nx
    Nx*=2

    print(f"Staggered scheme run time: {run_time:0.4f} seconds")


np.savetxt("mini_staggered_N.txt", timingN)
np.savetxt("mini_staggered_T.txt", timingT)    
#plt.show()
np.savetxt("mini_z_staggered.txt", z_staggered)
np.savetxt("mini_p_staggered.txt", p_staggered)
