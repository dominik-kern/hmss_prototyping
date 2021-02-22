# TODO 
#    run_solver(dim, N, Pu, Pp, dt)
#if __name__ == '__main__':
#    elasticity 3D
#    quadliteral mesh
"""
2D Minimal model (of 1D Terzaghi), hydromechanical (HM), staggered scheme (stress-split)
spatial FE discretization for both H and M (Taylor Hood)
temporal Euler-backward discretization
incompressible fluid, incompressible solid (but linear elastic bulk)
"""
from __future__ import print_function
import fenics as fe
import numpy as np
import time
#import matplotlib.pyplot as plt
import minimal_model_parameters as mmp
import stress_and_strain as sas


# DECLARATION
model=mmp.MMP()
ZeroScalar = fe.Constant((0))	
ZeroVector = fe.Constant((0,0,0))
p_ref, _, _, k, mu = model.get_physical_parameters()
Nx, dt, dt_prog, Nt, _, _, _ = model.get_fem_parameters()
Length, Width, Height, K, Lame1, Lame2, k_mu, cc = model.get_dependent_parameters()
p_ic, p_bc, p_load = model.get_icbc() 
material=sas.SAS(Lame1, Lame2, K)  # stress and strain
dT=fe.Constant(dt) #  make time step mutable

fe.set_log_level(30)  # control info/warning/error messages
vtkfile_p = fe.File('mini_mono_pressure.pvd')
vtkfile_u = fe.File('mini_mono_displacement.pvd')
vtkfile_s_total = fe.File('mini_mono_totalstress.pvd')    # xdmf for multiple fields

Nrefine=4
timingN=np.zeros(Nrefine)
timingT=np.zeros(Nrefine)
for nx in range(Nrefine):
    
    ## MESH (simplex elements in 2D=triangles) 
    mesh = fe.UnitCubeMesh(Nx, Nx, Nx)
    #mesh = fe.RectangleMesh.create([fe.Point(0, 0), fe.Point(Width, Length)], [Nx,Ny], fe.CellType.Type.quadrilateral)
    
    Pp = fe.FiniteElement('P', fe.tetrahedron, 1)
    Pu = fe.VectorElement('P', fe.tetrahedron, 1)
    element = fe.MixedElement([Pp, Pu])
    V = fe.FunctionSpace(mesh, element)
    
    Vsigma = fe.TensorFunctionSpace(mesh, "P", 1)     
    
    pu_ = fe.Function(V, name="pressure_displacement")    # function solved for and written to file
    sigma_ = fe.Function(Vsigma, name="total_stress")    # function solved for and written to file
    
    
    # INITIAL CONDITIONS: undeformed, at rest, same pressure everywhere
    pu_ic = fe.Expression(
            (
                p_ic,       		# p    
                "0.0","0.0","0.0", 	# (ux, uy, uz)
            ), degree = 2)
    pu_n = fe.interpolate(pu_ic, V)     # current value in time-stepping
    pu_.assign(fe.interpolate(pu_ic, V))      # previous value in time-stepping
    
    
    # BOUNDARY CONDITIONS
    # DirichletBC, assign geometry via functions
    tol = 1E-14
    def top(x, on_boundary):    
        return on_boundary and fe.near(x[2], Height, tol) 
    bc_top = fe.DirichletBC(V.sub(0), p_bc, top)  # drainage on top 
    
    def bottom(x, on_boundary):    
        return on_boundary and fe.near(x[2], 0.0, tol)   
    bc_bottom = fe.DirichletBC(V.sub(1), ZeroVector, bottom)   # fixed bottom
    
    def sides(x, on_boundary):    
        #return True
        return on_boundary and ( fe.near(x[0], 0.0, tol) or fe.near(x[0], Length, tol) or fe.near(x[1], 0.0, tol) or fe.near(x[1], Width, tol) )
    bc_sides_x = fe.DirichletBC(V.sub(1).sub(0), ZeroScalar, sides)   # rollers on side, i.e. fix in x-direction
    bc_sides_y = fe.DirichletBC(V.sub(1).sub(1), ZeroScalar, sides)   # rollers on side, i.e. fix in x-direction
    
    bc = [bc_top, bc_bottom, bc_sides_x, bc_sides_y]
    
    # Neumann BC, assign geometry via subdomains to have ds accessible in variational problem
    boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    topSD = fe.AutoSubDomain(lambda x: fe.near(x[2], Height, tol))
    topSD.mark(boundaries, 1)   # accessable via ds(1)
    ds = fe.ds(subdomain_data=boundaries)
    traction_topM = fe.Constant((0, 0, -p_load))		
    
    
    # FEM SYSTEM
    # Define variational HM problem a(v,p)=L(v)
    vp, vu = fe.TestFunctions(V)
    pu = fe.TrialFunction(V)
    p, u = fe.split(pu)
    #vp, vu =fe.split(vpvu)
    p_n, u_n = fe.split(pu_n)
    sigma_.assign(fe.project(material.sigma(p_n, u_n), Vsigma))
    
    Fdx = ( (fe.div(u)-fe.div(u_n))*vp
         + dT*k_mu*fe.dot(fe.grad(p/2+p_n/2), fe.grad(vp))   # midpoint
         + fe.inner(material.sigma(p,u), material.epsilon(vu)) )*fe.dx 
    Fds = - fe.dot(vu, traction_topM)*ds(1) 
    F=Fdx+Fds
    
    a, L=fe.lhs(F), fe.rhs(F)
    # no non-zero Neumann BC for H, i.e. no prescribed in-outflows (only flow via DirichletBC possible)
    # no sources (H)
    # no body forces (M)
    
    
    # TIME-STEPPING
    tic = time.perf_counter()
    t = 0.0
    
    #p_, u_ = pu_.split()
    #vtkfile_p << (p_, t)
    #vtkfile_u << (u_, t) 
    #vtkfile_s_total << (sigma_, t)
    
    z_mono=np.linspace(tol, Height-tol, Nx+1)
    p_mono=np.zeros((Nt, Nx+1))
    points=[ (0.5*Width, 0.5*Length, z_) for z_ in z_mono ]
    
    for n in range(Nt):     # time steps
        t += dt
        print(n+1,".step   t=",t)     
        fe.solve(a == L, pu_, bc, solver_parameters={'linear_solver' : 'mumps'})   #  solver_parameters={'linear_solver': 'gmres','preconditioner': 'ilu'}
        pu_n.assign(pu_)
        p_n, u_n = fe.split(pu_n)
        sigma_.assign(fe.project(material.sigma(p_n, u_n), Vsigma))
    
        p_mono[n, :] = np.array([ p_n(point) for point in points])
        
    #    p_, u_ = pu_.split()
    #    vtkfile_p << (p_, t)
    #    vtkfile_u << (u_, t)
    #    vtkfile_s_total << (sigma_, t)    
        dt*=dt_prog
        dT.assign(dt)
    
    toc = time.perf_counter()
    run_time=toc-tic
    timingT[nx]=run_time
    timingN[nx]=Nx
    Nx*=2

    print(f"Monolithic scheme run time: {run_time:0.4f} seconds")


np.savetxt("mini_mono_N.txt", timingN)
np.savetxt("mini_mono_T.txt", timingT)
   #plt.show()
np.savetxt("mini_z_mono.txt", z_mono)
np.savetxt("mini_p_mono.txt", p_mono)
