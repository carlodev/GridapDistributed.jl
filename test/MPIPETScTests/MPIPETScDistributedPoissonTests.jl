module MPIPETScDistributedPoissonTests

using Test
using Gridap
using Gridap.FESpaces
using GridapDistributed
using GridapDistributedPETScWrappers
using SparseArrays

using ArgParse

function PCFactorSetMatSolverType(arg1::GridapDistributedPETScWrappers.C.PC{Float64},
                                  arg2::Union{String,Cstring,Symbol,Array{UInt8},Ptr{UInt8}})
   err = ccall((:PCFactorSetMatSolverType,GridapDistributedPETScWrappers.C.petscRealDouble),
                 GridapDistributedPETScWrappers.C.PetscErrorCode,
                 (GridapDistributedPETScWrappers.C.PC{Float64},Cstring),
                 arg1,arg2)
   return err
end
function PCFactorSetUpMatSolverType(arg1::GridapDistributedPETScWrappers.C.PC{Float64})
   err = ccall((:PCFactorSetUpMatSolverType,GridapDistributedPETScWrappers.C.petscRealDouble),
                 GridapDistributedPETScWrappers.C.PetscErrorCode,
                 (GridapDistributedPETScWrappers.C.PC{Float64},),
               arg1)
   return err
end
function MatMumpsSetIcntl(arg1::GridapDistributedPETScWrappers.C.Mat{Float64},
                          arg2::GridapDistributedPETScWrappers.C.PetscInt,
                          arg3::GridapDistributedPETScWrappers.C.PetscInt)
    err = ccall((:MatMumpsSetIcntl,GridapDistributedPETScWrappers.C.petscRealDouble),
                 GridapDistributedPETScWrappers.C.PetscErrorCode,
                 (GridapDistributedPETScWrappers.C.Mat{Float64},
                  GridapDistributedPETScWrappers.C.PetscInt,
                  GridapDistributedPETScWrappers.C.PetscInt),
                 arg1,arg2,arg3)
    return err
end
function MatMumpsSetCntl(arg1::GridapDistributedPETScWrappers.C.Mat{Float64},
                         arg2::GridapDistributedPETScWrappers.C.PetscInt,
                         arg3::Cdouble)
    err = ccall((:MatMumpsSetIcntl,GridapDistributedPETScWrappers.C.petscRealDouble),
                 GridapDistributedPETScWrappers.C.PetscErrorCode,
                 (GridapDistributedPETScWrappers.C.Mat{Float64},
                  GridapDistributedPETScWrappers.C.PetscInt,
                  Cdouble),
                 arg1,arg2,arg3)
    return err
end

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--subdomains", "-s"
        help = "Tuple with the # of subdomains per Cartesian direction"
        arg_type = Int64
        default=[1,1]
        nargs='+'
        "--partition", "-p"
        help = "Tuple with the # of cells per Cartesian direction"
        arg_type = Int64
        default=[4,4]
        nargs='+'
    end
    return parse_args(s)
end

# High level API seems not to provide access to fine-tuning
# of MUMPS parameters (i.e., those which are set by MatMumpsSetIcntl/Cntl)
function solve_linear_system_petsc_mumps_high_level_API(op)
  ls = PETScLinearSolver(
    Float64;
    ksp_type = "preonly",
    ksp_error_if_not_converged = true,     # When a MUMPS factorization fails inside a KSP solve
                                           # the program will be stopped and the information printed
                                           # in an error message
    pc_type = "cholesky",                  # "cholesky" or "lu"
    pc_factor_mat_solver_type = "mumps"
  )
  fels = LinearFESolver(ls)
  uh = solve(fels, op)
end

function solve_linear_system_petsc_mumps_low_level_API(op)
  A=op.op.matrix
  b=op.op.vector
  ksp=Ref{GridapDistributedPETScWrappers.C.KSP{Float64}}()
  pc=Ref{GridapDistributedPETScWrappers.C.PC{Float64}}()
  mumpsmat=Ref{GridapDistributedPETScWrappers.C.Mat{Float64}}()

  GridapDistributedPETScWrappers.C.KSPCreate(comm(A),ksp)
  GridapDistributedPETScWrappers.C.KSPSetOperators(ksp[],A.p,A.p)
  GridapDistributedPETScWrappers.C.KSPSetType(ksp[],GridapDistributedPETScWrappers.C.KSPPREONLY)
  GridapDistributedPETScWrappers.C.KSPGetPC(ksp[],pc)

  # If system is SPD use the following two calls
  GridapDistributedPETScWrappers.C.PCSetType(pc[],GridapDistributedPETScWrappers.C.PCCHOLESKY)
  GridapDistributedPETScWrappers.C.MatSetOption(A.p,
                                                GridapDistributedPETScWrappers.C.MAT_SPD,GridapDistributedPETScWrappers.C.PETSC_TRUE);
  # Else ... use only the following one
  # GridapDistributedPETScWrappers.C.PCSetType(pc,GridapDistributedPETScWrappers.C.PCLU)

  PCFactorSetMatSolverType(pc[],GridapDistributedPETScWrappers.C.MATSOLVERMUMPS)
  PCFactorSetUpMatSolverType(pc[])
  GridapDistributedPETScWrappers.C.PCFactorGetMatrix(pc[],mumpsmat)
  MatMumpsSetIcntl(mumpsmat[],4 ,4)     # level of printing (0 to 4)
  MatMumpsSetIcntl(mumpsmat[],28,2)     # use 1 for sequential analysis and ictnl(7) ordering,
                                      # or 2 for parallel analysis and ictnl(29) ordering
  MatMumpsSetIcntl(mumpsmat[],29,2)     # parallel ordering 1 = ptscotch, 2 = parmetis
  MatMumpsSetCntl(mumpsmat[] ,3,1.0e-6)  # threshhold for row pivot detection
  GridapDistributedPETScWrappers.C.KSPSetUp(ksp[])

  x=copy(b)
  GridapDistributedPETScWrappers.C.KSPSolve(ksp[], b.p, x.p)
  uh = FEFunction(op.trial,x)
end

function run(comm,
             subdomains=(2, 2),
             cells=(4, 4),
             domain = (0, 1, 0, 1))

  # Manufactured solution
  u(x) = x[1] + x[2]
  f(x) = -Δ(u)(x)

  # Discretization
  domain = (0, 1, 0, 1)
  cells = (4, 4)
  model = CartesianDiscreteModel(comm, subdomains, domain, cells)

  # FE Spaces
  order=1
  reffe = ReferenceFE(lagrangian,Float64,order)
  V = FESpace(model=model,
              reffe=reffe,
              conformity=:H1,
              dirichlet_tags="boundary")
  U = TrialFESpace(V, u)

  trian=Triangulation(model)
  dΩ=Measure(trian,2*(order+1))

  function a(u,v)
    ∫(∇(v)⋅∇(u))dΩ
  end
  function l(v)
    ∫(v*f)dΩ
  end

  # FE solution
  op = AffineFEOperator(a,l,U,V)
  # uh = solve_linear_system_petsc_mumps_high_level_API(op)
  uh = solve_linear_system_petsc_mumps_low_level_API(op)

  # Error norms and print solution
  trian=Triangulation(OwnedCells,model)
  dΩ=Measure(trian,2*order)
  e = u-uh
  e_l2 = sum(∫(e*e)dΩ)
  tol = 1.0e-9
  @test e_l2 < tol
  if (i_am_master(comm)) println("$(e_l2) < $(tol)\n") end
end


parsed_args = parse_commandline()
subdomains = Tuple(parsed_args["subdomains"])
partition = Tuple(parsed_args["partition"])

MPIPETScCommunicator() do comm
  run(comm, subdomains,partition)
end

end # module
