
using Gridap
using Gridap.FESpaces
using Gridap.Algebra
using GridapDistributed
using GridapDistributed: DistributedFESpace
using GridapDistributed: DistributedAssemblyStrategy
using SparseArrays

function compute_subdomain_graph_dIS_and_lst_snd(space, dI)
  # List parts I have to send data to
  function compute_lst_snd(part, gids, I)
    lst_snd = Set{Int}()
    for i = 1:length(I)
      owner = gids.lid_to_owner[I[i]]
      if (owner != part)
        if (!(owner in lst_snd))
          push!(lst_snd, owner)
        end
      end
    end
    collect(lst_snd)
  end

  part_to_lst_snd = DistributedData(compute_lst_snd, space.gids, dI)
  part_to_num_snd = DistributedData(part_to_lst_snd) do part, lst_snd
    length(lst_snd)
  end

  offsets = gather(part_to_num_snd)
  num_edges = sum(offsets)
  GridapDistributed._fill_offsets!(offsets)
  part_to_offsets = scatter(get_comm(part_to_num_snd), offsets)

  part_to_owned_subdomain_graph_edge_gids =
    DistributedData(part_to_num_snd, part_to_offsets) do part, num_snd, offset
      owned_edge_gids = zeros(Int, num_snd)
      o = offset
      for i = 1:num_snd
        o += 1
        owned_edge_gids[i] = o
      end
      owned_edge_gids
    end

  function build_cell_to_edge_gids(
    part,
    cell_gids,
    dofgids,
    lspace,
    lst_snd,
    owned_edge_gids,
  )
    l2g = Dict{Int,Int}()
    for i = 1:length(lst_snd)
      l2g[lst_snd[i]] = owned_edge_gids[i]
    end
    cell_to_lids = Gridap.FESpaces.get_cell_dofs(lspace)
    cell_to_edge_gids = collect(cell_to_lids)
    cache = array_cache(cell_to_lids)
    for cell = 1:length(cell_to_lids)
      #If current cell is local
      if (cell_gids.lid_to_owner[cell] == part)
        current_cell_lids = getindex!(cache, cell_to_lids, cell)
        for j = 1:length(current_cell_lids)
          owner = part
          if (current_cell_lids[j] > 0)
            owner = dofgids.lid_to_owner[current_cell_lids[j]]
          end
          if (owner != part)
            cell_to_edge_gids[cell][j] = l2g[owner]
          else
            cell_to_edge_gids[cell][j] = 0
          end
        end
      end
    end
    cell_to_edge_gids
  end

  part_to_cell_to_edge_gids = DistributedVector(
    build_cell_to_edge_gids,
    space.model.gids,
    space.model.gids,
    space.gids,
    space.spaces,
    part_to_lst_snd,
    part_to_owned_subdomain_graph_edge_gids,
  )

  exchange!(part_to_cell_to_edge_gids)

  function compute_subdomain_graph_index_set(
    part,
    cell_gids,
    dof_gids,
    lspace,
    cell_to_edge_gids,
    owned_edge_gids,
  )
    cell_to_lids = Gridap.FESpaces.get_cell_dofs(lspace)
    cache = array_cache(cell_to_lids)

    lst_rcv = Int[]
    non_owned_edge_gids = Int[]
    neighbour_parts_touched = Set{Int}()

    #Traverse all cells
    for cell = 1:length(cell_to_edge_gids)
      current_cell_lids = getindex!(cache, cell_to_lids, cell)
      #If current cell is ghost
      if (cell_gids.lid_to_owner[cell] != part)
        #Go over Dofs of current cell
        for j = 1:length(cell_to_edge_gids[cell])
          edge_gid = cell_to_edge_gids[cell][j]
          lid = current_cell_lids[j]
          # Determine whether a dof touched by a ghost cell belongs to current part
          if (lid > 0 && dof_gids.lid_to_owner[lid] == part)
            neighbour_part = cell_gids.lid_to_owner[cell]
            if (!(neighbour_part in neighbour_parts_touched))
              push!(lst_rcv, neighbour_part)
              push!(neighbour_parts_touched, neighbour_part)
              push!(non_owned_edge_gids, edge_gid)
            end
          end
        end
      end
    end
    lid_to_gid = vcat(owned_edge_gids, non_owned_edge_gids)
    lid_to_owner = vcat([part for i = 1:length(owned_edge_gids)], lst_rcv)
    IndexSet(num_edges, lid_to_gid, lid_to_owner)
  end

  (
    DistributedIndexSet(
      compute_subdomain_graph_index_set,
      get_comm(space.model),
      num_edges,
      space.model.gids,
      space.gids,
      space.spaces,
      part_to_cell_to_edge_gids,
      part_to_owned_subdomain_graph_edge_gids,
    ),
    part_to_lst_snd,
  )
end

struct DistributedSparseMatrixAssemblerFullyAssembled{GM,GV,LM,LV} <: Assembler
  global_matrix_type::Type{GM}
  global_vector_type::Type{GV}
  local_matrix_type::Type{LM}
  local_vector_type::Type{LV}
  trial::GridapDistributed.DistributedFESpace
  test::GridapDistributed.DistributedFESpace
  assems::DistributedData{<:Assembler}
  strategy::DistributedAssemblyStrategy
end


function GridapDistributed.get_distributed_data(
  dassem::DistributedSparseMatrixAssemblerFullyAssembled,
)
  dassem.assems
end

function Gridap.FESpaces.SparseMatrixAssembler(
  global_matrix_type::Type,
  global_vector_type::Type,
  local_matrix_type::Type,
  local_vector_type::Type,
  dtrial::DistributedFESpace,
  dtest::DistributedFESpace,
  dstrategy::DistributedAssemblyStrategy,
)

  assems = DistributedData(
    dtrial.spaces,
    dtest.spaces,
    dstrategy,
  ) do part, U, V, strategy
    SparseMatrixAssembler(local_matrix_type, local_vector_type, U, V, strategy)
  end

  DistributedSparseMatrixAssemblerFullyAssembled(
    global_matrix_type,
    global_vector_type,
    local_matrix_type,
    local_vector_type,
    dtrial,
    dtest,
    assems,
    dstrategy,
  )
end


function Gridap.FESpaces.assemble_matrix_and_vector(
  dassem::DistributedSparseMatrixAssemblerFullyAssembled,
  ddata,
)
  # 1. Compute local portions
  # 2. Determine communication pattern
  # 3. Communicate entries
  # 4. Combine local + remote entries
  # 5. Build fully assembled local portions
  # 6. Combine fully assembled local portions into global data structure

  # 1.
  dIJVb = DistributedData(dassem.assems, ddata) do part, assem, data
    trial = assem.trial
    test = assem.test
    b = Gridap.Algebra.allocate_vector(assem, data)
    n = Gridap.FESpaces.count_matrix_and_vector_nnz_coo(assem, data)
    I, J, V = Gridap.FESpaces.allocate_coo_vectors(
      Gridap.FESpaces.get_matrix_type(assem),
      n,
    )
    Gridap.FESpaces.fill_matrix_and_vector_coo_numeric!(I, J, V, b, assem, data)
    Gridap.FESpaces.finalize_coo!(
      I,
      J,
      V,
      num_free_dofs(test),
      num_free_dofs(test),
    )
    (I, J, V, b)
  end

  # 2.
  dI = DistributedData(dIJVb) do part, IJVb
    first(IJVb)
  end
  dIS, part_to_lst_snd =
    compute_subdomain_graph_dIS_and_lst_snd(dassem.test, dI)

  # 3.
  # TODO: Isnt a bit strange that I have to pass dIS twice to the DistributedVector constructor?
  entries_exchange_vector = DistributedVector(
    dIS,
    dIS,
    part_to_lst_snd,
    dassem.test,
    dassem.trial,
    dIJVb,
  ) do part, IS, lst_snd, (test, test_gids), (trial, trial_gids), IJVb
    I, J, V, b = IJVb
    lpart = typeof((eltype(I)[], eltype(J)[], eltype(V)[], eltype(I)[],eltype(b)[]))[]
    for i=1:length(IS.lid_to_gid)
       push!(lpart,(eltype(I)[], eltype(J)[], eltype(V)[], eltype(I)[],eltype(b)[]))
    end
    for i = 1:length(I)
      owner = test_gids.lid_to_owner[I[i]]
      if (owner != part)
        edge_lid = findfirst((i) -> (i == owner), lst_snd)
        push!(lpart[edge_lid][1],test_gids.lid_to_gid[I[i]])
        push!(lpart[edge_lid][2],trial_gids.lid_to_gid[J[i]])
        push!(lpart[edge_lid][3],V[i])
      end
    end

    for i = 1:length(b)
      owner = test_gids.lid_to_owner[i]
      if (owner != part)
        edge_lid = findfirst((i) -> (i == owner), lst_snd)
        if (edge_lid != nothing)
          push!(lpart[edge_lid][4], test_gids.lid_to_gid[i])
          push!(lpart[edge_lid][5], b[i])
        end
      end
    end
    lpart
  end

  exchange!(entries_exchange_vector)

  # 4.
  dIJVb = DistributedData(
    dIJVb,
    entries_exchange_vector,
    dIS,
    dassem.test,
    dassem.trial,
  ) do part, IJVb, remote_entries, IS, (test, test_gids), (trial, trial_gids)
    I, J, V, b = IJVb
    GI = eltype(I)[]
    GJ = eltype(J)[]
    GV = eltype(V)[]

    #TODO: check with fverdugo if there is an already coded way of
    #      doing this vector pattern operation
    lid_to_owned_lid = fill(-1, length(test_gids.lid_to_owner))
    current = 1
    for i = 1:length(test_gids.lid_to_owner)
      if (test_gids.lid_to_owner[i] == part)
        lid_to_owned_lid[i] = current
        current += 1
      end
    end

    # Add local entries
    for i = 1:length(I)
      if (test_gids.lid_to_owner[I[i]] == part)
        push!(GI, lid_to_owned_lid[I[i]])
        push!(GJ, trial_gids.lid_to_gid[J[i]])
        push!(GV, V[i])
      end
    end
    # Add remote entries
    for edge_lid = 1:length(IS.lid_to_gid)
      if (IS.lid_to_owner[edge_lid] != part)
        for i = 1:length(remote_entries[edge_lid][1])
          push!(
            GI,
            lid_to_owned_lid[test_gids.gid_to_lid[remote_entries[edge_lid][1][i]]],
          )
          push!(GJ, remote_entries[edge_lid][2][i])
          push!(GV, remote_entries[edge_lid][3][i])
        end
        for i = 1:length(remote_entries[edge_lid][4])
          b[test_gids.gid_to_lid[remote_entries[edge_lid][4][i]]] +=
            remote_entries[edge_lid][5][i]
        end
      end
    end
    GI, GJ, GV, b
  end

  # 5.
  fully_assembled_local_portions = DistributedData(
    dassem,
    dIJVb,
    dassem.test,
    dassem.trial,
  ) do part, assem, IJVb, (test, test_gids), (trial, trial_gids)
    I, J, V, b = IJVb

    n_owned_dofs = count((i) -> (i == part), test_gids.lid_to_owner)

    Lb = allocate_vector(assem.vector_type, n_owned_dofs)
    current = 1
    for i = 1:length(test_gids.lid_to_owner)
      if (test_gids.lid_to_owner[i] == part)
        Lb[current] = b[i]
        current += 1
      end
    end
    LA = sparse_from_coo(
      assem.matrix_type,
      I,
      J,
      V,
      n_owned_dofs,
      trial_gids.ngids,
    )
    (LA, Lb)
  end

  # 6.
  dIJV = DistributedData(fully_assembled_local_portions) do part, LALb
    LA,_ = LALb
    #TODO: Swap J, I whenever findnz is fixed
    J, I, V = findnz(LA)
    I, J, V
  end

  dn = DistributedData(dIJV) do part, IJV
    I, _, _ = IJV
    length(I)
  end

  # 6.1. Assemble global A
  part_to_n = gather(dn)
  n = sum(part_to_n)
  gIJV = allocate_coo_vectors(dassem.global_matrix_type, n)
  GridapDistributed._fill_offsets!(part_to_n)
  offsets = scatter(get_comm(dn), part_to_n .+ 1)
  gI,gJ,gV = gIJV

  do_on_parts(dIJV,gI,gJ,gV,dassem.test,dassem.trial,offsets) do part, IJV, gI,gJ,gV, (test, test_gids), (trial, trial_gids), offset
    n_owned_dofs = count((i) -> (i == part), test_gids.lid_to_owner)
    owned_lid_to_lid = fill(-1, n_owned_dofs)
    current = 1
    for i = 1:length(test_gids.lid_to_owner)
      if (test_gids.lid_to_owner[i] == part)
        owned_lid_to_lid[current] = i
        current += 1
      end
    end

    I,J,V = IJV
    for i=1:length(I)
      gI[offset+i-1] = test_gids.lid_to_gid[owned_lid_to_lid[I[i]]]
      gJ[offset+i-1] = J[i]
      gV[offset+i-1] = V[i]
    end

  end


  sIJV = GridapDistributed.SequentialIJV(dIJV, gIJV)
  finalize_coo!(
    dassem.global_matrix_type,
    sIJV,
    dassem.test.gids,
    dassem.trial.gids,
  )
  A = sparse_from_coo(
    dassem.global_matrix_type,
    sIJV,
    dassem.test.gids,
    dassem.trial.gids,
  )

  # 6.2. Assemble global b
  b = allocate_vector(dassem.global_vector_type, dassem.test.gids)
  do_on_parts(
    dassem.test.gids,
    fully_assembled_local_portions,
    b,
  ) do part, gids, (LA, Lb), b
    current = 1
    for i = 1:length(gids.lid_to_owner)
      if (gids.lid_to_owner[i] == part)
        add_entry!(b, Lb[current], gids.lid_to_gid[i])
        current += 1
      end
    end
  end
  A, b
end


# Select matrix and vector types for discrete problem
# Note that here we use serial vectors and matrices
# but the assembly is distributed
T = Float64
global_vector_type = Vector{T}
global_matrix_type = SparseMatrixCSC{T,Int}

local_vector_type = Vector{T}
local_matrix_type = SparseMatrixCSR{1,T,Int}


# Manufactured solution
u(x) = x[1] + x[2]
f(x) = -Δ(u)(x)

# Discretization
subdomains = (2, 2)
domain = (0, 1, 0, 1)
cells = (4, 4)
comm = SequentialCommunicator(subdomains)
model = CartesianDiscreteModel(comm, subdomains, domain, cells)

# FE Spaces
order = 1
V = FESpace(
  global_vector_type,
  valuetype = Float64,
  reffe = :Lagrangian,
  order = order,
  model = model,
  conformity = :H1,
  dirichlet_tags = "boundary",
)

U = TrialFESpace(V, u)

# Terms in the weak form
terms = DistributedData(model) do part, (model, gids)

  trian = Triangulation(model)
  trian = remove_ghost_cells(trian, part, gids)

  degree = 2 * order
  quad = CellQuadrature(trian, degree)

  a(u, v) = ∇(v) * ∇(u)
  l(v) = v * f
  t1 = AffineFETerm(a, l, trian, quad)

  (t1,)
end

strategy = DistributedData(get_comm(model)) do part
  Gridap.FESpaces.DefaultAssemblyStrategy()
end
strategy = GridapDistributed.DistributedAssemblyStrategy(strategy)

assem = Gridap.FESpaces.SparseMatrixAssembler(
  global_matrix_type,
  global_vector_type,
  local_matrix_type,
  local_vector_type,
  U,
  V,
  strategy,
)

ddata = DistributedData(assem, terms) do part, assem, terms
  trial = assem.trial
  test = assem.test
  u = Gridap.FESpaces.get_cell_basis(trial)
  v = Gridap.FESpaces.get_cell_basis(test)
  uhd = zero(trial)
  data = Gridap.FESpaces.collect_cell_matrix_and_vector(uhd, u, v, terms)
end

A,b = Gridap.FESpaces.assemble_matrix_and_vector(assem, ddata)
