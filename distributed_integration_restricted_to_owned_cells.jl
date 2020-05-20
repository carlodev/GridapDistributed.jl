
using Gridap
using GridapDistributed
using SparseArrays

function compute_subdomain_graph_distributed_index_set(space,dIJV)
  # List parts I have to send data to
  function compute_lst_snd(part, gids, IJV)
    I,_,_ = IJV
    lst_snd = Set{Int}()
    for i = 1:length(I)
      owner = gids.lid_to_owner[I[i]]
      if (owner != part)
        if (!(owner in lst_snd))
          push!(lst_snd,owner)
        end
      end
    end
    collect(lst_snd)
  end

  part_to_lst_snd = DistributedData(compute_lst_snd, space.gids, dIJV)
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

  DistributedIndexSet(
    compute_subdomain_graph_index_set,
    get_comm(space.model),
    num_edges,
    space.model.gids,
    space.gids,
    space.spaces,
    part_to_cell_to_edge_gids,
    part_to_owned_subdomain_graph_edge_gids,
  )
end


# Select matrix and vector types for discrete problem
# Note that here we use serial vectors and matrices
# but the assembly is distributed
T = Float64
vector_type = Vector{T}
matrix_type = SparseMatrixCSC{T,Int}

# Manufactured solution
u(x) = x[1] + x[2]
f(x) = - Δ(u)(x)

# Discretization
subdomains = (2,2)
domain = (0,1,0,1)
cells = (4,4)
comm = SequentialCommunicator(subdomains)
model = CartesianDiscreteModel(comm,subdomains,domain,cells)

# FE Spaces
order = 1
V = FESpace(
  vector_type, valuetype=Float64, reffe=:Lagrangian, order=order,
  model=model, conformity=:H1, dirichlet_tags="boundary")

U = TrialFESpace(V,u)

# Terms in the weak form
terms = DistributedData(model) do part, (model,gids)

  trian = Triangulation(model)
  trian = remove_ghost_cells(trian,part,gids)

  degree = 2*order
  quad = CellQuadrature(trian,degree)

  a(u,v) = ∇(v)*∇(u)
  l(v) = v*f
  t1 = AffineFETerm(a,l,trian,quad)

  (t1,)
end

strategy = DistributedData(get_comm(model)) do part
   Gridap.FESpaces.DefaultAssemblyStrategy()
end


strategy = GridapDistributed.DistributedAssemblyStrategy(strategy)

assem = Gridap.FESpaces.SparseMatrixAssembler(matrix_type, vector_type, U, V, strategy)

dIJV = DistributedData(assem,terms) do part, assem, terms
   trial = assem.trial
   test =  assem.test
   u = Gridap.FESpaces.get_cell_basis(trial)
   v = Gridap.FESpaces.get_cell_basis(test)
   uhd = zero(trial)
   b = Gridap.Algebra.allocate_vector(assem,nothing)
   data = Gridap.FESpaces.collect_cell_matrix_and_vector(uhd,u,v,terms)
   n = Gridap.FESpaces.count_matrix_and_vector_nnz_coo(assem,data)
   I,J,V = Gridap.FESpaces.allocate_coo_vectors(Gridap.FESpaces.get_matrix_type(assem),n)
   Gridap.FESpaces.fill_matrix_and_vector_coo_numeric!(I,J,V,b,assem,data)
   (I,J,V)
end

DIS = compute_subdomain_graph_distributed_index_set(V,dIJV)
