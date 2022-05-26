"""
"""
function FESpaces.FESpace(model::DistributedDiscreteModel,
                          reffe::Tuple{RaviartThomas,Any,Any};
                          conformity=nothing,kwargs...)

  cell_reffes = map_parts(model.models) do m
    basis,reffe_args,reffe_kwargs = reffe
    cell_reffe = ReferenceFE(m,basis,reffe_args...;reffe_kwargs...)
  end
  _common_fe_space_constructor(model,cell_reffes;conformity,kwargs...)
end

function FESpace(model::DistributedDiscreteModel,
                 reffe::GenericRefFE{RaviartThomas};
                 conformity=nothing, kwargs...)
  cell_reffes = map_parts(model.models) do m
    Fill(reffe,num_cells(m))
  end
  _common_fe_space_constructor(model,cell_reffes;conformity,kwargs...)
end

function _common_fe_space_constructor(model,cell_reffes;conformity,kwargs...)
  sign_flips=_generate_sign_flips(model,cell_reffes)
  spaces = map_parts(model.models,sign_flips,cell_reffes) do m,sign_flip,cell_reffe
     conf = Conformity(testitem(cell_reffe),conformity)
     cell_fe = CellFE(m,cell_reffe,conf,sign_flip)
     FESpace(m, cell_fe; kwargs...)
  end
  gids =  generate_gids(model,spaces)
  vector_type = _find_vector_type(spaces,gids)
  DistributedSingleFieldFESpace(spaces,gids,vector_type)
end



function _generate_sign_flips(model,cell_reffes)
  sign_flips=map_parts(model.models,model.gids.partition,cell_reffes) do m, p, cell_reffe

    D = num_cell_dims(m)
    gtopo = get_grid_topology(m)

    # Extract composition among cells and facets
    cell_wise_facets_ids = get_faces(gtopo, D, D - 1)
    cache_cell_wise_facets_ids = array_cache(cell_wise_facets_ids)

    # Extract cells around facets
    cells_around_facets = get_faces(gtopo, D - 1, D)
    cache_cells_around_facets = array_cache(cells_around_facets)

    ncells = num_cells(m)
    ptrs = Vector{Int32}(undef,ncells+1)
    for cell in 1:ncells
      reffe=cell_reffe[cell]
      ptrs[cell+1] = num_dofs(reffe)
    end
    PArrays.length_to_ptrs!(ptrs)
    ndata = ptrs[end]-1
    data  = Vector{Bool}(undef,ndata)
    data .= false

    for cell in p.oid_to_lid
      sign_flip = view(data,ptrs[cell]:ptrs[cell+1]-1)
      reffe=cell_reffe[cell]
      D = num_dims(reffe)
      face_own_dofs = get_face_own_dofs(reffe)
      facet_lid = get_offsets(get_polytope(reffe))[D] + 1
      cell_facets_ids = getindex!(cache_cell_wise_facets_ids,
                                  cell_wise_facets_ids,
                                  cell)
      for facet_gid in cell_facets_ids
          facet_cells_around = getindex!(cache_cells_around_facets,
                cells_around_facets,
                facet_gid)
          is_slave=false
          if (length(facet_cells_around)==1)
            is_slave == false
          else
            mx=maximum(p.lid_to_gid[facet_cells_around])
            is_slave = (p.lid_to_gid[cell] == mx)
          end
          if is_slave
              for dof in face_own_dofs[facet_lid]
                  sign_flip[dof] = true
              end
          end
          facet_lid = facet_lid + 1
      end
    end
    PArrays.Table(data,ptrs)
  end
  exchange!(sign_flips,model.gids.exchanger)
  sign_flips
end
