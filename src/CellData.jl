
abstract type DistributedCellDatum <: GridapType end

# DistributedCellPoint
"""
"""
struct DistributedCellPoint{A<:AbstractArray{<:CellPoint}} <: DistributedCellDatum
  points::A
end

local_views(a::DistributedCellPoint) = a.points

# DistributedCellField
"""
"""
struct DistributedCellField{A,B} <: DistributedCellDatum
  fields::A
  metadata::B
  function DistributedCellField(
    fields::AbstractArray{<:CellField},
    metadata=nothing)

    A = typeof(fields)
    B = typeof(metadata)
    new{A,B}(fields,metadata)
  end
end

local_views(a::DistributedCellField) = a.fields

# Constructors

function CellData.CellField(f::Function,trian::DistributedTriangulation)
  fields = map(trian.trians) do t
    CellField(f,t)
  end
  DistributedCellField(fields)
end

function CellData.CellField(f::Number,trian::DistributedTriangulation)
  fields = map(trian.trians) do t
    CellField(f,t)
  end
  DistributedCellField(fields)
end

function CellData.CellField(
  f::AbstractArray{<:AbstractArray{<:Number}},trian::DistributedTriangulation)
  fields = map(f,trian.trians) do f,t
    CellField(f,t)
  end
  DistributedCellField(fields)
end

# Evaluation

function (f::DistributedCellField)(x::DistributedCellPoint)
  evaluate!(nothing,f,x)
end

function Arrays.evaluate!(cache,f::DistributedCellField,x::DistributedCellPoint)
  map(f.fields,x.points) do f,x
    evaluate!(nothing,f,x)
  end
end

# Operations

function Arrays.evaluate!(cache,k::Operation,a::DistributedCellField)
  fields = map(a.fields) do f
    evaluate!(nothing,k,f)
  end
  DistributedCellField(fields)
end

function Arrays.evaluate!(
  cache,k::Operation,a::DistributedCellField,b::DistributedCellField)
  fields = map(a.fields,b.fields) do f,g
    evaluate!(nothing,k,f,g)
  end
  DistributedCellField(fields)
end

function Arrays.evaluate!(
  cache,k::Operation,a::DistributedCellField,b::DistributedCellField,c::DistributedCellField)
  fields = map_parts(a.fields,b.fields,c.fields) do f,g,h
    evaluate!(nothing,k,f,g,h)
  end
  DistributedCellField(fields)
end

function Arrays.evaluate!(
  cache,k::Operation,a::DistributedCellField,b::DistributedCellField,c::DistributedCellField,d::DistributedCellField)
  fields = map_parts(a.fields,b.fields,c.fields,d.fields) do f,g,h,j
    evaluate!(nothing,k,f,g,h,j)
  end
  DistributedCellField(fields)
end

function Arrays.evaluate!(cache,k::Operation,a::DistributedCellField,b::Number)
  fields = map(a.fields) do f
    evaluate!(nothing,k,f,b)
  end
  DistributedCellField(fields)
end

function Arrays.evaluate!(cache,k::Operation,b::Number,a::DistributedCellField)
  fields = map(a.fields) do f
    evaluate!(nothing,k,b,f)
  end
  DistributedCellField(fields)
end

function Arrays.evaluate!(cache,k::Operation,a::DistributedCellField,b::Function)
  fields = map(a.fields) do f
    evaluate!(nothing,k,f,b)
  end
  DistributedCellField(fields)
end

function Arrays.evaluate!(cache,k::Operation,b::Function,a::DistributedCellField)
  fields = map(a.fields) do f
    evaluate!(nothing,k,b,f)
  end
  DistributedCellField(fields)
end

function Arrays.evaluate!(cache,k::Operation,a::DistributedCellField...)
  fields = map(map(i->i.fields,a)...) do f...
    evaluate!(nothing,k,f...)
  end
  DistributedCellField(fields)
end

# Composition

Base.:(∘)(f::Function,g::DistributedCellField) = Operation(f)(g)
Base.:(∘)(f::Function,g::Tuple{DistributedCellField,DistributedCellField}) = Operation(f)(g[1],g[2])
Base.:(∘)(f::Function,g::Tuple{DistributedCellField,Number}) = Operation(f)(g[1],g[2])
Base.:(∘)(f::Function,g::Tuple{Number,DistributedCellField}) = Operation(f)(g[1],g[2])
Base.:(∘)(f::Function,g::Tuple{DistributedCellField,Function}) = Operation(f)(g[1],g[2])
Base.:(∘)(f::Function,g::Tuple{Function,DistributedCellField}) = Operation(f)(g[1],g[2])
Base.:(∘)(f::Function,g::Tuple{Vararg{DistributedCellField}}) = Operation(f)(g...)

# Define some of the well known arithmetic ops

# Unary ops

for op in (:symmetric_part,:inv,:det,:abs,:abs2,:+,:-,:tr,:transpose,:adjoint,:grad2curl,:real,:imag,:conj)
  @eval begin
    ($op)(a::DistributedCellField) = Operation($op)(a)
  end
end

# Binary ops

for op in (:inner,:outer,:double_contraction,:+,:-,:*,:cross,:dot,:/)
  @eval begin
    ($op)(a::DistributedCellField,b::DistributedCellField) = Operation($op)(a,b)
    ($op)(a::DistributedCellField,b::Number) = Operation($op)(a,b)
    ($op)(a::Number,b::DistributedCellField) = Operation($op)(a,b)
    ($op)(a::DistributedCellField,b::Function) = Operation($op)(a,b)
    ($op)(a::Function,b::DistributedCellField) = Operation($op)(a,b)
  end
end

Base.broadcasted(f,a::DistributedCellField,b::DistributedCellField) = Operation((i,j)->f.(i,j))(a,b)
Base.broadcasted(f,a::Number,b::DistributedCellField) = Operation((i,j)->f.(i,j))(a,b)
Base.broadcasted(f,a::DistributedCellField,b::Number) = Operation((i,j)->f.(i,j))(a,b)
Base.broadcasted(f,a::Function,b::DistributedCellField) = Operation((i,j)->f.(i,j))(a,b)
Base.broadcasted(f,a::DistributedCellField,b::Function) = Operation((i,j)->f.(i,j))(a,b)
Base.broadcasted(::typeof(*),::typeof(∇),f::DistributedCellField) = Operation(Fields._extract_grad_diag)(∇(f))
Base.broadcasted(::typeof(*),s::Fields.ShiftedNabla,f::DistributedCellField) = Operation(Fields._extract_grad_diag)(s(f))

dot(::typeof(∇),f::DistributedCellField) = divergence(f)
outer(::typeof(∇),f::DistributedCellField) = gradient(f)
outer(f::DistributedCellField,::typeof(∇)) = transpose(gradient(f))
cross(::typeof(∇),f::DistributedCellField) = curl(f)

# Differential ops

function Fields.gradient(a::DistributedCellField)
  DistributedCellField(map(gradient,a.fields))
end

function Fields.divergence(a::DistributedCellField)
  DistributedCellField(map(divergence,a.fields))
end

function Fields.DIV(a::DistributedCellField)
  DistributedCellField(map(DIV,a.fields))
end

function Fields.∇∇(a::DistributedCellField)
  DistributedCellField(map(∇∇,a.fields))
end

function Fields.curl(a::DistributedCellField)
  DistributedCellField(map(curl,a.fields))
end

# Integration related
"""
"""
struct DistributedMeasure{A<:AbstractArray{<:Measure}} <: GridapType
  measures::A
end

local_views(a::DistributedMeasure) = a.measures

function CellData.Measure(t::DistributedTriangulation,args...)
  measures = map(t.trians) do trian
    Measure(trian,args...)
  end
  DistributedMeasure(measures)
end

function CellData.Measure(tt::DistributedTriangulation{Dc,Dp},it::DistributedTriangulation{Dc,Dp},args...) where {Dc,Dp}
  measures = map(local_views(tt),local_views(it)) do ttrian, itrian
    Measure(ttrian,itrian,args...)
  end
  return DistributedMeasure(measures)
end

function CellData.get_cell_points(a::DistributedMeasure)
  DistributedCellPoint(map(get_cell_points,a.measures))
end

"""
"""
struct DistributedDomainContribution{A<:AbstractArray{<:DomainContribution}} <: GridapType
  contribs::A
end

local_views(a::DistributedDomainContribution) = a.contribs

function Base.getindex(c::DistributedDomainContribution,t::DistributedTriangulation)
  map(getindex,c.contribs,t.trians)
end

function Fields.integrate(f::DistributedCellField,b::DistributedMeasure)
  contribs = map(f.fields,b.measures) do f,m
    integrate(f,m)
  end
  DistributedDomainContribution(contribs)
end

function Fields.integrate(f::Function,b::DistributedMeasure)
  contribs = map(b.measures) do m
    integrate(f,m)
  end
  DistributedDomainContribution(contribs)
end

function Fields.integrate(f::Number,b::DistributedMeasure)
  contribs = map(b.measures) do m
    integrate(f,m)
  end
  DistributedDomainContribution(contribs)
end

function (*)(a::Integrand,b::DistributedMeasure)
  integrate(a.object,b)
end

(*)(b::DistributedMeasure,a::Integrand) = a*b

function Base.sum(a::DistributedDomainContribution)
  sum(map(sum,a.contribs))
end

function (+)(a::DistributedDomainContribution,b::DistributedDomainContribution)
  contribs = map(+,a.contribs,b.contribs)
  DistributedDomainContribution(contribs)
end

function (-)(a::DistributedDomainContribution,b::DistributedDomainContribution)
  contribs = map(-,a.contribs,b.contribs)
  DistributedDomainContribution(contribs)
end

function (*)(a::Number,b::DistributedDomainContribution)
  contribs = map(b.contribs) do b
    a*b
  end
  DistributedDomainContribution(contribs)
end

(*)(a::DistributedDomainContribution,b::Number) = b*a

# Triangulation related

function CellData.get_cell_points(a::DistributedTriangulation)
  DistributedCellPoint(map(get_cell_points,a.trians))
end

function CellData.get_normal_vector(a::DistributedTriangulation)
  fields = map(get_normal_vector,a.trians)
  DistributedCellField(fields)
end

# Skeleton related

function DistributedCellField(a::AbstractArray{<:SkeletonPair})
  plus, minus = map(s->(s.plus,s.minus),a) |> tuple_of_arrays
  dplus = DistributedCellField(plus)
  dminus = DistributedCellField(minus)
  SkeletonPair(dplus,dminus)
end

function Base.getproperty(x::DistributedCellField, sym::Symbol)
  if sym in (:⁺,:plus)
    DistributedCellField(map(i->i.plus,x.fields))
  elseif sym in (:⁻, :minus)
    DistributedCellField(map(i->i.minus,x.fields))
  else
    getfield(x, sym)
  end
end

function Base.propertynames(x::DistributedCellField, private::Bool=false)
  (fieldnames(typeof(x))...,:⁺,:plus,:⁻,:minus)
end

for op in (:outer,:*,:dot)
  @eval begin
    ($op)(a::DistributedCellField,b::SkeletonPair{<:DistributedCellField}) = Operation($op)(a,b)
    ($op)(a::SkeletonPair{<:DistributedCellField},b::DistributedCellField) = Operation($op)(a,b)
  end
end

function Arrays.evaluate!(cache,k::Operation,a::DistributedCellField,b::SkeletonPair{<:DistributedCellField})
  plus = k(a.plus,b.plus)
  minus = k(a.minus,b.minus)
  SkeletonPair(plus,minus)
end

function Arrays.evaluate!(cache,k::Operation,a::SkeletonPair{<:DistributedCellField},b::DistributedCellField)
  plus = k(a.plus,b.plus)
  minus = k(a.minus,b.minus)
  SkeletonPair(plus,minus)
end

CellData.jump(a::DistributedCellField) = DistributedCellField(map(jump,a.fields))
CellData.jump(a::SkeletonPair{<:DistributedCellField}) = a.⁺ + a.⁻
CellData.mean(a::DistributedCellField) = DistributedCellField(map(mean,a.fields))


# DistributedCellDof

struct DistributedCellDof{A} <: DistributedCellDatum
  dofs::A
  function DistributedCellDof(dofs::AbstractArray{<:CellDof})
      A = typeof(dofs)
      new{A}(dofs)
  end
end

local_views(s::DistributedCellDof) = s.dofs

(a::DistributedCellDof)(f) = evaluate(a,f)

function Gridap.Arrays.evaluate!(cache,s::DistributedCellDof,f::DistributedCellField)
  map(local_views(s),local_views(f)) do s, f
      evaluate!(nothing,s,f)
  end
end

function Gridap.Arrays.evaluate!(cache, ::DistributedCellField, ::DistributedCellDof)
@unreachable """\n
CellField (f) objects cannot be evaluated at CellDof (s) objects.
However, CellDofs objects can be evaluated at CellField objects.
Did you mean evaluate(f,s) instead of evaluate(s,f), i.e.
f(s) instead of s(f)?
"""
end