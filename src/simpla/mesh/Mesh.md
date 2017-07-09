Manifold (Differential Manifold):
=================================
 A presentation of a _topological manifold_ is a second countable Hausdorff space that is locally homeomorphic
 to a linear space, by a collection (called an atlas) of homeomorphisms called _charts_. The composition of one
 _chart_ with the inverse of another m_chart_ is a function called a _transition map_, and defines a homeomorphism
  of an open subset of the linear space onto another open subset of the linear space.
  
 
 ## Requirements

~~~~~~~~~~~~~{.cpp}
template <typename BaseManifold,template<typename> class Policy1,template<typename> class Policy2>
class mesh:
public BaseManifold,
public Policy1<BaseManifold>,
public Policy2<BaseManifold>
{
.....
};
~~~~~~~~~~~~~
The following table lists requirements for a mesh type `M`,

 Pseudo-Signature  		| Semantics
 -----------------------|-------------
 `M( const M& )` 		| Copy constructor.
 `~M()` 				| Destructor.
 `mesh_type`		    | BaseManifold type of model, which describes coordinates and Metric
 `mesh_type`		    | Topology structure of model,   Topology of grid points
 `coordiantes_type` 	| m_data type of coordinates, i.e. nTuple<3,Real>
 `index_type`			| m_data type of the index of grid points, i.e. unsigned long
 `Domain  domain()`	    | Root domain of model
 
mesh policy concept {#concept_manifold_policy}
================================================
  Poilcies define the behavior of model , such as  interpolate or calculus;
  
 ~~~~~~~~~~~~~{.cpp}
 template <typename BaseManifold > class P;
 ~~~~~~~~~~~~~

  The following table lists requirements for a get_mesh policy type `P`,

  Pseudo-Signature  	 | Semantics
  -----------------------|-------------
  `P( BaseManifold  & )` | Constructor.
  `P( P const  & )`	   | Copy constructor.
  `~P( )` 			   | Copy Destructor.

 ## Interpolator policy
   Interpolator, map between discrete space and continue space, i.e. Gather & Scatter
   

 Pseudo-Signature  	     | Semantics
---------------------------|-----------------------------
`Gather(field_type const &f, coordinate_tuple x  )` 	    | Gather m_data from `f` at coordinates `x`.
`scatter(field_type &f, coordinate_tuple x ,value_type v)` 	| scatter `v` to field  `f` at coordinates `x`.
  

 ## Calculus  policy
  Define calculus operation of  fields on the model, such  as algebra or differential calculus.
  Differential calculus scheme , i.e. FDM,FVM,FEM,DG ....


  Pseudo-Signature        | Semantics
  ------------------------|-------------
  `diff_scheme(TOP op, field_type const &f, field_type const &f, index_type s ) `	| `diff_scheme`  binary operation `op` at grid point `s`.
  `diff_scheme(TOP op, field_type const &f,  index_type s )` 	| `diff_scheme`  unary operation  `op`  at grid point `s`.


  ## Differential Form
  @brief In the mathematical fields of @ref diff_geo and tensor calculus,
   differential forms are an approach to multivariable calculus that
     is independent of coordinates. --wiki


 ## Summary
 \note Let \f$M\f$ be a _smooth manifold_. A _differential form_ of degree \f$k\f$ is
  a smooth section of the \f$k\f$th exterior power of the cotangent bundle of \f$M\f$.
  At any point \f$p \in M\f$, a k-form \f$\beta\f$ defines an alternating multilinear map
 \f[
   \beta_p\colon T_p M\times \cdots \times T_p M \to \mathbb{R}
 \f]
 (with k factors of \f$T_p M\f$ in the product), where TpM is the tangent space to \f$M\f$ at \f$p\f$.
  Equivalently, \f$\beta\f$ is a totally antisymetric covariant tensor field of rank \f$k\f$.

  Differential form is a field

 ## Requirements