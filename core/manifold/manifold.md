Manifold  {#concept_manifold}
===========================
## Summary
  \note In mathematics, a _manifold_ is a topological space that resembles Euclidean space near each point. A _differentiable manifold_ is a type of manifold that is locally similar enough to a linear space to allow one to do calculus. 
   
## Requirements

 ~~~~~~~~~~~~~{.cpp}
 template <typename Geometry,template<typename> class Policy1,template<typename> class Policy2> 
 class Manifold:
	 public Geometry, 
	 public Policy1<Geometry>,
	 public Policy2<Geometry>
 {
   .....
 };
 ~~~~~~~~~~~~~
The following table lists requirements for a Manifold type `M`,  

 Pseudo-Signature  		| Semantics  
 -------------------|-------------  
 `M( const M& )` 		| Copy constructor.  
 `~M()` 				| Destructor. 
 `geometry_type`		| Geometry type of manifold, which describes coordinates and metric
 `topology_type`		| Topology structure of manifold,   topology of grid points
 `coordiantes_type` 	| data type of coordinates, i.e. nTuple<3,Real>
 `index_type`			| data type of the index of grid points, i.e. unsigned long
 `Domain  domain()`	| Root domain of manifold


Manifold policy concept {#concept_manifold_policy}
================================================
  Poilcies define the behavior of manifold , such as  interpolator or calculus;
 ~~~~~~~~~~~~~{.cpp}
 template <typename Geometry > class P;
 ~~~~~~~~~~~~~
 
 The following table lists requirements for a Manifold policy type `P`,  

 Pseudo-Signature  		| Semantics  
 -----------------------|-------------  
 `P( Geometry  & )` 	| Constructor.  
 `P( P const  & )`	| Copy constructor.  
 `~P( )` 				| Copy Destructor.  
 
## Interpolator policy
  Interpolator, map between discrete space and continue space, i.e. Gather & Scatter
  
   Pseudo-Signature  		| Semantics  
 ---------------------------|-------------  
 `gather(field_type const &f, coordinates_type x  )` 	| gather data from `f` at coordinates `x`.  
 `scatter(field_type &f, coordinates_type x ,value_type v)` 	| scatter `v` to field  `f` at coordinates `x`.  
  
## Calculus  policy
 Define calculus operation of  fields on the manifold, such  as algebra or differential calculus.
 Differential calculus scheme , i.e. FDM,FVM,FEM,DG ....


 Pseudo-Signature  		| Semantics  
 -----------------------|-------------  
 `calculate(TOP op, field_type const &f, field_type const &f, index_type s ) `	| `calculate`  binary operation `op` at grid point `s`.  
 `calculate(TOP op, field_type const &f,  index_type s )` 	| `calculate`  unary operation  `op`  at grid point `s`.   


## See aslo
- \subpage concept_geometry 
- \subpage concept_topology
- \subpage concept_interpolator
- \subpage concept_diff_scheme





