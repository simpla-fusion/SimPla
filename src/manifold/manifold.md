Manifold  {#concept_manifold}
===========================
## Summary
  \note In mathematics, a _manifold_ is a topological space that resembles Euclidean space near each point. A _differentiable manifold_ is a type of manifold that is locally similar enough to a linear space to allow one to do calculus. 
   
## Requirements

The following table lists requirements for a Manifold type M,  

 Pseudo-Signature  		| Semantics  
 -------------------|-------------  
 M::M( const M& ) 		| Copy constructor.  
 M::~M() 				| Destructor. 
 M::geometry_type		| Geometry type of manifold, which describes coordinates and metric
 M::topology_type		| Topology structure of manifold,   topology of grid points
 M::interpolator_policy	| Interpolator, map between discrete space and continue space, i.e. Gather & Scatter
 M::diff_scheme_policy	| Differential structure of manifold, difference scheme , i.e. FDM,FVM,FEM,DG ....
 M::coordiantes_type 	| data type of coordinates, i.e. nTuple<3,Real>
 M::index_type			| data type of the index of grid points, i.e. unsigned long
 Domain M::domain()		| Root domain of manifold

## See aslo
- \subpage concept_geometry 
- \subpage concept_topology
- \subpage concept_interpolator
- \subpage concept_diff_scheme
- \subpage concept_domain
 




