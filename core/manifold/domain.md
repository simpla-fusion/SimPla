Domain {#concept_domain}
=========================================

## Summary

 \note  "In mathematics, the  domain of definition or simply the domain of a function is the set of "input" or argument values for which the function is defined. That is, the function provides an "output" or value for each member of the domain.  Conversely, the set of values the function takes is termed the image of the function, which is sometimes also referred to as the  range of the function."  -- @ref http://en.wikipedia.org/wiki/Domain_of_a_function
 
  - Domain is a [parallel range](https://www.threadingbuildingblocks.org/docs/help/reference/containers_overview/container_range_concept.htm)

 	\note  "A Range can be recursively subdivided into two parts. It is recommended that the division be into nearly equal parts, but it is not required. Splitting as evenly as possible typically yields the best parallelism. Ideally, a range is recursively splittable until the parts represent portions of work that are more efficient to execute serially rather than split further. The amount of work represented by a Range typically depends upon higher level context, hence a typical type that models a Range should provide a way to control the degree of splitting. For example, the template class blocked_range has a grainsize parameter that specifies the biggest range considered indivisible." -- TBB  ( [Range concept in TBB](https://www.threadingbuildingblocks.org/docs/help/reference/algorithms/range_concept.htm) )
 	
 - Domain is a chain of simplex or polytope 
   \note In geometry, a simplex (plural simplexes or simplices) is a generalization of the notion of a triangle or tetrahedron to arbitrary dimensions. Specifically, a k-simplex is a k-dimensional polytope which is the convex hull of its k + 1 vertices. More formally, suppose the k + 1 points \f$ u_0,\dots, u_k \in \mathbb{R}^n \f$ are affinely independent, which means \f$ u_1 - u_0,\dots, u_k-u_0 \f$ are linearly independent. Then, the simplex determined by them is the set of points  \f$    C =\{\theta_0 u_0 + \dots+\theta_k u_k | \theta_i \ge 0, 0 \le i \le k, \sum_{i=0}^{k} \theta_i=1\}  \f$.

## Requirement 
The following table lists requirements for  Domain type D as _parallel range_ , which are same as Container Range in TBB.

 Pseudo-Signature  				| Semantics
 -------------------------------|--------------
 D( const D& ) 					| Copy constructor.
 ~D() 							| Destructor.
 bool empty() const 			| True if domain is empty.
 bool is_divisible() const 		| True if domain can be partitioned into two subdomains.
 D( D& d, split ) 				| Split d into two subdomains.
 difference_type 				| Type for difference of two iterators
 iterator 						| Iterator type for domain
 iterator begin(  ) const		| First item in domain.
 iterator end(  ) const 		| One past last item in domain. 
	



Additional requirements for  Domain type D 

 Pseudo-Signature  				| Semantics
 -------------------------------|-------------
 D const & parent()const		| Parent domain
 D operator &(D const & D1)const		| \f$D_0 \cap \D_1\f$
 D operator |(D const & D1)const		| \f$D_0 \cup \D_1\f$
 bool operator==(D const & D1)const		| \f$D_0 == \D_1\f$
 bool is_same(D const & D1)const		| \f$D_0 == \D_1\f$   

Requirements for  Domain type D as a geometric object, which could be a @ref concept_simplex or a chain of polytopes. 

 Pseudo-Signature  				| Semantics
 -------------------------------|-------------
 D const & parent()const		| Parent domain
 size_t hash(index_type)const 	| get relative  postion of  grid point s in the memory  
 size_t max_hash( )const 		| get max number of grid points in memory




 Pseudo-Signature  				| Semantics
 -------------------------------|-------------
 unsigned int iform				| type of form, VERTEX, EDGE, FACE,VOLUME
 geometry_typ					| Geometry 
 PD boundary(  D const& )		| Boundary of domain D, PD::ndims=D::ndims-1.
 boundbox() const				| boundbox on _this_ coordinates system
 cartesian_boundbox() const		| boundbox on _Cartesian_ coordinates system
 gather(coordinates_type x,TF const& f)const 	| get value at x
 scatter(coordiantes_type x,v,TF f *  )const 	| scatter v at x to f
 
 

##  See Also
- @ref concept_manifold
- @ref concept_topology
- @ref concept_simplex
- \subpage domain_iterator
- \subpage block_domain	 