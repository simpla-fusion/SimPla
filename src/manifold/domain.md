Domain {#concept_domain}
=========================================

## Summary

 \note  "In mathematics, the  domain of definition or simply the domain of a function is the set of "input" or argument values for which the function is defined. That is, the function provides an "output" or value for each member of the domain.  Conversely, the set of values the function takes is termed the image of the function, which is sometimes also referred to as the  range of the function."  -- @ref http://en.wikipedia.org/wiki/Domain_of_a_function
  
 Topology Domain  concept is same as the [Container range concept in TBB](https://www.threadingbuildingblocks.org/docs/help/reference/containers_overview/container_range_concept.htm)

 \note  "A Range can be recursively subdivided into two parts. It is recommended that the division be into nearly equal parts, but it is not required. Splitting as evenly as possible typically yields the best parallelism. Ideally, a range is recursively splittable until the parts represent portions of work that are more efficient to execute serially rather than split further. The amount of work represented by a Range typically depends upon higher level context, hence a typical type that models a Range should provide a way to control the degree of splitting. For example, the template class blocked_range has a grainsize parameter that specifies the biggest range considered indivisible." -- TBB  ( [Range concept in TBB](https://www.threadingbuildingblocks.org/docs/help/reference/algorithms/range_concept.htm) )

## Requirement 
The following table lists requirements for  Domain type D as _parallel range_ , which are same as Container Range in TBB.

 Pseudo-Signature  				| Semantics
 -------------------------------|--------------
 D::D( const D& ) 				| Copy constructor.
 D::~D() 						| Destructor.
 bool D::empty() const 			| True if domain is empty.
 bool D::is_divisible() const 	| True if domain can be partitioned into two subdomains.
 D::D( D& d, split ) 			| Split d into two subdomains.
 D::value_type 					| Item type
 D::reference 					| Item reference type
 D::const_reference 			| Item const reference type
 D::difference_type 			| Type for difference of two iterators
 D::iterator 					| Iterator type for domain
 D::iterator D::begin(  ) const	| First item in domain.
 D::iterator D::end(  ) const 	| One past last item in domain. 
	



Additional requirements for  Domain type D 

 Pseudo-Signature  				| Semantics
 -------------------------------|-------------
 D::iterator begin(  D const& )	| First item in domain.
 D::iterator end(  D const&)	| One past  last item in domain. 
 size_t size( D const&)			| number of items in the domain
 D const & parent()const		| Parent domain

Requirements for  Domain type D as a geometric object, which could be a @ref concept_simplex or a chain of polytopes. 

 Pseudo-Signature  				| Semantics
 -------------------------------|-------------
 unsigned int ndims 			| number of dimensions of domain D
 PD boundary(  D const& )		| Boundary of domain D, PD::ndims=D::ndims-1.
 D const & parent()const		| Parent domain
 boundbox() const				| boundbox on _this_ coordinates system
 cartesian_boundbox() const		| boundbox on this _Cartesian_ coordinates system
 

##  See Also
- @ref concept_manifold
- @ref concept_topology
- @ref concept_simplex
- \subpage domain_iterator
- \subpage block_domain	 