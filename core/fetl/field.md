Field {#concept_field}
=================================
## Summary
 \note A _Field_ assigns a scalar/vector/tensor to each point of a mathematical space (typically a Euclidean space or manifold).
 
 \note A _Field_ is a map / function \f$y=f(x)\f$, where \f$x\in D\f$ is coordinates defined in _domain_ \f$D\f$, and \f$y\f$ is a scalar/vector/tensor.
   
## Member types
 Member type	 				| Semantics
 -------------------------------|--------------
 coordinates_type				| Datatype of coordinates
 index_type						| Datatype of of grid points index
 value_type 					| Datatype of value 
 domain_type					| Domain  
 storage_type					| container type

 
## Member functions

###Constructor
 
 Pseudo-Signature 	 			| Semantics
 -------------------------------|--------------
 Field()						| Default constructor
 ~Field() 						| destructor.
 Field( const Field& ) 			| copy constructor.
 Field( Field && ) 				| move constructor.

 
### Domain &  Split

 Pseudo-Signature 	 			| Semantics
 -------------------------------|--------------
 Field( Domain & D ) 			| Construct a field on domain \f$D\f$.
 Field( Field &r,split)			| Split field into two part,  see @ref concept_domain
 domain_type const &domain() const 	| Get define domain of field
 void domain(domain_type cont&)  	| Reset define domain of field
 Field split(domain_type d)			| Sub-field on  domain \f$D \cap D_0\f$
 Field boundary()				| Sub-field on  boundary \f${\partial D}_0\f$
 
###   Capacity
 Pseudo-Signature 	 				| Semantics
 -------------------------------|--------------
 bool empty()   				| _true_ if memory is not allocated.
 allocate()						| allocate memory
 data()							| direct access to the underlying memory
 clear()						| set value to zero, allocate memory if empty() is _true_
 
 
### Element access	 
 Pseudo-Signature 				| Semantics
 -------------------------------|--------------
 value_type & at(index_type s)   			| access element on the grid points _s_ with bounds checking 
 value_type & operator[](index_type s)   	| access element on the grid points _s_
 field_value_type  operator()(coordiantes_type x) const | field value on coordinates \f$x\f$, which is interpolated from discrete points
 
### Assignment  
  Pseudo-Signature 	 				| Semantics
 -------------------------------|--------------
 Field & operator=(Function const & f)  	| assign values as \f$y[s]=f(x)\f$
 Field & operator=(FieldExpression const &)	| Assign operation, 
 Field operator=( const Field& )| copy-assignment operator.
 Field operator=( Field&& )		| move-assignment operator.
 Field & operator+=(Expression const &) | Assign operation +
 Field & operator-=(Expression const &) | Assign operation -
 Field & operator/=(Expression const &) | Assign operation /
 Field & operator*=(Expression const &) | Assign operation *
  
## Non-member functions
 Pseudo-Signature  				| Semantics
 -------------------------------|--------------
 swap(Field &,Field&)			| swap
 
## See also
 - \subpage FETL
 - @ref concept_manifold
 - @ref concept_domain