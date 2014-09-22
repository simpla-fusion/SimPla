Field {#concept_field}
=================================
## Summary
 \note A _Field_ assigns a scalar/vector/tensor to each point of a mathematical space (typically a Euclidean space or manifold).
 
 \note A _Field_ is a map / function \f$y=f(x)\f$, where \f$x\in D\f$ is coordinates defined in _domain_ \f$D\f$, and \f$y\f$ is a scalar/vector/tensor.
   
## Requirements

The following table lists requirements for  Field expression F  .

 Pseudo-Signature  				| Semantics
 -------------------------------|--------------
 coordinates_type				| Datatype of coordinates
 index_type						| Datatype of of grid points index
 value_type 					| Datatype of value 
 manifold_type					| manifold
 domain_type					| Domain on manifold
 Domain const &domain() const 	| Get define domain of field
 value_type operator()(coordiantes_type x) const | field value on coordinates \f$x\f$, which is interpolated from discrete points
 
 The following table lists requirements for  Field with real data.
 
 Pseudo-Signature  				| Semantics
 -------------------------------|--------------
 Field( Domain & D ) 				| Construct a field on domain \f$D\f$.
 Field( const Field& ) 				| Copy constructor.
 ~Field() 							| Destructor.
 Field(Field &r,split)				| Split field into two part,  see @ref concept_domain
 Field subset(Domain d)				| Sub-field on  domain \f$D \cap D_0\f$
 Field boundary()					| Sub-field on  boundary \f${\partial D}_0\f$
 value_type & at(index_type s)   			| access element on the grid points _s_ with bounds checking 
 value_type & operator[](index_type s)   	| access element on the grid points _s_
 bool empty()   							| _true_ if memory is not allocated.
 data()										| direct access to the underlying memory
 clear()									| set value to zero, allocate memory if empty() is _true_
 erase()									| deallocate memory
 fill(value_type v)							| set value to v
 Field & operator=(Function const & f)  	| assign values as \f$y[s]=f(x)\f$
 Field & operator=(FieldExpression const &)	| Assign operation, 
 Field & operator+=(Expression const &) | Assign operation +
 Field & operator-=(Expression const &) | Assign operation -
 Field & operator/=(Expression const &) | Assign operation /
 Field & operator*=(Expression const &) | Assign operation *
  
## Non-member functions
 
 
## See also
 - \subpage FETL
 - @ref concept_manifold
 - @ref concept_domain