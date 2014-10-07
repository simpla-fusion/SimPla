Expression Template Library {#FETL}
==========================
## Summary

  operation of fields/forms \f$\Omega^n\f$ on N-dimensional manifold \f$M\f$.
  
## Requirements

   
### Basic Algebra

  Pseudo-Signature  			| Semantics
 -------------------------------|--------------
 \f$\Omega^n\f$ operator-(\f$\Omega^n\f$  )	| negate operation
 \f$\Omega^n\f$ operator+(\f$\Omega^n\f$  )	| positive operation
 \f$\Omega^n\f$ operator+(\f$\Omega^n\f$ ,\f$\Omega^n\f$ )	| add  
 \f$\Omega^n\f$ operator-(\f$\Omega^n\f$ ,\f$\Omega^n\f$ )	| subtract  
 \f$\Omega^n\f$ operator*(\f$\Omega^n\f$ ,Scalar )	| multiply  
 \f$\Omega^n\f$ operator*(Scalar,\f$\Omega^n\f$ )	| multiply  
 \f$\Omega^n\f$ operator/(\f$\Omega^n\f$ ,Scalar )	| divide  
  
 
 
### Exterior Algebra
  Pseudo-Signature  			| Semantics
 -------------------------------|--------------
 \f$\Omega^{N-n}\f$ HodgeStar(\f$\Omega^n\f$ )	| hodge star, abbr. operator *
 \f$\Omega^{n-1}\f$ ExteriorDerivative(\f$\Omega^n\f$ )	| Exterior Derivative, abbr. d
 \f$\Omega^{n+1}\f$ Codifferential(\f$\Omega^n\f$ )	| Codifferential Derivative, abbr. delta
 \f$\Omega^{m+n}\f$ Wedge(\f$\Omega^m\f$ ,\f$\Omega^m\f$  )	| wedge product, abbr. operator^
 \f$\Omega^{n-1}\f$ InteriorProduct(Vector Field ,\f$\Omega^n\f$  )	| interior product, abbr. iv
 \f$\Omega^{N}\f$ InnerProduct(\f$\Omega^m\f$ ,\f$\Omega^m\f$ ) | inner product, 
 
### Three-dimensional Vector Algebra 

  Pseudo-Signature  			| Semantics
 -------------------------------|--------------
 \f$\Omega^{1}\f$ Grad(\f$\Omega^0\f$ )		| Grad  
 \f$\Omega^{0}\f$ Diverge(\f$\Omega^1\f$ )	| Diverge  
 \f$\Omega^{2}\f$ Curl(\f$\Omega^1\f$ )		| Curl  
 \f$\Omega^{1}\f$ Curl(\f$\Omega^2\f$ )		| Curl  
 
### Map between vector form  and scalar form

 Pseudo-Signature  			| Semantics
 -------------------------------|--------------
 \f$\Omega^{1}\f$ MapTo(\f${V\Omega}^0\f$ )	| map vector 0-form to 1-form  
 \f${V\Omega}^{0}\f$ MapTo(\f$\Omega^1\f$ )	| map 1-form to vector 0-form  

\f{eqnarray*}{
R &=& 1-\sum_{s}\frac{\omega_{ps}^{2}}{\omega\left(\omega+\Omega_{s}\right)}\\
L &=& 1+\sum_{s}\frac{\omega_{ps}^{2}}{\omega\left(\omega-\Omega_{s}\right)}\\
P &=& 1-\sum_{s}\frac{\omega_{ps}^{2}}{\omega^{2}}
\f}
 