/**
 *  @file diff_geometry.h
 *
 *  Created on:@date  2015年1月9日
 *   @author: salmon
 */

#ifndef CORE_MESH_DIFF_GEOMETRY_H_
#define CORE_MESH_DIFF_GEOMETRY_H_

namespace simpla
{

/**
 *  @addtogroup diff_geo Differential Geometry
 *  @brief Differential geometry is a mathematical discipline that
 *  uses the techniques of @ref diff_calculus,@ref integral_calculus,
 *  @ref linear_algebra and @ref multilinear_algebra to study problems in geometry.
 *  @details Differential geometry is a mathematical discipline that
 *  uses the techniques of differential calculus, integral calculus,
 *  linear algebra and multilinear algebra to study problems in geometry.
 *   The theory of plane and space curves and surfaces in the three-dimensional
 *    Euclidean space formed the basis for development of differential
 *     geometry during the 18th century and the 19th century.
 */
/** @ingroup diff_geo
 *  @addtogroup diff_form Differential Form
 *  @{
 *  @brief In the mathematical fields of @ref diff_geo and tensor calculus,
 *   differential forms are an approach to multivariable calculus that
 *     is independent of coordinates. --wiki
 *
 *
 * ## Summary
 * \note Let \f$M\f$ be a _smooth manifold_. A _differential form_ of degree \f$k\f$ is
 *  a smooth section of the \f$k\f$th exterior power of the cotangent bundle of \f$M\f$.
 *  At any point \f$p \in M\f$, a k-form \f$\beta\f$ defines an alternating multilinear map
 * \f[
 *   \beta_p\colon T_p M\times \cdots \times T_p M \to \mathbb{R}
 * \f]
 * (with k factors of \f$T_p M\f$ in the product), where TpM is the tangent space to \f$M\f$ at \f$p\f$.
 *  Equivalently, \f$\beta\f$ is a totally antisymetric covariant tensor field of rank \f$k\f$.
 *
 *  Differential form is a field
 *
 * ## Requirements
 *
 *  @}
 *
 */


/**
 * @ingroup diff_geo
 * \addtogroup  manifold Manifold
 *  @{
 *    \brief   Discrete spatial-temporal space
 *
 * ## Summary
 *  Manifold
 * ## Requirements
 *
 ~~~~~~~~~~~~~{.cpp}
 template <typename Geometry,template<typename> class Policy1,template<typename> class Policy2>
 class Mesh:
 public Geometry,
 public Policy1<Geometry>,
 public Policy2<Geometry>
 {
 .....
 };
 ~~~~~~~~~~~~~
 * The following table lists requirements for a Mesh type `M`,
 *
 *  Pseudo-Signature  		| Semantics
 *  ------------------------|-------------
 *  `M( const M& )` 		| Copy constructor.
 *  `~M()` 				    | Destructor.
 *  `geometry_type`		    | Geometry type of manifold, which describes coordinates and metric
 *  `topology_type`		    | Topology structure of manifold,   topology of grid points
 *  `coordiantes_type` 	    | data type of coordinates, i.e. nTuple<3,Real>
 *  `index_type`			| data type of the index of grid points, i.e. unsigned long
 *  `Domain  domain()`	    | Root domain of manifold
 *
 *
 * Mesh policy concept {#concept_manifold_policy}
 * ================================================
 *   Poilcies define the behavior of manifold , such as  interpolator or calculus;
 ~~~~~~~~~~~~~{.cpp}
 template <typename Geometry > class P;
 ~~~~~~~~~~~~~
 *
 *  The following table lists requirements for a Mesh policy type `P`,
 *
 *  Pseudo-Signature  	   | Semantics
 *  -----------------------|-------------
 *  `P( Geometry  & )` 	   | Constructor.
 *  `P( P const  & )`	   | Copy constructor.
 *  `~P( )` 			   | Copy Destructor.
 *
 * ## Interpolator policy
 *   Interpolator, map between discrete space and continue space, i.e. Gather & Scatter
 *
 *    Pseudo-Signature  	   | Semantics
 *  ---------------------------|-------------
 *  `gather(field_type const &f, coordinate_type x  )` 	    | gather data from `f` at coordinates `x`.
 *  `scatter(field_type &f, coordinate_type x ,value_type v)` 	| scatter `v` to field  `f` at coordinates `x`.
 *
 * ## Calculus  policy
 *  Define calculus operation of  fields on the manifold, such  as algebra or differential calculus.
 *  Differential calculus scheme , i.e. FDM,FVM,FEM,DG ....
 *
 *
 *  Pseudo-Signature  		| Semantics
 *  ------------------------|-------------
 *  `calculate(TOP op, field_type const &f, field_type const &f, index_type s ) `	| `calculate`  binary operation `op` at grid point `s`.
 *  `calculate(TOP op, field_type const &f,  index_type s )` 	| `calculate`  unary operation  `op`  at grid point `s`.
 *
 *  *
 *  @}
 */

template<size_t IFORM, typename TG, typename ...Policies> class Manifold;

}  // namespace simpla

#endif /* CORE_MESH_DIFF_GEOMETRY_H_ */
