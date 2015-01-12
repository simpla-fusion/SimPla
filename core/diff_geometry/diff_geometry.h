/**
 *  @file diff_geo.h
 *
 *  Created on: 2015年1月9日
 *      Author: salmon
 */

#ifndef CORE_DIFF_GEOMETRY_DIFF_GEOMETRY_H_
#define CORE_DIFF_GEOMETRY_DIFF_GEOMETRY_H_

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

}  // namespace simpla

#endif /* CORE_DIFF_GEOMETRY_DIFF_GEOMETRY_H_ */
