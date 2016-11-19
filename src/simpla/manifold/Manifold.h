/**
 * @file manifold.h
 *
 * @date 2015-2-9
 * @author salmon
 */

#ifndef CORE_MANIFOLD_H_
#define CORE_MANIFOLD_H_

#include <iostream>
#include <memory>
#include <simpla/toolbox/Log.h>
#include <simpla/toolbox/macro.h>
#include <simpla/toolbox/nTuple.h>
#include <simpla/toolbox/design_pattern/Observer.h>
#include <simpla/concept/Configurable.h>

#include <simpla/mesh/MeshBlock.h>
#include <simpla/mesh/Attribute.h>

#include "ManifoldTraits.h"


namespace simpla { namespace manifold
{


/**
 * @defgroup diff_geo Differential Geometry
 * @brief collection of get_mesh and differential scheme
 *
 * @detail
 * ## Summary
 * Differential geometry is a mathematical discipline that
 *  uses the techniques of differential calculus, integral calculus,
 *  linear algebra and multilinear algebra to study problems in geometry.
 *   The theory of plane and space curves and surfaces in the three-dimensional
 *    Euclidean space formed the basis for development of differential
 *     geometry during the 18th century and the 19th century.
 */
/**
 * @ingroup diff_geo
 * @addtogroup   Manifold Differential Manifold
 * @{
 *    @brief  Manifold
 *

 * ## Requirements
 *
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
 * The following table lists requirements for a mesh type `M`,
 *
 *  Pseudo-Signature  		| Semantics
 *  ------------------------|-------------
 *  `M( const M& )` 		| Copy constructor.
 *  `~M()` 				    | Destructor.
 *  `mesh_type`		    | BaseManifold type of geometry, which describes coordinates and Metric
 *  `mesh_type`		    | Topology structure of geometry,   Topology of grid points
 *  `coordiantes_type` 	    | m_data type of coordinates, i.e. nTuple<3,Real>
 *  `index_type`			| m_data type of the index of grid points, i.e. unsigned long
 *  `Domain  domain()`	    | Root domain of geometry
 *
 *
 * mesh policy concept {#concept_manifold_policy}
 * ================================================
 *   Poilcies define the behavior of geometry , such as  interpolate or calculus;
 ~~~~~~~~~~~~~{.cpp}
 template <typename BaseManifold > class P;
 ~~~~~~~~~~~~~
 *
 *  The following table lists requirements for a get_mesh policy type `P`,
 *
 *  Pseudo-Signature  	   | Semantics
 *  -----------------------|-------------
 *  `P( BaseManifold  & )` 	   | Constructor.
 *  `P( P const  & )`	   | Copy constructor.
 *  `~P( )` 			   | Copy Destructor.
 *
 * ## Interpolator policy
 *   Interpolator, map between discrete space and continue space, i.e. Gather & Scatter
 *
 *    Pseudo-Signature  	   | Semantics
 *  ---------------------------|-------------
 *  `gather(field_type const &f, coordinate_tuple x  )` 	    | gather m_data from `f` at coordinates `x`.
 *  `scatter(field_type &f, coordinate_tuple x ,value_type v)` 	| scatter `v` to field  `f` at coordinates `x`.
 *
 * ## Calculus  policy
 *  Define calculus operation of  fields on the geometry, such  as algebra or differential calculus.
 *  Differential calculus scheme , i.e. FDM,FVM,FEM,DG ....
 *
 *
 *  Pseudo-Signature  		| Semantics
 *  ------------------------|-------------
 *  `diff_scheme(TOP op, field_type const &f, field_type const &f, index_type s ) `	| `diff_scheme`  binary operation `op` at grid point `s`.
 *  `diff_scheme(TOP op, field_type const &f,  index_type s )` 	| `diff_scheme`  unary operation  `op`  at grid point `s`.
 *
 *
 *  ## Differential Form
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

 */

/**
 * Manifold
 */


template<typename TGeo, template<typename...> class ...Policies>
class Manifold :
        public mesh::AttributeHolder,
        public Policies<TGeo> ...
{
    typedef Manifold<TGeo, Policies ...> this_type;

public:
    typedef TGeo mesh_type;
    typedef TGeo geometry_type;
    geometry_type m_geo_;

    Manifold() : Policies<TGeo>(&m_geo_)... { m_geo_.connect(this); }

    virtual ~Manifold() {}

    Manifold(this_type const &other) = delete;

    this_type &operator=(const this_type &other) = delete;

    virtual bool is_a(std::type_info const &info) const { return typeid(this_type) == info; }

    virtual std::string get_class_name() const { return name(); }

    virtual std::string name() const
    {
        return "Manifold<" + traits::type_id_list<geometry_type, Policies<geometry_type>...>::name() + " > ";
    }

    geometry_type const &geometry() const { return m_geo_; }


    virtual void deploy()
    {
        this_type::calculus_policy::deploy();
        this_type::interpolate_policy::deploy();
    }


    virtual std::ostream &print(std::ostream &os, int indent = 1) const
    {
//        m_geo_->print(os, indent + 1);
        this_type::calculus_policy::print(os, indent + 1);
        this_type::interpolate_policy::print(os, indent + 1);
        return os;
    }


    virtual void destroy()
    {

    }


private:
//    std::shared_ptr<mesh::Atlas> m_atlas_;


}; //class Manifold

/* @}@} */

}}//namespace simpla::manifold

#endif /* CORE_MANIFOLD_H_ */
