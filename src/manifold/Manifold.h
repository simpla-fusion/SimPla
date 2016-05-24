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
#include "../gtl/Log.h"
#include "../gtl/macro.h"
#include "../gtl/nTuple.h"

#include "../model/CoordinateSystem.h"
#include "ManifoldTraits.h"


namespace simpla
{


/**
 * @defgroup diff_geo Differential Geometry
 * @brief collection of mesh and differential scheme
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
 *  `coordiantes_type` 	    | data type of coordinates, i.e. nTuple<3,Real>
 *  `index_type`			| data type of the index of grid points, i.e. unsigned long
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
 *  The following table lists requirements for a mesh policy type `P`,
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
 *  `gather(field_type const &f, coordinate_tuple x  )` 	    | gather data from `f` at coordinates `x`.
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
template<typename TMesh, template<typename> class ...Policies>
class Manifold
        : public TMesh,
          public Policies<TMesh> ...
{
    typedef Manifold<TMesh, Policies ...> this_type;

public:

    typedef TMesh mesh_type;

    using typename mesh_type::id_type;
    using typename mesh_type::range_type;
    using typename mesh_type::box_type;


    using typename mesh_type::scalar_type;
    using typename mesh_type::point_type;
    using typename mesh_type::vector_type;

    using mesh_type::ndims;
    using mesh_type::volume;
    using mesh_type::dual_volume;
    using mesh_type::inv_volume;
    using mesh_type::inv_dual_volume;
    using mesh_type::inner_product;

    Manifold() : Policies<mesh_type>(dynamic_cast<mesh_type &>(*this))... { }

    virtual ~Manifold() { }

    Manifold(this_type const &m) = delete;

    this_type &operator=(const this_type &other) = delete;

    virtual this_type &self() { return (*this); }

    virtual this_type const &self() const { return (*this); }


public:

    virtual bool is_a(std::type_info const &info) const { return typeid(this_type) == info || TMesh::is_a(info); }

    virtual std::string get_class_name() const { return "Manifold<" + mesh_type::get_class_name() + " ... >"; }

    virtual void deploy()
    {
        mesh_type::deploy();
        this->touch();
    }


    virtual std::ostream &print(std::ostream &os, int indent = 0) const
    {
//        os << std::setw(indent + 1) << " " << "Mesh = {";
//
//        os << std::setw(indent + 1) << " dt = " << m_dt_ << "," << std::endl;
//
//
//        TMesh::print(os, indent + 1);
//        properties().print(os, indent + 1);
//
//        os << "}  -- Mesh " << std::endl;

        mesh_type::print(os, indent);
        return os;
    }


//    virtual data_model::DataSet grid_vertices() const
//    {
//        auto ds = this->storage_policy::template data_set<point_type, VERTEX>();
//
//        ds.data = sp_alloc_memory(ds.memory_space.size() * sizeof(point_type));
//
//        point_type *p = reinterpret_cast<point_type *>(ds.data.get());
//
//        parallel::parallel_for(
//                this->template range<VERTEX>(),
//                [&](range_type const &r)
//                {
//                    for (auto const &s: r)
//                    {
//                        p[this->hash(s)] = this->map_to_cartesian(this->point(s));
//                    }
//                }
//        );
//
//        return std::move(ds);
//
//    };





}; //class Manifold

/* @}@} */

}//namespace simpla

#endif /* CORE_MANIFOLD_H_ */
