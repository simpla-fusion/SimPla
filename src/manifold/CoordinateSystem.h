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
 * @addtogroup   CoordinateSystem Differential CoordinateSystem
 * @{
 *    @brief  CoordinateSystem
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
 * CoordinateSystem
 */
template<typename TMesh, template<typename...> class ...Policies>
class CoordinateSystem
        : public TMesh,
          public Policies<TMesh> ...
{
    typedef CoordinateSystem<TMesh, Policies ...> this_type;

public:

    typedef TMesh mesh_type;

    CoordinateSystem() : Policies<mesh_type>(static_cast<mesh_type &>(*this))... { }

    virtual ~CoordinateSystem() { }

    CoordinateSystem(this_type const &other) : TMesh(other), Policies<mesh_type>(static_cast<mesh_type &>(*this))... { }

    this_type &operator=(const this_type &other) = delete;


    virtual bool is_a(std::type_info const &info) const { return typeid(this_type) == info || TMesh::is_a(info); }

    virtual std::string get_class_name() const { return class_name(); }

    static std::string class_name()
    {
        return "CoordinateSystem<" + traits::type_id_list<mesh_type, Policies<mesh_type>...>::name() + " > ";
    }

private:
    template<typename T> void deploy_dispatch() { T::deploy(); }

    template<typename T, typename T1, typename ...Others>
    void deploy_dispatch()
    {
        deploy_dispatch<T>();
        deploy_dispatch<T1, Others...>();
    }


public:
    virtual void deploy()
    {
        mesh_type::deploy();
        deploy_dispatch<Policies<mesh_type>...>();
        this->touch();
    }

    template<typename TDict>
    void setup(TDict const &dict)
    {
        mesh_type::setup(dict);
    }


    virtual std::ostream &print(std::ostream &os, int indent = 1) const
    {
        os << std::setw(indent) << " " << "CoordinateSystem = {" << std::endl;

        this_type::mesh_type::print(os, indent + 1);
        this_type::metric_policy::print(os, indent + 1);
        this_type::calculus_policy::print(os, indent + 1);
        this_type::interpolate_policy::print(os, indent + 1);

        os << std::setw(indent) << " " << "}  -- CoordinateSystem " << std::endl;


        return os;
    }

    virtual std::shared_ptr<mesh::MeshBase> clone() const
    {
        return std::dynamic_pointer_cast<mesh::MeshBase>(std::make_shared<this_type>(*this));
    };
//    virtual data_model::DataSet grid_vertices() const
//    {
//        auto ds = this->storage_policy::template dataset<point_type, VERTEX>();
//
//        ds.m_data = sp_alloc_memory(ds.memory_space.size() * sizeof(point_type));
//
//        point_type *p = reinterpret_cast<point_type *>(ds.m_data.get());
//
//        parallel::parallel_for(
//                this->template entity_id_range<VERTEX>(),
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





}; //class CoordinateSystem

/* @}@} */

}}//namespace simpla::manifold

#endif /* CORE_MANIFOLD_H_ */
