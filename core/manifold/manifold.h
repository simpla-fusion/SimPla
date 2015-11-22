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
#include "../gtl/macro.h"
#include "../gtl/ntuple.h"
#include "diff_scheme/diff_scheme.h"
#include "interpolate/interpolate.h"

namespace simpla
{


template<typename ...> struct Domain;
template<typename ...> struct Field;
template<typename ...> struct Expression;


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
 * @addtogroup   manifold Differential Manifold
 * @{
 *    @brief  manifold
 *

 * ## Requirements
 *
 ~~~~~~~~~~~~~{.cpp}
 template <typename BaseManifold,template<typename> class Policy1,template<typename> class Policy2>
 class Mesh:
 public BaseManifold,
 public Policy1<BaseManifold>,
 public Policy2<BaseManifold>
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
 *  `base_manifold_type`		    | BaseManifold type of geometry, which describes coordinates and Metric
 *  `mesh_type`		    | Topology structure of geometry,   Topology of grid points
 *  `coordiantes_type` 	    | data type of coordinates, i.e. nTuple<3,Real>
 *  `index_type`			| data type of the index of grid points, i.e. unsigned long
 *  `Domain  domain()`	    | Root domain of geometry
 *
 *
 * Mesh policy concept {#concept_manifold_policy}
 * ================================================
 *   Poilcies define the behavior of geometry , such as  interpolate or calculus;
 ~~~~~~~~~~~~~{.cpp}
 template <typename BaseManifold > class P;
 ~~~~~~~~~~~~~
 *
 *  The following table lists requirements for a Mesh policy type `P`,
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

template<typename ...> class Manifold;

/**
 * Manifold
 */
template<typename TMesh, typename ...Policies>
class Manifold<TMesh, Policies ...>
        : public TMesh, public Policies ...
{
    typedef TMesh base_manifold_type;

public:


    typedef TMesh mesh_type;

    typedef Manifold<mesh_type, Policies ...> this_type;

    typedef geometry::traits::coordinate_system_t<mesh_type> coordinates_system_type;

    typedef geometry::traits::scalar_type_t<coordinates_system_type> scalar_type;

    typedef geometry::traits::point_type_t<coordinates_system_type> point_type;

    typedef geometry::traits::vector_type_t<coordinates_system_type> vector_type;

    using mesh_type::ndims;
    using mesh_type::volume;
    using mesh_type::dual_volume;
    using mesh_type::inv_volume;
    using mesh_type::inv_dual_volume;
    using mesh_type::inner_product;

    Manifold() : Policies(static_cast<base_manifold_type &>(*this))... { }

    virtual ~Manifold() { }

    Manifold(this_type const &other) : base_manifold_type(other), Policies(other)... { }

    this_type &operator=(const this_type &other)
    {
        this_type(other).swap(*this);
        return *this;
    }


private:

    TEMPLATE_DISPATCH_DEFAULT(load)

    TEMPLATE_DISPATCH_DEFAULT(deploy)

    TEMPLATE_DISPATCH(swap, inline,)

    TEMPLATE_DISPATCH(print, inline, const)

public:
    void swap(const this_type &other) { _dispatch_swap<base_manifold_type, Policies...>(other); }

    template<typename TDict>
    void load(TDict const &dict)
    {
        auto d = dict["Manifold"];
        _dispatch_load<base_manifold_type, Policies...>(d);
    }


    void deploy()
    {
        mesh_type::deploy();
        _dispatch_deploy<base_manifold_type, Policies...>();
    }

    template<typename OS>
    OS &print(OS &os) const
    {
        os << "Manifold={" << std::endl;
        _dispatch_print<base_manifold_type, Policies...>(os);
        os << "}, # Manifold " << std::endl;
        return os;
    }

    template<typename T>
    inline constexpr T access(T const &v, id_t s) const { return v; }

    template<typename T, int ...N>
    inline constexpr nTuple<T, N...> const &
    access(nTuple<T, N...> const &v, id_t s) const { return v; }


    template<typename ...T>
    inline traits::primary_type_t<nTuple<Expression<T...>>>
    access(nTuple<Expression<T...>> const &v, id_t s) const
    {
        traits::primary_type_t<nTuple<Expression<T...> > > res;
        res = v;
        return std::move(res);
    }

    template<typename TV, typename ...Others>
    inline TV &access(Field<TV, Others...> &f, id_type s) const
    {
        return f[s];
    }


    template<typename TV, typename ...Others>
    inline TV access(Field<TV, Others...> const &f, id_type s) const
    {
        return f[s];
    }

    template<typename ...TD>
    inline auto access(Field<Expression<TD...> > const &f, id_type s) const
    DECL_RET_TYPE((this->calculus_policy::eval(f, s)))

    template<typename TOP, typename T, typename ...Args>
    void for_each(TOP const &op, T *self, Args &&... args) const
    {
        assert(self != nullptr);

        constexpr int iform = traits::iform<T>::value;

        self->deploy();
        for (auto const &s:this->template range<iform>())
        {
            op(access(*self, s), access(std::forward<Args>(args), s)...);
        }
//        for (auto const &m:m_patch_ghost_)
//        {
//            m.action(op, self, std::forward<Args>(args)...);
//        }
//
//        auto request = parallel_policy::sync(self);
//
//        for (auto const &m:m_patch_)
//        {
//            m.action(op, self, std::forward<Args>(args)...);
//        }
//        parallel_policy::wait(request);
    }

    template<typename TOP, typename T, typename ...Args>
    void for_each(TOP const &op, T const &self, Args &&... args) const
    {
        constexpr int iform = traits::iform<T>::value;
        for (auto const &s:this->template range<iform>())
        {
            op(access(self, s), access(std::forward<Args>(args), s)...);
        }
    }

//    typedef mesh::Patch<mesh_type> patch_type;
//    std::map<id_type, patch_type> m_patch_ghost_;
//    std::map<id_type, patch_type> m_patch_;


}; //class Manifold

/* @}@} */

}//namespace simpla

#endif /* CORE_MANIFOLD_H_ */
