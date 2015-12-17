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
#include "../parallel/parallel.h"
#include "../gtl/utilities/log.h"
#include "../gtl/macro.h"
#include "../gtl/ntuple.h"
#include "../geometry/coordinate_system.h"
#include "../gtl/utilities/memory_pool.h"
#include "../base/attribute.h"

#include "manifold_traits.h"


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
 *  `mesh_type`		    | BaseManifold type of geometry, which describes coordinates and Metric
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


    typedef geometry::traits::coordinate_system_t<mesh_type> coordinates_system_type;

    typedef geometry::traits::scalar_type_t<coordinates_system_type> scalar_type;

    typedef geometry::traits::point_type_t<coordinates_system_type> point_type;

    typedef geometry::traits::vector_type_t<coordinates_system_type> vector_type;


    typedef typename mesh_type::id_type id_type;
    typedef typename mesh_type::range_type range_type;
    typedef typename mesh_type::box_type box_type;

    using mesh_type::ndims;
    using mesh_type::volume;
    using mesh_type::dual_volume;
    using mesh_type::inv_volume;
    using mesh_type::inv_dual_volume;
    using mesh_type::inner_product;

    Manifold() : Policies<mesh_type>(dynamic_cast<mesh_type &>(*this))... { }

    Manifold(this_type const &m) : mesh_type(m), Policies<mesh_type>(
            dynamic_cast<mesh_type &>(*this))... { }

    virtual ~Manifold()
    {
    }

    virtual this_type &self() { return (*this); }

    virtual this_type const &self() const { return (*this); }


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

    void swap(const this_type &other) { _dispatch_swap<mesh_type, Policies<TMesh>...>(other); }

    template<typename TDict>
    void load(TDict const &dict)
    {
        TRY_IT((_dispatch_load<mesh_type, Policies<TMesh>...>(dict["Mesh"])));
    }

    void deploy()
    {
        mesh_type::deploy();
        _dispatch_deploy<mesh_type, Policies<TMesh>...>();
    }


    template<typename OS>
    OS &print(OS &os) const
    {
        os << "Mesh={" << std::endl;
        _dispatch_print<mesh_type, Policies<TMesh>...>(os);
        os << "}, # Mesh " << std::endl;
        return os;
    }


//    template<int IFORM, typename TOP, typename TF, typename   ...Args>
//    void apply(TOP const &op, TF &f, Args &&... args) const
//    {
////        static constexpr int IFORM =
////                traits::iform<typename
////                traits::unpack_type<0, Args...>::type>::value;
//
//        this->parallel_policy::template update<IFORM>(
//                [&](typename mesh_type::range_type const &r)
//                {
//                    for (auto const &s:r)
//                    {
//                        op(access(f, s), access(std::forward<Args>(args), s)...);
//                    }
//                }, f
//
//        );
//    }


    virtual DataSet grid_vertices() const
    {
        auto ds = this->storage_policy::template dataset<point_type, VERTEX>();

//        ds->deploy();
//
//        parallel::parallel_for(
//                this->template range<VERTEX>(),
//                [&](range_type const &r)
//                {
//                    for (auto const &s: r)
//                    {
//                        this->template at<point_type>(*ds, s) =
//                                this->map_to_cartesian(this->point(s));
//                        //   this->template at<point_type>(ds.data, s) = this->point(s);
//                    }
//                }
//        );

        return ds;

    };

    std::shared_ptr<this_type> refinement(int ratio, box_type const &box) const
    {
        auto res = std::make_shared<this_type>(*this);
        auto idx_box = res->index_box(box);
        res->box(box);
        res->dt(dt() / ratio);
        res->dimensions((std::get<1>(idx_box) - std::get<0>(idx_box)) * ratio);
        res->deploy();
        return res;
    }


    void next_time_step() { m_time_ += m_dt_; }

    double time() const { return m_time_; }

    void time(double t) { m_time_ = t; }

    double dt() const { return m_dt_; }

    void dt(double p_dt) { m_dt_ = p_dt; }


private:
    double m_dt_;
    double m_time_;


private:
    class AttributeInternal_;

    std::map<std::string, std::weak_ptr<AttributeInternal_>> m_registered_attributes_;

public:

    template<typename TV, int IFORM> class Attribute;

    template<typename TV, int IFORM>
    std::shared_ptr<Attribute<TV, IFORM>> create_attribute(std::string const &name)
    {
        auto res = std::make_shared<Attribute<TV, IFORM>>(*this);

        if (name != "") { enroll(name, std::dynamic_pointer_cast<AttributeInternal_>(res)); }

        return res;
    }

    template<typename TV, int IFORM>
    std::shared_ptr<Attribute<TV, IFORM>> create_attribute() const
    {
        return std::make_shared<Attribute<TV, IFORM>>(*this);
    }

    template<typename TF>
    void enroll(std::string const &name, std::shared_ptr<TF> p)
    {
        m_registered_attributes_.insert(
                std::make_pair(name,
                               std::dynamic_pointer_cast<AttributeInternal_>(p)));
    };


}; //class Manifold

template<typename TMesh, template<typename> class ...Policies>
class Manifold<TMesh, Policies ...>::AttributeInternal_ : public AttributeBase
{
    typedef Manifold<TMesh, Policies ...> mesh_type;

    typedef typename mesh_type::id_type id_type;

public:
    AttributeInternal_(mesh_type const &m) : m_mesh_(&m) { }

    AttributeInternal_(AttributeInternal_ const &other) : m_mesh_(other.m_mesh_) { }

    AttributeInternal_(AttributeInternal_ &&other) : m_mesh_(other.m_mesh_) { }

    virtual ~AttributeInternal_() { };

    void swap(AttributeInternal_ &other)
    {
        std::swap(m_mesh_, other.m_mesh_);
    }

    AttributeInternal_ &operator=(AttributeInternal_ const &other)
    {
        AttributeInternal_(other).swap(*this);
        return *this;
    }

    virtual DataSet dataset() const = 0;

    virtual int center_type() const { return 0; };

    virtual int rank() const { return 0; };

    virtual int extent(int i) const { return 1; }


    mesh_type const &mesh() const { return *m_mesh_; }

private:
    mesh_type const *m_mesh_;

};

template<typename TMesh, template<typename> class ...Policies>
template<typename TV, int IFORM>
class Manifold<TMesh, Policies ...>::Attribute : public AttributeInternal_
{

private:
    typedef Manifold<TMesh, Policies ...> mesh_type;

    typedef typename mesh_type::id_type id_type;


public:
    using AttributeInternal_::mesh;
    typedef TV value_type;

    Attribute(mesh_type const &m, std::string const &) : AttributeInternal_(m), m_data_(nullptr) { }

    Attribute(mesh_type const &m) : AttributeInternal_(m), m_data_(nullptr) { }

    Attribute(Attribute const &other) : AttributeInternal_(other), m_data_(other.m_data_) { }

    Attribute(Attribute &&other) : AttributeInternal_(other), m_data_(other.m_data_) { }

    virtual ~Attribute() { }

    Attribute &operator=(Attribute const &other)
    {
        AttributeInternal_(other).swap(*this);
        return *this;
    }

    void swap(Attribute &other)
    {
        AttributeInternal_::swap(other);
        std::swap(m_data_, other.m_data_);
    }

    virtual int center_type() const { return IFORM; };

    virtual int rank() const { return traits::rank<TV>::value; }

    virtual int extent(int i) const { return traits::seq_value<typename traits::extents<TV>::type>::value[i]; }


    virtual DataSet dataset() const { return this->mesh().template dataset<TV, IFORM>(m_data_); };

    virtual DataSet dataset() { return this->mesh().template dataset<TV, IFORM>(m_data_); };

    virtual bool empty() const { return m_data_ == nullptr; }

    virtual void deploy() { if (empty()) { m_data_ = this->mesh().template data<value_type, IFORM>(); }}

    virtual void clear()
    {
        deploy();
        memset(m_data_.get(), 0, this->mesh().template memory_size<IFORM>() * sizeof(value_type));
    }

    virtual void sync()
    {
        auto ds = dataset();
        this->mesh().sync(ds);
    }

    value_type &at(id_type const &s)
    {
        return m_data_.get()[this->mesh().hash(s)];
    }

    value_type const &at(id_type const &s) const
    {
        return m_data_.get()[this->mesh().hash(s)];
    }

    value_type &operator[](id_type const &s) { return at(s); }

    value_type const &operator[](id_type const &s) const { return at(s); }

    typename mesh_type::range_type range() { return this->mesh().template range<IFORM>(); }

    template<typename TRange, typename Func>
    void accept(TRange const &r0, Func const &fun)
    {
        deploy();
        this->mesh().template for_each_value<value_type, IFORM>(*this, r0, fun);
    };
protected:
    std::shared_ptr<value_type> m_data_;

};
/* @}@} */

}//namespace simpla

#endif /* CORE_MANIFOLD_H_ */
