/**
 * @file field.h
 * @author salmon
 * @date 2015-10-16.
 */

#ifndef SIMPLA_FIELD_H
#define SIMPLA_FIELD_H

#include <simpla/SIMPLA_config.h>

#include <type_traits>
#include <cassert>

#include <simpla/toolbox/Log.h>
#include <simpla/mesh/Attribute.h>

#include "simpla/manifold/schemes/CalculusPolicy.h"
#include "simpla/manifold/schemes/InterpolatePolicy.h"

#include "Algebra.h"
#include "nTuple.h"

namespace simpla
{
namespace algebra { namespace declare { template<typename, typename, size_type ...I> struct Field_; }}

template<typename TV, typename TM, size_type IFORM = VERTEX, size_type DOF = 1>
using Field=algebra::declare::Field_<TV, TM, IFORM, DOF>;


namespace algebra
{
namespace traits
{
template<typename TV, typename TM, size_type IFORM, size_type DOF>
struct is_field<declare::Field_<TV, TM, IFORM, DOF> > : public std::integral_constant<bool, true> {};

template<typename TV, typename TM, size_type IFORM, size_type DOF>
struct reference<declare::Field_<TV, TM, IFORM, DOF> > { typedef declare::Field_<TV, TM, IFORM, DOF> const &type; };

template<typename TV, typename TM, size_type IFORM, size_type DOF>
struct value_type<declare::Field_<TV, TM, IFORM, DOF>> { typedef TV type; };

template<typename TV, typename TM, size_type IFORM, size_type DOF>
struct rank<declare::Field_<TV, TM, IFORM, DOF> > : public index_const<3> {};

template<typename TV, typename TM, size_type IFORM, size_type DOF>
struct iform<declare::Field_<TV, TM, IFORM, DOF> > : public index_const<IFORM> {};

template<typename TV, typename TM, size_type IFORM, size_type DOF>
struct dof<declare::Field_<TV, TM, IFORM, DOF> > : public index_const<DOF> {};


template<typename TV, typename TM, size_type IFORM, size_type DOF>
struct field_value_type<declare::Field_<TV, TM, IFORM, DOF>>
{
    typedef std::conditional_t<DOF == 1, TV, declare::nTuple_<TV, DOF> > cell_tuple;
    typedef std::conditional_t<(IFORM == VERTEX || IFORM == VOLUME),
            cell_tuple, declare::nTuple_<cell_tuple, 3> > type;
};

template<typename TV, typename TM, size_type ...I>
struct mesh_type<declare::Field_<TV, TM, I...> > { typedef TM type; };
} //namespace   traits

template<typename ...T> struct _engine;

namespace declare
{
template<typename TV, typename TM, size_type IFORM, size_type DOF>
class Field_<TV, TM, IFORM, DOF>
{
private:
    typedef Field_<TV, TM, IFORM, DOF> this_type;

public:

    typedef traits::field_value_t<this_type> field_value;
    typedef TV value_type;
    typedef TM mesh_type;


private:
    typedef typename mesh_type::template data_block_type<TV, IFORM, DOF> data_block_type;

    data_block_type *m_data_;
    mesh_type const *m_mesh_;
    friend struct _engine<Field_<TV, TM, IFORM, DOF>>;
public:


    explicit Field_() : m_mesh_(nullptr), m_data_(nullptr) {};


    virtual ~Field() {}

    Field_(this_type const &other) = delete;

    Field_(this_type &&other) = delete;

    virtual void pre_process()
    {

        ASSERT(m_data_ != nullptr);
        ASSERT(m_mesh_ != nullptr);

    }

    virtual void post_process()
    {

        m_mesh_ = nullptr;
        m_data_ = nullptr;

    }

    /** @name as_function  @{*/
    template<typename ...Args> field_value
    gather(Args &&...args) const { return m_mesh_->gather(*this, std::forward<Args>(args)...); }

    template<typename ...Args> field_value
    operator()(Args &&...args) const { return gather(std::forward<Args>(args)...); }

    /**@}*/

    /** @name as_array   @{*/

    value_type &
    get_value(mesh::MeshEntityId s) { return m_data_->get(mesh::MeshEntityIdCoder::unpack_index4(s, DOF)); }

    value_type const &
    get(mesh::MeshEntityId s) const { return m_data_->get(mesh::MeshEntityIdCoder::unpack_index4(s, DOF)); }


    value_type &
    get(index_type i, index_type j, index_type k = 0, index_type l = 0) { return m_data_->get(i, j, k, l); }

    value_type const &
    get(index_type i, index_type j, index_type k = 0, index_type l = 0) const { return m_data_->get(i, j, k, l); }

    template<typename TI>
    inline value_type &operator[](TI const &s) { return get(s); }

    template<typename TI>
    inline value_type const &operator[](TI const &s) const { return get(s); }

    template<typename TR>
    inline this_type &operator=(TR const &rhs)
    {
        _engine<this_type>::apply(m_mesh_, (*this), tags::_assign(), rhs);
        return (*this);
    }

    template<typename TR>
    inline this_type &operator+=(TR const &rhs)
    {
        _engine<this_type>::apply(m_mesh_, (*this), tags::plus_assign(), rhs);

        return (*this);
    }

    template<typename TR>
    inline this_type &operator-=(TR const &rhs)
    {
        _engine<this_type>::apply(m_mesh_, (*this), tags::minus_assign(), rhs);


        return (*this);
    }

    template<typename TR>
    inline this_type &operator*=(TR const &rhs)
    {
        _engine<this_type>::apply(m_mesh_, (*this), tags::multiplies_assign(), rhs);

        return (*this);
    }

    template<typename TR>
    inline this_type &operator/=(TR const &rhs)
    {
        _engine<this_type>::apply(m_mesh_, (*this), tags::divides_assign(), rhs);

        return (*this);
    }

    template<typename ...Args>
    void apply(Args &&...args) { _engine<this_type>::apply(m_mesh_, *this, std::forward<Args>(args)...); }

}; // class Field_
} //namespace declare
namespace st=simpla::traits;

template<typename TV, typename TM, size_type IFORM, size_type DOF>
struct _engine<declare::Field_<TV, TM, IFORM, DOF> >
{
    typedef TM mesh_type;
    typedef TV value_type;
    typedef declare::Field_<TV, TM, IFORM, DOF> self_type;
    typedef manifold::schemes::CalculusPolicy<mesh_type> calculus_policy;

    typedef manifold::schemes::InterpolatePolicy<mesh_type> interpolate_policy;

public:


    template<typename T> static T &
    get_value(T &v) { return v; };

    template<typename T, typename I0> static st::remove_all_extents_t<T, I0> &
    get_value(T &v, I0 const *s, ENABLE_IF((st::is_indexable<T, I0>::value)))
    {
        return get_value(v[*s], s + 1);
    };

    template<typename T, typename I0> static st::remove_all_extents_t<T, I0> &
    get_value(T &v, I0 const *s, ENABLE_IF((!st::is_indexable<T, I0>::value)))
    {
        return v;
    };
private:
    template<typename T, typename ...Args> static T &
    get_value_(std::integral_constant<bool, false> const &, T &v, Args &&...)
    {
        return v;
    }


    template<typename T, typename I0, typename ...Idx> static st::remove_extents_t<T, I0, Idx...> &
    get_value_(std::integral_constant<bool, true> const &, T &v, I0 const &s0, Idx &&...idx)
    {
        return get_value(v[s0], std::forward<Idx>(idx)...);
    };
public:
    template<typename T, typename I0, typename ...Idx> static st::remove_extents_t<T, I0, Idx...> &
    get_value(T &v, I0 const &s0, Idx &&...idx)
    {
        return get_value_(std::integral_constant<bool, st::is_indexable<T, I0>::value>(),
                          v, s0, std::forward<Idx>(idx)...);
    };

    template<typename T, size_type N> static T &
    get_value(declare::nTuple_<T, N> &v, size_type const &s0) { return v[s0]; };

    template<typename T, size_type N> static T const &
    get_value(declare::nTuple_<T, N> const &v, size_type const &s0) { return v[s0]; };
public:
    template<typename TOP, typename ...Others, size_type ... index, typename ...Idx> static auto
    _invoke_helper(declare::Expression<TOP, Others...> const &expr, index_sequence<index...>, Idx &&... s)
    DECL_RET_TYPE((TOP::eval(get_value(std::get<index>(expr.m_args_), std::forward<Idx>(s)...)...)))

    template<typename TOP, typename   ...Others, typename ...Idx> static auto
    get_value(declare::Expression<TOP, Others...> const &expr, Idx &&... s)
    DECL_RET_TYPE((_invoke_helper(expr, index_sequence_for<Others...>(), std::forward<Idx>(s)...)))


    template<typename TOP, typename ...Others> static void
    apply(mesh_type const *m, self_type &self, TOP const &op, Others &&...others)
    {
        self.m_mesh_->range().foreach(
                [&](mesh::MeshEntityId const &s)
                {
                    interpolate_policy::assign(m, self, s, std::forward<Others>(others)...);
                });
    }
//
//    template<typename TFun> void
//    assign(TFun const &fun, mesh::EntityIdRange const &r0,
//           std::result_of_t<TFun(point_type const &)> *p = nullptr)
//    {
//        r0.foreach([&](mesh::MeshEntityId const &s)
//                   {
//                       interpolate_policy::assign(*this, *m_mesh_, s, fun(m_mesh_->point(s)));
//                   });
//
//    }
//
//    template<typename U> void
//    assign(U const &v, mesh::EntityIdRange const &r0,
//           ENABLE_IF((std::is_convertible<U, value_type>::value || std::is_same<U, field_value>::value)))
//    {
//        pre_process();
//
//        r0.foreach([&](mesh::MeshEntityId const &s)
//                   {
//                       interpolate_policy::assign(*this, *m_mesh_, s, v);
//                   });
//
//    }
//
//    typedef mesh::MeshEntityIdCoder M;
//
//    void assign(this_type const &other, mesh::EntityIdRange const &r0)
//    {
//        pre_process();
//
//        r0.foreach([&](mesh::MeshEntityId const &s)
//                   {
//                       for (int i = 0; i < DOF; ++i) { get(M::sw(s, i)) = other.get(M::sw(s, i)); }
//                   });
//
//    }
//
//    template<typename ...U>
//    void assign(Expression<U...> const &expr, mesh::EntityIdRange const &r0)
//    {
//        pre_process();
//
//        r0.foreach([&](mesh::MeshEntityId const &s)
//                   {
//                       for (int i = 0; i < DOF; ++i)
//                       {
//                           get(M::sw(s, i)) = calculus_policy::eval(*m_mesh_, expr, M::sw(s, i));
//                       }
//                   });
//
//    }
//
//
//    template<typename Other> void
//    assign(Other const &other, mesh::MeshZoneTag const &tag = mesh::SP_ES_ALL)
//    {
//        pre_process();
//        if (tag == mesh::SP_ES_ALL)
//        {
//            assign(other, m_data_->range());
//        } else
//        {
//            assign(other, m_mesh_->mesh_block()->range(entity_type(), tag));
//        }
//    }
//
//    void copy(mesh::EntityIdRange const &r0, this_type const &g)
//    {
//        UNIMPLEMENTED;
////        r0.assign([&](mesh::MeshEntityId const &s) { get(s) = g.get(s); });
//    }
//
//
//    virtual void copy(mesh::EntityIdRange const &r0, mesh::DataBlock const &other)
//    {
//        UNIMPLEMENTED;
////        assert(other.is_a(typeid(this_type)));
////
////        this_type const &g = static_cast<this_type const & >(other);
////
////        copy(r0, static_cast<this_type const & >(other));
//
//    }
};

}
}//namespace simpla::algebra







#endif //SIMPLA_FIELD_H
