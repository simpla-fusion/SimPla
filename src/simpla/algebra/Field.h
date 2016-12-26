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
//#include <simpla/toolbox/Log.h>
//#include <simpla/mesh/Attribute.h>
//#include "simpla/manifold/schemes/CalculusPolicy.h"
//#include "simpla/manifold/schemes/InterpolatePolicy.h"
#include "Algebra.h"
#include "nTuple.h"

namespace simpla { namespace algebra
{
namespace schemes
{
template<typename ...> struct CalculusPolicy;
template<typename ...> struct InterpolatePolicy;
}  //namespace schemes

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

    typedef typename mesh_type::template data_block_type<TV, IFORM, DOF> data_type;


    typedef algebra::schemes::CalculusPolicy<mesh_type> calculus_policy;
    typedef algebra::schemes::InterpolatePolicy<mesh_type> interpolate_policy;

    data_type *m_data_;
    mesh_type const *m_mesh_;

    std::shared_ptr<mesh_type> m_mesh_holder_;
    std::shared_ptr<data_type> m_data_holder_;

public:
    template<typename ...Args>
    Field_(Args &&...args) : m_mesh_(nullptr), m_data_(nullptr) {};

    explicit Field_(mesh_type const *m, data_type *d = nullptr) :
            m_mesh_holder_(m, simpla::tags::do_nothing()), m_data_holder_(d, simpla::tags::do_nothing()),
            m_mesh_(nullptr), m_data_(nullptr) {};

    explicit Field_(std::shared_ptr<mesh_type> const &m, std::shared_ptr<data_type> const &d = nullptr) :
            m_mesh_holder_(m), m_data_holder_(d), m_mesh_(nullptr), m_data_() {};


    virtual ~Field_() {}

    Field_(this_type const &other) = delete;

    Field_(this_type &&other) = delete;

    virtual void pre_process()
    {
        assert(m_data_ != nullptr);
        assert(m_mesh_ != nullptr);
    }

    virtual void post_process()
    {
        m_mesh_ = nullptr;
        m_data_ = nullptr;
    }

    virtual void deploy()
    {
        m_mesh_ = m_mesh_holder_.get();
        m_data_ = m_data_holder_.get();
    }

    virtual void clear() {}

    /** @name as_function  @{*/
    template<typename ...Args> inline field_value
    gather(Args &&...args) const
    {
        return interpolate_policy::gather(m_mesh_, *this, std::forward<Args>(args)...);
    }

    template<typename ...Args> inline field_value
    scatter(field_value const &v, Args &&...args)
    {
        return interpolate_policy::scatter(m_mesh_, *this, v, std::forward<Args>(args)...);
    }

    template<typename ...Args> field_value
    operator()(Args &&...args) const { return gather(std::forward<Args>(args)...); }

    /**@}*/

    /** @name as_array   @{*/
    template<typename TID> value_type &
    at(TID const &s) { return calculus_policy::get_value((*this), s); }


    template<typename TID> value_type &
    at(TID const &s) const { return calculus_policy::get_value((*this), s); }


    value_type &
    get(index_type i, index_type j, index_type k = 0, index_type l = 0)
    {
        return calculus_policy::get_value((*this), i, j, k, l);
    }

    value_type const &
    get(index_type i, index_type j, index_type k = 0, index_type l = 0) const
    {
        return calculus_policy::get_value((*this), i, j, k, l);
    }

    template<typename TI> inline value_type &
    operator[](TI const &s) { return at(s); }

    template<typename TI> inline value_type const &
    operator[](TI const &s) const { return at(s); }

    template<typename TR> inline this_type &
    operator=(TR const &rhs)
    {
        calculus_policy::apply(m_mesh_, (*this), tags::_assign(), rhs);
        return (*this);
    }

    template<typename TR> inline this_type &
    operator+=(TR const &rhs)
    {
        calculus_policy::apply(m_mesh_, (*this), tags::plus_assign(), rhs);

        return (*this);
    }

    template<typename TR> inline this_type &
    operator-=(TR const &rhs)
    {
        calculus_policy::apply(m_mesh_, (*this), tags::minus_assign(), rhs);


        return (*this);
    }

    template<typename TR> inline this_type &
    operator*=(TR const &rhs)
    {
        calculus_policy::apply(m_mesh_, (*this), tags::multiplies_assign(), rhs);

        return (*this);
    }

    template<typename TR> inline this_type &
    operator/=(TR const &rhs)
    {
        calculus_policy::apply(m_mesh_, (*this), tags::divides_assign(), rhs);

        return (*this);
    }

    template<typename ...Args> void
    apply(Args &&...args) { calculus_policy::apply(m_mesh_, *this, std::forward<Args>(args)...); }

    template<typename ...Args> void
    assign(Args &&...args) { apply(tags::_assign(), std::forward<Args>(args)...); }

}; // class Field_
}
}} //namespace simpla::algebra::declare

namespace simpla { template<typename TV, typename TM, size_type IFORM = VERTEX, size_type DOF = 1> using Field=algebra::declare::Field_<TV, TM, IFORM, DOF>; }
//
//namespace simpla { namespace algebra
//{
//
//
//
//namespace st=simpla::traits;
//
//template<typename TV, typename TM, size_type IFORM, size_type DOF>
//struct _engine<declare::Field_<TV, TM, IFORM, DOF> >
//{
//    typedef TM mesh_type;
//    typedef TV value_type;
//    typedef declare::Field_<TV, TM, IFORM, DOF> self_type;
//    typedef schemes::CalculusPolicy<mesh_type> calculus_policy;
//    typedef schemes::InterpolatePolicy<mesh_type> interpolate_policy;
//
//public:
//
//
//    template<typename TOP, typename ...Others> static void
//    apply(self_type &self, TOP const &op, Others &&...others)
//    {
//        self.m_mesh_->range().foreach(
//                [&](mesh::MeshEntityId const &s)
//                {
//                    TOP::eval(calculus_policy::get_value(self, s),
//                              calculus_policy::get_value(std::forward<Others>(others), s)...));
//                });
//    }
//
//    static void clear(self_type &) {}
////
////    template<typename TFun> void
////    assign(TFun const &fun, mesh::EntityIdRange const &r0,
////           std::result_of_t<TFun(point_type const &)> *p = nullptr)
////    {
////        r0.foreach([&](mesh::MeshEntityId const &s)
////                   {
////                       interpolate_policy::assign(*this, *m_mesh_, s, fun(m_mesh_->point(s)));
////                   });
////
////    }
////
////    template<typename U> void
////    assign(U const &v, mesh::EntityIdRange const &r0,
////           ENABLE_IF((std::is_convertible<U, value_type>::value || std::is_same<U, field_value>::value)))
////    {
////        pre_process();
////
////        r0.foreach([&](mesh::MeshEntityId const &s)
////                   {
////                       interpolate_policy::assign(*this, *m_mesh_, s, v);
////                   });
////
////    }
////
////    typedef mesh::MeshEntityIdCoder M;
////
////    void assign(this_type const &other, mesh::EntityIdRange const &r0)
////    {
////        pre_process();
////
////        r0.foreach([&](mesh::MeshEntityId const &s)
////                   {
////                       for (int i = 0; i < DOF; ++i) { get(M::sw(s, i)) = other.get(M::sw(s, i)); }
////                   });
////
////    }
////
////    template<typename ...U>
////    void assign(Expression<U...> const &expr, mesh::EntityIdRange const &r0)
////    {
////        pre_process();
////
////        r0.foreach([&](mesh::MeshEntityId const &s)
////                   {
////                       for (int i = 0; i < DOF; ++i)
////                       {
////                           get(M::sw(s, i)) = calculus_policy::eval(*m_mesh_, expr, M::sw(s, i));
////                       }
////                   });
////
////    }
////
////
////    template<typename Other> void
////    assign(Other const &other, mesh::MeshZoneTag const &tag = mesh::SP_ES_ALL)
////    {
////        pre_process();
////        if (tag == mesh::SP_ES_ALL)
////        {
////            assign(other, m_data_->range());
////        } else
////        {
////            assign(other, m_mesh_->mesh_block()->range(entity_type(), tag));
////        }
////    }
////
////    void copy(mesh::EntityIdRange const &r0, this_type const &g)
////    {
////        UNIMPLEMENTED;
//////        r0.assign([&](mesh::MeshEntityId const &s) { get(s) = g.get(s); });
////    }
////
////
////    virtual void copy(mesh::EntityIdRange const &r0, mesh::DataBlock const &other)
////    {
////        UNIMPLEMENTED;
//////        assert(other.is_a(typeid(this_type)));
//////
//////        this_type const &g = static_cast<this_type const & >(other);
//////
//////        copy(r0, static_cast<this_type const & >(other));
////
////    }
//};
//}}//namespace simpla::algebra







#endif //SIMPLA_FIELD_H
