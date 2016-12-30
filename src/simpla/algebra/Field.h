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
#include "Algebra.h"
//#include <simpla/toolbox/Log.h>
//#include <simpla/mesh/Attribute.h>
//#include "simpla/manifold/schemes/CalculusPolicy.h"
//#include "simpla/manifold/schemes/InterpolatePolicy.h"
#include "Algebra.h"
#include "nTuple.h"

namespace simpla { namespace algebra
{
namespace declare { template<typename, typename, size_type ...> struct Field_; }

namespace traits
{


//***********************************************************************************************************************

template<typename T> struct field_value_type { typedef T type; };

template<typename T> using field_value_t=typename field_value_type<T>::type;

template<typename> struct mesh_type { typedef void type; };

template<typename TV, typename TM, size_type ...I>
struct mesh_type<declare::Field_<TV, TM, I...> > { typedef TM type; };


template<typename TV, typename TM, size_type IFORM, size_type DOF>
struct is_field<declare::Field_<TV, TM, IFORM, DOF> > : public std::integral_constant<bool, true> {};

template<typename TV, typename TM, size_type IFORM, size_type DOF>
struct reference<declare::Field_<TV, TM, IFORM, DOF> >
{
    typedef declare::Field_<TV, TM, IFORM, DOF> const &type;
};

template<typename TV, typename TM, size_type IFORM, size_type DOF>
struct reference<const declare::Field_<TV, TM, IFORM, DOF> > { typedef declare::Field_<TV, TM, IFORM, DOF> const &type; };

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
    typedef std::conditional_t<(IFORM == VERTEX || IFORM == VOLUME), cell_tuple, declare::nTuple_<cell_tuple, 3> > type;
};

//template<typename T> struct primary_type<T, typename std::enable_if<is_field<T>::value>::type>
//{
//    typedef typename declare::Field_<
//            value_type_t < T>, typename mesh_type<T>::type, iform<T>::value, dof<T>::value> type;
//};
//template<typename TV, typename TM, size_type IFORM, size_type DOF>
//struct data_block_type
//{
//    typedef declare::Array_<TV,
//            rank<TM>::value + ((IFORM == VERTEX || IFORM == VOLUME ? 0 : 1) * DOF > 1 ? 1 : 0)> type;
//};
//template<typename TV, typename TM, size_type IFORM, size_type DOF> using data_block_t=typename data_block_type<TV, TM, IFORM, DOF>::type;

}//namespace traits{

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

    typedef typename mesh_type::id_type mesh_id_type;

    typedef calculus::calculator<this_type> calculus_policy;

    typedef typename calculus_policy::data_block_type data_type;

    data_type *m_data_;

    mesh_type const *m_mesh_;

    std::shared_ptr<data_type> m_data_holder_;

public:
    Field_() : m_data_holder_(nullptr), m_mesh_(nullptr), m_data_(nullptr) {};

    explicit Field_(mesh_type const *m, data_type *d) :
            m_data_holder_(d, simpla::tags::do_nothing()),
            m_mesh_(m), m_data_(nullptr) {};

    explicit Field_(mesh_type const *m, std::shared_ptr<data_type> const &d = nullptr) :
            m_data_holder_(d), m_mesh_(m), m_data_() {};


    virtual ~Field_() {}

    Field_(this_type const &other) = delete;

    Field_(this_type &&other) = delete;

    virtual std::ostream &print(std::ostream &os, int indent = 0) const { return os; }

    template<typename TR> inline this_type &
    operator=(TR const &rhs) { return assign(rhs); }

    virtual void pre_process()
    {
        deploy();
        assert(m_data_holder_ != nullptr);
        assert(m_mesh_ != nullptr);
    }

    virtual void post_process()
    {
        m_mesh_ = nullptr;
        m_data_ = nullptr;
        m_data_holder_.reset();
    }

    virtual void accept(mesh_type const *m, std::shared_ptr<data_type> const &d = nullptr)
    {
        post_process();
        m_data_holder_ = d;
        m_mesh_ = m;
        pre_process();
    }

    virtual void accept(mesh_type const *m, data_type *d)
    {
        post_process();
        m_data_holder_ = std::shared_ptr<data_type>(d, simpla::tags::do_nothing());
        m_mesh_ = m;
        pre_process();
    }

    virtual void deploy()
    {
        if (!m_data_holder_) { m_data_holder_ = calculus_policy::create_data_block(m_mesh_); }
        m_data_ = m_data_holder_.get();
    }

    virtual void clear() { apply(tags::_clear()); }


    /** @name as_function  @{*/
    template<typename ...Args> inline auto
    gather(Args &&...args) const
    DECL_RET_TYPE((apply(tags::_gather(), std::forward<Args>(args)...)))


    template<typename ...Args> inline auto
    scatter(field_value const &v, Args &&...args)
    DECL_RET_TYPE((apply(tags::_scatter(), v, std::forward<Args>(args)...)))


//    auto
//    operator()(mesh_type::point_type const &x) const DECL_RET_TYPE((gather(x)))


    /**@}*/

    /** @name as_array   @{*/


    template<typename ...TID> value_type &
    at(TID &&...s) { return calculus_policy::get_value(*m_mesh_, *m_data_, std::forward<TID>(s)...); }

    template<typename ...TID> value_type const &
    at(TID &&...s) const { return calculus_policy::get_value(*m_mesh_, *m_data_, std::forward<TID>(s)...); }

    template<typename ...Args> auto
    operator()(Args &&...args) DECL_RET_TYPE((at(std::forward<Args>(args)...)))

    template<typename ...Args> auto
    operator()(Args &&...args) const DECL_RET_TYPE((at(std::forward<Args>(args)...)))

    template<typename TI> inline value_type &
    operator[](TI const &s) { return at(s); }

    template<typename TI> inline value_type const &
    operator[](TI const &s) const { return at(s); }

    /**@}*/


    template<typename ...Args> this_type &
    assign(Args &&...args) { return apply(tags::_assign(), std::forward<Args>(args)...); }

    template<typename ...Args> this_type &
    apply(Args &&...args)
    {
        pre_process();
        calculus_policy::apply(*this, *m_mesh_, std::forward<Args>(args)...);
        return *this;
    }
}; // class Field_
} //namespace declare

}} //namespace simpla::algebra

namespace simpla
{
template<typename TV, typename TM, size_type IFORM = VERTEX, size_type DOF = 1> using Field=algebra::declare::Field_<TV, TM, IFORM, DOF>;
}


#endif //SIMPLA_FIELD_H
