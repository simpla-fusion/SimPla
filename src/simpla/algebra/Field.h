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

    typedef traits::field_value_t <this_type> field_value;
    typedef TV value_type;
    typedef TM mesh_type;
private:

    typedef typename mesh_type::template data_block_type<TV, IFORM, DOF> data_type;

    friend mesh_type;

    data_type *m_data_;

    mesh_type const *m_mesh_;

    std::shared_ptr<data_type> m_data_holder_;

public:
//    template<typename ...Args>
//    Field_(Args &&...args) : m_mesh_(nullptr), m_data_(nullptr) {};

    explicit Field_(mesh_type const *m, data_type *d = nullptr) :
            m_data_holder_(d, simpla::tags::do_nothing()),
            m_mesh_(m), m_data_(nullptr) {};

    explicit Field_(std::shared_ptr<mesh_type> const &m, std::shared_ptr<data_type> const &d = nullptr) :
            m_data_holder_(d), m_mesh_(m.get()), m_data_() {};


    virtual ~Field_() {}

    Field_(this_type const &other) = delete;

    Field_(this_type &&other) = delete;

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
    }

    virtual void deploy()
    {
        m_mesh_->template create_data_block<TV, IFORM, DOF>(&m_data_holder_);
        m_data_ = m_data_holder_.get();
    }

    virtual void move_to(mesh_type const *m, std::shared_ptr<data_type> const &d)
    {
        post_process();
        m_data_holder_ = d;
        m_mesh_ = m;
        pre_process();
    }

    virtual void clear() { apply(tags::_clear()); }

    template<typename TR> inline this_type &
    operator=(TR const &rhs) { return assign(rhs); }

    /** @name as_function  @{*/
    template<typename ...Args> inline auto
    gather(Args &&...args) const
    DECL_RET_TYPE((apply(tags::_gather(), std::forward<Args>(args)...)))


    template<typename ...Args> inline auto
    scatter(field_value const &v, Args &&...args)
    DECL_RET_TYPE((apply(tags::_scatter(), v, std::forward<Args>(args)...)))


    template<typename ...Args> auto
    operator()(Args &&...args) const DECL_RET_TYPE((gather(std::forward<Args>(args)...)))

    /**@}*/

    /** @name as_array   @{*/
    template<typename ...TID> value_type &
    at(TID &&...s) { return m_mesh_->access((*this), std::forward<TID>(s)...); }

    template<typename ...TID> value_type const &
    at(TID &&...s) const { return m_mesh_->access((*this), std::forward<TID>(s)...); }


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
        m_mesh_->apply(*this, std::forward<Args>(args)...);
        return *this;
    }


}; // class Field_
}
}} //namespace simpla::algebra::declare

namespace simpla
{
template<typename TV, typename TM, size_type IFORM = VERTEX, size_type DOF = 1> using Field=algebra::declare::Field_<TV, TM, IFORM, DOF>;
}


#endif //SIMPLA_FIELD_H
