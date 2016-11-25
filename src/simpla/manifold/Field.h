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

#include <simpla/toolbox/type_traits.h>
#include <simpla/toolbox/Log.h>
#include <simpla/toolbox/design_pattern/Observer.h>
#include <simpla/mesh/Attribute.h>

#include "FiberBundle.h"
#include "FieldTraits.h"
#include "FieldExpression.h"
#include "schemes/CalculusPolicy.h"
#include "schemes/InterpolatePolicy.h"


namespace simpla
{


template<typename ...> class Field;


template<typename TV, typename TM, size_type I, size_type DOF>
class Field<TV, TM, index_const<I>, index_const<DOF>> :
        public mesh::FiberBundle<TV, static_cast<mesh::MeshEntityType>(I), DOF>
{
private:
    static constexpr mesh::MeshEntityType IFORM = static_cast<mesh::MeshEntityType>(I);
    typedef Field<TV, TM, index_const<I>, index_const<DOF>> this_type;
    typedef mesh::FiberBundle<TV, static_cast<mesh::MeshEntityType>(I), DOF> base_type;
    using base_type::view_type;
public:
    typedef TV value_type;
    typedef TM mesh_type;
    typedef typename std::conditional<DOF == 1, value_type, nTuple<value_type, DOF> >::type cell_tuple;
    typedef typename std::conditional<(IFORM == mesh::VERTEX || IFORM == mesh::VOLUME),
            cell_tuple, nTuple<cell_tuple, 3> >::type field_value_type;

private:
    typedef typename mesh_type::template data_block_type<TV, IFORM, DOF> data_block;

    data_block *m_data_;
    mesh_type const *m_mesh_;

public:

    typedef manifold::schemes::CalculusPolicy<mesh_type> calculus_policy;

    typedef manifold::schemes::InterpolatePolicy<mesh_type> interpolate_policy;


//    explicit Field(mesh::Chart<mesh_type> *chart, std::shared_ptr<view_type> const &attr = nullptr)
//            : base_type(chart, attr),
//              m_mesh_(nullptr),
//              m_data_(nullptr) {};

    template<typename ...Args>
    explicit Field(mesh::Chart<mesh_type> *chart, Args &&...args):
            base_type(chart, std::forward<Args>(args)...),
            m_mesh_(nullptr),
            m_data_(nullptr) {};


    virtual ~Field() {}

    Field(this_type const &other) = delete;

    Field(this_type &&other) = delete;

    virtual bool is_a(std::type_info const &t_info) const
    {
        return t_info == typeid(this_type) || base_type::is_a(t_info);
    };

    bool is_valid() const { return m_data_ != nullptr && m_mesh_ != nullptr; };

    using base_type::entity_type;

    using base_type::value_type_info;

    using base_type::dof;


    void deploy()
    {
        m_mesh_ = base_type::template mesh_as<mesh_type>();
        m_data_ = base_type::template data_as<data_block>();
        m_data_->deploy();
    }


    virtual void clear()
    {
        deploy();
        m_data_->clear();
    }


    /** @name as_function  @{*/
    template<typename ...Args> field_value_type
    gather(Args &&...args) const { return m_mesh_->mesh_block()->gather(*this, std::forward<Args>(args)...); }

    template<typename ...Args> field_value_type
    operator()(Args &&...args) const { return gather(std::forward<Args>(args)...); }

    /**@}*/

    /** @name as_array   @{*/

    template<typename ...Args>
    inline value_type &get(Args &&...args) { return m_data_->get(std::forward<Args>(args)...); }

    template<typename ...Args>
    inline value_type const &get(Args &&...args) const { return m_data_->get(std::forward<Args>(args)...); }

    template<typename ...Args>
    inline value_type &operator()(Args &&...args) { return m_data_->get(std::forward<Args>(args)...); }

    template<typename ...Args>
    inline value_type const &operator()(Args &&...args) const { return m_data_->get(std::forward<Args>(args)...); }

    template<typename TI>
    inline value_type &operator[](TI const &s) { return m_data_->get(s); }

    template<typename TI>
    inline value_type const &operator[](TI const &s) const { return m_data_->get(s); }

//    this_type &operator=(this_type const &other)
//    {
//        assign(mesh::SP_ES_ALL, other);
//        return *this;
//    }

    template<typename ...U> inline this_type &
    operator=(Field<U...> const &other)
    {
        assign(mesh::SP_ES_ALL, other);
        return *this;
    }

    template<typename Other> inline this_type &
    operator=(Other const &other)
    {
        assign(mesh::SP_ES_ALL, other);
        return *this;
    }

    template<typename Other> inline this_type &
    operator+=(Other const &other)
    {
        *this = *this + other;
        return *this;
    }

    template<typename Other> inline this_type &
    operator-=(Other const &other)
    {
        *this = *this - other;
        return *this;
    }

    template<typename Other> inline this_type &
    operator*=(Other const &other)
    {
        *this = *this * other;
        return *this;
    }

    template<typename Other> inline this_type &
    operator/=(Other const &other)
    {
        *this = *this / other;
        return *this;
    }

    inline this_type &
    operator=(this_type const &other)
    {
        deploy();

        m_data_->foreach(
                [&](value_type &v, index_type i, index_type j, index_type k, index_type l)
                {
                    v = other.get(i, j, l, l);
                });
        return *this;

    }

    /* @}*/

    template<typename U> void
    assign(mesh::EntityIdRange const &r0, U const &v,
           ENABLE_IF((std::is_arithmetic<U>::value || std::is_same<U, value_type>::value)))
    {
        deploy();
        r0.foreach([&](mesh::MeshEntityId const &s) { get(s) = v; });
    }


    template<typename ...U>
    void assign(mesh::EntityIdRange const &r0, Field<Expression<U...>> const &expr)
    {
        deploy();
        r0.foreach([&](mesh::MeshEntityId const &s) { get(s) = calculus_policy::eval(*m_mesh_, expr, s); });

    }

//    template<typename TFun, typename ...U> void
//    assign(mesh::EntityIdRange const &r0, std::function<value_type(point_type const &, U const &...)> const &fun,
//            U &&...args)
//    {
//        deploy();
//        r0.foreach([&](mesh::MeshEntityId const &s) { get(s) = fun(m_frame_->point(s), std::forward<U>(args)...); });
//    }


    template<typename TFun> void
    assign(mesh::EntityIdRange const &r0, TFun const &fun,
           ENABLE_IF((std::is_same<typename std::result_of<TFun(point_type const &)>::type, field_value_type>::value))
    )
    {
        deploy();
        r0.foreach([&](mesh::MeshEntityId const &s)
                   {
                       get(s) = interpolate_policy::template sample<IFORM>(*m_mesh_, s, fun(m_mesh_->point(s)));
                   });
    }

    template<typename TFun> void
    assign(mesh::EntityIdRange const &r0, TFun const &fun,
           ENABLE_IF(
                   (std::is_same<typename std::result_of<TFun(mesh::MeshEntityId const &)>::type, value_type>::value))
    )
    {
        deploy();
        r0.foreach([&](mesh::MeshEntityId const &s) { get(s) = fun(s); });
    }


    template<typename U> void
    assign(U const &v,
           ENABLE_IF((std::is_arithmetic<U>::value || std::is_same<U, value_type>::value)))
    {
        deploy();
        m_data_->assign([&](index_type i, index_type j, index_type k, index_type l) { return v; });
    }


    template<typename ...U>
    void assign(Field<Expression<U...>> const &expr)
    {
        deploy();
        auto b = std::get<0>(m_mesh_->mesh_block()->outer_index_box());
        index_type gw[4] = {1, 1, 1, 0};

        m_data_->foreach(
                [&](value_type &v, index_type i, index_type j, index_type k, index_type l)
                {
                    auto s = mesh::MeshEntityIdCoder::pack_index4<IFORM>(i, j, k, l / DOF, l % DOF);

//                    VERBOSE << FILE_LINE_STAMP
//                            << " [" << i << "," << j << "," << k << "," << l / DOF << "," << l % DOF << "]=>"
//                            << mesh::MeshEntityIdCoder::unpack_index4(mesh::MeshEntityIdCoder::sw(s, 2), DOF)
//                            << std::endl;

                    v = calculus_policy::eval(*m_mesh_, expr, s);
                });

    }


    template<typename TFun> void
    assign(TFun const &fun,
           ENABLE_IF((std::is_same<typename std::result_of<TFun(point_type const &)>::type, value_type>::value)))
    {
        deploy();

        m_data_->foreach(
                [&](value_type &v, index_type i, index_type j, index_type k, index_type l)
                {
                    v = fun(m_mesh_->point(mesh::MeshEntityIdCoder::pack_index4<IFORM>(i, j, k, l / DOF, l % DOF)));
                });


    }

    template<typename TFun> void
    assign(TFun const &fun,
           ENABLE_IF((std::is_same<typename std::result_of<TFun(index_type, index_type, index_type,
                                                                index_type)>::type, value_type>::value)))
    {
        deploy();

        m_data_->foreach(
                [&](value_type &v, index_type i, index_type j, index_type k, index_type l)
                {
                    v = fun(i, j, k, l);
                });


    }

    template<typename TFun> void
    assign(TFun const &fun,
           ENABLE_IF(((!std::is_same<typename std::result_of<TFun(point_type const &)>::type, value_type>::value) &&
                      (std::is_same<typename std::result_of<TFun(point_type const &)>::type, field_value_type>::value)
                     ))
    )
    {
        deploy();

        m_data_->foreach(
                [&](value_type &v, index_type i, index_type j, index_type k, index_type l)
                {
                    auto s = mesh::MeshEntityIdCoder::pack_index4<IFORM>(i, j, k, l / DOF, l % DOF);
                    v = interpolate_policy::template sample<IFORM>(*m_mesh_, s, fun(m_mesh_->point(s)));;
                });


    }


    template<typename TFun> void
    assign(TFun const &fun,
           ENABLE_IF((std::is_same<typename std::result_of<TFun(mesh::MeshEntityId const &)>::type,
                   value_type>::value))
    )
    {
        deploy();
        m_data_->foreach(
                [&](value_type &v, index_type i, index_type j, index_type k, index_type l)
                {
                    v = fun(mesh::MeshEntityIdCoder::pack_index4<IFORM>(i, j, k, l));
                });
    }
//
//    template<typename U> void
//    assign_ghost(U const &v,
//                  ENABLE_IF((std::is_arithmetic<U>::value || std::is_same<U, value_type>::value)))
//    {
//        deploy();
//
//        m_data_->assign_ghost([&](index_type i, index_type j, index_type k, index_type l) { return v; });
//    }
//
//
//    template<typename ...U>
//    void assign_ghost(Field<Expression<U...>> const &expr)
//    {
//        deploy();
//
//        m_data_->assign_ghost(
//                [&](index_type i, index_type j, index_type k, index_type l)
//                {
//                    return calculus_policy::eval(*m_mesh_, expr,
//                                                 mesh::MeshEntityIdCoder::pack_index4<IFORM>(i, j, k, l));
//                });
//
//    }
//
//
//    template<typename TFun> void
//    assign_ghost(TFun const &fun,
//                  ENABLE_IF((std::is_same<typename std::result_of<TFun(point_type const &)>::type,
//                          field_value_type>::value)))
//    {
//        deploy();
//        m_data_->assign_ghost(
//                [&](index_type i, index_type j, index_type k, index_type l)
//                {
//                    auto s = mesh::MeshEntityIdCoder::pack_index4<IFORM>(i, j, k, l);
//                    return interpolate_policy::template sample<IFORM>(*m_mesh_, s, fun(m_mesh_->point(s)));
//                });
//
//
//    }
//
//    template<typename TFun> void
//    assign_ghost(TFun const &fun,
//                  ENABLE_IF(
//                          (std::is_same<typename std::result_of<TFun(
//                                  mesh::MeshEntityId const &)>::type, value_type>::value))
//    )
//    {
//        deploy();
//
//        m_data_->assign_ghost(
//                [&](index_type i, index_type j, index_type k, index_type l)
//                {
//                    return fun(mesh::MeshEntityIdCoder::pack_index4<IFORM>(i, j, k, l));
//                });
//    }

    template<typename ...Args> void
    assign(mesh::MeshZoneTag const &tag, Args &&...args)
    {
        deploy();
        if (tag == mesh::SP_ES_ALL)
        {
            assign(std::forward<Args>(args)...);
        } else
        {
            assign(m_mesh_->mesh_block()->range(entity_type(), tag, DOF), std::forward<Args>(args)...);
        }
    }

    void copy(mesh::EntityIdRange const &r0, this_type const &g)
    {
        UNIMPLEMENTED;
//        r0.assign([&](mesh::MeshEntityId const &s) { get(s) = g.get(s); });
    }


    virtual void copy(mesh::EntityIdRange const &r0, mesh::DataBlock const &other)
    {
        UNIMPLEMENTED;
//        assert(other.is_a(typeid(this_type)));
//
//        this_type const &g = static_cast<this_type const & >(other);
//
//        copy(r0, static_cast<this_type const & >(other));

    }


};


}//namespace simpla







#endif //SIMPLA_FIELD_H
