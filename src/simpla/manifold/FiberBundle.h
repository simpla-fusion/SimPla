//
// Created by salmon on 16-11-24.
//

#ifndef SIMPLA_FIBERBUNDLE_H
#define SIMPLA_FIBERBUNDLE_H

#include <simpla/toolbox/Log.h>
#include <simpla/mesh/MeshCommon.h>
#include <simpla/mesh/DataBlock.h>
#include <simpla/mesh/Attribute.h>

#include "Chart.h"

namespace simpla { namespace mesh
{
class ChartBase;

template<typename> class Chart;

template<typename TV, MeshEntityType IFORM, size_type DOF = 1>
class FiberBundle : public AttributeView<TV, IFORM, DOF>
{
private:
    typedef FiberBundle<TV, IFORM, DOF> this_type;

    typedef AttributeView <TV, IFORM, DOF> base_type;
public:

    template<typename ...Args>
    explicit FiberBundle(ChartBase *chart, Args &&...args) :
            base_type(std::forward<Args>(args)...), m_chart_(chart)
    {
        m_chart_->connect(this);
    };


    virtual ~FiberBundle() { m_chart_->disconnect(this); }


    virtual bool is_a(std::type_info const &t_info) const { return t_info == typeid(this_type); }

    template<typename U> U const *mesh_as() { return m_chart_->mesh_as<U>(); }

    ChartBase const *chart() const { return m_chart_; }

    template<typename TG>
    Chart<TG> const *chart_as() const { return static_cast<Chart<TG> const *>( m_chart_); }

    void move_to(std::shared_ptr<MeshBlock> const &m, std::shared_ptr<DataBlock> const &d)
    {
        base_type::move_to(m, d);
//        deploy();
    }


    virtual void deploy()
    {
//        base_type::move_to(m_chart_->coordinate_frame()->mesh_block()->id());
        base_type::deploy();
    }


private:
    ChartBase *m_chart_;


};
}} //namespace simpla {

#endif //SIMPLA_FIBERBUNDLE_H
