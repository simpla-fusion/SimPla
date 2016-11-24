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


class FiberBundle
{
public:
    FiberBundle(ChartBase *chart, std::shared_ptr<AttributeViewBase> const &attr) : m_chart_(chart), m_attr_(attr)
    {
        chart->connect(m_attr_);

    };

    virtual ~FiberBundle() {}

    virtual MeshEntityType entity_type() const =0;

    virtual std::type_info const &value_type_info() const =0;

    virtual size_type dof() const =0;

    virtual bool is_a(std::type_info const &t_info) const { return t_info == typeid(FiberBundle); }


    template<typename U> U const *data_as() const { return m_attr_->data_as<U>(); }

    template<typename U> U *data_as() { return m_attr_->data_as<U>(); }

    template<typename U> U const *mesh_as() { return m_chart_->mesh_as<U>(); }

    ChartBase const *chart() const { return m_chart_; }

    template<typename TG>
    Chart <TG> const *chart_as() const { return static_cast<Chart<TG> const *>( m_chart_); }

    template<typename ...Args>
    void move_to(Args &&...args) { m_attr_->move_to(std::forward<Args>(args)...); }


    virtual void deploy() { move_to(m_chart_->coordinate_frame()->mesh_block()->id()); }

private:
    ChartBase const *m_chart_;
    std::shared_ptr<AttributeViewBase> m_attr_;


};
}} //namespace simpla {

#endif //SIMPLA_FIBERBUNDLE_H
