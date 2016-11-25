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
class Bundle : public AttributeView<TV, IFORM, DOF>
{
private:
    typedef Bundle<TV, IFORM, DOF> this_type;

    typedef AttributeView <TV, IFORM, DOF> base_type;
public:
    using base_type::deploy;
    using base_type::move_to;

    template<typename ...Args>
    explicit Bundle(ChartBase *chart, Args &&...args) :
            base_type(std::forward<Args>(args)...), m_chart_(chart)
    {
        m_chart_->connect(this);
    };


    virtual ~Bundle() { m_chart_->disconnect(this); }


    virtual bool is_a(std::type_info const &t_info) const
    {
        return t_info == typeid(this_type) || base_type::is_a(t_info);
    }

    template<typename U> U const *mesh_as() { return m_chart_->mesh_as<U>(); }

    ChartBase const *chart() const { return m_chart_; }

    template<typename TG>
    Chart<TG> const *chart_as() const { return static_cast<Chart<TG> const *>( m_chart_); }


private:
    ChartBase *m_chart_;


};
}} //namespace simpla {

#endif //SIMPLA_FIBERBUNDLE_H
