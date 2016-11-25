//
// Created by salmon on 16-11-24.
//
#include "Chart.h"

#include <simpla/mesh/Attribute.h>
#include <simpla/mesh/MeshBlock.h>

#include "CoordinateFrame.h"

namespace simpla { namespace mesh
{


ChartBase::ChartBase() {}

ChartBase::~ChartBase() {}

bool ChartBase::is_a(std::type_info const &info) const { return typeid(ChartBase) == info; }

void ChartBase::initialize() { DO_NOTHING; }

void ChartBase::deploy() { DO_NOTHING; }


AttributeViewBase *
ChartBase::connect(AttributeViewBase *attr)
{

    m_attr_views_.insert(attr);
    return attr;

}

void ChartBase::disconnect(AttributeViewBase *attr)
{
    m_attr_views_.erase(attr);
}

void ChartBase::move_to(std::shared_ptr<MeshBlock> const &m)
{
    coordinate_frame()->move_to(m);

    for (auto &item:m_attr_views_) { item->move_to(m->id()); }
};


std::set<AttributeViewBase *> &ChartBase::attributes() { return m_attr_views_; };

std::set<AttributeViewBase *> const &ChartBase::attributes() const { return m_attr_views_; };


}}//namespace simpla {namespace mesh
