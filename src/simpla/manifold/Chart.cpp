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


std::shared_ptr<AttributeViewBase>
ChartBase::connect(std::shared_ptr<AttributeViewBase> const &attr, std::string const &key)
{
    if (key == "") { m_attr_views_[attr->attribute()->name()] = attr; }
    else { m_attr_views_[key] = attr; }
    return attr;

}

void ChartBase::move_to(std::shared_ptr<MeshBlock> const &m)
{
    coordinate_frame()->move_to(m);

    for (auto &item:m_attr_views_) { item.second->move_to(m->id()); }
};


std::map<std::string, std::shared_ptr<AttributeViewBase> > &
ChartBase::attributes() { return m_attr_views_; };

std::map<std::string, std::shared_ptr<AttributeViewBase> > const &
ChartBase::attributes() const { return m_attr_views_; };


}}//namespace simpla {namespace mesh
