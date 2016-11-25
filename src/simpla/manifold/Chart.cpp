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

std::ostream &ChartBase::print(std::ostream &os, int indent) const
{
    os << std::setw(indent + 1) << " " << "Mesh = { ";
    coordinate_frame()->print(os, indent + 1);
    os << std::setw(indent + 1) << " " << "}," << std::endl;
    os << std::setw(indent + 1) << " " << "Attribute= { ";

    for (auto const &item:attributes())
    {
        os << "\"" << item->attribute()->name() << "\" , ";
    }

    os << std::setw(indent + 1) << " " << "} , " << std::endl;
};


bool ChartBase::is_a(std::type_info const &info) const { return typeid(ChartBase) == info; }

void ChartBase::initialize(Real data_time) { DO_NOTHING; }

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

    for (auto &item:m_attr_views_) { item->move_to(m); }
};


std::set<AttributeViewBase *> &ChartBase::attributes() { return m_attr_views_; };

std::set<AttributeViewBase *> const &ChartBase::attributes() const { return m_attr_views_; };


}}//namespace simpla {namespace mesh
