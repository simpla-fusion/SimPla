//
// Created by salmon on 16-11-24.
//
#include "Chart.h"

#include <simpla/mesh/Attribute.h>
#include <simpla/mesh/MeshBlock.h>


namespace simpla { namespace mesh
{


Chart::Chart() {}

Chart::~Chart() {}

std::ostream &Chart::print(std::ostream &os, int indent) const
{
    os << std::setw(indent + 1) << " " << "Mesh = { ";
    os << "Type = \"" << get_class_name() << "\",";
    if (m_mesh_block_ != nullptr)
    {
        os << std::endl;
        os << std::setw(indent + 1) << " " << " Block = {";
        m_mesh_block_->print(os, indent + 1);
        os << std::setw(indent + 1) << " " << "},";
    }    os << std::setw(indent + 1) << " " << "}," << std::endl;

    os << std::setw(indent + 1) << " " << "Attribute= { ";

    for (auto const &item:attributes())
    {
        os << "\"" << item->attribute()->name() << "\" , ";
    }

    os << std::setw(indent + 1) << " " << "} , " << std::endl;
};


bool Chart::is_a(std::type_info const &info) const { return typeid(Chart) == info; }


AttributeViewBase *
Chart::connect(AttributeViewBase *attr)
{

    m_attr_views_.insert(attr);
    return attr;

}

void Chart::disconnect(AttributeViewBase *attr) { m_attr_views_.erase(attr); }

void Chart::initialize(Real data_time) { DO_NOTHING; }

void Chart::update() { DO_NOTHING; }

void Chart::move_to(std::shared_ptr<MeshBlock> const &m)
{
    m_mesh_block_ = m;
    for (auto &item:m_attr_views_) { item->move_to(m); }
};


std::set<AttributeViewBase *> &Chart::attributes() { return m_attr_views_; };

std::set<AttributeViewBase *> const &Chart::attributes() const { return m_attr_views_; };


}}//namespace simpla {namespace mesh
