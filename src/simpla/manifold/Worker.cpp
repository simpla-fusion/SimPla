//
// Created by salmon on 16-11-4.
//
#include "Worker.h"
#include <set>
#include <simpla/mesh/MeshBlock.h>
#include <simpla/mesh/Attribute.h>

namespace simpla { namespace mesh
{


Worker::Worker(std::shared_ptr<Chart> const &c) : m_chart_(c) {}

Worker::~Worker() {};

std::ostream &Worker::print(std::ostream &os, int indent) const
{

    os << std::setw(indent + 1) << " " << " [" << get_class_name() << " : " << name() << "]" << std::endl;

    os << std::setw(indent + 1) << " " << "Config = {" << db << "}" << std::endl;

    if (m_chart_ != nullptr)
    {
        os << std::setw(indent + 1) << " " << "Chart = " << std::endl
           << std::setw(indent + 1) << " " << "{ " << std::endl;
        m_chart_->print(os, indent + 1);
        os << std::setw(indent + 1) << " " << "}," << std::endl;
    }
    return os;
}

void Worker::move_to(std::shared_ptr<mesh::MeshBlock> const &m)
{
    ASSERT (m_chart_ != nullptr);
    m_chart_->move_to(m);
}


void Worker::update()
{
    if (m_chart_ != nullptr)
    {
        m_chart_->update();
        for (auto &item:m_chart_->attributes()) { item->update(); }
    }

}

void Worker::initialize(Real data_time)
{
    if (m_chart_ != nullptr)
    {
        m_chart_->initialize(data_time);

        for (auto &item:m_chart_->attributes()) { item->clear(); }
    }

}
//
//
//void Worker::deploy()
//{
////    move_to(m_pimpl_->m_mesh_);
////    foreach([&](AttributeViewBase &ob) { ob.deploy(); });
//
//}
//
//void Worker::destroy()
//{
////    foreach([&](AttributeViewBase &ob) { ob.destroy(); });
//    m_pimpl_->m_frame_ = nullptr;
//}
//

}}//namespace simpla { namespace mesh1
