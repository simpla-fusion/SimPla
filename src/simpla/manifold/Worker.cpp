//
// Created by salmon on 16-11-4.
//
#include "Worker.h"
#include <set>
#include <simpla/mesh/MeshBlock.h>
#include <simpla/mesh/Attribute.h>

namespace simpla { namespace mesh
{


Worker::Worker() {}

Worker::~Worker() {};

std::ostream &Worker::print(std::ostream &os, int indent) const
{

    os << std::setw(indent + 1) << " " << " [" << get_class_name() << " : " << name() << "]" << std::endl;
    if (chart() != nullptr)
    {
        os << std::setw(indent + 1) << " " << "Chart = " << std::endl
           << std::setw(indent + 1) << " " << "{ " << std::endl;
        chart()->print(os, indent + 1);
        os << std::setw(indent + 1) << " " << "}," << std::endl;
    }
    return os;
}

void Worker::move_to(std::shared_ptr<mesh::MeshBlock> const &m)
{
    if (chart() != nullptr) { chart()->move_to(m); }
}

void Worker::initialize(Real data_time)
{

    if (model() != nullptr) model()->initialize(data_time);

    ASSERT (chart() != nullptr)

    chart()->initialize(data_time);

    for (auto &item:chart()->attributes()) { item->clear(); }

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
