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
//    if (m_pimpl_->m_frame_ != nullptr)
//    {
//        os << std::setw(indent + 1) << " Mesh = " << m_pimpl_->m_frame_->name() << ", "
//           << " type = \"" << get_class_name() << "\", ";
//
//    }
    os << std::setw(indent + 1) << " " << " [" << get_class_name() << " : " << name() << "]" << std::endl;
    os << std::setw(indent + 1) << " " << "Chart = " << std::endl
       << std::setw(indent + 1) << " " << "{ " << std::endl;
    chart()->print(os, indent + 1);
    os << std::setw(indent + 1) << " " << "}," << std::endl;

    return os;
}

void Worker::initialize(Real data_time)
{

    chart()->initialize(data_time);
    model()->initialize(data_time);

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
