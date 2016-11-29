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
    postprocess();
    m_chart_->move_to(m);
    preprocess();
}

void Worker::preprocess()
{
    if (is_valid()) { return; } else { concept::Deployable::preprocess(); }
    m_chart_->preprocess();
}

void Worker::postprocess()
{
    if (!is_valid()) { return; } else { concept::Deployable::postprocess(); }
    m_chart_->postprocess();
}

void Worker::initialize(Real data_time)
{
    preprocess();
    ASSERT (m_chart_ != nullptr);
    m_chart_->initialize(data_time);
}

void Worker::finalize(Real data_time)
{
    m_chart_->finalize(data_time);
    postprocess();
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
