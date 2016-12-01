//
// Created by salmon on 16-11-4.
//
#include "Worker.h"
#include <set>
#include <simpla/mesh/MeshBlock.h>
#include <simpla/mesh/Attribute.h>

namespace simpla { namespace mesh
{


Worker::Worker(std::shared_ptr<Chart> const &c) : m_chart_(nullptr), m_model_(nullptr) { set_chart(c); }

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

std::shared_ptr<Chart> const &
Worker::get_chart() const
{
    return m_chart_;
}

void Worker::set_chart(std::shared_ptr<Chart> const &c)
{
    m_chart_ = c;
    m_model_ = std::make_shared<model::Model>(m_chart_);
}

std::shared_ptr<model::Model> const &Worker::get_model() const { return m_model_; }


void Worker::move_to(std::shared_ptr<mesh::MeshBlock> const &m)
{
    postprocess();
    m_chart_->move_to(m);
    preprocess();
}

void Worker::deploy()
{
    concept::Deployable::deploy();
    m_chart_->deploy();
    m_model_->deploy();
}

void Worker::preprocess()
{
    if (is_valid()) { return; } else { concept::Deployable::preprocess(); }
    m_chart_->preprocess();
    m_model_->preprocess();
}

void Worker::postprocess()
{
    if (!is_valid()) { return; } else { concept::Deployable::postprocess(); }
    m_model_->postprocess();
    m_chart_->postprocess();
}

void Worker::initialize(Real data_time)
{
    preprocess();
    ASSERT (m_chart_ != nullptr);
    m_chart_->initialize(data_time);
    m_model_->initialize(data_time);
}

void Worker::finalize(Real data_time)
{

    m_model_->finalize(data_time);
    m_chart_->finalize(data_time);
    postprocess();
}


}}//namespace simpla { namespace mesh1
