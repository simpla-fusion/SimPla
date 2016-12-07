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
    post_process();
    m_chart_->move_to(m);
    pre_process();
}

void Worker::deploy()
{
    concept::LifeControllable::deploy();
    m_chart_->deploy();
    m_model_->deploy();
}

void Worker::pre_process()
{
    if (is_valid()) { return; } else { concept::LifeControllable::pre_process(); }
    m_chart_->pre_process();
    m_model_->pre_process();
}

void Worker::post_process()
{
    if (!is_valid()) { return; } else { concept::LifeControllable::post_process(); }
    m_model_->post_process();
    m_chart_->post_process();
}

void Worker::initialize(Real data_time, Real dt)
{
    pre_process();
    ASSERT (m_chart_ != nullptr);
    m_chart_->initialize(data_time, dt);
    m_model_->initialize(data_time, dt);
}

virtual void Worker::sync() {}

unsigned int Worker::next_phase(Real data_time, Real dt, unsigned int inc_phase)
{
    unsigned int start_phase = current_phase_num();
    unsigned int end_phase = concept::LifeControllable::next_phase(data_time, dt, inc_phase);

    switch (start_phase)
    {
        #define NEXT_PHASE(_N_) case _N_: phase##_N_(data_time, dt);sync();++start_phase;if (start_phase >=end_phase )break;

        NEXT_PHASE(0);
        NEXT_PHASE(1);
        NEXT_PHASE(2);
        NEXT_PHASE(3);
        NEXT_PHASE(4);
        NEXT_PHASE(5);
        NEXT_PHASE(6);
        NEXT_PHASE(7);
        NEXT_PHASE(8);
        NEXT_PHASE(9);

        #undef NEXT_PHASE
        default:
            break;
    }
    return end_phase;
};

void Worker::finalize(Real data_time, Real dt)
{
    next_phase(data_time, dt, max_phase_num() - current_phase_num());

    m_model_->finalize(data_time, 0);
    m_chart_->finalize(data_time, 0);
    post_process();
}


}}//namespace simpla { namespace mesh1
