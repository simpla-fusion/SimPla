//
// Created by salmon on 16-11-4.
//
#include "Worker.h"
//#include <set>
//#include <simpla/concept/Printable.h>
//#include <simpla/data/DataTable.h>
//#include <simpla/mesh/MeshBlock.h>
//#include <simpla/mesh/Attribute.h>

namespace simpla {
namespace mesh {

Worker::Worker(Mesh *m) : m_mesh_(m) {}

Worker::~Worker(){};

std::ostream &Worker::print(std::ostream &os, int indent) const {
    os << std::setw(indent + 1) << " "
       << " [" << get_class_name() << " : " << name() << "]" << std::endl;

    os << std::setw(indent + 1) << " "
       << "Config = {" << db << "}" << std::endl;

    if (m_mesh_ != nullptr) {
        os << std::setw(indent + 1) << " "
           << "Mesh = " << std::endl
           << std::setw(indent + 1) << " "
           << "{ " << std::endl;
        m_mesh_->print(os, indent + 1);
        os << std::setw(indent + 1) << " "
           << "}," << std::endl;
    }
    return os;
}

void Worker::accept(Patch *p) {
    post_process();
    //    auto m = p.mesh();
    //    auto id = m->id();
    //    m_mesh_->move_to(p.mesh());
    //    for (auto &item:attributes()) { item->move_to(m, p.data(item->id())); }
    ASSERT(m_mesh_ != nullptr);
    m_mesh_->accept(p);
    pre_process();
}

void Worker::deploy() {
    concept::LifeControllable::deploy();
    if (m_mesh_ != nullptr) m_mesh_->deploy();
}

void Worker::destroy() {
    m_mesh_ = nullptr;
    concept::LifeControllable::destroy();
}

void Worker::pre_process() {
    if (!is_valid()) { concept::LifeControllable::pre_process(); }
    ASSERT(m_mesh_ != nullptr);
    m_mesh_->pre_process();
}

void Worker::post_process() {
    ASSERT(m_mesh_ != nullptr);
    m_mesh_->post_process();
    if (is_valid()) { concept::LifeControllable::post_process(); }
}

void Worker::initialize(Real data_time, Real dt) {
    pre_process();
    ASSERT(m_mesh_ != nullptr);
    m_mesh_->initialize(data_time, dt);
}

void Worker::finalize(Real data_time, Real dt) {
    m_mesh_->finalize(data_time, dt);
    post_process();
}

void Worker::sync() {}
//
// void Worker::phase(unsigned int num, Real data_time, Real dt)
//{
//    concept::LifeControllable::phase(num);
//    switch (num)
//    {
//        #define PHASE(_N_) case _N_: phase##_N_(data_time, dt); break;
//
//        PHASE(0);
//        PHASE(1);
//        PHASE(2);
//        PHASE(3);
//        PHASE(4);
//        PHASE(5);
//        PHASE(6);
//        PHASE(7);
//        PHASE(8);
//        PHASE(9);
//
//        #undef NEXT_PHASE
//        default:
//            break;
//    }
//}
//
// unsigned int Worker::next_phase(Real data_time, Real dt, unsigned int inc_phase)
//{
//    unsigned int start_phase = current_phase_num();
//    unsigned int end_phase = concept::LifeControllable::next_phase(inc_phase);
//
//    switch (start_phase)
//    {
//        #define NEXT_PHASE(_N_) case _N_: phase##_N_(data_time, dt);sync();++start_phase;if
//        (start_phase >=end_phase )break;
//
//        NEXT_PHASE(0);
//        NEXT_PHASE(1);
//        NEXT_PHASE(2);
//        NEXT_PHASE(3);
//        NEXT_PHASE(4);
//        NEXT_PHASE(5);
//        NEXT_PHASE(6);
//        NEXT_PHASE(7);
//        NEXT_PHASE(8);
//        NEXT_PHASE(9);
//
//        #undef NEXT_PHASE
//        default:
//            break;
//    }
//    return end_phase;
//};
}
}  // namespace simpla { namespace mesh1
