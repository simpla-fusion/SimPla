//
// Created by salmon on 16-11-4.
//
#include "Worker.h"

namespace simpla {
namespace mesh {

Worker::Worker(Mesh *m) : m_mesh_(m) {}

Worker::~Worker(){};

std::ostream &Worker::Print(std::ostream &os, int indent) const {
    os << std::setw(indent + 1) << " "
       << " [" << getClassName() << " : " << name() << "]" << std::endl;

    os << std::setw(indent + 1) << " "
       << "Config = {" << db << "}" << std::endl;

    if (m_mesh_ != nullptr) {
        os << std::setw(indent + 1) << " "
           << "Mesh = " << std::endl
           << std::setw(indent + 1) << " "
           << "{ " << std::endl;
        m_mesh_->Print(os, indent + 1);
        os << std::setw(indent + 1) << " "
           << "}," << std::endl;
    }
    return os;
}

void Worker::Accept(Patch *p) {
    PostProcess();
    ASSERT(m_mesh_ != nullptr);
    m_mesh_->Accept(p);
    PreProcess();
}

void Worker::Deploy() {
    concept::LifeControllable::Deploy();
    ASSERT(m_mesh_ != nullptr);
    m_mesh_->Deploy();
}

void Worker::Destroy() { concept::LifeControllable::Destroy(); }

void Worker::PreProcess() {
    if (!isValid()) { concept::LifeControllable::PreProcess(); }
    ASSERT(m_mesh_ != nullptr);
    m_mesh_->PreProcess();
}

void Worker::PostProcess() {
    ASSERT(m_mesh_ != nullptr);
    m_mesh_->PostProcess();
    if (isValid()) { concept::LifeControllable::PostProcess(); }
}

void Worker::Initialize(Real data_time, Real dt) {
    PreProcess();
    ASSERT(m_mesh_ != nullptr);
    m_mesh_->Initialize(data_time, dt);
}

void Worker::Finalize(Real data_time, Real dt) {
    ASSERT(m_mesh_ != nullptr);
    m_mesh_->Finalize(data_time, dt);
    PostProcess();
}

void Worker::Sync() {}
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
//        #define NEXT_PHASE(_N_) case _N_: phase##_N_(data_time, dt);Sync();++start_phase;if
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
