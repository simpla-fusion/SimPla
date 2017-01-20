//
// Created by salmon on 16-11-4.
//
#include "Worker.h"
#include "Attribute.h"
#include "Mesh.h"
#include "Patch.h"
namespace simpla {
namespace mesh {

Worker::Worker() {}

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

void Worker::Accept(std::shared_ptr<Patch> p) {
    Finalize();
    m_patch_ = p;
    Initialize();
}
void Worker::Release() {
    Finalize();
    m_patch_.reset();
}

void Worker::Deploy() {
    Object::Deploy();
    if (m_mesh_ == nullptr) { m_mesh_ = create_mesh(); };
    if (m_patch_ == nullptr) { m_patch_ = std::make_shared<Patch>(); }

    m_mesh_->Deploy();
}

void Worker::Destroy() {
    m_mesh_.reset();
    m_patch_.reset();
    Object::Destroy();
}
void Worker::Initialize() {
    Object::Initialize();
    ASSERT(m_mesh_ != nullptr);
    ASSERT(m_patch_ != nullptr);
    m_mesh_->mesh_block(m_patch_->mesh_block());
    for (auto attr : m_attrs_) { attr->data_block(m_patch_->data(attr->description().id())); }
    m_mesh_->Initialize();
}

void Worker::Finalize() {
    ASSERT(m_mesh_ != nullptr);
    m_patch_->mesh_block(m_mesh_->mesh_block());
    for (auto attr : m_attrs_) { m_patch_->data(attr->description().id(), attr->data_block()); }
    m_mesh_->Finalize();
    Object::Finalize();
}
void Worker::PreProcess() {
    if (isReady()) { return; }
    Object::PreProcess();
    ASSERT(m_mesh_ != nullptr);
    m_mesh_->PreProcess();
}

void Worker::PostProcess() {
    ASSERT(m_mesh_ != nullptr);
    m_mesh_->PostProcess();
    Object::PostProcess();
}
void Worker::Sync() {}
//
// void Worker::phase(unsigned int num, Real data_time, Real dt)
//{
//    Object::phase(num);
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
//    unsigned int end_phase = Object::next_phase(inc_phase);
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
