//
// Created by salmon on 16-11-4.
//
#include "Worker.h"
#include "AttributeView.h"
#include "DomainView.h"
#include "Patch.h"

namespace simpla {
namespace engine {
struct Worker::pimpl_s {
    std::shared_ptr<Worker> m_next_ = nullptr;
    DomainView *m_domain_;
};
Worker::Worker() : m_pimpl_(new pimpl_s) {}
Worker::~Worker(){};
std::shared_ptr<Worker> &Worker::next() { return m_pimpl_->m_next_; }
std::shared_ptr<Worker> const &Worker::next() const { return m_pimpl_->m_next_; }

std::ostream &Worker::Print(std::ostream &os, int indent) const {
    //    os << std::setw(indent + 1) << " "
    //       << " [" << getClassName() << " : " << name() << "]" << std::endl;
    os << std::setw(indent + 1) << "  type = \"" << getClassName() << "\", config = {" << db << "},";
    os << std::setw(indent + 1) << " attributes = { ";
    AttributeViewBundle::Print(os, indent);
    os << "  } , ";
    return os;
}

/**
 * @startuml
 * title Initialize/Finalize
 * actor  Main
 * participant Worker as Worker << base >>
 * participant Worker as EMWorker << Generated >>
 * participant Patch
 * participant MeshView
 * Main -> Worker : << Initialize >>
 * activate Worker
 * Worker -> EMWorker : << Initialize >>
 *    activate EMWorker
 *          create MeshView
 *          EMWorker -> MeshView :  <<create MeshView>>
 *          create AttributeView
 *          EMWorker -> AttributeView :  <<create AttributeView>>
 *          EMWorker --> Worker :Done
 *    deactivate EMWorker
 *    Worker -> MeshView : Set up MeshView with Patch
 *    activate MeshView
 *          MeshView -> Patch      : << require mesh block >>
 *          activate Patch
 *              Patch --> MeshView  : return mesh block
 *          deactivate Patch
 *          MeshView -> MeshView : Initialize
 *          MeshView --> Worker : Done
 *    deactivate MeshView
 *    Worker -> AttributeView  : Set up AttributeView with Patch
 *    activate AttributeView
 *          AttributeView -> Worker: <<require MeshView>>
 *          Worker --> AttributeView : return MeshView
 *         AttributeView -> Patch   : require DataBlock  at AttributeId
 *         Patch --> AttributeView  : return DataBlock
 *         alt if DataBlock ==nullptr
 *              AttributeView -> MeshView : << Create DataBlock of AttributeId >>
 *              MeshView --> AttributeView : return DataBlock
 *         end
 *         AttributeView --> Worker : Done
 *    deactivate AttributeView
 *    Worker -> EMWorker : Initialize
 *    activate EMWorker
 *          EMWorker -> EMWorker : Initialize
 *          EMWorker --> Worker   : Done
 *    deactivate EMWorker
 *    Worker --> Main: Done
 * deactivate Worker
 * @enduml
 */

void Worker::Update() {
    if (isUpdated()) { return; }
    AttributeViewBundle::Update();
    Initialize();

    if (next() != nullptr) { next()->Update(); }
}
void Worker::Evaluate() {
    Update();
    Process();
    if (next() != nullptr) { next()->Evaluate(); }
};

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
