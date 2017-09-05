//
// Created by salmon on 17-9-5.

#include "SimpleTimeIntegrator.h"
#include "simpla/engine/Domain.h"
#include "simpla/engine/Mesh.h"
#include "simpla/engine/MeshBlock.h"
namespace simpla {
void SimpleTimeIntegrator::DoInitialize() { base_type::DoSetUp(); }
void SimpleTimeIntegrator::DoFinalize() { base_type::DoFinalize(); }
void SimpleTimeIntegrator::DoUpdate() { base_type::DoUpdate(); }
void SimpleTimeIntegrator::DoTearDown() { base_type::DoTearDown(); }

void SimpleTimeIntegrator::Advance(Real time_now, Real time_dt) {
    GetAtlas()->Foreach([&](std::shared_ptr<engine::MeshBlock> const &blk) {
        auto p = GetPatch(blk->GetGUID());

        for (auto &item : GetDomains()) {
            if (item.second->Push(blk, p)) {
                item.second->Advance(time_now, time_dt);
                p = item.second->Pop();
            };
        }
        SetPatch(blk->GetGUID(), p);

    });
}

}  // namespace simpla