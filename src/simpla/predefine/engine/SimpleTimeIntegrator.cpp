//
// Created by salmon on 17-9-5.

#include "SimpleTimeIntegrator.h"
#include "simpla/engine/Domain.h"
#include "simpla/engine/Mesh.h"
#include "simpla/engine/MeshBlock.h"
namespace simpla {

SimpleTimeIntegrator::SimpleTimeIntegrator() {}

SimpleTimeIntegrator::~SimpleTimeIntegrator() {}

std::shared_ptr<simpla::data::DataNode> SimpleTimeIntegrator::Serialize() const { return base_type::Serialize(); }
void SimpleTimeIntegrator::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
}

void SimpleTimeIntegrator::DoSetUp() { base_type::DoSetUp(); }
void SimpleTimeIntegrator::DoUpdate() { base_type::DoUpdate(); }
void SimpleTimeIntegrator::DoTearDown() { base_type::DoTearDown(); }

void SimpleTimeIntegrator::Synchronize() { Update(); }

void SimpleTimeIntegrator::Advance(Real time_now, Real time_dt) {
    Update();
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