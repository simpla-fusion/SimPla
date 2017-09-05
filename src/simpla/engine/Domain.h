//
// Created by salmon on 17-4-5.
//

#ifndef SIMPLA_DOMAINBASE_H
#define SIMPLA_DOMAINBASE_H

#include "simpla/SIMPLA_config.h"

#include <memory>

#include "simpla/algebra/Array.h"
#include "simpla/data/Data.h"
#include "simpla/geometry/Chart.h"
#include "simpla/utilities/Signal.h"

#include "Attribute.h"
#include "EngineObject.h"
#include "Model.h"
#include "PoliciesCommon.h"

namespace simpla {
namespace engine {
class Model;
class MeshBase;

class DomainBase : public EngineObject {
    SP_OBJECT_HEAD(DomainBase, EngineObject)

   protected:
    explicit DomainBase(std::shared_ptr<MeshBase> const &m, std::shared_ptr<Model> const &model = nullptr);

   public:
    void SetMesh(std::shared_ptr<MeshBase> const &m);
    std::shared_ptr<MeshBase> GetMesh() const;
    void SetModel(std::shared_ptr<Model> const &m);
    std::shared_ptr<Model> GetModel() const;

    void SetBlock(const std::shared_ptr<MeshBlock> &blk);
    std::shared_ptr<MeshBlock> GetBlock() const;

    bool CheckOverlap(const std::shared_ptr<MeshBlock> &blk) const;
    bool Push(std::shared_ptr<engine::MeshBlock> const &, std::shared_ptr<data::DataNode> const &);
    std::shared_ptr<data::DataNode> Pop() override;

    void DoSetUp() override;
    void DoUpdate() override;
    void DoTearDown() override;

    design_pattern::Signal<void(DomainBase *, Real)> PreInitialCondition;
    virtual void DoInitialCondition(Real time_now) {}
    design_pattern::Signal<void(DomainBase *, Real)> PostInitialCondition;
    void InitialCondition(Real time_now);

    design_pattern::Signal<void(DomainBase *, Real)> PreTagRefinementCells;
    virtual void DoTagRefinementCells(Real time_now) {}
    design_pattern::Signal<void(DomainBase *, Real)> PostTagRefinementCells;
    void TagRefinementCells(Real time_now);

    design_pattern::Signal<void(DomainBase *, Real, Real)> PreBoundaryCondition;
    virtual void DoBoundaryCondition(Real time_now, Real dt) {}
    design_pattern::Signal<void(DomainBase *, Real, Real)> PostBoundaryCondition;
    void BoundaryCondition(Real time_now, Real time_dt);

    design_pattern::Signal<void(DomainBase *, Real, Real)> PreComputeFluxes;
    virtual void DoComputeFluxes(Real time_now, Real dt) {}
    design_pattern::Signal<void(DomainBase *, Real, Real)> PostComputeFluxes;

    void ComputeFluxes(Real time_now, Real time_dt);

    Real ComputeStableDtOnPatch(Real time_now, Real time_dt) const;

    design_pattern::Signal<void(DomainBase *, Real, Real)> PreAdvance;
    virtual Real DoAdvance(Real time_now, Real dt) {}
    design_pattern::Signal<void(DomainBase *, Real, Real)> PostAdvance;
    Real Advance(Real time_now, Real time_dt);

};  // class DomainBase

template <typename TM, template <typename> class... Policies>
class Domain : public DomainBase, public Policies<Domain<TM, Policies...>>... {
    typedef TM mesh_type;
    SP_OBJECT_HEAD(Domain, DomainBase);

   protected:
    template <typename... Args>
    explicit Domain(Args &&... args) : DomainBase(std::forward<Args>(args)...), Policies<this_type>(this)... {}

   public:
    void DoSetUp() override;
    void DoUpdate() override;
    void DoTearDown() override;

    void DoInitialCondition(Real time_now) override;
    void DoBoundaryCondition(Real time_now, Real dt) override;
    void DoAdvance(Real time_now, Real dt) override;
    void DoTagRefinementCells(Real time_now) override;

    mesh_type const *mesh() const { return dynamic_cast<mesh_type const *>(DomainBase::GetMesh()); }
    mesh_type *mesh() { return dynamic_cast<mesh_type *>(DomainBase::GetMesh()); }

    template <typename TL, typename TR>
    void Fill(TL &lhs, TR &&rhs) const {
        FillBody(lhs, std::forward<TR>(rhs));
    };

    template <typename TL, typename TR, typename... Others>
    void FillRange(TL &lhs, TR &&rhs, Others &&... others) const {
        mesh()->FillRange(lhs, std::forward<TR>(rhs), std::forward<Others>(others)...);
    };

    template <typename TL, typename TR>
    void FillBody(TL &lhs, TR &&rhs) const {
        mesh()->FillBody(lhs, std::forward<TR>(rhs), GetName());
    };

    template <typename TL, typename TR>
    void FillBoundary(TL &lhs, TR &&rhs) const {
        mesh()->FillBoundary(lhs, std::forward<TR>(rhs), GetName());
    };

};  // class Domain
template <typename TM, template <typename> class... Policies>
Domain<TM, Policies...>::Domain() : Policies<this_type>(this)... {};
template <typename TM, template <typename> class... Policies>
Domain<TM, Policies...>::~Domain(){};

template <typename TM, template <typename> class... Policies>
std::shared_ptr<data::DataNode> Domain<TM, Policies...>::Serialize() const {
    auto cfg = DomainBase::Serialize();
    traits::_try_invoke_Serialize<Policies...>(this, cfg);
    return cfg;
};

template <typename TM, template <typename> class... Policies>
void Domain<TM, Policies...>::Deserialize(std::shared_ptr<data::DataNode> const &cfg) {
    DomainBase::Deserialize(cfg);
    traits::_try_invoke_Deserialize<Policies...>(this, cfg);
};

template <typename TM, template <typename> class... Policies>
void Domain<TM, Policies...>::DoSetUp() {
    if (GetMesh() == nullptr) { SetMesh(TM::New(this)); }
    base_type::DoSetUp();
};
template <typename TM, template <typename> class... Policies>
void Domain<TM, Policies...>::DoUpdate() {
    base_type::DoUpdate();
};
template <typename TM, template <typename> class... Policies>
void Domain<TM, Policies...>::DoTearDown() {
    base_type::DoTearDown();
};

template <typename TM, template <typename> class... Policies>
void Domain<TM, Policies...>::DoInitialCondition(Real time_now) {
    simpla::traits::_try_invoke_InitialCondition<Policies...>(this, time_now);
}

template <typename TM, template <typename> class... Policies>
void Domain<TM, Policies...>::DoBoundaryCondition(Real time_now, Real dt) {
    traits::_try_invoke_BoundaryCondition<Policies...>(this, time_now, dt);
}

template <typename TM, template <typename> class... Policies>
void Domain<TM, Policies...>::DoAdvance(Real time_now, Real dt) {
    traits::_try_invoke_Advance<Policies...>(this, time_now, dt);
}

template <typename TM, template <typename> class... Policies>
void Domain<TM, Policies...>::DoTagRefinementCells(Real time_now) {
    traits::_try_invoke_TagRefinementCells<Policies...>(this, time_now);
}

}  // namespace engine
}  // namespace simpla
#endif  // SIMPLA_DOMAINBASE_H
