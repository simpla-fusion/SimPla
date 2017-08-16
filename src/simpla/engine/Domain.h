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
#include "Model.h"
#include "PoliciesCommon.h"

namespace simpla {
namespace engine {
class Patch;
class Model;
class MeshBase;

class DomainBase : public SPObject {
    SP_OBJECT_DECLARE_MEMBERS(DomainBase, SPObject)
   protected:
    explicit DomainBase(std::shared_ptr<MeshBase> const &m, std::shared_ptr<Model> const &model = nullptr);

   public:
    void DoInitialize() override;
    void DoFinalize() override;
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
    virtual void DoAdvance(Real time_now, Real dt) {}
    design_pattern::Signal<void(DomainBase *, Real, Real)> PostAdvance;
    void Advance(Real time_now, Real time_dt);

    void SetGeoBody(const std::shared_ptr<geometry::GeoObject> &b) { m_geo_body_ = b; }
    std::shared_ptr<geometry::GeoObject> GetGeoBody() const { return m_geo_body_; }

    void SetModel(std::shared_ptr<engine::Model> const &m) { m_model_ = m; }
    std::shared_ptr<Model> GetModel() const { return m_model_; }

    virtual const MeshBase *GetMesh() const { return m_mesh_.get(); }
    virtual MeshBase *GetMesh() { return m_mesh_.get(); }

   private:
    std::shared_ptr<MeshBase> m_mesh_ = nullptr;
    std::shared_ptr<engine::Model> m_model_ = nullptr;
    std::shared_ptr<geometry::GeoObject> m_geo_body_ = nullptr;

};  // class DomainBase

template <typename TM, template <typename> class... Policies>
class Domain : public DomainBase, public Policies<Domain<TM, Policies...>>... {
    typedef TM mesh_type;
    SP_OBJECT_DECLARE_MEMBERS(Domain, DomainBase);

   protected:
    template <typename... Args>
    explicit Domain(Args &&... args) : DomainBase(std::forward<Args>(args)...), Policies<this_type>(this)... {}

   public:
    void DoInitialCondition(Real time_now) override;
    void DoBoundaryCondition(Real time_now, Real dt) override;
    void DoAdvance(Real time_now, Real dt) override;
    void DoTagRefinementCells(Real time_now) override;

    mesh_type const *GetMesh() const override { return dynamic_cast<mesh_type const *>(DomainBase::GetMesh()); }
    mesh_type *GetMesh() override { return dynamic_cast<mesh_type *>(DomainBase::GetMesh()); }

    template <typename TL, typename TR>
    void Fill(TL &lhs, TR &&rhs) const {
        FillBody(lhs, std::forward<TR>(rhs));
    };

    template <typename TL, typename TR, typename... Others>
    void FillRange(TL &lhs, TR &&rhs, Others &&... others) const {
        GetMesh()->FillRange(lhs, std::forward<TR>(rhs), std::forward<Others>(others)...);
    };

    template <typename TL, typename TR>
    void FillBody(TL &lhs, TR &&rhs) const {
        GetMesh()->FillBody(lhs, std::forward<TR>(rhs), GetName());
    };

    template <typename TL, typename TR>
    void FillBoundary(TL &lhs, TR &&rhs) const {
        GetMesh()->FillBoundary(lhs, std::forward<TR>(rhs), GetName());
    };

};  // class Domain

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

template <typename TM, template <typename> class... Policies>
void Domain<TM, Policies...>::Serialize(std::shared_ptr<data::DataEntity> const &cfg) const {
    DomainBase::Serialize(cfg);
    traits::_try_invoke_Serialize<Policies...>(this, cfg);
};

template <typename TM, template <typename> class... Policies>
void Domain<TM, Policies...>::Deserialize(std::shared_ptr<const data::DataEntity> const &cfg)  {
    DomainBase::Deserialize(cfg);
    traits::_try_invoke_Deserialize<Policies...>(this, cfg);
};
}  // namespace engine
}  // namespace simpla
#endif  // SIMPLA_DOMAINBASE_H
