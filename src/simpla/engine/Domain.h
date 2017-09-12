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
#include "MeshBlock.h"
namespace simpla {
namespace engine {

class DomainBase : public EngineObject, public AttributeGroup {
    SP_OBJECT_HEAD(DomainBase, EngineObject)

   public:
    void Push(const std::shared_ptr<data::DataNode> &) override;
    std::shared_ptr<data::DataNode> Pop() const override;
    int GetNDIMS() const;
    void SetChart(std::shared_ptr<geometry::Chart> const &c);
    virtual std::shared_ptr<geometry::Chart> GetChart();
    virtual std::shared_ptr<const geometry::Chart> GetChart() const;

    void SetBlock(const std::shared_ptr<const MeshBlock> &blk);
    std::shared_ptr<const MeshBlock> GetBlock() const;

    void SetBoundary(std::shared_ptr<geometry::GeoObject> const &g);
    std::shared_ptr<geometry::GeoObject> GetBoundary() const;

    bool CheckOverlap(const std::shared_ptr<MeshBlock> &blk) const;

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
    virtual void DoAdvance(Real time_now, Real dt) {}
    design_pattern::Signal<void(DomainBase *, Real, Real)> PostAdvance;
    void Advance(Real time_now, Real time_dt);

    virtual void InitializeAttribute(Attribute *) const;

};  // class DomainBase

template <typename TChart, template <typename> class... Policies>
class Domain : public DomainBase, public Policies<Domain<TChart, Policies...>>... {
    typedef TChart chart_type;
    SP_OBJECT_HEAD(Domain, DomainBase);

   public:
    std::shared_ptr<const geometry::Chart> GetChart() const override { return DomainBase::GetChart(); };
    virtual std::shared_ptr<const engine::MeshBlock> GetBlock() const override { return DomainBase::GetBlock(); };

    void DoSetUp() override;
    void DoUpdate() override;
    void DoTearDown() override;

    void DoInitialCondition(Real time_now) override;
    void DoBoundaryCondition(Real time_now, Real dt) override;
    void DoAdvance(Real time_now, Real dt) override;
    void DoTagRefinementCells(Real time_now) override;

    void SetRange(std::string const &, Range<EntityId> const &);
    Range<EntityId> GetRange(std::string const &k) const;
    template <typename TL, typename TR>
    void Fill(TL &lhs, TR &&rhs) const {
        FillRange(lhs, std::forward<TR>(rhs), Range<EntityId>{}, true);
        //        FillRange(lhs, 0, "PATCH_BOUNDARY_" + std::to_string(TL::iform), false);
    };

    template <typename TL, typename TR>
    void FillRange(TL &lhs, TR &&rhs, Range<EntityId> r, bool full_fill_if_range_is_null = false) const;

    template <typename TL, typename TR>
    void FillRange(TL &lhs, TR &&rhs, std::string const &k = "", bool full_fill_if_range_is_null = false) const {
        FillRange(lhs, std::forward<TR>(rhs), GetRange(k), full_fill_if_range_is_null);
    };

    template <typename TL, typename TR>
    void FillBody(TL &lhs, TR &&rhs, std::string const &prefix = "") const {
        FillRange(lhs, std::forward<TR>(rhs), prefix + "_BODY_" + std::to_string(TL::iform), false);
    };

    template <typename TL, typename TR>
    void FillBoundary(TL &lhs, TR &&rhs, std::string const &prefix = "") const {
        FillRange(lhs, std::forward<TR>(rhs), prefix + "_BOUNDARY_" + std::to_string(TL::iform), false);
    };

    template <typename U, int IFORM, int... DOF>
    void InitializeAttribute(AttributeT<U, IFORM, DOF...> *attr) const {
        //        base_type::InitializeAttribute(attr);
    };
};  // class Domain

#define SP_DOMAIN_HEAD(_CLASS_NAME_, _BASE_NAME_)              \
    SP_OBJECT_HEAD(_CLASS_NAME_, _BASE_NAME_);                 \
    void DoSetUp() override;                                   \
    void DoUpdate() override;                                  \
    void DoTearDown() override;                                \
                                                               \
    void DoInitialCondition(Real time_now) override;           \
    void DoBoundaryCondition(Real time_now, Real dt) override; \
    void DoAdvance(Real time_now, Real dt) override;           \
    void DoTagRefinementCells(Real time_now) override;

template <typename TChart, template <typename> class... Policies>
Domain<TChart, Policies...>::Domain() : DomainBase() {}
template <typename TChart, template <typename> class... Policies>
Domain<TChart, Policies...>::~Domain(){};

template <typename TChart, template <typename> class... Policies>
std::shared_ptr<data::DataNode> Domain<TChart, Policies...>::Serialize() const {
    auto cfg = DomainBase::Serialize();
    return cfg;
};

template <typename TChart, template <typename> class... Policies>
void Domain<TChart, Policies...>::Deserialize(std::shared_ptr<data::DataNode> const &cfg) {
    DomainBase::Deserialize(cfg);
};

template <typename TChart, template <typename> class... Policies>
void Domain<TChart, Policies...>::DoSetUp() {
    base_type::DoSetUp();
};
template <typename TChart, template <typename> class... Policies>
void Domain<TChart, Policies...>::DoUpdate() {
    base_type::DoUpdate();
};
template <typename TChart, template <typename> class... Policies>
void Domain<TChart, Policies...>::DoTearDown() {
    base_type::DoTearDown();
};

template <typename TChart, template <typename> class... Policies>
void Domain<TChart, Policies...>::DoInitialCondition(Real time_now) {}

template <typename TChart, template <typename> class... Policies>
void Domain<TChart, Policies...>::DoBoundaryCondition(Real time_now, Real dt) {}

template <typename TChart, template <typename> class... Policies>
void Domain<TChart, Policies...>::DoAdvance(Real time_now, Real dt) {}

template <typename TChart, template <typename> class... Policies>
void Domain<TChart, Policies...>::DoTagRefinementCells(Real time_now) {}

template <typename TM, template <typename> class... Policies>
template <typename LHS, typename RHS>
void Domain<TM, Policies...>::FillRange(LHS &lhs, RHS &&rhs, Range<EntityId> r, bool full_fill_if_range_is_null) const {
    //    if (r.isFull() || (r.isNull() && full_fill_if_range_is_null)) {
    //        this_type::Calculate(this, lhs, std::forward<RHS>(rhs));
    //    } else {
    //        this_type::Calculate(this, lhs, std::forward<RHS>(rhs), r);
    //    }
};

template <typename TM, template <typename> class... Policies>
void Domain<TM, Policies...>::SetRange(std::string const &, Range<EntityId> const &){};
template <typename TM, template <typename> class... Policies>
Range<EntityId> Domain<TM, Policies...>::GetRange(std::string const &k) const {
    return Range<EntityId>{};
};
}  // namespace engine
}  // namespace simpla
#endif  // SIMPLA_DOMAINBASE_H
