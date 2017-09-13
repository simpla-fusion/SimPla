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
    void InitializeAttribute(AttributeT<U, IFORM, DOF...> *attr) const;
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

#define SP_DOMAIN_POLICY_HEAD(_NAME_)                  \
   private:                                            \
    typedef THost host_type;                           \
    typedef _NAME_<THost> this_type;                   \
    THost *m_host_;                                    \
                                                       \
   public:                                             \
    _NAME_(THost *h) : m_host_(h) {}                   \
    virtual ~_NAME_() = default;                       \
    _NAME_(_NAME_ const &other) = delete;              \
    _NAME_(_NAME_ &&other) = delete;                   \
    _NAME_ &operator=(_NAME_ const &other) = delete;   \
    _NAME_ &operator=(_NAME_ &&other) = delete;        \
    std::shared_ptr<data::DataNode> Serialize() const; \
    void Deserialize(std::shared_ptr<data::DataNode> const &cfg);

template <typename TChart, template <typename> class... Policies>
Domain<TChart, Policies...>::Domain() : DomainBase(), Policies<this_type>(this)... {}
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
namespace detail {
template <typename U, typename SFS>
void InitializeArray(Array<U> &v, SFS const &sfc) {
    Array<U>(sfc).swap(v);
}
template <typename U, int N0, int... N, typename SFC>
void InitializeArray(nTuple<Array<U>, N0, N...> &v, SFC const &sfc) {
    for (int i = 0; i < N0; ++i) { InitializeArray(v[i], sfc); }
}
}  // namespace detail{

template <typename TChart, template <typename> class... Policies>
template <typename U, int IFORM, int... DOF>
void Domain<TChart, Policies...>::InitializeAttribute(AttributeT<U, IFORM, DOF...> *attr) const {
    switch (IFORM) {
        case NODE:
            detail::InitializeArray(*attr, this->GetSpaceFillingCurve(NODE));
            break;
        case CELL:
            detail::InitializeArray(*attr, this->GetSpaceFillingCurve(CELL));
            break;
        case EDGE:
            detail::InitializeArray((*attr)[0], this->GetSpaceFillingCurve(EDGE, 0));
            detail::InitializeArray((*attr)[1], this->GetSpaceFillingCurve(EDGE, 1));
            detail::InitializeArray((*attr)[2], this->GetSpaceFillingCurve(EDGE, 2));
            break;
        case FACE:
            detail::InitializeArray((*attr)[0], this->GetSpaceFillingCurve(FACE, 0));
            detail::InitializeArray((*attr)[1], this->GetSpaceFillingCurve(FACE, 1));
            detail::InitializeArray((*attr)[2], this->GetSpaceFillingCurve(FACE, 2));
            break;
        default:
            UNIMPLEMENTED;
            break;
    }
};

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
