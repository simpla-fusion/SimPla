//
// Created by salmon on 17-4-5.
//

#ifndef SIMPLA_DOMAINBASE_H
#define SIMPLA_DOMAINBASE_H

#include "simpla/SIMPLA_config.h"

#include <memory>

#include "simpla/algebra/Array.h"
#include "simpla/algebra/nTuple.h"
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
    std::string TypeName() const final { return "DomainBase"; }

   public:
    //    void Push(const std::shared_ptr<data::DataNode> &) override;
    //    std::shared_ptr<data::DataNode> Pop() const override;

    void Push(const std::shared_ptr<Patch> &) override;
    std::shared_ptr<Patch> Pop() const override;

    int GetNDIMS() const;
    void SetChart(std::shared_ptr<geometry::Chart> const &c);
    virtual std::shared_ptr<geometry::Chart> GetChart();
    virtual std::shared_ptr<const geometry::Chart> GetChart() const;

    void SetMeshBlock(const std::shared_ptr<const MeshBlock> &blk);
    virtual std::shared_ptr<const MeshBlock> GetMeshBlock() const;

    virtual int CheckBoundary() const;
    bool isOutOfBoundary() const;
    bool isOnBoundary() const;
    bool isFirstTime() const;

    void SetBoundary(std::shared_ptr<geometry::GeoObject> const &g);
    std::shared_ptr<geometry::GeoObject> GetBoundary() const;
    box_type GetBlockBox() const;
    std::shared_ptr<geometry::GeoObject> GetBlockBoundingBox() const;

    void DoSetUp() override;
    void DoUpdate() override;
    void DoTearDown() override;
    design_pattern::Signal<void(DomainBase *, std::shared_ptr<simpla::data::DataNode> const &)> OnDeserialize;
    design_pattern::Signal<void(DomainBase const *, std::shared_ptr<simpla::data::DataNode> &)> OnSerialize;

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
    std::shared_ptr<const engine::MeshBlock> GetMeshBlock() const override { return DomainBase::GetMeshBlock(); };

    std::shared_ptr<data::DataNode> db() const override { return SPObject::db(); }
    std::shared_ptr<data::DataNode> db() override { return SPObject::db(); }
    void db(std::shared_ptr<data::DataNode> const &d) override { SPObject::db(d); }

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
    void Fill(TL &lhs, TR const &rhs) const {
        Fill(lhs, rhs, Range<EntityId>{});
        //        Fill(lhs, 0, "PATCH_BOUNDARY_" + std::to_string(TL::iform));
    };

    template <typename V, int IFORM, int... DOF, typename TR>
    void Fill(AttributeT<V, IFORM, DOF...> &lhs, TR const &rhs, const Range<EntityId> &r) const;

    template <typename TL, typename TR>
    void Fill(TL &lhs, TR const &rhs, std::string const &k) const {
        Fill(lhs, (rhs), GetRange(k));
    };

    template <typename TL, typename TR>
    void FillBody(TL &lhs, TR const &rhs, std::string const &prefix = "") const {
        //        FillRange(lhs, (rhs), prefix + "_BODY_" + std::to_string(TL::iform));
    };

    template <typename TL, typename TR>
    void FillBoundary(TL &lhs, TR const &rhs, std::string const &prefix = "") const {
        //        FillRange(lhs, (rhs), prefix + "_BOUNDARY_" + std::to_string(TL::iform));
    };

    template <typename U, int IFORM, int... DOF>
    void InitializeAttribute(AttributeT<U, IFORM, DOF...> *attr) const;

};  // class Domain

#define SP_DOMAIN_HEAD(_CLASS_NAME_, _BASE_NAME_)    \
    SP_OBJECT_HEAD(_CLASS_NAME_, _BASE_NAME_);       \
    void DoSetUp() override;                         \
    void DoUpdate() override;                        \
    void DoTearDown() override;                      \
                                                     \
    void DoInitialCondition(Real time_now) override; \
    void DoAdvance(Real time_now, Real dt) override; \
    void DoTagRefinementCells(Real time_now) override;

#define SP_DOMAIN_POLICY_HEAD(_NAME_)                  \
   private:                                            \
    typedef THost host_type;                           \
    typedef _NAME_<THost> this_type;                   \
    THost *m_host_;                                    \
                                                       \
   public:                                             \
    _NAME_(THost *h);                                  \
    virtual ~_NAME_();                                 \
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
template <typename U, typename SFC>
void InitializeArray_(Array<U, SFC> &v, SFC const &sfc) {
    Array<U, SFC>(sfc).swap(v);
    v.alloc();
}
template <typename U, int N0, int... N, typename SFC>
void InitializeArray_(nTuple<simpla::Array<U, SFC>, N0, N...> &v, SFC const &sfc) {
    for (int i = 0; i < N0; ++i) { InitializeArray_(v[i], sfc); }
}
template <typename TArray, typename THost>
void InitializeArray(std::integral_constant<int, NODE>, TArray &v, THost const *host) {
    InitializeArray_(v, host->GetSpaceFillingCurve(0b000));
}
template <typename TArray, typename THost>
void InitializeArray(std::integral_constant<int, CELL>, TArray &v, THost const *host) {
    InitializeArray_(v, host->GetSpaceFillingCurve(0b000));
}
template <typename TArray, typename THost>
void InitializeArray(std::integral_constant<int, EDGE>, TArray &v, THost const *host) {
    InitializeArray_(v[0], host->GetSpaceFillingCurve(0b001));
    InitializeArray_(v[1], host->GetSpaceFillingCurve(0b010));
    InitializeArray_(v[2], host->GetSpaceFillingCurve(0b100));
}
template <typename TArray, typename THost>
void InitializeArray(std::integral_constant<int, FACE>, TArray &v, THost const *host) {
    InitializeArray_(v[0], host->GetSpaceFillingCurve(0b110));
    InitializeArray_(v[1], host->GetSpaceFillingCurve(0b101));
    InitializeArray_(v[2], host->GetSpaceFillingCurve(0b011));
}

template <int IFORM, typename TArray, typename THost>
void InitializeArray(std::integral_constant<int, IFORM>, TArray &v, THost const *host) {
    UNIMPLEMENTED;
}

template <typename THost, typename U, int IFORM, int... DOF, typename RHS>
void DomainAssign(THost *self, engine::AttributeT<U, IFORM, DOF...> &lhs, RHS const &rhs, const Range<EntityId> &r,
                  ENABLE_IF((!simpla::traits::is_invocable<RHS, point_type>::value))) {
    lhs.Assign(rhs);
}
template <typename THost, typename U, int IFORM, int... DOF, typename RHS>
void DomainAssign(THost *self, engine::AttributeT<U, IFORM, DOF...> &lhs, RHS const &rhs, const Range<EntityId> &r,
                  ENABLE_IF((simpla::traits::is_invocable<RHS, point_type>::value))) {
    auto chart = self->GetChart();
    lhs.Assign([&](int w, index_type x, index_type y, index_type z) {
        return rhs(chart->local_coordinates(EntityIdCoder::m_sub_index_to_id_[IFORM][w], x, y, z));
    });
}
template <typename THost, typename U, typename RHS>
void DomainAssign(THost *self, engine::AttributeT<U, NODE> &lhs, RHS const &rhs, const Range<EntityId> &r,
                  ENABLE_IF((simpla::traits::is_invocable<RHS, point_type>::value))) {
    auto chart = self->GetChart();
    lhs.Assign([&](index_type x, index_type y, index_type z) { return rhs(chart->local_coordinates(0, x, y, z)); });
}
template <typename THost, typename U, typename RHS>
void DomainAssign(THost *self, engine::AttributeT<U, CELL> &lhs, RHS const &rhs, const Range<EntityId> &r,
                  ENABLE_IF((simpla::traits::is_invocable<RHS, point_type>::value))) {
    auto chart = self->GetChart();
    lhs.Assign([&](index_type x, index_type y, index_type z) { return rhs(chart->local_coordinates(7, x, y, z)); });
}

template <typename THost, typename V, int... DOF, typename... U>
void DomainAssign(THost *self, AttributeT<V, NODE, DOF...> &lhs, Expression<U...> const &rhs,
                  const Range<EntityId> &r) {
    //    if (r.isFull()) {
    lhs.Assign(self->template Calculate<0>(rhs));
    //    } else {
    //        //        this_type::Calculate(lhs, rhs, r);
    //    }
};
template <typename THost, typename V, int... DOF, typename... U>
void DomainAssign(THost *self, AttributeT<V, EDGE, DOF...> &lhs, Expression<U...> const &rhs,
                  const Range<EntityId> &r) {
    //    if (r.isFull()) {
    lhs.template AssignSub<0>(self->template Calculate<0>(rhs));
    lhs.template AssignSub<1>(self->template Calculate<1>(rhs));
    lhs.template AssignSub<2>(self->template Calculate<2>(rhs));

    //    } else {
    //        //        this_type::Calculate(lhs, rhs, r);
    //    }
};
template <typename THost, typename V, int... DOF, typename... U>
void DomainAssign(THost *self, AttributeT<V, FACE, DOF...> &lhs, Expression<U...> const &rhs,
                  const Range<EntityId> &r) {
    //    if (r.isFull()) {
    lhs.template AssignSub<0>(self->template Calculate<0>(rhs));
    lhs.template AssignSub<1>(self->template Calculate<1>(rhs));
    lhs.template AssignSub<2>(self->template Calculate<2>(rhs));
    //    } else {
    //        //        this_type::Calculate(lhs, rhs, r);
    //    }
};
template <typename THost, typename V, int... DOF, typename... U>
void DomainAssign(THost *self, AttributeT<V, CELL, DOF...> &lhs, Expression<U...> const &rhs,
                  const Range<EntityId> &r) {
    //    if (r.isFull()) {
    lhs.Assign(self->template Calculate<1>(rhs));

    //    } else {
    //        //        this_type::Calculate(lhs, rhs, r);
    //    }
};

// template <typename THost, typename U, int IFORM, int... DOF, typename... RHS>
// void DomainAssign(THost *self, engine::AttributeT<U, IFORM, DOF...> &lhs, Expression<RHS...> const &rhs) {
//    lhs.Assign(rhs);
//}

}  // namespace detail {

template <typename TChart, template <typename> class... Policies>
template <typename U, int IFORM, int... DOF>
void Domain<TChart, Policies...>::InitializeAttribute(AttributeT<U, IFORM, DOF...> *attr) const {
    detail::InitializeArray(std::integral_constant<int, IFORM>(), *attr, this);
};
template <typename TM, template <typename> class... Policies>
template <typename V, int IFORM, int... DOF, typename RHS>
void Domain<TM, Policies...>::Fill(AttributeT<V, IFORM, DOF...> &lhs, RHS const &rhs, const Range<EntityId> &r) const {
    detail::DomainAssign(this, lhs, rhs, r);
};

//
// template <typename TM, template <typename> class... Policies>
// template <typename V, int... DOF, typename... U>
// void Domain<TM, Policies...>::Fill(AttributeT<V, NODE, DOF...> &lhs, Expression<U...> const &rhs,
//                                   const Range<EntityId> &r) const {
//    //    if (r.isFull()) {
//    //    Assign(lhs, this->Calculate<0b000>(rhs));
//
//    //    } else {
//    //        //        this_type::Calculate(lhs, rhs, r);
//    //    }
//};
// template <typename TM, template <typename> class... Policies>
// template <typename V, int... DOF, typename... U>
// void Domain<TM, Policies...>::Fill(AttributeT<V, EDGE, DOF...> &lhs, Expression<U...> const &rhs,
//                                   const Range<EntityId> &r) const {
//    //    if (r.isFull()) {
//    //    Assign(lhs[0], this->Calculate<0b001>(rhs));
//    //    Assign(lhs[1], this->Calculate<0b010>(rhs));
//    //    Assign(lhs[2], this->Calculate<0b100>(rhs));
//
//    //    } else {
//    //        //        this_type::Calculate(lhs, rhs, r);
//    //    }
//};
// template <typename TM, template <typename> class... Policies>
// template <typename V, int... DOF, typename... U>
// void Domain<TM, Policies...>::Fill(AttributeT<V, FACE, DOF...> &lhs, Expression<U...> const &rhs,
//                                   const Range<EntityId> &r) const {
//    //    if (r.isFull()) {
//    //    Assign(lhs[0], this->template Calculate<0b110>(rhs));
//    //    Assign(lhs[1], this->template Calculate<0b101>(rhs));
//    //    Assign(lhs[2], this->template Calculate<0b011>(rhs));
//
//    //    } else {
//    //        //        this_type::Calculate(lhs, rhs, r);
//    //    }
//};
// template <typename TM, template <typename> class... Policies>
// template <typename V, int... DOF, typename... U>
// void Domain<TM, Policies...>::Fill(AttributeT<V, CELL, DOF...> &lhs, Expression<U...> const &rhs,
//                                   const Range<EntityId> &r) const {
//    //    if (r.isFull()) {
//    //    Assign(lhs, this->Calculate<0b111>(rhs));
//
//    //    } else {
//    //        //        this_type::Calculate(lhs, rhs, r);
//    //    }
//};
template <typename TM, template <typename> class... Policies>
void Domain<TM, Policies...>::SetRange(std::string const &, Range<EntityId> const &){};
template <typename TM, template <typename> class... Policies>
Range<EntityId> Domain<TM, Policies...>::GetRange(std::string const &k) const {
    return Range<EntityId>{};
};
}  // namespace engine
}  // namespace simpla
#endif  // SIMPLA_DOMAINBASE_H
