//
// Created by salmon on 17-4-5.
//

#ifndef SIMPLA_DOMAINBASE_H
#define SIMPLA_DOMAINBASE_H

#include "simpla/SIMPLA_config.h"

#include <simpla/geometry/CutCell.h>
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
using namespace simpla::data;

class DomainBase : public EngineObject, public AttributeGroup {
    SP_ENABLE_CREATE_HEAD(EngineObject, DomainBase)

    void Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) override;
    std::shared_ptr<simpla::data::DataEntry> Serialize() const override;

   private:
    struct pimpl_s;
    pimpl_s *m_pimpl_ = nullptr;

   public:
    void Push(const std::shared_ptr<Patch> &) override;
    std::shared_ptr<Patch> Pop() const override;
    bool IsInitialized() const override;

    void SetChart(std::shared_ptr<const geometry::Chart> const &c);
    virtual std::shared_ptr<const geometry::Chart> GetChart() const;

    void SetMeshBlock(const std::shared_ptr<const MeshBlock> &blk);
    virtual std::shared_ptr<const MeshBlock> GetMeshBlock() const;
    box_type GetBlockBox() const;

    virtual int CheckBlockInBoundary() const;
    void SetBoundary(std::shared_ptr<const geometry::GeoObject> const &g);
    std::shared_ptr<const geometry::GeoObject> GetBoundary() const;

    void DoSetUp() override;
    void DoUpdate() override;
    void DoTearDown() override;

    design_pattern::Signal<void(DomainBase *, std::shared_ptr<const simpla::data::DataEntry> const &)> OnDeserialize;
    design_pattern::Signal<void(DomainBase const *, std::shared_ptr<simpla::data::DataEntry> &)> OnSerialize;

    virtual void DoInitialCondition(Real time_now);

    design_pattern::Signal<void(DomainBase *, Real)> PreInitialCondition;
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

    std::shared_ptr<DomainBase> AddEmbeddedDomain(std::string const &k, std::shared_ptr<DomainBase> const &b);

};  // class DomainBase

template <typename TChart, template <typename> class... Policies>
class Domain : public DomainBase, public Policies<Domain<TChart, Policies...>>... {
    typedef TChart chart_type;
    SP_ENABLE_NEW_HEAD(DomainBase, Domain);

   public:
    std::shared_ptr<const geometry::Chart> GetChart() const override { return DomainBase::GetChart(); };
    std::shared_ptr<const engine::MeshBlock> GetMeshBlock() const override { return DomainBase::GetMeshBlock(); };

    void DoSetUp() override;
    void DoUpdate() override;
    void DoTearDown() override;

    void DoInitialCondition(Real time_now) override;
    void DoBoundaryCondition(Real time_now, Real dt) override;
    void DoAdvance(Real time_now, Real dt) override;
    void DoTagRefinementCells(Real time_now) override;

    template <typename V, int IFORM, int... DOF, typename TR>
    void Fill(AttributeT<V, IFORM, DOF...> &lhs, TR const &rhs) const;

    template <typename U, int IFORM, int... DOF>
    void InitializeAttribute(AttributeT<U, IFORM, DOF...> *attr) const;

    AttributeT<unsigned int, NODE> m_node_tag_{this, "Name"_ = "node_tag"};
    AttributeT<Real, EDGE> m_edge_frac_{this, "Name"_ = "edge_frac"};
    AttributeT<Real, FACE> m_face_frac_{this, "Name"_ = "face_frac"};
    AttributeT<Real, CELL> m_cell_frac_{this, "Name"_ = "cell_frac"};

};  // class Domain

#define SP_DOMAIN_HEAD(_CLASS_NAME_, _BASE_NAME_)                                                                      \
    SP_SERIALIZABLE_HEAD(_BASE_NAME_, _CLASS_NAME_)                                                                    \
   protected:                                                                                                          \
    _CLASS_NAME_();                                                                                                    \
                                                                                                                       \
   public:                                                                                                             \
    ~_CLASS_NAME_() override;                                                                                          \
    template <typename... Args>                                                                                        \
    static std::shared_ptr<this_type> New(Args &&... args) {                                                           \
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));                                 \
    }                                                                                                                  \
                                                                                                                       \
    static bool _is_registered;                                                                                        \
                                                                                                                       \
    void DoSetUp() override;                                                                                           \
    void DoUpdate() override;                                                                                          \
    void DoTearDown() override;                                                                                        \
                                                                                                                       \
    void DoInitialCondition(Real time_now) override;                                                                   \
    void DoAdvance(Real time_now, Real dt) override;                                                                   \
    void DoTagRefinementCells(Real time_now) override;                                                                 \
    void AddOnDeserialize(                                                                                             \
        std::function<void(this_type *, std::shared_ptr<const simpla::data::DataEntry> const &)> const &fun) {         \
        simpla::engine::DomainBase::OnDeserialize.Connect(                                                             \
            [=](simpla::engine::DomainBase *self, std::shared_ptr<const simpla::data::DataEntry> const &cfg) {         \
                if (auto d = dynamic_cast<this_type *>(self)) { fun(d, cfg); };                                        \
            });                                                                                                        \
    }                                                                                                                  \
    void AddOnSerialize(                                                                                               \
        std::function<void(this_type const *, std::shared_ptr<const simpla::data::DataEntry> const &)> const &fun) {   \
        simpla::engine::DomainBase::OnDeserialize.Connect(                                                             \
            [=](simpla::engine::DomainBase const *self, std::shared_ptr<const simpla::data::DataEntry> const &cfg) {   \
                if (auto d = dynamic_cast<_CLASS_NAME_ const *>(self)) { fun(d, cfg); };                               \
            });                                                                                                        \
    }                                                                                                                  \
    void AddPreInitialCondition(std::function<void(this_type *, Real)> const &fun) {                                   \
        simpla::engine::DomainBase::PreInitialCondition.Connect([=](simpla::engine::DomainBase *self, Real time_now) { \
            if (auto d = dynamic_cast<this_type *>(self)) { fun(d, time_now); };                                       \
        });                                                                                                            \
    }                                                                                                                  \
    void AddPostInitialCondition(std::function<void(this_type *, Real)> const &fun) {                                  \
        simpla::engine::DomainBase::PostInitialCondition.Connect(                                                      \
            [=](simpla::engine::DomainBase *self, Real time_now) {                                                     \
                if (auto d = dynamic_cast<this_type *>(self)) { fun(d, time_now); };                                   \
            });                                                                                                        \
    }

#define SP_DOMAIN_POLICY_HEAD(_NAME_)                   \
   private:                                             \
    typedef THost host_type;                            \
    typedef _NAME_<THost> this_type;                    \
    THost *m_host_;                                     \
                                                        \
   public:                                              \
    _NAME_(THost *h);                                   \
    virtual ~_NAME_();                                  \
    _NAME_(_NAME_ const &other) = delete;               \
    _NAME_(_NAME_ &&other) = delete;                    \
    _NAME_ &operator=(_NAME_ const &other) = delete;    \
    _NAME_ &operator=(_NAME_ &&other) = delete;         \
    std::shared_ptr<data::DataEntry> Serialize() const; \
    void Deserialize(std::shared_ptr<const data::DataEntry> const &cfg);

template <typename TChart, template <typename> class... Policies>
Domain<TChart, Policies...>::Domain() : DomainBase(), Policies<this_type>(this)... {}
template <typename TChart, template <typename> class... Policies>
Domain<TChart, Policies...>::~Domain(){};

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
void Domain<TChart, Policies...>::DoInitialCondition(Real time_now) {
    //    if (CheckBlockInBoundary() == 0)
    {
        DomainBase::DoInitialCondition(time_now);

        InitializeAttribute(&m_node_tag_);
        InitializeAttribute(&m_edge_frac_);
        m_edge_frac_[0].Fill(1.0);
        m_edge_frac_[1].Fill(1.0);
        m_edge_frac_[2].Fill(1.0);

        //        geometry::CutCellTagNode(&m_node_tag_, &m_edge_frac_[0], GetChart(),
        //        GetMeshBlock()->GetIndexBox(), GetBoundary(), 0b001);
    }
}

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
void InitializeArray(std::integral_constant<int, NODE>, THost const *host, TArray &v) {
    InitializeArray_(v, host->GetSpaceFillingCurve(0b000));
}
template <typename TArray, typename THost>
void InitializeArray(std::integral_constant<int, CELL>, THost const *host, TArray &v) {
    InitializeArray_(v, host->GetSpaceFillingCurve(0b000));
}
template <typename TArray, typename THost>
void InitializeArray(std::integral_constant<int, EDGE>, THost const *host, TArray &v) {
    InitializeArray_(v[0], host->GetSpaceFillingCurve(0b001));
    InitializeArray_(v[1], host->GetSpaceFillingCurve(0b010));
    InitializeArray_(v[2], host->GetSpaceFillingCurve(0b100));
}
template <typename TArray, typename THost>
void InitializeArray(std::integral_constant<int, FACE>, THost const *host, TArray &v) {
    InitializeArray_(v[0], host->GetSpaceFillingCurve(0b110));
    InitializeArray_(v[1], host->GetSpaceFillingCurve(0b101));
    InitializeArray_(v[2], host->GetSpaceFillingCurve(0b011));
}

template <int IFORM, typename TArray, typename THost>
void InitializeArray(std::integral_constant<int, IFORM>, THost const *host, TArray &v) {
    UNIMPLEMENTED;
}

template <typename THost, typename U, int IFORM, int... DOF, typename RHS>
void DomainAssign(THost *self, AttributeT<U, IFORM, DOF...> &lhs, RHS const &rhs,
                  ENABLE_IF((!simpla::traits::is_invocable<RHS, point_type>::value))) {
    UNIMPLEMENTED;
}

template <typename THost, typename U, int... DOF, typename RHS>
void DomainAssign(THost *self, AttributeT<U, NODE, DOF...> &lhs, RHS const &rhs,
                  ENABLE_IF((!simpla::traits::is_invocable<RHS, point_type>::value))) {
    lhs.Assign(rhs, self->GetSpaceFillingCurve(0b000));
}
template <typename THost, typename U, int... DOF, typename RHS>
void DomainAssign(THost *self, AttributeT<U, EDGE, DOF...> &lhs, RHS const &rhs,
                  ENABLE_IF((!simpla::traits::is_invocable<RHS, point_type>::value))) {
    lhs.template AssignSub<0>(rhs, self->GetSpaceFillingCurve(0b001));
    lhs.template AssignSub<1>(rhs, self->GetSpaceFillingCurve(0b010));
    lhs.template AssignSub<2>(rhs, self->GetSpaceFillingCurve(0b100));
}
template <typename THost, typename U, int... DOF, typename RHS>
void DomainAssign(THost *self, AttributeT<U, FACE, DOF...> &lhs, RHS const &rhs,
                  ENABLE_IF((!simpla::traits::is_invocable<RHS, point_type>::value))) {
    lhs.template AssignSub<0>(rhs, self->GetSpaceFillingCurve(0b110));
    lhs.template AssignSub<1>(rhs, self->GetSpaceFillingCurve(0b101));
    lhs.template AssignSub<2>(rhs, self->GetSpaceFillingCurve(0b011));
}
template <typename THost, typename U, int... DOF, typename RHS>
void DomainAssign(THost *self, AttributeT<U, CELL, DOF...> &lhs, RHS const &rhs,
                  ENABLE_IF((!simpla::traits::is_invocable<RHS, point_type>::value))) {
    lhs.Assign(rhs, self->GetSpaceFillingCurve(0b111));
}
template <typename THost, typename U, typename RHS>
void DomainAssign(THost *self, AttributeT<U, NODE> &lhs, RHS const &rhs,
                  ENABLE_IF((simpla::traits::is_invocable<RHS, point_type>::value))) {
    auto chart = self->GetChart();
    lhs.Assign([&](auto &&... idx) { return rhs(chart->local_coordinates(0, std::forward<decltype(idx)>(idx)...)); },
               self->GetSpaceFillingCurve(0b000));
}
template <typename THost, typename U, int... DOF, typename RHS>
void DomainAssign(THost *self, AttributeT<U, EDGE, DOF...> &lhs, RHS const &rhs,
                  ENABLE_IF((simpla::traits::is_invocable<RHS, point_type>::value))) {
    auto chart = self->GetChart();
    lhs.template AssignSub<0>(
        [&](auto &&... idx) {
            return simpla::traits::nt_get_r<0>(rhs(chart->local_coordinates(EntityIdCoder::m_sub_index_to_id_[EDGE][0],
                                                                            std::forward<decltype(idx)>(idx)...)));
        },
        self->GetSpaceFillingCurve(0b001));
    lhs.template AssignSub<1>(
        [&](auto &&... idx) {
            return simpla::traits::nt_get_r<1>(rhs(chart->local_coordinates(EntityIdCoder::m_sub_index_to_id_[EDGE][1],
                                                                            std::forward<decltype(idx)>(idx)...)));
        },
        self->GetSpaceFillingCurve(0b010));
    lhs.template AssignSub<2>(
        [&](auto &&... idx) {
            return simpla::traits::nt_get_r<2>(rhs(chart->local_coordinates(EntityIdCoder::m_sub_index_to_id_[EDGE][2],
                                                                            std::forward<decltype(idx)>(idx)...)));
        },
        self->GetSpaceFillingCurve(0b100));
}
template <typename THost, typename U, int... DOF, typename RHS>
void DomainAssign(THost *self, AttributeT<U, FACE, DOF...> &lhs, RHS const &rhs,
                  ENABLE_IF((simpla::traits::is_invocable<RHS, point_type>::value))) {
    auto chart = self->GetChart();
    lhs.template AssignSub<0>(
        [&](auto &&... idx) {
            return simpla::traits::nt_get_r<0>(rhs(chart->local_coordinates(EntityIdCoder::m_sub_index_to_id_[FACE][0],
                                                                            std::forward<decltype(idx)>(idx)...)));
        },
        self->GetSpaceFillingCurve(0b110));
    lhs.template AssignSub<1>(
        [&](auto &&... idx) {
            return simpla::traits::nt_get_r<1>(rhs(chart->local_coordinates(EntityIdCoder::m_sub_index_to_id_[FACE][1],
                                                                            std::forward<decltype(idx)>(idx)...)));
        },
        self->GetSpaceFillingCurve(0b101));
    lhs.template AssignSub<2>(
        [&](auto &&... idx) {
            return simpla::traits::nt_get_r<2>(rhs(chart->local_coordinates(EntityIdCoder::m_sub_index_to_id_[FACE][2],
                                                                            std::forward<decltype(idx)>(idx)...)));
        },
        self->GetSpaceFillingCurve(0b011));
}

template <typename THost, typename U, typename RHS>
void DomainAssign(THost *self, AttributeT<U, CELL> &lhs, RHS const &rhs,
                  ENABLE_IF((simpla::traits::is_invocable<RHS, point_type>::value))) {
    auto chart = self->GetChart();
    lhs.Assign([&](auto &&... idx) { return rhs(chart->local_coordinates(7, std::forward<decltype(idx)>(idx)...)); },
               self->GetSpaceFillingCurve(0b111));
}

template <typename THost, typename V, int... DOF, typename... U>
void DomainAssign(THost *self, AttributeT<V, NODE, DOF...> &lhs, Expression<U...> const &rhs) {
    lhs.Assign(self->template Calculate<0>(rhs), self->GetSpaceFillingCurve(0b000));
};
template <typename THost, typename V, int... DOF, typename... U>
void DomainAssign(THost *self, AttributeT<V, EDGE, DOF...> &lhs, Expression<U...> const &rhs) {
    lhs.template AssignSub<0>(self->template Calculate<0>(rhs), self->GetSpaceFillingCurve(0b001));
    lhs.template AssignSub<1>(self->template Calculate<1>(rhs), self->GetSpaceFillingCurve(0b010));
    lhs.template AssignSub<2>(self->template Calculate<2>(rhs), self->GetSpaceFillingCurve(0b100));
};
template <typename THost, typename V, int... DOF, typename... U>
void DomainAssign(THost *self, AttributeT<V, FACE, DOF...> &lhs, Expression<U...> const &rhs) {
    lhs.template AssignSub<0>(self->template Calculate<0>(rhs), self->GetSpaceFillingCurve(0b110));
    lhs.template AssignSub<1>(self->template Calculate<1>(rhs), self->GetSpaceFillingCurve(0b101));
    lhs.template AssignSub<2>(self->template Calculate<2>(rhs), self->GetSpaceFillingCurve(0b011));
};
template <typename THost, typename V, int... DOF, typename... U>
void DomainAssign(THost *self, AttributeT<V, CELL, DOF...> &lhs, Expression<U...> const &rhs) {
    lhs.Assign(self->template Calculate<0>(rhs), self->GetSpaceFillingCurve(0b111));
};
}  // namespace detail {

template <typename TChart, template <typename> class... Policies>
template <typename U, int IFORM, int... DOF>
void Domain<TChart, Policies...>::InitializeAttribute(AttributeT<U, IFORM, DOF...> *attr) const {
    detail::InitializeArray(std::integral_constant<int, IFORM>(), this, *attr);
};
template <typename TM, template <typename> class... Policies>
template <typename V, int IFORM, int... DOF, typename RHS>
void Domain<TM, Policies...>::Fill(AttributeT<V, IFORM, DOF...> &lhs, RHS const &rhs) const {
    detail::DomainAssign(this, lhs, rhs);
};
}  // namespace engine
}  // namespace simpla
#endif  // SIMPLA_DOMAINBASE_H
