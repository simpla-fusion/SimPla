//
// Created by salmon on 17-7-16.
//

#ifndef SIMPLA_MESH_H
#define SIMPLA_MESH_H

#include "simpla/SIMPLA_config.h"

#include "Attribute.h"
#include "SPObject.h"
#include "simpla/algebra/Field.h"
#include "simpla/data/Data.h"
#include "simpla/data/EnableCreateFromDataTable.h"

#include "PoliciesCommon.h"

namespace simpla {
namespace geometry {
struct Chart;
struct GeoObject;
}
namespace engine {
class MeshBlock;
class Patch;
using namespace simpla::data;

struct MeshBase : public AttributeGroup, public data::EnableCreateFromDataTable<MeshBase> {
    SP_OBJECT_HEAD(MeshBase, data::EnableCreateFromDataTable<MeshBase>)
   public:
    using AttributeGroup::attribute_type;

    MeshBase();
    ~MeshBase() override;

    MeshBase(MeshBase const &other) = delete;
    MeshBase(MeshBase &&other) noexcept = delete;
    void swap(MeshBase &other) = delete;
    MeshBase &operator=(this_type const &other) = delete;
    MeshBase &operator=(this_type &&other) noexcept = delete;

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(std::shared_ptr<data::DataTable> const &t) override;

    virtual geometry::Chart *GetChart() = 0;
    virtual geometry::Chart const *GetChart() const = 0;

    virtual this_type *GetMesh() { return this; }
    virtual this_type const *GetMesh() const { return this; }

    virtual void AddEmbeddedBoundary(std::string const &prefix, const geometry::GeoObject *g){};
    virtual std::shared_ptr<geometry::GeoObject> GetGeoBody() const;
    virtual Real CheckOverlap(const geometry::GeoObject *) const;

    virtual index_box_type GetIndexBox(int tag = 0) const;
    virtual box_type GetBox(int tag = 0) const;

    void SetBlock(const MeshBlock &blk);
    virtual const MeshBlock *GetBlock() const;

    void DoInitialize() override;
    void DoFinalize() override;
    void DoUpdate() override;
    void DoTearDown() override;

    virtual void DoInitialCondition(Real time_now) {}
    virtual void DoBoundaryCondition(Real time_now, Real dt) {}
    virtual void DoAdvance(Real time_now, Real dt) {}
    virtual void DoTagRefinementCells(Real time_now) {}
    virtual void TagRefinementRange(Range<EntityId> const &r){};

    void InitialCondition(Real time_now);
    void BoundaryCondition(Real time_now, Real dt);
    void Advance(Real time_now, Real dt);
    void TagRefinementCells(Real time_now);

    void Pull(Patch *p) override;
    void Push(Patch *p) override;

    void InitialCondition(Patch *patch, Real time_now);
    void BoundaryCondition(Patch *patch, Real time_now, Real dt);
    void Advance(Patch *patch, Real time_now, Real dt);

    void SetRange(std::string const &, Range<EntityId> const &);
    Range<EntityId> &GetRange(std::string const &k);
    Range<EntityId> GetRange(std::string const &k) const;

   private:
    MeshBlock m_mesh_block_{index_box_type{{0, 0, 0}, {1, 1, 1}}};

    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

template <typename TChart, template <typename> class... Policies>
class Mesh : public MeshBase, public Policies<Mesh<TChart, Policies...>>... {
    SP_OBJECT_HEAD(Mesh, MeshBase);

   public:
    typedef Mesh<TChart, Policies...> mesh_type;
    typedef TChart chart_type;

    chart_type m_chart_;

    Mesh() : Policies<this_type>(this)... {};
    ~Mesh() override = default;

    void Deserialize(std::shared_ptr<data::DataTable> const &cfg) override;
    std::shared_ptr<data::DataTable> Serialize() const override;

    void DoUpdate() override;

    chart_type *GetChart() override { return &m_chart_; }
    chart_type const *GetChart() const override { return &m_chart_; }
    this_type *GetMesh() override { return this; }
    this_type const *GetMesh() const override { return this; }

    const MeshBlock *GetBlock() const override { return MeshBase::GetBlock(); }

    index_box_type GetIndexBox(int tag) const override { return MeshBase::GetIndexBox(tag); };

    void DoInitialCondition(Real time_now) override;
    void DoBoundaryCondition(Real time_now, Real dt) override;
    void DoAdvance(Real time_now, Real dt) override;
    void DoTagRefinementCells(Real time_now) override;

    template <typename TL, typename TR>
    void Fill(TL &lhs, TR &&rhs) const {
        FillRange(lhs, std::forward<TR>(rhs), Range<EntityId>{}, true);
        //        FillRange(lhs, 0, "PATCH_BOUNDARY_" + std::to_string(TL::iform), false);
    };

    template <typename TL, typename TR>
    void FillRange(TL &lhs, TR &&rhs, Range<EntityId> r = Range<EntityId>{},
                   bool full_fill_if_range_is_null = false) const;

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

    Field<mesh_type, int, VOLUME> m_refinement_tags_{this, "name"_ = "_refinement_tags_", "IS_NOT_OWNED"_};
    Field<mesh_type, Real, VOLUME> m_workload_{this, "name"_ = "_workload_", "IS_NOT_OWNED"_};

    void TagRefinementRange(Range<EntityId> const &r) override;

    void AddEmbeddedBoundary(std::string const &prefix, const geometry::GeoObject *g) override;

    std::shared_ptr<geometry::GeoObject> GetGeoBody() const override { return base_type::GetGeoBody(); }
};
template <typename TM, template <typename> class... Policies>
void Mesh<TM, Policies...>::TagRefinementRange(Range<EntityId> const &r) {
    if (!m_refinement_tags_.isNull() && !r.isNull()) {
        r.foreach ([&](EntityId s) {
            if (m_refinement_tags_[0].in_box(s.x, s.y, s.z)) { m_refinement_tags_[0](s.x, s.y, s.z) = 1; }
        });
    }
};
template <typename TM, template <typename> class... Policies>
void Mesh<TM, Policies...>::DoUpdate() {
    MeshBase::DoUpdate();
    if (!m_refinement_tags_.isNull()) { m_refinement_tags_.Clear(); }
};
namespace _detail {
DEFINE_INVOKE_HELPER(SetEmbeddedBoundary)
DEFINE_INVOKE_HELPER(Calculate)
}

template <typename TM, template <typename> class... Policies>
void Mesh<TM, Policies...>::DoInitialCondition(Real time_now) {
    traits::_try_invoke_InitialCondition<Policies...>(this, time_now);
}
template <typename TM, template <typename> class... Policies>
void Mesh<TM, Policies...>::DoBoundaryCondition(Real time_now, Real dt) {
    traits::_try_invoke_BoundaryCondition<Policies...>(this, time_now, dt);
}
template <typename TM, template <typename> class... Policies>
void Mesh<TM, Policies...>::DoAdvance(Real time_now, Real dt) {
    traits::_try_invoke_Advance<Policies...>(this, time_now, dt);
}

template <typename TM, template <typename> class... Policies>
void Mesh<TM, Policies...>::DoTagRefinementCells(Real time_now) {
    traits::_try_invoke_TagRefinementCells<Policies...>(this, time_now);
}

template <typename TM, template <typename> class... Policies>
std::shared_ptr<data::DataTable> Mesh<TM, Policies...>::Serialize() const {
    auto res = base_type::Serialize();
    res->Set("Chart", m_chart_.Serialize());
    traits::_try_invoke_Serialize<Policies...>(this, res.get());
    return res;
};
template <typename TM, template <typename> class... Policies>
void Mesh<TM, Policies...>::Deserialize(std::shared_ptr<data::DataTable> const &cfg) {
    base_type::Deserialize(cfg);
    traits::_try_invoke_Deserialize<Policies...>(this, cfg);
};
template <typename TM, template <typename> class... Policies>
void Mesh<TM, Policies...>::AddEmbeddedBoundary(std::string const &prefix, const geometry::GeoObject *g) {
    if (MeshBase::CheckOverlap(g) < EPSILON) { return; }
    _detail::_try_invoke_SetEmbeddedBoundary<Policies...>(this, prefix, g);
};

template <typename TM, template <typename> class... Policies>
template <typename LHS, typename RHS>
void Mesh<TM, Policies...>::FillRange(LHS &lhs, RHS &&rhs, Range<EntityId> r, bool full_fill_if_range_is_null) const {
    if (r.isFull() || (r.isNull() && full_fill_if_range_is_null)) {
        _detail::_try_invoke_once_Calculate<Policies...>(this, lhs, std::forward<RHS>(rhs));
    } else {
        _detail::_try_invoke_once_Calculate<Policies...>(this, lhs, std::forward<RHS>(rhs), r);
    }
};

}  // namespace mesh
}  // namespace simpla{

#endif  // SIMPLA_MESH_H
