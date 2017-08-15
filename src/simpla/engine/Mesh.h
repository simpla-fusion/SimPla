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
#include "simpla/utilities/Factory.h"

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
class Curve;
struct MeshBase : public engine::SPObject, public AttributeGroup, public Factory<MeshBase> {
    SP_OBJECT_HEAD(MeshBase, SPObject)
   protected:
    MeshBase();

   public:
    ~MeshBase() override;
    SP_DEFAULT_CONSTRUCT(MeshBase);
    void Serialize(data::DataTable &cfg) const override;
    void Deserialize(const data::DataTable &cfg) override;

    using AttributeGroup::attribute_type;

    int GetNDIMS() const;

    virtual std::shared_ptr<geometry::Chart> GetChart() = 0;
    virtual std::shared_ptr<const geometry::Chart> GetChart() const = 0;

    virtual this_type *GetMesh() { return this; }
    virtual this_type const *GetMesh() const { return this; }

    virtual void AddEmbeddedBoundary(std::string const &prefix, const geometry::GeoObject *g){};

    virtual index_box_type IndexBox(int tag) const;
    virtual box_type GetBox(int tag) const;
    box_type BoundingBox(int tag) const { return GetBox(tag); }

    virtual std::tuple<Real, index_box_type> CheckOverlap(geometry::GeoObject const *) const;

    virtual void SetBlock(const std::shared_ptr<MeshBlock> &blk);
    virtual std::shared_ptr<const MeshBlock> GetBlock() const;
    virtual std::shared_ptr<MeshBlock> GetBlock();

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

    void Pop(const std::shared_ptr<Patch> &p) override;
    void Push(const std::shared_ptr<Patch> &p) override;

    void InitialCondition(const std::shared_ptr<Patch> &patch, Real time_now);
    void BoundaryCondition(const std::shared_ptr<Patch> &patch, Real time_now, Real dt);
    void Advance(const std::shared_ptr<Patch> &patch, Real time_now, Real dt);

    void SetRange(std::string const &, Range<EntityId> const &);
    Range<EntityId> &GetRange(std::string const &k);
    Range<EntityId> GetRange(std::string const &k) const;

   private:
    std::shared_ptr<MeshBlock> m_mesh_block_ = nullptr;

    struct pimpl_s;
    pimpl_s *m_pimpl_;
};

template <typename TChart, template <typename> class... Policies>
class Mesh : public MeshBase, public Policies<Mesh<TChart, Policies...>>... {
    SP_OBJECT_HEAD(Mesh, MeshBase);

   protected:
    Mesh() : Policies<this_type>(this)... {};

   public:
    ~Mesh() override = default;
    SP_DEFAULT_CONSTRUCT(Mesh);
    static std::shared_ptr<this_type> New() { return std::shared_ptr<this_type>(new this_type); };

   public:
    typedef Mesh<TChart, Policies...> mesh_type;
    typedef TChart chart_type;

    std::shared_ptr<chart_type> m_chart_;
    void Deserialize(data::DataTable const &cfg) override;
    void Serialize(data::DataTable &cfg) const override;

    void DoUpdate() override;

    std::shared_ptr<geometry::Chart> GetChart() override { return m_chart_; }
    std::shared_ptr<geometry::Chart const> GetChart() const override { return m_chart_; }

    std::shared_ptr<MeshBlock> GetBlock() override { return MeshBase::GetBlock(); }
    std::shared_ptr<const MeshBlock> GetBlock() const override { return MeshBase::GetBlock(); }

    this_type *GetMesh() override { return this; }
    this_type const *GetMesh() const override { return this; }

    index_box_type IndexBox(int tag) const override { return MeshBase::IndexBox(tag); };

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

    Field<mesh_type, int, CELL> m_refinement_tags_{this, "name"_ = "_refinement_tags_", "IS_NOT_OWNED"_};
    Field<mesh_type, Real, CELL> m_workload_{this, "name"_ = "_workload_", "IS_NOT_OWNED"_};

    void TagRefinementRange(Range<EntityId> const &r) override;

    void AddEmbeddedBoundary(std::string const &prefix, const geometry::GeoObject *g) override;
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
void Mesh<TM, Policies...>::Serialize(data::DataTable &cfg) const {
    base_type::Serialize(cfg);
    m_chart_->Serialize(cfg.GetTable("Chart"));
    traits::_try_invoke_Serialize<Policies...>(this, cfg);
};
template <typename TM, template <typename> class... Policies>
void Mesh<TM, Policies...>::Deserialize(const DataTable &cfg) {
    base_type::Deserialize(cfg);
    traits::_try_invoke_Deserialize<Policies...>(this, cfg);
};
template <typename TM, template <typename> class... Policies>
void Mesh<TM, Policies...>::AddEmbeddedBoundary(std::string const &prefix, const geometry::GeoObject *g) {
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
