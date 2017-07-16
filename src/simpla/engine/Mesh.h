//
// Created by salmon on 17-7-16.
//

#ifndef SIMPLA_MESH_H
#define SIMPLA_MESH_H

#include "Attribute.h"
#include "SPObject.h"
#include "simpla/data/EnableCreateFromDataTable.h"

namespace simpla {
namespace geometry {
struct Chart;
}
namespace engine {
class MeshBlock;
class Patch;

struct MeshBase : public SPObject, public AttributeGroup, public data::EnableCreateFromDataTable<MeshBase> {
    SP_OBJECT_HEAD(MeshBase, SPObject)
    DECLARE_REGISTER_NAME(MeshBase)
   public:
    using AttributeGroup::attribute_type;

    MeshBase();
    ~MeshBase() override;

    MeshBase(MeshBase const &other) = delete;
    MeshBase(MeshBase &&other) noexcept = delete;
    void swap(MeshBase &other) = delete;
    MeshBase &operator=(this_type const &other) = delete;
    MeshBase &operator=(this_type &&other) noexcept = delete;

    virtual const geometry::Chart &GetChart() const = 0;
    virtual geometry::Chart &GetChart() = 0;

    void SetBlock(const MeshBlock &blk);
    virtual const MeshBlock &GetBlock() const;
    virtual id_type GetBlockId() const;

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(std::shared_ptr<data::DataTable> const &t) override;

    void DoInitialize() override;
    void DoFinalize() override;
    void DoUpdate() override;
    void DoTearDown() override;

    virtual void DoInitialCondition(Real time_now) {}
    virtual void DoBoundaryCondition(Real time_now, Real dt) {}
    virtual void DoAdvance(Real time_now, Real dt) {}

    void InitialCondition(Real time_now);
    void BoundaryCondition(Real time_now, Real dt);
    void Advance(Real time_now, Real dt);

    void Pull(Patch *p) override;
    void Push(Patch *p) override;

    void InitialCondition(Patch *patch, Real time_now);
    void BoundaryCondition(Patch *patch, Real time_now, Real dt);
    void Advance(Patch *patch, Real time_now, Real dt);

    void SetRange(std::string const &, Range<EntityId> const &);
    Range<EntityId> &GetRange(std::string const &k);
    Range<EntityId> GetRange(std::string const &k) const;

   private:
    MeshBlock m_mesh_block_;
    std::shared_ptr<geometry::Chart> m_chart_ = nullptr;

    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

template <typename TChart, template <typename> class... Policies>
class Mesh : public MeshBase, public Policies<Mesh<TChart, Policies...>>... {
   public:
    DECLARE_REGISTER_NAME(Mesh)

    typedef Mesh<TChart, Policies...> this_type;
    Mesh() : Policies<this_type>(this)... {};
    ~Mesh() = default;
    const TChart &GetChart() const override { return m_chart_; };
    TChart &GetChart() override { return m_chart_; };
    const engine::MeshBlock &GetBlock() const override { return MeshBase::GetBlock(); }

   private:
    TChart m_chart_;
};

}  // namespace mesh

}  // namespace simpla{

#endif  // SIMPLA_MESH_H
