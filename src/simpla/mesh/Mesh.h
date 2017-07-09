//
// Created by salmon on 16-11-19.
//

#ifndef SIMPLA_MESHBASE_H
#define SIMPLA_MESHBASE_H

#include <simpla/data/all.h>
#include <simpla/engine/Attribute.h>
#include <simpla/model/Chart.h>
#include <simpla/model/GeoObject.h>
#include <simpla/utilities/SPObject.h>

namespace simpla {

namespace engine {
class AttributeGroup;
class SPObject;
}

struct MeshBase;
struct EBMeshBase;
template <typename...>
struct Mesh;
template <typename...>
struct EBMesh;

/**
 *  Define:
 *   A bundle is a triple \f$(E, p, B)\f$ where \f$E\f$, \f$B\f$ are sets and \f$p:E \rightarrow B\f$ a map
 *   - \f$E\f$ is called the total space
 *   - \f$B\f$ is the base space of the bundle
 *   - \f$p\f$ is the projection
 *
 */
class MeshBase : public engine::SPObject,
                 public engine::AttributeGroup,
                 public data::EnableCreateFromDataTable<MeshBase> {
    SP_OBJECT_HEAD(MeshBase, engine::SPObject);

   public:
    typedef engine::Attribute attribute_type;

    explicit MeshBase(std::shared_ptr<model::Chart> const &c = nullptr, std::string const &s_name = "");
    ~MeshBase() override;
    SP_DEFAULT_CONSTRUCT(MeshBase);
    DECLARE_REGISTER_NAME(MeshBase);

    virtual bool empty() const { return false; }

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(const std::shared_ptr<data::DataTable> &t) override;

    void DoUpdate() override;

    virtual unsigned int GetNDims() const { return 3; }

    void SetChart(std::shared_ptr<model::Chart> const &);
    std::shared_ptr<model::Chart> GetChart() const;

    point_type const &GetCellWidth() const;
    point_type const &GetOrigin() const;

    void FitBoundBox(box_type const &);

    index_tuple GetIndexOffset() const;
    void SetDimensions(index_tuple const &);
    index_tuple GetDimensions() const;

    void SetPeriodicDimension(size_tuple const &x);
    size_tuple const &GetPeriodicDimension() const;

    void SetDefaultGhostWidth(index_tuple const &);
    index_tuple GetDefaultGhostWidth() const;
    index_tuple GetGhostWidth(int tag = VERTEX) const;

    void SetBlock(const engine::MeshBlock &);
    const engine::MeshBlock &GetBlock() const;
    id_type GetBlockId() const;

    box_type GetBox() const;

    virtual index_box_type GetIndexBox(int tag) const = 0;

    virtual point_type local_coordinates(EntityId s, Real const *r) const = 0;

    virtual point_type map(point_type const &) const;

    point_type local_coordinates(EntityId s, point_type const &r) const { return local_coordinates(s, &r[0]); };

    template <typename... Args>
    point_type point(Args &&... args) const {
        return local_coordinates(std::forward<Args>(args)...);
    }
    template <typename... Args>
    point_type global_coordinates(Args &&... args) const {
        return map(local_coordinates(std::forward<Args>(args)...));
    }

    virtual void InitializeData(Real time_now);
    virtual void SetBoundaryCondition(Real time_now, Real time_dt);

    void Push(engine::Patch *p) override;
    void Pull(engine::Patch *p) override;

    MeshBase const *GetMesh() const override;

    template <typename LHS, typename RHS>
    void Fill(LHS &lhs, RHS &&rhs) const {
        if (isA(typeid(typename LHS::mesh_type))) {
            dynamic_cast<typename LHS::mesh_type const *>(this)->FillBody(lhs, std::forward<RHS>(rhs));
        } else if (isA(typeid(EBMesh<typename LHS::mesh_type>))) {
            dynamic_cast<EBMesh<typename LHS::mesh_type> const *>(this)->FillBody(lhs, std::forward<RHS>(rhs));
        } else {
            RUNTIME_ERROR << "Unknown mesh type! " << typeid(LHS).name() << std::endl;
        }
    }

   protected:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
template <typename>
struct CalculusPolicy;

struct EBMeshBase : public MeshBase {
    SP_OBJECT_HEAD(EBMeshBase, MeshBase);

    explicit EBMeshBase(MeshBase const *m) : m_base_mesh_(m){};

    ~EBMeshBase() override = default;

    SP_DEFAULT_CONSTRUCT(EBMeshBase);

    this_type const *GetMesh() const override { return this; }

    bool empty() const override { return true; }

    virtual MeshBase const *GetBaseMesh() const { return m_base_mesh_; }

   private:
    MeshBase const *m_base_mesh_;
};

template <typename TM>
struct EBMesh<TM> : public EBMeshBase {
    SP_OBJECT_HEAD(EBMesh<TM>, EBMeshBase);

    typedef TM base_mesh_type;
    static constexpr unsigned int NDIMS = base_mesh_type::NDIMS;

    explicit EBMesh(TM const *m) : EBMeshBase(m){};

    ~EBMesh() override = default;

    SP_DEFAULT_CONSTRUCT(EBMesh);

    this_type const *GetMesh() const override { return this; }

    base_mesh_type const *GetBaseMesh() const override {
        return dynamic_cast<base_mesh_type const *>(base_type::GetBaseMesh());
    }

    template <typename TL, typename TR>
    void FillBody(TL &lhs, TR &&rhs) const {
        return CalculusPolicy<this_type>::Fill(*this, lhs, std::forward<TR>(rhs));
    }
};

template <typename... TPolicy>
struct Mesh : public MeshBase, public TPolicy... {
   public:
    SP_OBJECT_HEAD(Mesh<TPolicy...>, MeshBase)

    typedef engine::Attribute attribute_type;

    template <typename... Args>
    explicit Mesh(Args &&... args) : base_type(std::forward<Args>(args)...), TPolicy()... {};

    ~Mesh() override = default;

    SP_DEFAULT_CONSTRUCT(Mesh)
    DECLARE_REGISTER_NAME(Mesh);

    void InitializeData(Real time_now) override{};
    void SetBoundaryCondition(Real time_now, Real time_dt) override{};

    template <typename LHS, typename RHS>
    void FillBody(LHS &lhs, RHS &&rhs) const {
        return CalculusPolicy<this_type>::Fill(*this, lhs.Get(), std::forward<RHS>(rhs));
    }

    template <typename TL, typename... Args>
    decltype(auto) GetEntity(TL &lhs, Args &&... args) const {
        return CalculusPolicy<this_type>::GetEntity(*this, lhs, std::forward<Args>(args)...);
    }
};
}  // namespace simpla

#endif  // SIMPLA_MESHBASE_H
