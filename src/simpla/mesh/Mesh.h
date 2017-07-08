//
// Created by salmon on 16-11-19.
//

#ifndef SIMPLA_MESHBASE_H
#define SIMPLA_MESHBASE_H

#include <simpla/concept/Printable.h>
#include <simpla/data/all.h>
#include <simpla/geometry/Chart.h>
#include <simpla/geometry/GeoObject.h>
#include "simpla/engine/Attribute.h"
#include "simpla/engine/SPObject.h"

namespace simpla {
class MeshBase;

template <typename>
struct CalculusPolicy;

struct EBMeshBase;
template <typename>
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

    explicit MeshBase(std::shared_ptr<geometry::Chart> const &c = nullptr, std::string const &s_name = "");
    ~MeshBase() override;
    SP_DEFAULT_CONSTRUCT(MeshBase);
    DECLARE_REGISTER_NAME(MeshBase);

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(const std::shared_ptr<data::DataTable> &t) override;

    void DoUpdate() override;

    virtual void InitializeData(Real time_now);
    virtual void SetBoundaryCondition(Real time_now, Real time_dt);

    void Push(engine::Patch *) override;
    void Pull(engine::Patch *) override;

    virtual MeshBase *GetMesh() override;
    virtual MeshBase const *GetMesh() const override;
    virtual EBMeshBase const *GetCenter() const;
    virtual EBMeshBase const *GetBoundary() const;

    virtual unsigned int GetNDims() const { return 3; }

    void SetChart(std::shared_ptr<geometry::Chart> const &);
    std::shared_ptr<geometry::Chart> GetChart() const;

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

    virtual index_box_type GetIndexBox(int tag = VERTEX) const = 0;

    virtual point_type local_coordinates(EntityId s, Real const *r = 0) const = 0;

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

    template <typename LHS, typename RHS>
    void Fill(LHS &lhs, RHS const &rhs) const {
        //        ASSERT(isA(typeid(simpla::traits::mesh_type_t<LHS>)));
        //        dynamic_cast<simpla::traits::mesh_type_t<LHS> const *>(this)->DoFill(lhs, rhs);
    }

    virtual void AddGeometryObject(std::shared_ptr<geometry::GeoObject> const &g, std::string const &prefix);

   protected:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

struct EBMeshBase {
    SP_OBJECT_BASE(EBMeshBase);

    EBMeshBase(MeshBase const *m);
    virtual ~EBMeshBase() = default;

    template <typename TM>
    EBMeshBase(TM const *m) : EBMeshBase(m){};

    virtual MeshBase const *GetBaseMesh() const;
    virtual bool empty() const;

    void AddGeometryObject(std::shared_ptr<geometry::GeoObject> const &g, std::string const &prefix);

    template <typename LHS, typename RHS>
    void Fill(LHS &lhs, RHS const &rhs) const {
        if (isA(typeid(EBMesh<typename LHS::mesh_type>))) {
            dynamic_cast<EBMesh<typename LHS::mesh_type> const *>(this)->DoFill(lhs, rhs);
        } else if (isA(typeid(typename LHS::mesh_type))) {
            dynamic_cast<typename LHS::mesh_type const *>(this)->DoFill(lhs, rhs);
        }
    }

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

template <typename TM>
struct EBMesh : public EBMeshBase {
    SP_OBJECT_HEAD(EBMesh<TM>, EBMeshBase);
    typedef TM base_mesh_type;

    EBMesh(TM const *m) : EBMeshBase(m){};

    virtual ~EBMesh() = default;

    base_mesh_type const *GetBaseMesh() const override {
        return dynamic_cast<base_mesh_type const *>(base_type::GetBaseMesh());
    }

    template <typename LHS, typename RHS>
    void DoFill(LHS &lhs, RHS const &rhs) const {}
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

    template <typename TL, typename TR>
    void DoFill(TL &lhs, TR const &rhs) const;

    //    {
    //        if (GetCenter()->empty()) {
    //            CalculusPolicy<this_type>::Fill<simpla::traits::iform<TL>::value>(*this, lhs.Get(), rhs);
    //        } else {
    //            GetCenter()->Fill(lhs, rhs);
    //        };
    //    }

    template <typename TL, typename... Args>
    decltype(auto) GetEntity(TL &lhs, Args &&... args) const {
        //        return CalculusPolicy<this_type>::GetEntity<simpla::traits::iform<TL>::value>(*this, lhs.Get(),
        return lhs(std::forward<Args>(args)...);
    }

    EBMesh<this_type> const *GetCenter() const {
        return dynamic_cast<EBMesh<this_type> const *>(base_type::GetCenter());
    }
    EBMesh<this_type> const *GetBoundary() const {
        return dynamic_cast<EBMesh<this_type> const *>(base_type::GetBoundary());
    }
};
}  // namespace simpla

#endif  // SIMPLA_MESHBASE_H
