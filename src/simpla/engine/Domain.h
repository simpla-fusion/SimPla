//
// Created by salmon on 17-4-5.
//

#ifndef SIMPLA_DOMAIN_H
#define SIMPLA_DOMAIN_H

#include <simpla/data/all.h>
#include <simpla/geometry/GeoObject.h>
#include <simpla/utilities/Signal.h>
#include <memory>
#include "Attribute.h"
#include "MeshBase.h"
namespace simpla {
namespace engine {
class MeshBase;
class Patch;
class AttributeGroup;

/**
* @brief
*/
class Domain : public SPObject,
               public AttributeGroup,
               public data::EnableCreateFromDataTable<Domain, std::shared_ptr<geometry::GeoObject>> {
    SP_OBJECT_HEAD(Domain, SPObject)
   public:
    explicit Domain(std::shared_ptr<geometry::GeoObject> const &g = nullptr);
    ~Domain() override;

    SP_DEFAULT_CONSTRUCT(Domain);
    DECLARE_REGISTER_NAME("Domain")

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(const std::shared_ptr<data::DataTable> &t) override;

    virtual MeshBase *GetMesh() = 0;
    virtual MeshBase const *GetMesh() const = 0;

    void AddGeoObject(std::string const &k, std::shared_ptr<geometry::GeoObject> const &g);
    std::shared_ptr<geometry::GeoObject> GetGeoObject(std::string const &k = "") const;
    EntityRange GetRange(std::string const &k = "") const;
    EntityRange GetBodyRange(int IFORM = VERTEX, std::string const &k = "") const;
    EntityRange GetBoundaryRange(int IFORM = VERTEX, std::string const &k = "", bool is_parallel = true) const;
    EntityRange GetParallelBoundaryRange(int IFORM = VERTEX, std::string const &k = "") const;
    EntityRange GetPerpendicularBoundaryRange(int IFORM = VERTEX, std::string const &k = "") const;

    void Initialize() override;
    void Finalize() override;
    void SetUp() override;
    void TearDown() override;

    design_pattern::Signal<void(Domain *, Real)> OnInitialCondition;
    design_pattern::Signal<void(Domain *, Real, Real)> OnBoundaryCondition;
    design_pattern::Signal<void(Domain *, Real, Real)> OnAdvance;

    virtual void InitialCondition(Real time_now) {}
    virtual void BoundaryCondition(Real time_now, Real dt) {}
    virtual void Advance(Real time_now, Real dt) {}

    void Push(const std::shared_ptr<Patch> &);
    std::shared_ptr<Patch> PopPatch();

    std::shared_ptr<Patch> ApplyInitialCondition(const std::shared_ptr<Patch> &, Real time_now);
    std::shared_ptr<Patch> ApplyBoundaryCondition(const std::shared_ptr<Patch> &, Real time_now, Real dt);
    std::shared_ptr<Patch> DoAdvance(const std::shared_ptr<Patch> &, Real time_now, Real dt);

    template <typename T>
    T GetAttribute(std::string const &k, EntityRange const &r = EntityRange()) const {
        return T(Get(k)->cast_as<T>(), r);
    };

    template <typename T>
    T GetAttribute(std::string const &k, std::string const &s) const {
        return GetAttribute<T>(k, GetBodyRange(T::iform, s));
    };

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

#define DOMAIN_HEAD(_DOMAIN_NAME_, _BASE_TYPE_, _MESH_TYPE_)                                                           \
   public:                                                                                                             \
    explicit _DOMAIN_NAME_(std::shared_ptr<geometry::GeoObject> const &g = nullptr) : _BASE_TYPE_(g), m_mesh_(this) {} \
    ~_DOMAIN_NAME_() override = default;                                                                               \
    SP_DEFAULT_CONSTRUCT(_DOMAIN_NAME_);                                                                               \
    DECLARE_REGISTER_NAME(std::string(__STRING(_DOMAIN_NAME_)) + "<" + _MESH_TYPE_::RegisterName() + ">")              \
    template <int IFORM, int DOF = 1>                                                                                  \
    using field_type = Field<_MESH_TYPE_, typename _MESH_TYPE_::scalar_type, IFORM, DOF>;                              \
    _MESH_TYPE_ m_mesh_;                                                                                               \
    MeshBase *GetMesh() override { return &m_mesh_; }                                                                  \
    MeshBase const *GetMesh() const override { return &m_mesh_; }
}
}
#endif  // SIMPLA_DOMAIN_H
