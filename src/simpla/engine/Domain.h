//
// Created by salmon on 17-4-5.
//

#ifndef SIMPLA_DOMAIN_H
#define SIMPLA_DOMAIN_H

#include <simpla/geometry/GeoObject.h>
#include <simpla/utilities/Signal.h>
#include <memory>
#include "Attribute.h"
#include "simpla/data/all.h"
namespace simpla {
namespace engine {
class MeshBase;
class Patch;
class AttributeGroup;
/**
* @brief
*/
class Domain : public data::Serializable,
               public data::EnableCreateFromDataTable<Domain, std::shared_ptr<geometry::GeoObject> > {
    SP_OBJECT_BASE(Domain)
   public:
    explicit Domain(std::shared_ptr<geometry::GeoObject> const &g = nullptr,
                    std::shared_ptr<MeshBase> const &m = nullptr);
    ~Domain() override;

    SP_DEFAULT_CONSTRUCT(Domain);
    DECLARE_REGISTER_NAME("Domain")

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(const std::shared_ptr<data::DataTable> &t) override;

    std::shared_ptr<MeshBase> &GetMesh() { return m_mesh_; }
    std::shared_ptr<MeshBase> const &GetMesh() const { return m_mesh_; }
    std::shared_ptr<geometry::GeoObject> &GetGeoObject() { return m_geo_object_; }
    std::shared_ptr<geometry::GeoObject> const &GetGeoObject() const { return m_geo_object_; }

    void Push(Patch *);
    void Pop(Patch *);

    virtual void Initialize();
    virtual void Finalize();
    virtual void SetUp();
    virtual void TearDown();

    design_pattern::Signal<void(Domain *)> OnInitialize;
    design_pattern::Signal<void(Domain *)> OnFinalize;
    design_pattern::Signal<void(Domain *)> OnSetUp;
    design_pattern::Signal<void(Domain *)> OnTearDown;

    design_pattern::Signal<void(Domain *, Real)> OnInitialCondition;
    design_pattern::Signal<void(Domain *, Real, Real)> OnBoundaryCondition;
    design_pattern::Signal<void(Domain *, Real, Real)> OnAdvance;
    virtual void InitialCondition(Real time_now);
    virtual void BoundaryCondition(Real time_now, Real dt);
    virtual void Advance(Real time_now, Real dt);

   private:
    std::shared_ptr<MeshBase> m_mesh_;
    std::shared_ptr<geometry::GeoObject> m_geo_object_;
};

#define DOMAIN_HEAD(_DOMAIN_NAME_, _BASE_TYPE_)                                                                    \
   public:                                                                                                         \
    explicit _DOMAIN_NAME_(std::shared_ptr<geometry::GeoObject> const &g,                                          \
                           std::shared_ptr<engine::MeshBase> const &m = nullptr)                                   \
        : _BASE_TYPE_(                                                                                             \
              g, (m != nullptr) ? m : std::dynamic_pointer_cast<engine::MeshBase>(std::make_shared<mesh_type>())), \
          m_mesh_(std::dynamic_pointer_cast<mesh_type>(engine::Domain::GetMesh())) {}                              \
    ~_DOMAIN_NAME_() override = default;                                                                           \
    SP_DEFAULT_CONSTRUCT(_DOMAIN_NAME_);                                                                           \
    DECLARE_REGISTER_NAME(std::string(__STRING(_DOMAIN_NAME_)) + "<" + mesh_type::RegisterName() + ">")            \
    std::shared_ptr<mesh_type> m_mesh_;                                                                            \
    template <int IFORM, int DOF = 1>                                                                              \
    using field_type = Field<mesh_type, typename mesh_type::scalar_type, IFORM, DOF>;
}
}
#endif  // SIMPLA_DOMAIN_H
