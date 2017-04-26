//
// Created by salmon on 17-4-5.
//

#ifndef SIMPLA_DOMAIN_H
#define SIMPLA_DOMAIN_H

#include <simpla/geometry/GeoObject.h>
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
class Domain
    : public data::Serializable,
      public data::EnableCreateFromDataTable<Domain, std::shared_ptr<geometry::GeoObject>, std::shared_ptr<MeshBase> > {
    SP_OBJECT_BASE(Domain)
   public:
    explicit Domain(std::shared_ptr<geometry::GeoObject> g = nullptr, std::shared_ptr<MeshBase> m = nullptr);
    ~Domain() override;

    SP_DEFAULT_CONSTRUCT(Domain);
    DECLARE_REGISTER_NAME("Domain")

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(std::shared_ptr<data::DataTable> t) override;

    virtual std::shared_ptr<Domain> Clone() const;

    std::shared_ptr<MeshBase> &GetMesh() { return m_mesh_; }
    std::shared_ptr<MeshBase> const &GetMesh() const { return m_mesh_; }
    std::shared_ptr<geometry::GeoObject> &GetGeoObject() { return m_geo_object_; }
    std::shared_ptr<geometry::GeoObject> const &GetGeoObject() const { return m_geo_object_; }

    virtual void Initialize();
    virtual void Finalize();

    virtual void Push(Patch *);
    virtual void Pop(Patch *);
    virtual void SetUp();
    virtual void TearDown();
    virtual void InitializeCondition(Real time_now);
    virtual void BoundaryCondition(Real time_now, Real dt);
    virtual void Advance(Real time_now, Real dt);

   private:
    std::shared_ptr<MeshBase> m_mesh_;
    std::shared_ptr<geometry::GeoObject> m_geo_object_;
};

#define DOMAIN_HEAD(_DOMAIN_NAME_)                                                                                 \
   public:                                                                                                         \
    explicit _DOMAIN_NAME_(std::shared_ptr<geometry::GeoObject> g, std::shared_ptr<engine::MeshBase> m = nullptr)  \
        : engine::Domain(                                                                                          \
              g, (m != nullptr) ? m : std::dynamic_pointer_cast<engine::MeshBase>(std::make_shared<mesh_type>())), \
          m_mesh_(std::dynamic_pointer_cast<mesh_type>(engine::Domain::GetMesh())) {                               \
        if (m == nullptr && g != nullptr) {                                                                        \
            GetMesh()->GetChart()->SetOrigin(std::get<0>(GetGeoObject()->GetBoundBox()));                          \
        }                                                                                                          \
    }                                                                                                              \
    ~_DOMAIN_NAME_() override = default;                                                                           \
    SP_DEFAULT_CONSTRUCT(_DOMAIN_NAME_);                                                                           \
    DECLARE_REGISTER_NAME(std::string(__STRING(_DOMAIN_NAME_)) + "<" + mesh_type::ClassName() + ">")               \
    std::shared_ptr<mesh_type> m_mesh_;                                                                            \
    template <int IFORM, int DOF = 1>                                                                              \
    using field_type = Field<mesh_type, typename mesh_type::scalar_type, IFORM, DOF>;
}
}
#endif  // SIMPLA_DOMAIN_H
