//
// Created by salmon on 16-11-27.
//

#ifndef SIMPLA_MODEL_H
#define SIMPLA_MODEL_H

#include <simpla/geometry/GeoObject.h>
#include <simpla/mesh/EntityId.h>
#include "Attribute.h"
#include "Mesh.h"

namespace simpla {
namespace model {
class GeoObject;
}
namespace engine {

using namespace data;

class Model : public concept::Configurable {
    typedef Model this_type;

   public:
    Model(std::shared_ptr<data::DataTable> const &t = nullptr);
    virtual ~Model();
    virtual bool Update();
    virtual void Initialize();
    box_type const &bound_box() const;
    std::shared_ptr<data::DataTable> GetMaterial(std::string const &k) const;
    std::shared_ptr<data::DataTable> SetMaterial(std::string const &k, std::shared_ptr<DataTable> const &p = nullptr);
    id_type GetMaterialId(std::string const &k) const;

    id_type AddObject(std::string const &material_type_name, std::shared_ptr<geometry::GeoObject> const &);
    std::shared_ptr<geometry::GeoObject> GetObject(std::string const &k) const;
    size_type DeleteObject(std::string const &);

    //    id_type GetObjectMaterialId(id_type) const;

    //    geometry::GeoObject SelectObjectByMaterial(std::string const &material_type_name) const;
    //    //    geometry::GeoObject SelectObjectByMaterial(id_type) const;
    //    size_type RemoveObjectByMaterial(id_type);

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}  // namespace engine {
}  // namespace simpla{namespace model{

#endif  // SIMPLA_MODEL_H
