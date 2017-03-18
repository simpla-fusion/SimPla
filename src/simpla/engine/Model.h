//
// Created by salmon on 16-11-27.
//

#ifndef SIMPLA_MODEL_H
#define SIMPLA_MODEL_H

#include <simpla/geometry/GeoObject.h>
#include <simpla/mesh/EntityId.h>
#include "AttributeView.h"
#include "MeshView.h"

namespace simpla {
namespace model {
class GeoObject;
}
namespace engine {

using namespace data;

class Model : public SPObject, public concept::Printable {
    typedef Model this_type;

   public:
    Model();
    virtual ~Model();
    virtual std::ostream &Print(std::ostream &os, int indent = 0) const;
    virtual bool Update();
    virtual void Initialize();
    box_type const &bound_box() const;
    std::shared_ptr<data::DataTable> GetMaterial(std::string const &k) const;
    std::shared_ptr<data::DataTable> SetMaterial(std::string const &k, std::shared_ptr<DataTable> const &p = nullptr);
    id_type GetMaterialId(std::string const &k) const;

    //    std::map<std::string, id_type> &GetMaterialListByName() const;
    //    std::map<id_type, std::string> &GetMaterialListById() const;

    id_type AddObject(std::string const &material_type_name, std::shared_ptr<geometry::GeoObject> const &);
    id_type AddObject(id_type material_type_id, std::shared_ptr<geometry::GeoObject> const &);

    std::shared_ptr<geometry::GeoObject> GetObject(id_type) const;
    std::shared_ptr<geometry::GeoObject> GetObject(std::string const &k) const;
    size_type RemoveObject(id_type);
    size_type RemoveObject(std::string const &);

    //    id_type GetObjectMaterialId(id_type) const;

    geometry::GeoObject SelectObjectByMaterial(std::string const &material_type_name) const;
    //    geometry::GeoObject SelectObjectByMaterial(id_type) const;
    size_type RemoveObjectByMaterial(id_type);

    std::shared_ptr<MeshView> GetMesh(std::string const &) const;
    std::shared_ptr<MeshView> SetMesh(std::string const &, std::shared_ptr<engine::MeshView>);

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}  // namespace engine {
}  // namespace simpla{namespace model{

#endif  // SIMPLA_MODEL_H
