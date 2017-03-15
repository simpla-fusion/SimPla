//
// Created by salmon on 16-11-27.
//

#ifndef SIMPLA_MODEL_H
#define SIMPLA_MODEL_H

#include <simpla/engine/AttributeView.h>
#include <simpla/engine/MeshView.h>
#include <simpla/mesh/EntityId.h>
#include "geometry/GeoObject.h"

namespace simpla {
namespace model {

using namespace data;

class GeoObject;

class Model : public SPObject, public concept::Printable {
    typedef Model this_type;

   public:
    Model();
    virtual ~Model();
    virtual std::ostream &Print(std::ostream &os, int indent = 0) const;
    virtual bool Update();
    bool Valid();
    box_type const &bound_box() const;
    data::DataTable const &GetMaterial(std::string const &k) const;
    data::DataTable &GetMaterial(std::string const &k);
    id_type GetMaterialId(std::string const &k) const;
    id_type GetMaterialId(std::string const &k);

    std::map<std::string, id_type> &GetMaterialListByName() const;
    std::map<id_type, std::string> &GetMaterialListById() const;

    id_type AddObject(geometry::GeoObject const &, std::string const &material_type_name);
    id_type AddObject(geometry::GeoObject const &, id_type material_type_id = NULL_ID);
    geometry::GeoObject const &GetObject(id_type) const;
    size_type RemoveObject(id_type);
    size_type RemoveObject(std::string const &);

    id_type GetObjectMaterialId(id_type) const;

    geometry::GeoObject SelectObjectByMaterial(std::string const &material_type_name) const;
    geometry::GeoObject SelectObjectByMaterial(id_type) const;
    size_type RemoveObjectByMaterial(id_type);

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}
}  // namespace simpla{namespace model{

#endif  // SIMPLA_MODEL_H
