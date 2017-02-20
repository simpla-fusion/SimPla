//
// Created by salmon on 16-11-27.
//

#ifndef SIMPLA_MODEL_H
#define SIMPLA_MODEL_H

#include <simpla/concept/Configurable.h>
#include <simpla/engine/AttributeView.h>
#include <simpla/engine/MeshView.h>
#include <simpla/mesh/EntityId.h>
#include "geometry/GeoObject.h"

namespace simpla {
namespace model {

using namespace data;

class GeoObject;

class Model : public concept::Printable, public concept::Configurable {
    typedef Model this_type;

   public:
    Model();
    virtual ~Model();
    virtual std::ostream &Print(std::ostream &os, int indent = 0) const;
    virtual void Update();

    box_type const &bound_box() const;
    id_type GetMaterialId(std::string const &) const;

    void AddObject(std::string const &domain_type_name, std::shared_ptr<geometry::GeoObject> const &);
    void AddObject(id_type, std::shared_ptr<geometry::GeoObject> const &);

    std::shared_ptr<geometry::GeoObject> const &GetObject(std::string const &key) const;
    std::shared_ptr<geometry::GeoObject> const &GetObject(id_type) const;
    void RemoveObject(std::string const &key);

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}
}  // namespace simpla{namespace model{

#endif  // SIMPLA_MODEL_H
