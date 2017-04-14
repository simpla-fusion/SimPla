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

class Model : public data::Serializable {
    typedef Model this_type;

   public:
    Model();
    ~Model();
    void Update();
    void Initialize();
    void Finalize();

    int GetNDims() const;
    box_type const &GetBoundBox() const;

    //    std::shared_ptr<data::DataTable> GetMaterial(std::string const &k = "") const;
    //    std::shared_ptr<data::DataTable> SetMaterial(std::string const &k, std::shared_ptr<DataTable> p = nullptr);
    //    id_type GetMaterialId(std::string const &k) const;

    std::pair<std::shared_ptr<geometry::GeoObject>, bool> AddObject(std::string const &k, std::shared_ptr<DataTable>);

    id_type AddObject(std::string const &k, std::shared_ptr<geometry::GeoObject> const &);
    std::shared_ptr<geometry::GeoObject> GetObject(std::string const &k) const;
    size_type DeleteObject(std::string const &);

    std::map<std::string, std::shared_ptr<geometry::GeoObject>> const &GetAll() const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}  // namespace engine {
}  // namespace simpla{namespace model{

#endif  // SIMPLA_MODEL_H
