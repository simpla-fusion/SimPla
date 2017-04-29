//
// Created by salmon on 16-11-27.
//

#ifndef SIMPLA_MODEL_H
#define SIMPLA_MODEL_H

#include <simpla/geometry/GeoObject.h>
#include "Attribute.h"
#include "MeshBase.h"

namespace simpla {
namespace model {
class GeoObject;
}
namespace engine {

using namespace data;

class Model : public data::Serializable {
    SP_OBJECT_BASE(Model);

   public:
    Model();
    ~Model() override;

    SP_DEFAULT_CONSTRUCT(Model)

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(const std::shared_ptr<data::DataTable> &cfg) override;
    using data::Serializable::Serialize;

    void Initialize();
    void SetUp();
    void TearDown();

    void Finalize();

    int GetNDims() const;

    box_type const &GetBoundBox() const;

    void SetObject(std::string const &k, std::shared_ptr<DataTable>);
    void SetObject(std::string const &k,
                                                                    std::shared_ptr<geometry::GeoObject> const &);
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
