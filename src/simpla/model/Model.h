//
// Created by salmon on 16-11-27.
//

#ifndef SIMPLA_MODEL_H
#define SIMPLA_MODEL_H

#include <simpla/data/Serializable.h>
#include <simpla/engine/SPObject.h>
#include <simpla/geometry/GeoObject.h>
namespace simpla {
namespace model {

using namespace data;

class Model : public  engine::SPObject, public data::Serializable {
    SP_OBJECT_HEAD(Model,  engine::SPObject);

   public:
    Model();
    ~Model() override;

    SP_DEFAULT_CONSTRUCT(Model)

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(const std::shared_ptr<data::DataTable> &cfg) override;
    using data::Serializable::Serialize;

    void Initialize() override;
    void Update() override;
    void TearDown() override;
    void Finalize() override;

    int GetNDims() const;

    box_type const &GetBoundBox() const;

    void SetObject(std::string const &k, std::shared_ptr<geometry::GeoObject> const &);
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
