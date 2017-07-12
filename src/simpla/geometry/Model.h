//
// Created by salmon on 16-11-27.
//

#ifndef SIMPLA_MODEL_H
#define SIMPLA_MODEL_H

#include <simpla/data/Serializable.h>
#include <simpla/utilities/SPObject.h>
#include <functional>
#include "GeoObject.h"
namespace simpla {
namespace geometry {

using namespace data;

class Model : public engine::SPObject, public data::EnableCreateFromDataTable<Model> {
    SP_OBJECT_HEAD(Model, engine::SPObject);

   public:
    Model();
    ~Model() override;

    SP_DEFAULT_CONSTRUCT(Model)
    DECLARE_REGISTER_NAME(Model)

    std::shared_ptr<DataTable> Serialize() const override;
    void Deserialize(const std::shared_ptr<DataTable> &cfg) override;

    void DoInitialize() override;
    void DoUpdate() override;
    void DoTearDown() override;
    void DoFinalize() override;

    int GetNDims() const;

    box_type const &GetBoundBox() const;

    void GetAttribute(std::string const &attr_name) const;
    std::function<nTuple<Real, 3>(point_type const &)> GetVectorAttribute(std::string const &attr_name) const;

    void SetObject(std::string const &k, std::shared_ptr<geometry::GeoObject> const &);
    std::shared_ptr<geometry::GeoObject> GetObject(std::string const &k) const;
    size_type DeleteObject(std::string const &);

    std::map<std::string, std::shared_ptr<geometry::GeoObject>> const &GetAll() const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}  // namespace engine {
}  // namespace simpla{namespace geometry{

#endif  // SIMPLA_MODEL_H