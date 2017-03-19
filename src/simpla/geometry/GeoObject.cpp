//
// Created by salmon on 17-2-21.
//
#include "GeoObject.h"
#include "Cube.h"

namespace simpla {
namespace geometry {

struct GeoObjectFactory::pimpl_s {
    std::map<std::string, std::function<std::shared_ptr<GeoObject>(std::shared_ptr<data::DataEntity> const &)>>
        m_mesh_factory_;
};

GeoObjectFactory::GeoObjectFactory() : m_pimpl_(new pimpl_s){};
GeoObjectFactory::~GeoObjectFactory(){};

bool GeoObjectFactory::RegisterCreator(
    std::string const &k,
    std::function<std::shared_ptr<GeoObject>(std::shared_ptr<data::DataEntity> const &)> const &fun) {
    return m_pimpl_->m_mesh_factory_.emplace(k, fun).second;
};

std::shared_ptr<GeoObject> GeoObjectFactory::Create(std::shared_ptr<data::DataEntity> const &t) const {
    std::shared_ptr<GeoObject> res = nullptr;
    if (t == nullptr) {
        res = std::make_shared<Cube>(box_type{{0, 0, 0}, {1, 1, 1}});
    } else if (t->value_type_info() == typeid(std::string)) {
        res = m_pimpl_->m_mesh_factory_.at(data::data_cast<std::string>(*t))(t);
    } else if (t->isTable()) {
        res = m_pimpl_->m_mesh_factory_.at(t->cast_as<data::DataTable>().GetValue<std::string>("name"))(t);
    }

    if (res != nullptr) { LOGGER << "GeoObject [" << res->getClassName() << "] is created!" << std::endl; }
    return res;
}

}  // namespace geometry {
}  // namespace simpla {