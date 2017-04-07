//
// Created by salmon on 17-2-21.
//
#include "GeoObject.h"
#include "Cube.h"

namespace simpla {
namespace geometry {

struct GeoObjectFactory {
    std::map<std::string, std::function<GeoObject *(std::shared_ptr<data::DataTable> const &)>> m_geo_factory_;
};

bool GeoObject::RegisterCreator(std::string const &k,
                                std::function<GeoObject *(std::shared_ptr<data::DataTable> const &)> const &fun) {
    return SingletonHolder<GeoObjectFactory>::instance().m_geo_factory_.emplace(k, fun).second;
};

GeoObject *GeoObject::Create(std::shared_ptr<data::DataTable> const &cfg) {
    GeoObject *res = nullptr;

    try {
        if (cfg != nullptr) {
            res = SingletonHolder<GeoObjectFactory>::instance().m_geo_factory_.at(
                cfg->GetValue<std::string>("name", ""))(cfg);
        }
    } catch (std::out_of_range const &) {
        RUNTIME_ERROR << "GeoObject creator  [] is missing!" << std::endl;
        return nullptr;
    }
    if (res != nullptr) { LOGGER << "GeoObject [" << res->GetClassName() << "] is created!" << std::endl; }
    return res;
}
}  // namespace geometry {
}  // namespace simpla {