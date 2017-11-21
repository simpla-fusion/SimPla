//
// Created by salmon on 17-11-1.
//
#include "GeoEngine.h"
#include <simpla/data/DataEntry.h>
#include <simpla/utilities/SingletonHolder.h>
#include "GeoAlgorithm.h"
#include "GeoObject.h"
namespace simpla {
namespace geometry {
void Initialize(std::string const &key) {
    auto p = Factory<GeoEngineAPI>::Create(key.empty() ? "OCE" : key);
    GEO_ENGINE.swap(p);
    if (GEO_ENGINE == nullptr) {
        RUNTIME_ERROR << "Create GeoEngine Fail! [" << key << "]" << std::endl
                      << Factory<GeoEngineAPI>::ShowDescription();
    } else {
        VERBOSE << "Create Geometry Engine : " << GEO_ENGINE->GetRegisterName();
    }
}
void Initialize(std::shared_ptr<data::DataEntry> const &d) {
    if (d != nullptr) {
        Initialize(d->GetValue<std::string>("_TYPE_", ""));
        GEO_ENGINE->Deserialize(d);
    }
    if (GEO_ENGINE == nullptr) {
        RUNTIME_ERROR << "Create GeoEngine Fail! " << *d << std::endl << Factory<GeoEngineAPI>::ShowDescription();
    }
}

void Finalize() {
    GEO_ENGINE->CloseFile();
    GEO_ENGINE.reset();
}

GeoEngineAPI::GeoEngineAPI() = default;
GeoEngineAPI::~GeoEngineAPI() = default;
void GeoEngineAPI::Deserialize(std::shared_ptr<simpla::data::DataEntry> const &cfg) {}
std::shared_ptr<simpla::data::DataEntry> GeoEngineAPI::Serialize() const {
    return data::DataEntry::New(data::DataEntry::DN_TABLE);
}
std::shared_ptr<GeoObject> GeoEngineAPI::GetBoundary(std::shared_ptr<const GeoObject> const &) const {
    UNIMPLEMENTED;
    return nullptr;
}
bool GeoEngineAPI::CheckIntersection(std::shared_ptr<const GeoObject> const &g, point_type const &p,
                                     Real tolerance) const {
    return CheckPointInBox(p, g->GetBoundingBox());
}
bool GeoEngineAPI::CheckIntersection(std::shared_ptr<const GeoObject> const &g, box_type const &box,
                                     Real tolerance) const {
    return CheckBoxOverlapped(box, g->GetBoundingBox());
}
bool GeoEngineAPI::CheckIntersection(std::shared_ptr<const GeoObject> const &g0,
                                     std::shared_ptr<const GeoObject> const &g1, Real tolerance) const {
    ASSERT(g0 != nullptr && g1 != nullptr);
    return CheckIntersection(g0, g1->GetBoundingBox(), tolerance) && GetIntersection(g0, g1, tolerance) != nullptr;
}
std::shared_ptr<GeoObject> GeoEngineAPI::GetUnion(std::shared_ptr<const GeoObject> const &g0,
                                                  std::shared_ptr<const GeoObject> const &g1, Real tolerance) const {
    DUMMY << "Union :" << g0->FancyTypeName() << " || " << g1->FancyTypeName();
    return nullptr;
}
std::shared_ptr<GeoObject> GeoEngineAPI::GetDifference(std::shared_ptr<const GeoObject> const &g0,
                                                       std::shared_ptr<const GeoObject> const &g1,
                                                       Real tolerance) const {
    DUMMY << "Difference : " << g0->FancyTypeName() << " - " << g1->FancyTypeName();
    return nullptr;
}
std::shared_ptr<GeoObject> GeoEngineAPI::GetIntersection(std::shared_ptr<const GeoObject> const &g0,
                                                         std::shared_ptr<const GeoObject> const &g1,
                                                         Real tolerance) const {
    DUMMY << "Intersection : " << g0->FancyTypeName() << " && " << g1->FancyTypeName();
    return nullptr;
}
void GeoEngineAPI::OpenFile(std::string const &path) { m_is_opened_ = true; }
void GeoEngineAPI::CloseFile() { m_is_opened_ = false; }
void GeoEngineAPI::FlushFile() { DUMMY << "Dump File"; }
std::string GeoEngineAPI::GetFilePath() const { return ""; }
void GeoEngineAPI::Save(std::shared_ptr<const GeoObject> const &geo, std::string const &name) const {
    DUMMY << "Save GeoObject to:" << name;
}
std::shared_ptr<GeoObject> GeoEngineAPI::Load(std::string const &name) const {
    DUMMY << "Load GeoObject from:" << name;
    return nullptr;
}
}  // namespace geometry
}  // namespace simpla