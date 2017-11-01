//
// Created by salmon on 17-11-1.
//
#include "GeoEngine.h"
#include "GeoAlgorithm.h"
#include "GeoObject.h"
namespace simpla {
namespace geometry {

struct GeoEngineHolder {
    std::shared_ptr<GeoEngine> m_engine_ = nullptr;
    void Initialize(std::shared_ptr<data::DataNode> const &d = nullptr);
    void Initialize(int argc, char **argv);
    void Finalize();
    GeoEngine &get();
    GeoEngine const &get() const;
};
void GeoEngineHolder::Initialize(std::shared_ptr<data::DataNode> const &d) { m_engine_ = GeoEngine::Create(d); }
void GeoEngineHolder::Initialize(int argc, char **argv) { UNIMPLEMENTED; }
void GeoEngineHolder::Finalize() {}
GeoEngine &GeoEngineHolder::get() {
    Initialize();
    return *m_engine_;
}
GeoEngine const &GeoEngineHolder::get() const {
    ASSERT(m_engine_ != nullptr);
    return *m_engine_;
}

GeoEngine::GeoEngine() = default;
// GeoEngine::GeoEngine(GeoEngine const &) = default;
GeoEngine::~GeoEngine() = default;

void GeoEngine::Initialize(std::shared_ptr<data::DataNode> const &d) {
    SingletonHolder<GeoEngineHolder>::instance().Initialize(d);
}
void GeoEngine::Initialize(int argc, char **argv) {
    SingletonHolder<GeoEngineHolder>::instance().Initialize(argc, argv);
}
void GeoEngine::Finalize() { SingletonHolder<GeoEngineHolder>::instance().Finalize(); }
GeoEngine &GeoEngine::entry() { return SingletonHolder<GeoEngineHolder>::instance().get(); }
std::shared_ptr<GeoObject> GeoEngine::GetBoundary(std::shared_ptr<const GeoObject> const &g) {
    return entry().GetBoundaryInterface(g);
}
bool GeoEngine::CheckIntersection(std::shared_ptr<const GeoObject> const &g, point_type const &x, Real tolerance) {
    return entry().CheckIntersectionInterface(g, x, tolerance);
}
bool GeoEngine::CheckIntersection(std::shared_ptr<const GeoObject> const &g, box_type const &b, Real tolerance) {
    return entry().CheckIntersectionInterface(g, b, tolerance);
}

std::shared_ptr<GeoObject> GeoEngine::GetUnion(std::shared_ptr<const GeoObject> const &g0,
                                               std::shared_ptr<const GeoObject> const &g1, Real tolerance) {
    return entry().GetIntersectionInterface(g0, g1, tolerance);
}
std::shared_ptr<GeoObject> GeoEngine::GetDifference(std::shared_ptr<const GeoObject> const &g0,
                                                    std::shared_ptr<const GeoObject> const &g1, Real tolerance) {
    return entry().GetIntersectionInterface(g0, g1, tolerance);
}
std::shared_ptr<GeoObject> GeoEngine::GetIntersection(std::shared_ptr<const GeoObject> const &g0,
                                                      std::shared_ptr<const GeoObject> const &g1, Real tolerance) {
    return entry().GetIntersectionInterface(g0, g1, tolerance);
}

std::shared_ptr<GeoObject> GeoEngine::GetBoundaryInterface(std::shared_ptr<const GeoObject> const &) const {
    UNIMPLEMENTED;
    return nullptr;
}
bool GeoEngine::CheckIntersectionInterface(std::shared_ptr<const GeoObject> const &g, point_type const &p,
                                           Real tolerance) const {
    return TestPointInBox(p, g->GetBoundingBox());
}
bool GeoEngine::CheckIntersectionInterface(std::shared_ptr<const GeoObject> const &g, box_type const &box,
                                           Real tolerance) const {
    return TestBoxOverlapped(box, g->GetBoundingBox());
}
std::shared_ptr<GeoObject> GeoEngine::GetUnionInterface(std::shared_ptr<const GeoObject> const &g0,
                                                        std::shared_ptr<const GeoObject> const &g1,
                                                        Real tolerance) const {
    DUMMY << "Union :" << g0->FancyTypeName() << " || " << g1->FancyTypeName();
    return nullptr;
}
std::shared_ptr<GeoObject> GeoEngine::GetDifferenceInterface(std::shared_ptr<const GeoObject> const &g0,
                                                             std::shared_ptr<const GeoObject> const &g1,
                                                             Real tolerance) const {
    DUMMY << "Difference : " << g0->FancyTypeName() << " - " << g1->FancyTypeName();
    return nullptr;
}
std::shared_ptr<GeoObject> GeoEngine::GetIntersectionInterface(std::shared_ptr<const GeoObject> const &g0,
                                                               std::shared_ptr<const GeoObject> const &g1,
                                                               Real tolerance) const {
    DUMMY << "Intersection : " << g0->FancyTypeName() << " && " << g1->FancyTypeName();
    return nullptr;
}
}  // namespace geometry
}  // namespace simpla