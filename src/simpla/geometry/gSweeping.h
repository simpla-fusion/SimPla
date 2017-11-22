//
// Created by salmon on 17-11-22.
//

#ifndef SIMPLA_GSWEEPING_H
#define SIMPLA_GSWEEPING_H

#include "GeoEntity.h"
#include "gBody.h"
#include "gCurve.h"
#include "gSurface.h"
namespace simpla {
namespace geometry {

struct gSweeping {
    explicit gSweeping(std::shared_ptr<const GeoEntity> const& basis_entity, std::shared_ptr<const gCurve> const& curve,
                       vector_type const& Nx = {1, 0, 0}, vector_type const& Ny = {0, 1, 1})
        : m_basis_entity_(basis_entity), m_curve_(curve), m_Nx_(Nx), m_Ny_(Ny) {}

    vector_type m_Nx_;
    vector_type m_Ny_;

    void SetCurve(std::shared_ptr<const gCurve> const& c, vector_type const& Nx = {1, 0, 0},
                  vector_type const& Ny = {0, 1, 1}) {
        m_curve_ = c;
        m_Nx_ = (Nx);
        m_Ny_ = (Ny);
    };
    std::shared_ptr<const gCurve> GetCurve() const { return m_curve_; };
    void SetBasisEntity(std::shared_ptr<const GeoEntity> const& b) { m_basis_entity_ = b; };
    std::shared_ptr<const GeoEntity> GetBasisEntity() const { return m_basis_entity_; };

   private:
    std::shared_ptr<const gCurve> m_curve_;
    std::shared_ptr<const GeoEntity> m_basis_entity_;
};

struct gSweepingSurface : public gSurface, public gSweeping {
    SP_GEO_ENTITY_HEAD(gSurface, gSweepingSurface, SweepingSurface);

    explicit gSweepingSurface(std::shared_ptr<const gCurve> const& basis_entity,
                              std::shared_ptr<const gCurve> const& curve, vector_type const& Nx = {1, 0, 0},
                              vector_type const& Ny = {0, 1, 1})
        : gSweeping(basis_entity, curve, m_Nx_, m_Ny_) {}
};
struct gSweepingBody : public gBody, public gSweeping {
    SP_GEO_ENTITY_HEAD(gBody, gSweepingBody, SweepingBody);
    explicit gSweepingBody(std::shared_ptr<const gSurface> const& basis_entity,
                           std::shared_ptr<const gCurve> const& curve, vector_type const& Nx = {1, 0, 0},
                           vector_type const& Ny = {0, 1, 1})
        : gSweeping(basis_entity, curve, m_Nx_, m_Ny_){} {}
};
std::shared_ptr<GeoEntity> gMakeRevolution(std::shared_ptr<const GeoEntity> const& geo, vector_type const& Nr,
                                           vector_type const& Nz);

std::shared_ptr<GeoEntity> gMakePrism(std::shared_ptr<const GeoEntity> const& geo, vector_type const& direction);

std::shared_ptr<GeoEntity> gMakePipe(std::shared_ptr<const GeoEntity> const& geo,
                                     std::shared_ptr<const gCurve> const& curve);
}  // namespace geometry {
}  // namespace simpla {
#endif  // SIMPLA_GSWEEPING_H
