//
// Created by salmon on 17-7-9.
//
#include "Tokamak.h"
namespace simpla {
namespace model {
REGISTER_CREATOR(Tokamak)
struct Tokamak::pimpl_s {
    GEqdsk geqdsk;
    Real m_phi0_ = 0, m_phi1_ = TWOPI;
};
Tokamak::Tokamak() : m_pimpl_(new pimpl_s) {}

std::shared_ptr<data::DataTable> Tokamak::Serialize() const { return std::make_shared<data::DataTable>(); }

void Tokamak::Deserialize(const std::shared_ptr<DataTable> &cfg) {
    model::Model::Deserialize(cfg);

    nTuple<Real, 2> phi = cfg->GetValue("Phi", nTuple<Real, 2>{0, TWOPI});

    m_pimpl_->m_phi0_ = phi[0];
    m_pimpl_->m_phi1_ = phi[1];

    m_pimpl_->geqdsk.load(cfg->GetValue<std::string>("gfile", "gfile"));
    Update();
}

void Tokamak::DoUpdate() {
    model::Model::SetObject(GetName() + ".Limiter",
                            std::make_shared<model::RevolveZ>(m_pimpl_->geqdsk.limiter(), m_pimpl_->geqdsk.PhiAxis,
                                                              m_pimpl_->m_phi0_, m_pimpl_->m_phi1_));

    model::Model::SetObject(GetName() + ".Center",
                            std::make_shared<model::RevolveZ>(m_pimpl_->geqdsk.boundary(), m_pimpl_->geqdsk.PhiAxis,
                                                              m_pimpl_->m_phi0_, m_pimpl_->m_phi1_));

    model::Model::DoUpdate();
}
}  // namespace model {
}  // namespace simpla {
