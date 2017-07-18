//
// Created by salmon on 17-7-9.
//
#include "Tokamak.h"
#include "simpla/data/Data.h"
namespace simpla {

bool Tokamak::is_registered = engine::Model::RegisterCreator<Tokamak>("Tokamak");

struct Tokamak::pimpl_s {
    GEqdsk geqdsk;
    Real m_phi0_ = 0, m_phi1_ = TWOPI;
};
Tokamak::Tokamak() : m_pimpl_(new pimpl_s) {}

std::shared_ptr<data::DataTable> Tokamak::Serialize() const {
    auto res = base_type::Serialize();

    res->SetValue("Type", "Tokamak");

    return res;
}

void Tokamak::Deserialize(const std::shared_ptr<data::DataTable> &cfg) {
    nTuple<Real, 2> phi = cfg->GetValue("Phi", nTuple<Real, 2>{0, TWOPI});

    m_pimpl_->m_phi0_ = phi[0];
    m_pimpl_->m_phi1_ = phi[1];

    LoadGFile(cfg->GetValue<std::string>("gfile", "gfile"));

    Update();
}
engine::Model::attr_fun Tokamak::GetAttribute(std::string const &attr_name) const {
    attr_fun res = nullptr;

    if (attr_name == "psi") {
        res = [&](point_type const &x) { return m_pimpl_->geqdsk.psi(x); };
    } else if (attr_name == "JT") {
        res = [&](point_type const &x) { return m_pimpl_->geqdsk.JT(x[0], x[1]); };
    } else {
        res = m_pimpl_->geqdsk.GetAttribute(attr_name);
    }

    return res;
};
    engine::Model::vec_attr_fun Tokamak::GetAttributeVector(std::string const &attr_name) const {
    vec_attr_fun res = nullptr;
    if (attr_name == "B0") {
        res = [&](point_type const &x) { return m_pimpl_->geqdsk.B(x); };
    }

    return res;
};
void Tokamak::LoadGFile(std::string const &file) { m_pimpl_->geqdsk.load(file); }

void Tokamak::DoUpdate() {
    engine::Model::SetObject("Limiter",
                             std::make_shared<geometry::RevolveZ>(m_pimpl_->geqdsk.limiter(), m_pimpl_->geqdsk.PhiAxis,
                                                                  m_pimpl_->m_phi0_, m_pimpl_->m_phi1_));

    engine::Model::SetObject("Plasma",
                             std::make_shared<geometry::RevolveZ>(m_pimpl_->geqdsk.boundary(), m_pimpl_->geqdsk.PhiAxis,
                                                                  m_pimpl_->m_phi0_, m_pimpl_->m_phi1_));

    engine::Model::DoUpdate();
}
}  // namespace simpla {
