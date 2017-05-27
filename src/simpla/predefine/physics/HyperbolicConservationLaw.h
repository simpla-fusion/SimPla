//
// Created by salmon on 17-5-27.
//

#ifndef SIMPLA_HYPERBOLICCONSERVATIONLAW_H
#define SIMPLA_HYPERBOLICCONSERVATIONLAW_H

#include "simpla/SIMPLA_config.h"
#include "simpla/algebra/all.h"
#include "simpla/engine/all.h"
#include "simpla/physics/PhysicalConstants.h"
namespace simpla {
template <typename TM>
class HyperbolicConservationLaw : public engine::Domain {
    SP_OBJECT_HEAD(HyperbolicConservationLaw<TM>, engine::Domain)

   public:
    DOMAIN_HEAD(HyperbolicConservationLaw, engine::Domain, TM)

    std::shared_ptr<data::DataTable> Serialize() const override;

    void Deserialize(std::shared_ptr<data::DataTable> const &cfg) override;

    void InitialCondition(Real time_now) override;

    void BoundaryCondition(Real time_now, Real dt) override;

    void Advance(Real time_now, Real dt) override;

    DOMAIN_DECLARE_FIELD(ne, VOLUME, 1);
    DOMAIN_DECLARE_FIELD(B0v, VOLUME, 3);
    DOMAIN_DECLARE_FIELD(J, EDGE, 1);

    DOMAIN_DECLARE_FIELD(rho, VOLUME, 1);
    DOMAIN_DECLARE_FIELD(rhoU, VOLUME, 1);
    DOMAIN_DECLARE_FIELD(rhoW, VOLUME, 1);
    DOMAIN_DECLARE_FIELD(rhoV, VOLUME, 1);
    DOMAIN_DECLARE_FIELD(rhoE, VOLUME, 1);

    DOMAIN_DECLARE_FIELD(fRho, FACE, 1);
    DOMAIN_DECLARE_FIELD(fRhoU, FACE, 1);
    DOMAIN_DECLARE_FIELD(fRhoW, FACE, 1);
    DOMAIN_DECLARE_FIELD(fRhoV, FACE, 1);
    DOMAIN_DECLARE_FIELD(fRhoE, FACE, 1);
};

template <typename TM>
bool HyperbolicConservationLaw<TM>::is_registered = engine::Domain::RegisterCreator<HyperbolicConservationLaw<TM>>();

template <typename TM>
std::shared_ptr<data::DataTable> HyperbolicConservationLaw<TM>::Serialize() const {
    auto res = std::make_shared<data::DataTable>();
    res->SetValue<std::string>("Type", "HyperbolicConservationLaw<" + TM::RegisterName() + ">");

    return res;
};

template <typename TM>
void HyperbolicConservationLaw<TM>::Deserialize(std::shared_ptr<data::DataTable> const &cfg) {
    DoInitialize();

    Click();
}

template <typename TM>
void HyperbolicConservationLaw<TM>::InitialCondition(Real time_now) {
    Domain::InitialCondition(time_now);
}

template <typename TM>
void HyperbolicConservationLaw<TM>::BoundaryCondition(Real time_now, Real dt) {}

template <typename TF, typename TPL, typename TPR>
void RS(TF &f, TPL const &Pl, TPR const &Pr, double gamma) {
    double fl[5], fr[5], ul[5], ur[5];
    double lambda, c;
    c = std::sqrt(gamma * std::max(Pl[4], Pr[4]) / std::min(Pl[0], Pr[0]));
    lambda = c + max(fabs(Pl[1]), fabs(Pr[1]));
    //
    ul[0] = Pl[0];
    ur[0] = Pr[0];
    ul[1] = Pl[0] * Pl[1];
    ur[1] = Pr[0] * Pr[1];
    ul[2] = Pl[0] * Pl[2];
    ur[2] = Pr[0] * Pr[2];
    ul[3] = Pl[0] * Pl[3];
    ur[3] = Pr[0] * Pr[3];
    ul[4] = Pl[4] / (gamma - 1.0) + 0.5 * Pl[0] * (Pl[1] * Pl[1] + Pl[2] * Pl[2] + Pl[3] * Pl[3]);
    ur[4] = Pr[4] / (gamma - 1.0) + 0.5 * Pr[0] * (Pr[1] * Pr[1] + Pr[2] * Pr[2] + Pr[3] * Pr[3]);
    //
    fl[0] = ul[1];
    fl[1] = Pl[4] + Pl[0] * Pl[1] * Pl[1];
    fl[2] = Pl[0] * Pl[1] * Pl[2];
    fl[3] = Pl[0] * Pl[1] * Pl[3];
    fl[4] = (ul[4] + Pl[4]) * Pl[1];
    fr[0] = ur[1];
    fr[1] = Pr[4] + Pr[0] * Pr[1] * Pr[1];
    fr[2] = Pr[0] * Pr[1] * Pr[2];
    fr[3] = Pr[0] * Pr[1] * Pr[3];
    fr[4] = (ur[4] + Pr[4]) * Pr[1];

    for (int i = 0; i < 5; ++i) { f[i] = 0.5 * (fl[i] + fr[i] + lambda * (ul[i] - ur[i])); }
}
template <typename TM>
void HyperbolicConservationLaw<TM>::Advance(Real time_now, Real dt) {
    DEFINE_PHYSICAL_CONST

    rho -= diverge(fRho) * dt;
    rhoU -= diverge(fRhoU) * dt;
    rhoV -= diverge(fRhoV) * dt;
    rhoW -= diverge(fRhoW) * dt;
    rhoE -= diverge(fRhoE) * dt;
}

}  // namespace simpla{
#endif  // SIMPLA_HYPERBOLICCONSERVATIONLAW_H
