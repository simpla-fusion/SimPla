//
// Created by salmon on 16-5-21.
//

#ifndef SIMPLA_PML_H
#define SIMPLA_PML_H

#include "../gtl/Log.h"
#include "../gtl/primitives.h"
#include "../physics/PhysicalConstants.h"
#include "../field/Field.h"
#include "../manifold/Calculus.h"
#include "../task_flow/Context.h"

namespace simpla { namespace phy_solver
{
using namespace field;

/**
 *  @ingroup FieldSolver
 *  @brief absorb boundary condition, PML
 */
template<typename TM>
class PML : public task_flow::Context::Worker
{
    typedef TM mesh_type;
    typedef typename mesh_type::scalar_type scalar_type;
    typedef typename mesh_type::point_type point_type;


    mesh_type const &m_mesh_;
    Field<scalar_type, mesh_type, mesh::EDGE> E{*this, "E"};
    Field<scalar_type, mesh_type, mesh::FACE> B{*this, "B"};


    Field<scalar_type, mesh_type, mesh::EDGE> X10, X11, X12;
    Field<scalar_type, mesh_type, mesh::FACE> X20, X21, X22;

    // alpha
    Field<scalar_type, mesh_type, mesh::VERTEX> a0{*this, "PML_a0"};
    Field<scalar_type, mesh_type, mesh::VERTEX> a1{*this, "PML_a1"};
    Field<scalar_type, mesh_type, mesh::VERTEX> a2{*this, "PML_a2"};
    // sigma
    Field<scalar_type, mesh_type, mesh::VERTEX> s0{*this, "PML_s0"};
    Field<scalar_type, mesh_type, mesh::VERTEX> s1{*this, "PML_s1"};
    Field<scalar_type, mesh_type, mesh::VERTEX> s2{*this, "PML_s2"};

public:
    PML(task_flow::Context &ctx);

    virtual ~PML();

    virtual void view(mesh::uuid id);

    virtual void work(Real dt);

private:


    inline Real sigma_(Real r, Real expN, Real dB)
    {
        return (0.5 * (expN + 2.0) * 0.1 * dB * std::pow(r, expN + 1.0));
    }

    inline Real alpha_(Real r, Real expN, Real dB)
    {
        return (1.0 + 2.0 * std::pow(r, expN));
    }

    void init();

};

template<typename TM>
PML<TM>::PML(task_flow::Context &ctx)
{
    init();
}

template<typename TM>
PML<TM>::~PML()
{
}

template<typename TM>
void PML<TM>::view(mesh::uuid id)
{
    if (a0.empty()) { init() };
}

template<typename TM>
void PML<TM>::init()
{

    mesh::point_type xmin, xmax;

    std::tie(xmin, xmax) = m_mesh_.box();

    LOGGER << "create PML solver [" << xmin << " , " << xmax << " ]";

    DEFINE_PHYSICAL_CONST

    Real dB = 100, expN = 2;

    a0.fill(1.0);
    a1.fill(1.0);
    a2.fill(1.0);
    s0.fill(0.0);
    s1.fill(0.0);
    s2.fill(0.0);
    X10.fill(0.0);
    X11.fill(0.0);
    X12.fill(0.0);
    X20.fill(0.0);
    X21.fill(0.0);
    X22.fill(0.0);

    point_type ymin, ymax;
    std::tie(ymin, ymax) = m_mesh_->extents();

    for (auto s : m_mesh_->template domain<mesh::VERTEX>())
    {
        point_type x = m_mesh_->coordinates(s);

#define DEF(_N_)                                                                    \
        if (x[_N_] < xmin[_N_])                                                         \
        {                                                                           \
            Real r = (xmin[_N_] - x[_N_]) / (xmin[_N_] - ymin[_N_]);                        \
            a##_N_[s] = alpha_(r, expN, dB);                                            \
            s##_N_[s] = sigma_(r, expN, dB) * speed_of_light / (xmin[_N_] - ymin[_N_]);     \
        }                                                                           \
        else if (x[_N_] > xmax[_N_])                                                    \
        {                                                                           \
            Real r = (x[_N_] - xmax[_N_]) / (ymax[_N_] - xmax[_N_]);                        \
            a##_N_[s] = alpha_(r, expN, dB);                                            \
            s##_N_[s] = sigma_(r, expN, dB) * speed_of_light / (ymax[_N_] - xmax[_N_]);     \
        };

        DEF(0)
        DEF(1)
        DEF(2)
#undef DEF
    }


    LOGGER << DONE;

}

template<typename TM>
void PML<TM>::work(Real dt)
{
    VERBOSE << "PML push E" << std::endl;

    DEFINE_PHYSICAL_CONST

    field <scalar_type, mesh_type, mesh::EDGE> dX1{*this};

    dX1 = (-2.0 * dt * s0 * X10 + curl_pdx(B) / (mu0 * epsilon0) * dt)
          / (a0 + s0 * dt);
    X10 += dX1;
    E += dX1;

    dX1 = (-2.0 * dt * s1 * X11 + curl_pdy(B) / (mu0 * epsilon0) * dt)
          / (a1 + s1 * dt);
    X11 += dX1;
    E += dX1;

    dX1 = (-2.0 * dt * s2 * X12 + curl_pdz(B) / (mu0 * epsilon0) * dt)
          / (a2 + s2 * dt);
    X12 += dX1;
    E += dX1;

    VERBOSE << "PML Push B" << std::endl;

    field <scalar_type, mesh_type, mesh::FACE> dX2{*this};

    dX2 = (-2.0 * dt * s0 * X20 + curl_pdx(E) * dt) / (a0 + s0 * dt);
    X20 += dX2;
    B -= dX2;

    dX2 = (-2.0 * dt * s1 * X21 + curl_pdy(E) * dt) / (a1 + s1 * dt);
    X21 += dX2;
    B -= dX2;

    dX2 = (-2.0 * dt * s2 * X22 + curl_pdz(E) * dt) / (a2 + s2 * dt);
    X22 += dX2;
    B -= dX2;

}
}} //namespace simpla { namespace solver


#endif //SIMPLA_PML_H
