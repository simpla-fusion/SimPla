//
// Created by salmon on 16-5-21.
//

#ifndef SIMPLA_PML_H
#define SIMPLA_PML_H

#include "../../src/gtl/Log.h"
#include "../../src/gtl/primitives.h"
#include "../../src/physics/PhysicalConstants.h"
#include "../../src/field/Field.h"
#include "../../src/manifold/Calculus.h"
#include "../../src/simulation/Context.h"

namespace simpla
{
using namespace mesh;

/**
 *  @ingroup FieldSolver
 *  @brief absorb boundary condition, PML
 */
template<typename TM>
class PML : public ProblemDomain
{
    typedef ProblemDomain base_type;
    typedef TM mesh_type;
public:
    template<typename ValueType, size_t IFORM> using field_t =  Field<ValueType, TM, std::integral_constant<size_t, IFORM> >;;

    typedef typename mesh_type::scalar_type scalar_type;
    typedef typename mesh_type::point_type point_type;

    field_t<scalar_type, mesh::EDGE> E{*this, "E"};
    field_t<scalar_type, mesh::FACE> B{*this, "B"};


    field_t<scalar_type, mesh::EDGE> X10{m}, X11{m}, X12{m};
    field_t<scalar_type, mesh::FACE> X20{m}, X21{m}, X22{m};

    // alpha
    field_t<scalar_type, mesh::VERTEX> a0{*this, "PML_a0"};
    field_t<scalar_type, mesh::VERTEX> a1{*this, "PML_a1"};
    field_t<scalar_type, mesh::VERTEX> a2{*this, "PML_a2"};
    // sigma
    field_t<scalar_type, mesh::VERTEX> s0{*this, "PML_s0"};
    field_t<scalar_type, mesh::VERTEX> s1{*this, "PML_s1"};
    field_t<scalar_type, mesh::VERTEX> s2{*this, "PML_s2"};


public:

    PML(mesh_type const &mesh);

    virtual ~PML();

    virtual void next_step(Real dt);

    virtual void setup();

    virtual void tear_down();

    virtual io::IOStream &check_point(io::IOStream &) const;

private:


    inline Real sigma_(Real r, Real expN, Real dB)
    {
        return (0.5 * (expN + 2.0) * 0.1 * dB * std::pow(r, expN + 1.0));
    }

    inline Real alpha_(Real r, Real expN, Real dB)
    {
        return (1.0 + 2.0 * std::pow(r, expN));
    }


};

template<typename TM> PML<TM>::PML(mesh_type const &m) : base_type(m) { }

template<typename TM> PML<TM>::~PML() { }


template<typename TM> void
PML<TM>::setup()
{

    mesh::point_type xmin, xmax;

    std::tie(xmin, xmax) = m->box();

    LOGGER << "create PML solver [" << xmin << " , " << xmax << " ]";

    DEFINE_PHYSICAL_CONST

    Real dB = 100, expN = 2;

    a0 = 1.0;
    a1 = 1.0;
    a2 = 1.0;
    s0 = 0.0;
    s1 = 0.0;
    s2 = 0.0;
    X10 = 0.0;
    X11 = 0.0;
    X12 = 0.0;
    X20 = 0.0;
    X21 = 0.0;
    X22 = 0.0;

    point_type ymin, ymax;
    std::tie(ymin, ymax) = m->box();

    for (auto s : m->range(mesh::VERTEX))
    {
        point_type x = m->point(s);

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

template<typename TM> void
PML<TM>::tear_down() { }

template<typename TM> io::IOStream &
PML<TM>::check_point(io::IOStream &) const
{

}

template<typename TM>
void PML<TM>::next_step(Real dt)
{
    VERBOSE << "PML push E" << std::endl;

    DEFINE_PHYSICAL_CONST

    field_t<scalar_type, mesh::EDGE> dX1{*this};

    dX1 = (-2.0 * dt * s0 * X10 + curl_pdx(B) / (mu0 * epsilon0) * dt) / (a0 + s0 * dt);
    X10 += dX1;
    E += dX1;

    dX1 = (-2.0 * dt * s1 * X11 + curl_pdy(B) / (mu0 * epsilon0) * dt) / (a1 + s1 * dt);
    X11 += dX1;
    E += dX1;

    dX1 = (-2.0 * dt * s2 * X12 + curl_pdz(B) / (mu0 * epsilon0) * dt) / (a2 + s2 * dt);
    X12 += dX1;
    E += dX1;

    VERBOSE << "PML Push B" << std::endl;

    field_t<scalar_type, mesh::FACE> dX2{*this};

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
} //namespace simpla


#endif //SIMPLA_PML_H
