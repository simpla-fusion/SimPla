//
// Created by salmon on 16-5-21.
//

#ifndef SIMPLA_PML_H
#define SIMPLA_PML_H

#include "../../src/sp_def.h"
#include "../../src/gtl/Log.h"
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
class PML : public simulation::ProblemDomain
{
    typedef simulation::ProblemDomain base_type;
public:
    typedef TM mesh_type;

    template<typename ValueType, size_t IFORM> using field_t =  Field <ValueType, TM, index_const<IFORM>>;;

    PML(const mesh_type *mp);

    PML &setup_center_domain(box_type const &center_box);

    virtual ~PML();

    virtual void next_step(Real dt);

    virtual void deploy();

    virtual void sync(mesh::TransitionMap const &, simulation::ProblemDomain const &other);

    virtual std::string get_class_name() const { return class_name(); }

    static std::string class_name() { return "PML<" + traits::type_id<TM>::name() + ">"; }


    mesh_type const *m = nullptr;


    field_t<scalar_type, mesh::EDGE> E{m};
    field_t<scalar_type, mesh::FACE> B{m};


    field_t<scalar_type, mesh::EDGE> X10{m}, X11{m}, X12{m};
    field_t<scalar_type, mesh::FACE> X20{m}, X21{m}, X22{m};

    // alpha
    field_t<scalar_type, mesh::VERTEX> a0{m};
    field_t<scalar_type, mesh::VERTEX> a1{m};
    field_t<scalar_type, mesh::VERTEX> a2{m};
    // sigma
    field_t<scalar_type, mesh::VERTEX> s0{m};
    field_t<scalar_type, mesh::VERTEX> s1{m};
    field_t<scalar_type, mesh::VERTEX> s2{m};

    field_t<scalar_type, mesh::EDGE> dX1{m};
    field_t<scalar_type, mesh::FACE> dX2{m};


private:
    inline Real sigma_(Real r, Real expN, Real dB)
    {
        return static_cast<Real>((0.5 * (expN + 2.0) * 0.1 * dB * std::pow(r, expN + 1.0)));
    }

    inline Real alpha_(Real r, Real expN, Real dB)
    {
        return static_cast<Real>(1.0 + 2.0 * std::pow(r, expN));
    }
};


template<typename TM>
PML<TM>::~PML() { }

template<typename TM>
PML<TM>::PML(const mesh_type *mp) : base_type(mp), m(mp)
{
    assert(mp != nullptr);


}

template<typename TM> PML<TM> &
PML<TM>::setup_center_domain(box_type const &center_box)
{
    a0.clear();
    a1.clear();
    a2.clear();
    s0.clear();
    s1.clear();
    s2.clear();
    DEFINE_PHYSICAL_CONST
    Real dB = 100, expN = 2;
    point_type m_xmin, m_xmax;
    point_type c_xmin, c_xmax;

    std::tie(m_xmin, m_xmax) = m->box();
    std::tie(c_xmin, c_xmax) = center_box;
    auto dims = m->dimensions();

    m->range(mesh::VERTEX, SP_ES_ALL).foreach(
            [&](mesh::MeshEntityId s)
            {
                point_type x = m->point(s);

#define DEF(_N_)     a##_N_[s]=1;   s##_N_[s]=0;                                                     \
        if(dims[_N_]>1)                                                                              \
        {                                                                                            \
                if (x[_N_] <c_xmin[_N_])                                                             \
                {                                                                                    \
                    Real r = (c_xmin[_N_] - x[_N_]) / (m_xmax[_N_] - m_xmin[_N_]);                   \
                    a##_N_[s] = alpha_(r, expN, dB);                                                 \
                    s##_N_[s] = sigma_(r, expN, dB) * speed_of_light/ (m_xmax[_N_] - m_xmin[_N_]);  \
                }                                                                                    \
                else if (x[_N_] >c_xmax[_N_])                                                        \
                {                                                                                    \
                    Real r = (x[_N_] - c_xmax[_N_]) / (m_xmax[_N_] - m_xmin[_N_]);                   \
                    a##_N_[s] = alpha_(r, expN, dB);                                                 \
                    s##_N_[s] = sigma_(r, expN, dB) * speed_of_light/ (m_xmax[_N_] - m_xmin[_N_]);  \
                }                                                                                    \
        }
                DEF(0)
                DEF(1)
                DEF(2)
#undef DEF
            }
    );

    return *this;
}

template<typename TM>
void PML<TM>::deploy()
{
    X10.clear();
    X11.clear();
    X12.clear();
    X20.clear();
    X21.clear();
    X22.clear();
    E.clear();
    B.clear();
    dX1.clear();
    dX2.clear();
    declare_global(&E, "E");
    declare_global(&B, "B");
}


template<typename TM>
void PML<TM>::sync(mesh::TransitionMap const &t_map, simulation::ProblemDomain const &other)
{
    auto const &E2 = *static_cast<field_t<scalar_type, mesh::EDGE> const *>( other.attribute("E"));
    auto const &B2 = *static_cast<field_t<scalar_type, mesh::FACE> const *>( other.attribute("B"));


    t_map.direct_map(mesh::EDGE,
                     [&](mesh::MeshEntityId const &s1, mesh::MeshEntityId const &s2) { E[s1] = E2[s2]; });

    t_map.direct_map(mesh::FACE,
                     [&](mesh::MeshEntityId const &s1, mesh::MeshEntityId const &s2) { B[s1] = B2[s2]; });


}

template<typename TM>
void PML<TM>::next_step(Real dt)
{
    DEFINE_PHYSICAL_CONST

    dX2 = (X20 * (-2.0 * dt * s0) + curl_pdx(E) * dt) / (a0 + s0 * dt);
    X20 += dX2;
    B -= dX2;

    dX2 = (X21 * (-2.0 * dt * s0) + curl_pdy(E) * dt) / (a1 + s1 * dt);
    X21 += dX2;
    B -= dX2;

    dX2 = (X22 * (-2.0 * dt * s0) + curl_pdz(E) * dt) / (a2 + s2 * dt);
    X22 += dX2;
    B -= dX2;

    dX1 = (X10 * (-2.0 * dt * s0) + curl_pdx(B) / (mu0 * epsilon0) * dt) / (a0 + s0 * dt);
    X10 += dX1;
    E += dX1;

    dX1 = (X11 * (-2.0 * dt * s0) + curl_pdy(B) / (mu0 * epsilon0) * dt) / (a1 + s1 * dt);
    X11 += dX1;
    E += dX1;

    dX1 = (X12 * (-2.0 * dt * s0) + curl_pdz(B) / (mu0 * epsilon0) * dt) / (a2 + s2 * dt);
    X12 += dX1;
    E += dX1;
}
} //namespace simpla


#endif //SIMPLA_PML_H
