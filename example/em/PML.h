//
// Created by salmon on 16-5-21.
//

#ifndef SIMPLA_PML_H
#define SIMPLA_PML_H

#include "../../src/gtl/Log.h"
#include "../../src/sp_def.h"
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
    typedef TM mesh_type;
public:
    template<typename ValueType, size_t IFORM> using field_t =  Field<ValueType, TM, std::integral_constant<size_t, IFORM> >;;


    mesh_type const *m = nullptr;


    PML(const mesh_type *mp) : base_type(mp), m(mp) { }

    virtual ~PML() { }

    virtual void next_step(Real dt);

    virtual void set_direction(int const *od);

    virtual void setup(ConfigParser const &options);

    virtual void tear_down();

    virtual io::IOStream &check_point(io::IOStream &) const;

    virtual io::IOStream &save(io::IOStream &os) const { return os; };

    virtual io::IOStream &load(io::IOStream &is) const { return is; };

    virtual void sync(mesh::TransitionMap const &, simulation::ProblemDomain const &other) { };

    virtual std::string get_class_name() const { return class_name(); }

    static std::string class_name() { return "PML<" + traits::type_id<TM>::name() + ">"; }

    field_t<scalar_type, mesh::EDGE> E{*this, "E"};
    field_t<scalar_type, mesh::FACE> B{*this, "B"};


    field_t<scalar_type, mesh::EDGE> X10{*this}, X11{*this}, X12{*this};
    field_t<scalar_type, mesh::FACE> X20{*this}, X21{*this}, X22{*this};

    // alpha
    field_t<scalar_type, mesh::VERTEX> a0{*this};
    field_t<scalar_type, mesh::VERTEX> a1{*this};
    field_t<scalar_type, mesh::VERTEX> a2{*this};
    // sigma
    field_t<scalar_type, mesh::VERTEX> s0{*this};
    field_t<scalar_type, mesh::VERTEX> s1{*this};
    field_t<scalar_type, mesh::VERTEX> s2{*this};

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
void PML<TM>::setup(ConfigParser const &options)
{
    base_type::setup(options);

    E.clear();
    B.clear();


    X10.clear();
    X11.clear();
    X12.clear();
    X20.clear();
    X21.clear();
    X22.clear();


    a0.clear();
    a1.clear();
    a2.clear();
    s0.clear();
    s1.clear();
    s2.clear();
}

template<typename TM> void
PML<TM>::set_direction(int const *od)
{
    point_type xmin, xmax;

    std::tie(xmin, xmax) = m->box();

    LOGGER << "create PML solver [" << xmin << " , " << xmax << " ]" << std::endl;

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

    m->range(mesh::VERTEX).foreach(
            [&](mesh::MeshEntityId s)
            {
                point_type x = m->point(s);

#define DEF(_N_)                                                                    \
        if (od[_N_] ==-1)                                                         \
        {                                                                           \
            Real r = (xmin[_N_] - x[_N_]) / (xmin[_N_] - ymin[_N_]);                        \
            a##_N_[s] = alpha_(r, expN, dB);                                            \
            s##_N_[s] = sigma_(r, expN, dB) * speed_of_light / (xmin[_N_] - ymin[_N_]);     \
        }                                                                           \
        else if (od[_N_] ==1)                                                    \
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
    );


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
    VERBOSE << "Mesh Block[" << m->short_id() << "] PML push E" << std::endl;

    DEFINE_PHYSICAL_CONST

    field_t<scalar_type, mesh::EDGE> dX1{*this};
    dX1.clear();

    dX1 = (-2.0 * dt * s0 * X10 + curl_pdx(B) / (mu0 * epsilon0) * dt) / (a0 + s0 * dt);
    X10 += dX1;
    E += dX1;

    dX1 = (-2.0 * dt * s1 * X11 + curl_pdy(B) / (mu0 * epsilon0) * dt) / (a1 + s1 * dt);
    X11 += dX1;
    E += dX1;

    dX1 = (-2.0 * dt * s2 * X12 + curl_pdz(B) / (mu0 * epsilon0) * dt) / (a2 + s2 * dt);
    X12 += dX1;
    E += dX1;

    VERBOSE << "Mesh Block[" << m->short_id() << "] PML Push B" << std::endl;

    field_t<scalar_type, mesh::FACE> dX2{*this};
    dX2.clear();

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
