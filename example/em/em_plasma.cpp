/**
 * @file em_plasma.cpp
 * @author salmon
 * @date 2015-11-20.
 */

#include "em_plasma.h"

#include "../../core/gtl/utilities/utilities.h"
#include "../../core/parallel/parallel.h"
#include "../../core/io/io.h"

#include "../../core/manifold/pre_define/predefine.h"
#include "../../core/field/field.h"

#include "../../core/particle/particle.h"
#include "../../core/particle/particle_proxy.h"
#include "../../core/particle/particle_engine.h"
#include "../../core/dataset/datatype_ext.h"
#include "../../core/particle/particle_generator.h"

namespace simpla
{
struct PICDemo
{
    SP_DEFINE_STRUCT(point_type,
                     Vec3, x,
                     Vec3, v);

    SP_DEFINE_STRUCT(sample_type,
                     point_type, z,
                     Real, f,
                     Real, w
    );

    SP_DEFINE_PROPERTIES(
            Real, mass,
            Real, charge,
            Real, temperature
    )

    void update() { }

    Vec3 project(sample_type const &p) const { return p.z.x; }

    Real function_value(sample_type const &p) const { return p.f * p.w; }


    point_type lift(Vec3 const &x, Vec3 const &v) const
    {
        return point_type{x, v};
    }

    point_type lift(std::tuple<Vec3, Vec3> const &z) const
    {
        return point_type{std::get<0>(z), std::get<1>(z)};
    }

    sample_type sample(Vec3 const &x, Vec3 const &v, Real f) const
    {
        return sample_type{lift(x, v), f, 0};
    }

    sample_type sample(point_type const &z, Real f) const
    {
        return sample_type{z, f, 0};
    }

    template<typename TE, typename TB>
    void push(sample_type *p, Real dt, Real t, TE const &E, TB const &B)
    {
        p->z.x += p->z.v * dt * 0.5;
        p->z.v += E(p->z.x) * dt;
        p->z.x += p->z.v * dt * 0.5;
    };

};

struct EMPlasma
{
    EMPlasma() { }

    ~EMPlasma() { }

    std::string description() { return ""; }

    virtual bool check_type(std::type_info const &info) { return info == typeid(EMPlasma); }

    void setup(int argc, char **argv);

    void next_time_step();

    void tear_down();

    void check_point();


    typedef manifold::CartesianManifold mesh_type;

    typedef Real scalar_type;

    typedef nTuple<scalar_type, 3> vector_type;

    mesh_type m;

    traits::field_t<vector_type, mesh_type, VERTEX> Bv{m};
    traits::field_t<vector_type, mesh_type, VERTEX> B0v{m};
    traits::field_t<vector_type, mesh_type, VERTEX> Ev{m};

    traits::field_t<scalar_type, mesh_type, FACE> B0{m};
    traits::field_t<scalar_type, mesh_type, FACE> B1{m};
    traits::field_t<scalar_type, mesh_type, EDGE> E1{m};
    traits::field_t<scalar_type, mesh_type, EDGE> pdE{m};

    typedef traits::field_t<scalar_type, mesh_type, FACE> TB;
    typedef traits::field_t<scalar_type, mesh_type, EDGE> TE;
    typedef traits::field_t<scalar_type, mesh_type, EDGE> TJ;
    typedef traits::field_t<scalar_type, mesh_type, VERTEX> TRho;


    TJ J1{m};

    Particle<PICDemo, mesh_type> ion{m};

    struct particle_s
    {
        std::shared_ptr<ParticleProxyBase<TB, TE, TJ, TRho>> p;
        Real mass;
        Real charge;
        TJ J1;
        TRho rho1;
        traits::field_t<scalar_type, mesh_type, VERTEX> rhos;
        traits::field_t<vector_type, mesh_type, VERTEX> Js;
    };
    std::map<std::string, particle_s> particles;
};

void EMPlasma::setup(int argc, char **argv)
{

    ConfigParser options;

    options.init(argc, argv);

    m.load(options);

    m.deploy();


    MESSAGE << std::endl << "[ Configuration ]" << std::endl << m << std::endl;

    try
    {
        Bv.clear();
        B0v.clear();
        Ev.clear();

        B0.clear();
        B1.clear();
        E1.clear();
        pdE.clear();

        J1.clear();

        B0v = map_to<VERTEX>(B0);

    } catch (std::exception const &error)
    {
        THROW_EXCEPTION_RUNTIME_ERROR("Field init error", error.what());
    }

    ParticleGenerator<PICDemo> gen(ion);

    ion.generator(gen, 100, 1.0);
}

void EMPlasma::tear_down()
{
    io::cd("/dump/");
    LOGGER << SAVE(ion) << std::endl;
}


void EMPlasma::check_point()
{
    io::cd("/record/");

    LOGGER << SAVE_RECORD(Bv) << std::endl;
    LOGGER << SAVE_RECORD(B0v) << std::endl;
    LOGGER << SAVE_RECORD(Ev) << std::endl;

    LOGGER << SAVE_RECORD(B0) << std::endl;
    LOGGER << SAVE_RECORD(B1) << std::endl;
    LOGGER << SAVE_RECORD(E1) << std::endl;

}

void EMPlasma::next_time_step()
{
    VERBOSE << "Push one step" << std::endl;

    DEFINE_PHYSICAL_CONST

    Real dt = m.dt();

    ion.push(dt, 0, E1, B1);

    ion.integral(&J1);

    B1 -= curl(E1) * (dt * 0.5);
    E1 += curl(B1) * dt;
    B1 -= curl(E1) * (dt * 0.5);


    Ev = map_to<VERTEX>(E1);
    Bv = map_to<VERTEX>(B1);

    traits::field_t<scalar_type, mesh_type, VERTEX> BB{m};
    BB = dot(B0, B0);

    traits::field_t<vector_type, mesh_type, VERTEX> Q{m};
    traits::field_t<vector_type, mesh_type, VERTEX> K{m};

    Q = map_to<VERTEX>(pdE);

    traits::field_t<scalar_type, mesh_type, VERTEX> a{m};
    traits::field_t<scalar_type, mesh_type, VERTEX> b{m};
    traits::field_t<scalar_type, mesh_type, VERTEX> c{m};

    a.clear();
    b.clear();
    c.clear();

    for (auto &p :   particles)
    {

        p.second.p->integral(&p.second.J1);

        auto &rhos = p.second.rhos;
        auto &Js = p.second.Js;

        Real ms = p.second.mass;
        Real qs = p.second.charge;

        Real as = (dt * qs) / (2.0 * ms);

        K = (Ev * rhos * 2.0 + cross(Js, B0v)) * as + Js;

        Q -= 0.5 * dt / epsilon0
             * (K + cross(K, B0v) * as + B0v * (dot(K, B0v) * as * as) / (BB * as * as + 1)) + Js;

        a += rhos / BB * (as / (as * as + 1));
        b += rhos / BB * (as * as / (as * as + 1));
        c += rhos / BB * (as * as * as / (as * as + 1));

    }

    a *= 0.5 * dt / epsilon0;
    b *= 0.5 * dt / epsilon0;
    c *= 0.5 * dt / epsilon0;
    a += 1;

    auto dEv = traits::make_field<vector_type, VERTEX>(m);

    dEv = (Q * a - cross(Q, B0v) * b +
           B0v * (dot(Q, B0v) * (b * b - c * a) / (a + c * BB))) / (b * b * BB + a * a);

    Ev += dEv;

    pdE = map_to<EDGE>(dEv);


}

}

// namespace simpla
int main(int argc, char **argv)
{
    using namespace simpla;

    ConfigParser options;

    try
    {

        logger::init(argc, argv);

        parallel::init(argc, argv);

        io::init(argc, argv);

        options.init(argc, argv);
    }
    catch (std::exception const &error)
    {
        THROW_EXCEPTION_RUNTIME_ERROR("Initial error", error.what());
    }

    INFORM << ShowCopyRight() << std::endl;

    if (options["V"] || options["version"])
    {
        MESSAGE << "SIMPla " << ShowVersion();
        TheEnd(0);
        return TERMINATE;
    }
    else if (options["h"] || options["help"])
    {

        MESSAGE << " Usage: " << argv[0] << "   <options> ..." << std::endl << std::endl;

        MESSAGE << " Options:" << std::endl

        << "\t -h,\t--help            \t, Print a usage message and exit.\n"

        << "\t -v,\t--version         \t, Print version information exit. \n"

        << std::endl;


        TheEnd(0);

    }


    simpla::EMPlasma ctx;

    int num_of_step = options["number_of_step"].as<int>(2);

    int check_point = options["check_point"].as<int>(1);

    ctx.setup(argc, argv);

    int count = 0;

    ctx.check_point();

    MESSAGE << "====================================================" << std::endl;
    INFORM << "\t >>> START <<< " << std::endl;

    while (count < num_of_step)
    {
        ctx.next_time_step();

        if (count % check_point == 0)
            ctx.check_point();

        ++count;
    }
    ctx.tear_down();

    INFORM << "\t >>> Done <<< " << std::endl;
    MESSAGE << "====================================================" << std::endl;


    io::close();
    parallel::close();
    logger::close();

    return 0;
}

