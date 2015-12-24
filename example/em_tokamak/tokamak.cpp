/**
 * @file em_plasma.cpp
 * @author salmon
 * @date 2015-11-20.
 *
 * @example  em/em_plasma.cpp
 *    This is an example of EM plasma
 */

#include "tokamak.h"

#include "../../core/gtl/utilities/utilities.h"
#include "../../core/parallel/Parallel.h"
#include "../../core/io/IO.h"

#include "../../core/manifold/pre_define/PreDefine.h"
#include "../../core/particle/pre_define/PICBoris.h"

#include "../../core/model/GEqdsk.h"
#include "../../core/model/Constraint.h"
#include "../../core/io/XDMFStream.h"

namespace simpla
{


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

    typedef Real scalar_type;

    typedef manifold::CylindricalManifold mesh_type;

    typedef typename mesh_type::id_type id_type;
    typedef typename mesh_type::point_type point_type;
    typedef typename mesh_type::box_type box_type;
    typedef typename mesh_type::range_type range_type;
    typedef nTuple<scalar_type, 3> vector_type;

    mesh_type m;

    io::XDMFStream out_stream;

    model::Surface<mesh_type> limiter_boundary;
    model::IdSet<mesh_type> vertex_boundary;
    model::IdSet<mesh_type> edge_boundary;
    model::IdSet<mesh_type> face_boundary;

    model::IdSet<mesh_type> plasma_region_volume;

    model::IdSet<mesh_type> plasma_region_vertex;

    model::IdSet<mesh_type> J_src;

    std::function<Vec3(Real, point_type const &)> J_src_fun;

    traits::field_t<scalar_type, mesh_type, EDGE> E0{m, "E0"};
    traits::field_t<scalar_type, mesh_type, FACE> B0{m, "B0"};
    traits::field_t<vector_type, mesh_type, VERTEX> B0v{m, "B0v"};
    traits::field_t<scalar_type, mesh_type, VERTEX> BB{m, "BB"};

    traits::field_t<vector_type, mesh_type, VERTEX> Ev{m, "Ev"};
    traits::field_t<vector_type, mesh_type, VERTEX> Bv{m, "Bv"};

    traits::field_t<scalar_type, mesh_type, FACE> B1{m, "B1"};
    traits::field_t<scalar_type, mesh_type, EDGE> E1{m, "E1"};
    traits::field_t<scalar_type, mesh_type, EDGE> J1{m, "J1"};

    traits::field_t<scalar_type, mesh_type, VERTEX> rho0{m};

    typedef traits::field_t<scalar_type, mesh_type, FACE> TB;
    typedef traits::field_t<scalar_type, mesh_type, EDGE> TE;
    typedef traits::field_t<scalar_type, mesh_type, EDGE> TJ;
    typedef traits::field_t<scalar_type, mesh_type, VERTEX> TRho;
    typedef traits::field_t<vector_type, mesh_type, VERTEX> TJv;


    typedef particle::ParticleProxyBase<TE, TB, TJv, TRho> particle_proxy_type;

    struct particle_s
    {
        Real mass;
        Real charge;
        traits::field_t<scalar_type, mesh_type, VERTEX> rho1;
        traits::field_t<vector_type, mesh_type, VERTEX> J1;
        std::shared_ptr<particle_proxy_type> f;
        std::shared_ptr<particle_proxy_type> test;

    };

    std::map<std::string, particle_s> particles;


    std::pair<typename std::map<std::string, particle_s>::iterator, bool>
    add_particle(std::string const &name, Real mass,
                 Real charge)
    {
        return particles.emplace(
                std::make_pair(
                        name,
                        particle_s{
                                mass, charge,
                                traits::field_t<scalar_type, mesh_type, VERTEX>(m, "n_" + name),
                                traits::field_t<vector_type, mesh_type, VERTEX>(m, "J_" + name)
                        }));

    }

    template<typename TP, typename TDict>
    std::shared_ptr<particle_proxy_type>
    create_particle(std::string const &key, TDict const &dict)
    {
        VERBOSE << "Create particle [" << key << "]" << std::endl;

        TP pic(m, key);

        dict.as(&pic.properties());

        pic.deploy();

        auto gen = particle::make_generator(pic.engine(), 1.0);

        pic.generator(plasma_region_volume, gen, pic.properties()["PIC"].template as<size_t>(10),
                      pic.properties()["T"].template as<Real>(1));


        return particle_proxy_type::create(pic.data());

    }

    size_t m_count = 0;

    bool disable_particle = false;

    bool disable_field = false;
};

void EMPlasma::setup(int argc, char **argv)
{

    ConfigParser options;


    options.init(argc, argv);

    disable_particle = options["DisableParticle"].template as<bool>(false);
    disable_field = options["DisableField"].template as<bool>(false);

    options["Mesh"].as(&m.properties());

    nTuple<Real, 2, 3> box;

    Real phi0 = 0, phi1 = TWOPI;

    if (m.properties()["Geometry"]["Box"].as(&box))
    {
        phi0 = box[0][2];
        phi1 = box[1][2];
    }


    GEqdsk geqdsk;

    geqdsk.load(options["GEQDSK"].as<std::string>(""));

    box = geqdsk.limiter().box();


    box[0][2] = phi0;
    box[1][2] = phi1;


    m.box(box);

    m.deploy();

    VERBOSE << "Clear fields" << std::endl;


//        if (m_geo_.topology_type() == "CoRectMesh")
//        {
//            int ndims = m_geo_.ndims;
//
//            nTuple<size_t, 3> dims;
//
//            dims = m_geo_.dimensions();
//
//            nTuple<Real, 3> xmin, dx;
//
//            std::tie(xmin, std::ignore) = m_geo_.box();
//
//            dx = m_geo_.dx();
//
//            base_type::set_grid(ndims, &dims[0], &xmin[0], &dx[0]);
//        }
//
//        else if (m_geo_.topology_type() == "SMesh")
//        {
//
//        }
    out_stream.open(options["output"].as<std::string>("tokamak"), "GEqdsk");

    out_stream.set_topology_geometry("Main", m.grid_vertices());

    out_stream.open_grid("back_ground", io::XDMFStream::UNIFORM);


    E0.clear();
    Ev.clear();
    B1.clear();
    E1.clear();
    J1.clear();

    B0.clear();

    parallel::parallel_for(
            m.template range<FACE>(),
            [&](range_type const &r)
            {
                for (auto const &s:r)
                {
                    B0.assign(s, geqdsk.B(m.point(s)));

                }
            }
    );

    B0.sync();

    out_stream.write("B0", *B0.data());

    rho0.clear();

    auto const &boundary = geqdsk.boundary();
    parallel::parallel_for(
            m.template range<VERTEX>(),
            [&](range_type const &r)
            {
                for (auto const &s:r)
                {
                    auto x = m.point(s);

                    if (boundary.within(x))
                    {
                        rho0.assign(s, geqdsk.profile("ne", x));
                    }

                }
            }
    );

    rho0.sync();

    out_stream.write("rho0", *rho0.data());


    B0v = map_to<VERTEX>(B0);

    BB = dot(B0v, B0v);

    out_stream.write("B0v", *B0v.data());
    out_stream.write("BB", *BB.data());


    {
        model::Cache<mesh_type> cache;

        model::update_cache(m, geqdsk.limiter(), &cache);

        model::get_cell_on_surface<EDGE>(m, cache, &edge_boundary);

        model::get_cell_on_surface<FACE>(m, cache, &face_boundary);

        model::get_cell_on_surface<VERTEX>(m, cache, &vertex_boundary);

        model::get_cell_on_surface(m, cache, &limiter_boundary);


    }

    {
        model::Cache<mesh_type> cache;

        model::update_cache(m, geqdsk.boundary(), &cache);

        model::get_cell_in_surface<VOLUME>(m, cache, &plasma_region_volume);

        model::get_cell_in_surface<VERTEX>(m, cache, &plasma_region_vertex);

    }

    {

        auto dict = options["Constraints"]["J"];

        if (dict)
        {
            model::create_id_set(
                    m, m.template make_range<EDGE>(
                            m.index_box(dict["Box"].template as<box_type>())),
                    &J_src);

            dict["Value"].as(&J_src_fun);
        }

    }

    {
        DEFINE_PHYSICAL_CONST;

        auto ps = options["Particles"];

        for (auto const &dict:ps)
        {

            std::string key = dict.first.template as<std::string>();

            auto &p = particles[key];

            p.mass = dict.second["mass"].template as<Real>();

            p.charge = dict.second["charge"].template as<Real>();


            traits::field_t<scalar_type, mesh_type, VERTEX>(m, "n_" + key).swap(p.rho1);

            traits::field_t<vector_type, mesh_type, VERTEX>(m, "J_" + key).swap(p.J1);

            if (dict.second["Density"])
            {
                p.rho1 = traits::make_field_function_from_config<scalar_type, VERTEX>(m, dict.second["Density"]);
            }
            else { p.rho1 = rho0; }

            if (dict.second["Type"].template as<std::string>() == "Boris")
            {
                p.f = create_particle<particle::BorisParticle<mesh_type>>(key, dict.second);
                if (dict.second["EnableTracking"])
                {
                    p.test = create_particle<particle::BorisTrackingParticle<mesh_type>>(key, dict.second);
                }

            }

        }
    }

    BB.data()->properties()["DisableCheckPoint"] = true;
    B0v.data()->properties()["DisableCheckPoint"] = true;
    E0.data()->properties()["DisableCheckPoint"] = true;
    B0.data()->properties()["DisableCheckPoint"] = true;

    MESSAGE << std::endl << "[ Configuration ]" << std::endl << m << std::endl;

    MESSAGE << "Particles = {" << std::endl;
    for (auto const &item:particles)
    {
        MESSAGE << "  " << item.first << " =  ";

        if ((item.second.f == nullptr))
        {
            MESSAGE << "{"
            << " mass =" << item.second.mass << " , "
            << " charge = " << item.second.charge << " , "
            << " type =   \"Fluid\" " << "}";
        }
        else
        {
            MESSAGE << *item.second.f;
        }


        MESSAGE << "," << std::endl;
    }
    MESSAGE << "}" << std::endl;


    Ev = map_to<VERTEX>(E1);


    out_stream.close_grid();
    out_stream.open_grid("record", io::XDMFStream::COLLECTION_TEMPORAL);

}

void EMPlasma::tear_down()
{
    out_stream.close_grid();

    out_stream.open_grid("dump", io::XDMFStream::UNIFORM);

    out_stream.reference_topology_geometry("Main");

    out_stream.time(m.time());

    for (auto const &item:m.attributes())
    {
        if (!item.second.lock()->properties()["DisableXDMFOutput"])
        {
            out_stream.write(item.first, *std::dynamic_pointer_cast<base::AttributeObject>(item.second.lock()));

        } else
        {
            out_stream.hdf5().write(item.first, item.second.lock()->dump(), io::SP_RECORD);
        }
    }
    out_stream.close_grid();

    out_stream.close();
}


void EMPlasma::check_point()
{
    if (!disable_field)
    {

        out_stream.open_grid(type_cast<std::string>(m_count), io::XDMFStream::UNIFORM);

        out_stream.reference_topology_geometry("Main");

        out_stream.time(m.time());

        for (auto const &item:m.attributes())
        {
            auto attr = item.second.lock();
            if (!attr->properties()["DisableCheckPoint"])
            {
                if (!attr->properties()["IsParticle"])
                {
                    out_stream.write(item.first, *std::dynamic_pointer_cast<base::AttributeObject>(attr));

                }

            }
        }

        out_stream.close_grid();
    }
    if (!disable_particle)
    {


        for (auto const &item:m.attributes())
        {
            auto attr = item.second.lock();
            if (!attr->properties()["DisableCheckPoint"])
            {
                if (attr->properties()["IsParticle"])
                {
                    out_stream.hdf5().write(item.first, attr->checkpoint(), io::SP_RECORD);
                }

            }
        }
    }
    m.next_time_step();

    ++m_count;

}

void EMPlasma::next_time_step()
{
    VERBOSE << "Push one step" << std::endl;

    DEFINE_PHYSICAL_CONST

    Real dt = m.dt();

    Real t = m.time();
    if (!disable_particle)
    {
        for (auto &p:particles)
        {

            if (p.second.f != nullptr)
            {
                p.second.rho1.clear();
                p.second.J1.clear();

                p.second.f->push(dt, m.time(), E0, B0);
//                p.second.f->integral(&p.second.J1);
            }

            if (p.second.test != nullptr)
            {
                p.second.test->push(dt, m.time(), E0, B0);
            }
        }
    }
    if (!disable_field)
    {


        J1.clear();

        LOG_CMD(B1 -= curl(E1) * (dt * 0.5));

        B1.accept(face_boundary.range(), [&](id_type, Real &v) { v = 0; });

        J1.accept(J_src.range(), [&](id_type s, Real &v) { J1.add(s, J_src_fun(t, m.point(s))); });

        LOG_CMD(E1 += (curl(B1) * speed_of_light2 - J1 / epsilon0) * dt);

        E1.accept(edge_boundary.range(), [&](id_type, Real &v) { v = 0; });


        traits::field_t<vector_type, mesh_type, VERTEX> dE{m};



        //particle::absorb(ion, limiter_boundary);
        if (particles.size() > 0)
        {

            traits::field_t<vector_type, mesh_type, VERTEX> Q{m};
            traits::field_t<vector_type, mesh_type, VERTEX> K{m};

            traits::field_t<scalar_type, mesh_type, VERTEX> a{m};
            traits::field_t<scalar_type, mesh_type, VERTEX> b{m};
            traits::field_t<scalar_type, mesh_type, VERTEX> c{m};

            a.clear();
            b.clear();
            c.clear();

            Q = map_to<VERTEX>(E1) - Ev;


            for (auto &p :   particles)
            {

                Real ms = p.second.mass;
                Real qs = p.second.charge;


                traits::field_t<scalar_type, mesh_type, VERTEX> &ns = p.second.rho1;

                traits::field_t<vector_type, mesh_type, VERTEX> &Js = p.second.J1;;


                Real as = (dt * qs) / (2.0 * ms);

                Q -= 0.5 * dt / epsilon0 * Js;

                K = (Ev * qs * ns * 2.0 + cross(Js, B0v)) * as + Js;

                Js = (K + cross(K, B0v) * as + B0v * (dot(K, B0v) * as * as)) / (BB * as * as + 1);

                Q -= 0.5 * dt / epsilon0 * Js;

                a += qs * ns * (as / (BB * as * as + 1));
                b += qs * ns * (as * as / (BB * as * as + 1));
                c += qs * ns * (as * as * as / (BB * as * as + 1));


            }

            a *= 0.5 * dt / epsilon0;
            b *= 0.5 * dt / epsilon0;
            c *= 0.5 * dt / epsilon0;
            a += 1;


            LOG_CMD(dE = (Q * a - cross(Q, B0v) * b + B0v * (dot(Q, B0v) * (b * b - c * a) / (a + c * BB))) /
                         (b * b * BB + a * a));

            for (auto &p :   particles)
            {
                Real ms = p.second.mass;
                Real qs = p.second.charge;
                traits::field_t<scalar_type, mesh_type, VERTEX> &ns = p.second.rho1;
                traits::field_t<vector_type, mesh_type, VERTEX> &Js = p.second.J1;;


                Real as = (dt * qs) / (2.0 * ms);

                K = dE * ns * qs * as;
                Js += (K + cross(K, B0v) * as + B0v * (dot(K, B0v) * as * as)) / (BB * as * as + 1);
            }
            Ev += dE;

            LOG_CMD(E1 += map_to<EDGE>(Ev) - E1);
        }


        LOG_CMD(B1 -= curl(E1) * (dt * 0.5));

        B1.accept(face_boundary.range(), [&](id_type const &, Real &v) { v = 0; });
    }
    m.next_time_step();
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


        options.init(argc, argv);
    }
    catch (std::exception const &error)
    {
        RUNTIME_ERROR << "Initial error" << error.what() << std::endl;
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


    int num_of_steps = options["number_of_steps"].as<int>(20);

    int check_point = options["check_point"].as<int>(1);

    try
    {
        ctx.setup(argc, argv);
    }
    catch (std::exception const &error)
    {
        RUNTIME_ERROR << "Context setup error!" << error.what() << std::endl;
    }


    int count = 0;

    ctx.check_point();

    MESSAGE << "====================================================" << std::endl;
    INFORM << "\t >>> START <<< " << std::endl;

    while (count < num_of_steps)
    {
        ctx.next_time_step();

        if (count % check_point == 0)
            ctx.check_point();

        ++count;
    }
    ctx.tear_down();

    INFORM << "\t >>> Done <<< " << std::endl;
    MESSAGE << "====================================================" << std::endl;


    parallel::close();

    logger::close();

    return 0;
}
