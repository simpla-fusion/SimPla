/*
 * explicit_em_impl.h
 *
 * @date  2013-12-29
 *      @author  salmon
 */

#ifndef EXPLICIT_EM_IMPL_H_
#define EXPLICIT_EM_IMPL_H_

#include <cmath>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <utility>

// Misc
#include "../../core/application/context_base.h"
#include "../../core/diff_geometry/fetl.h"
#include "utilities.h"

// Data IO
#include "../../core/io/io.h"

// Field
#include "../../core/field/loadField.h"
#include "geometry_obj"
#include "../../core/geometry/model.h"

// Particle
//#include "../../core/particle/particle.h"

// Model
#include "../../core/numeric/geometric_algorithm.h"
//#include "../../experimental/particle_factory.h"

// Solver
#include "../field_solver/pml.h"
#include "../field_solver/implicitPushE.h"
#include "../particle_solver/register_particle.h"

namespace simpla
{

/**
 * @ingroup Application
 *
 * \brief Electromagnetic solver
 */
template<typename TM>
struct ExplicitEMContext : public ContextBase
{
public:

    typedef TM mesh_type;
    typedef typename mesh_type::scalar_type scalar_type;
    typedef ExplicitEMContext<mesh_type> this_type;
    typedef ContextBase base_type;

    ExplicitEMContext();

    template<typename ...Args>
    ExplicitEMContext(Args &&...args) :
            ExplicitEMContext()
    {
        load(std::forward<Args>(args)...);
    }

    ~ExplicitEMContext();

    double CheckCourantDt() const;

    template<typename ...Args>
    static std::shared_ptr<base_type> create(Args &&... args)
    {
        return std::dynamic_pointer_cast<base_type>(
                std::shared_ptr<this_type>(
                        new this_type(std::forward<Args>(args)...)));
    }

    template<typename ...Args>
    static std::pair<std::string,
            std::function<std::shared_ptr<base_type>(Args const &...)>> CreateFactoryFun()
    {
        std::function<std::shared_ptr<base_type>(Args const &...)> call_back =
                [](Args const &...args)
                {
                    return this_type::create(args...);
                };
        return std::move(std::make_pair(get_type_as_string_static(), call_back));
    }

    // interface begin

    template<typename TDict> void load(TDict const &dict);

    virtual std::ostream &print(std::ostream &os) const;


//	std::string load(std::string const & path = "");

    std::string save(std::string const &path = "") const;

    static std::string get_type_as_string_static()
    {
        return "ExplicitEMContext_" + mesh_type::get_type_as_string_static();
    }

    std::string get_type_as_string() const
    {
        return get_type_as_string_static();
    }

    std::ostream &print(std::ostream &os) const
    {
        print_(os);
        return os;
    }

    void next_timestep();

    bool pre_process();

    bool post_process();

    bool empty() const
    {
        return !model->is_valid();
    }

    operator bool() const
    {
        return model->is_valid();
    }

    // interface end

    void InitPECboundary();

public:

    std::string description;

    std::shared_ptr<Model < mesh_type>> model;

    template<typename TV, size_t iform> using field=Field <Domain<Model < mesh_type>, iform>,std::shared_ptr<TV>>;

    field<scalar_type, EDGE> E1, dE;
    field<scalar_type, FACE> B1, dB;

    field<scalar_type, EDGE> J1; //!< current density
    field<Real, EDGE> Jext; //!< external current

//	field<VERTEX, scalar_type>  phi; //!< electrostatic potential

    ImplicitPushE<Model < mesh_type>> implicit_push_E;

    field<Real, VERTEX> n0; //!< background  equilibrium electron density
    field<nTuple<Real, 3>, VERTEX> E0; //!<background  equilibrium electoric field  (B0)=0
    field<nTuple<Real, 3>, VERTEX> B0; //!<background  equilibrium magnetic field J0+curl(B0)=0
//	field<EDGE, Real> J0; //!<background  equilibrium current density J0+curl(B0)=0

//	PML<mesh_type> pml_push;

private:
    typedef decltype(E1) E_type;
    typedef decltype(B1) B_type;
    typedef decltype(J1) J_type;

    typedef std::map<std::string, std::shared_ptr<ParticleBase> > TParticles;

    template<typename TBatch>
    void ExcuteCommands(TBatch const &batch)
    {
        if (batch.size() == 0)
            return;
        VERBOSE << "Apply constraints";
        for (auto const &command : batch)
        {
            command();
        }
    }

    std::list<std::function<void()> > commandToE_;

    std::list<std::function<void()> > commandToB_;

    std::list<std::function<void()> > commandToJ_;

    std::map<std::string, std::shared_ptr<ParticleBase>> particles_;

};

template<typename TM>
ExplicitEMContext<TM>::ExplicitEMContext() :
        E1(model), B1(model), J1(model), dE(model), dB(model),

        B0(model), E0(model),

        n0(model), Jext(model),

        implicit_push_E(model)
{
}

template<typename TM>
ExplicitEMContext<TM>::~ExplicitEMContext()
{
}

template<typename TM>
template<typename OS>
OS &ExplicitEMContext<TM>::print_(OS &os) const
{

    os

    << "Description = \"" << description << "\"," << std::endl

    << " Model = { " << model << "} ," << std::endl;

    if (particles_.size() > 0)
    {

        os << "\n , Particles = { \n";
        for (auto const &p : particles_)
        {
            os << p.first << " = { ";
            p.second->print(os);
            os << "}," << std::endl;
//			os << " { " << p.second << " }, " << std::endl;
        }
        os << "\n} ";
    }

    return os;

}

template<typename TM>
std::string ExplicitEMContext<TM>::save(std::string const &path) const
{

    auto abs_path = (cd(path));

    VERBOSE << SAVE(E1);
    VERBOSE << SAVE(B1);
    VERBOSE << SAVE(J1);
    VERBOSE << SAVE(Jext);
    VERBOSE << SAVE(dE);
    VERBOSE << SAVE(dB);

    for (auto const &p : particles_)
    {
        VERBOSE << p.second->save(abs_path + p.first + "/");
    }

    return path;
}

template<typename TM> template<typename TDict>
void ExplicitEMContext<TM>::load(TDict const &dict)
{
    DEFINE_PHYSICAL_CONST

    LOGGER << "Load ExplicitEMContext ";

    description = dict["Description"].template as<std::string>();

    LOGGER << description;

    field<Real, VERTEX> ne0(model);
    field<Real, VERTEX> Te0(model);
    field<Real, VERTEX> Ti0(model);

    if (dict["Model"]["GFile"])
    {
        model->mesh_type::load(dict["Model"]["Mesh"]);

        GEqdsk geqdsk;

        geqdsk.load(dict["Model"]["GFile"].template as<std::string>());

        geqdsk.save("/Geqdsk/");

        typename mesh_type::coordinate_tuple src_min;
        typename mesh_type::coordinate_tuple src_max;
        typename mesh_type::coordinate_tuple min1, min2, max1, max2;

        std::tie(src_min, src_max) = geqdsk.extents();

        min1 = model->MapTo(geqdsk.InvMapTo(src_min));
        max1 = model->MapTo(geqdsk.InvMapTo(src_max));

        std::tie(min2, max2) = model->extents();

        Clipping(min2, max2, &min1, &max1);

        auto dims = model->dimensions();

//		if (model->enable_spectral_method)
//		{
//
//			/**
//			 *  @bug Lua can not handle field with complex value!!
//			 */
//
//			for (int i = 0; i < model->NDIMS; ++i)
//			{
//				if (dims[i] <= 1)
//				{
//					min1[i] = min2[i];
//					max1[i] = max2[i];
//				}
//			}
//		}

        if (dims[2] > 1)
        {
            min1[2] = min2[2];
            max1[2] = max2[2];
        }

        model->extents(min1, max1);

        model->update();

        geqdsk.SetUpMaterial(&model);

        B1.clear();
        E1.clear();
        J1.clear();
        B0.clear();
        ne0.clear();
        Te0.clear();
        Ti0.clear();

        geqdsk.GetProfile("B", &B0);
        geqdsk.GetProfile("ne", &ne0);
        geqdsk.GetProfile("Te", &Te0);
        geqdsk.GetProfile("Ti", &Ti0);

        description = description + "\n GEqdsk ID:" + geqdsk.description();
        InitPECboundary();
    }
    else
    {
        if (!model->load(dict["Model"]))
        {
            THROW_EXCEPTION_PARSER_ERROR("Configure 'Model' fail!");
        }

        B1.clear();
        E1.clear();
        J1.clear();
        B0.clear();

        VERBOSE_CMD(loadField(dict["InitValue"]["B"], &B1));

        VERBOSE_CMD(loadField(dict["InitValue"]["E"], &E1));

        VERBOSE_CMD(loadField(dict["InitValue"]["J"], &J1));

        VERBOSE_CMD(loadField(dict["InitValue"]["ne"], &ne0));

        VERBOSE_CMD(loadField(dict["InitValue"]["Te"], &Te0));

        VERBOSE_CMD(loadField(dict["InitValue"]["Ti"], &Ti0));

    }

    dB.clear();
    dE.clear();
    E0.clear();
    Jext.clear();

    cd("/Input/");

    VERBOSE << SAVE(ne0);
    VERBOSE << SAVE(Te0);
    VERBOSE << SAVE(Ti0);
    VERBOSE << SAVE(B0);
    VERBOSE << SAVE(E0);

    LOGGER << "Load Particles";

    auto particle_factory = RegisterAllParticles<mesh_type, TDict,
    Model < mesh_type > const &, decltype(ne0), decltype(Te0) > ();

    /**
     * @todo load particle engine plugin
     *
     *  add new creator at here
     *
     */
    for (auto opt : dict["Particles"])
    {
        auto id = opt.first.template as<std::string>("unnamed");

        auto type_str = opt.second["Type"].template as<std::string>("");

//		try
//		{
        auto p = particle_factory.create(type_str, opt.second, model, ne0, Te0);

        if (p != nullptr)
        {
            particles_.emplace(id, p);

        }

//		} catch (...)
//		{
//
//			THROW_EXCEPTION_PARSER_ERROR("Particle={" + id + " = { Type = " + type_str + "}}" + "  ");
//
//		}

    }

    LOGGER << "Load Constraints";

//	for (auto const & item : dict["Constraints"])
//	{
//		try
//		{
//
//			auto dof = item.second["DOF"].template as<std::string>("");
//
//			VERBOSE << "Add constraint to " << dof;
//
//			if (dof == "E")
//			{
//				commandToE_.push_back(
//						E1.CreateCommand(
//								model->select_by_config(E1.IForm,
//										item.second["Select"]),
//								item.second["Operation"]));
//			}
//			else if (dof == "B")
//			{
//
//				commandToB_.push_back(
//						B1.CreateCommand(
//								model->select_by_config(B1.IForm,
//										item.second["Select"]),
//								item.second["Operation"]));
//			}
//			else if (dof == "J")
//			{
//
//				commandToJ_.push_back(
//						Jext.CreateCommand(
//								model->select_by_config(Jext.IForm,
//										item.second["Select"]),
//								item.second["Operation"]));
//			}
//			else
//			{
//				THROW_EXCEPTION_PARSER_ERROR("Unknown DOF!");
//			}
//
//		} catch (std::runtime_error const & e)
//		{
//
//			THROW_EXCEPTION_PARSER_ERROR("Load 'Constraints' error! ");
//		}
//	}
//	bool enableImplicit = false;
//
//	for (auto const &p : particles_)
//	{
//		enableImplicit = enableImplicit || p.second->is_implicit();
//	}
//	bool enablePML = false;
//
//	try
//	{
//		LOGGER << "Load electromagnetic fields solver";
//
//		using namespace std::placeholders;
//
//		Real ic2 = 1.0 / (mu0 * epsilon0);
//
//		if (dict["FieldSolver"]["PML"])
//		{
//			auto solver = std::shared_ptr<PML<TM> >(new PML<TM>(model, dict["FieldSolver"]["PML"]));
//
//			E_plus_CurlB = std::bind(&PML<TM>::next_timestepE, solver, _1, _2, _3, _4);
//
//			B_minus_CurlE = std::bind(&PML<TM>::next_timestepB, solver, _1, _2, _3, _4);
//
//		}
//		else
//		{
//			E_plus_CurlB = [mu0 , epsilon0](Real dt, TE const & E , TB const & B, TE* pdE)
//			{
//				auto & dE=*pdE;
//				VERBOSE_CMD(dE += curl(B)/(mu0 * epsilon0) *dt);
//			};
//
//			B_minus_CurlE = [](Real dt, TE const & E, TB const &, TB* pdB)
//			{
//				auto & dB=*pdB;
//				VERBOSE_CMD( dB -= curl(E)*dt);
//			};
//		}
//
//	} catch (std::runtime_error const & e)
//	{
//		THROW_EXCEPTION_PARSER_ERROR("Configure field solver error! ");
//	}
//	if (enableImplicit)
//	{
//
//		auto solver = std::shared_ptr<ImplicitPushE<mesh_type>>(new);
//		Implicit_PushE = [solver] ( TE const & pE, TB const & pB, TParticles const&p, TE*dE)
//		{	solver->next_time_step( pE,pB,p,dE);};
//	}

}

template<typename TM>
bool ExplicitEMContext<TM>::pre_process()
{
    return true;
}

template<typename TM>
void ExplicitEMContext<TM>::InitPECboundary()
{
    std::vector<typename mesh_type::index_type> conduct_wall_E_;

    for (auto s : E1.domain())
    {
        if (model->get(s) == model->null_material)
        {
            conduct_wall_E_.push_back(s);
        }
    }
    if (conduct_wall_E_.size() > 0)
    {
        std::function<void()> fun = [=]()
        {
            VERBOSE << "Apply PEC to E1";
            for (auto s : conduct_wall_E_)
            {
                traits::index(this->E1, s) = 0;

            };
        };
        commandToE_.push_back(fun);
    }

    std::vector<typename mesh_type::index_type> conduct_wall_B_;

    for (auto s : B1.domain())
    {
        if (model->get(s) == model->null_material)
        {
            conduct_wall_B_.push_back(s);
        }
    }

    if (conduct_wall_B_.size() > 0)
    {
        std::function<void()> fun = [=]()
        {
            VERBOSE << "Apply PEC to B1 ";
            for (auto s : conduct_wall_B_)
            {
                traits::index(this->B1, s) = 0;

            };
        };
        commandToE_.push_back(fun);
    }

}

template<typename TM>
bool ExplicitEMContext<TM>::post_process()
{
    return true;
}

template<typename TM>
void ExplicitEMContext<TM>::next_timestep()
{
    DEFINE_PHYSICAL_CONST

    INFORM

    << "[" << model->get_clock() << "]"

    << "Simulation Time = " << (model->get_time() / CONSTANTS["s"]) << "[s]";

    Real dt = model->get_dt();

// Compute Cycle Begin

    LOG_CMD(dE = (curl(B1) / mu0 - J1) / epsilon0 * dt);

//   particle 1/2 -> 1  . To n[1/2], J[1/2]
//	implicit_push_E.next_time_step(&dE);

    LOG_CMD(E1 += dE);    // E(t=0 -> 1)

    ExcuteCommands(commandToE_);

    LOG_CMD(dB = -curl(E1) * dt);

    LOG_CMD(B1 += dB * 0.5);    //	B(t=1/2 -> 1)
    ExcuteCommands(commandToB_);

    J1.clear();

    ExcuteCommands(commandToJ_);

    //   particle 0-> 1. Get J[1/2]
    for (auto &p : particles_)
    {
        p.second->next_timestep();
    }

    LOG_CMD(B1 += dB * 0.5);    //  B(t=0 -> 1/2)
    ExcuteCommands(commandToB_);
// Compute Cycle End
    model->next_timestep();

}

}
// namespace simpla

#endif /* EXPLICIT_EM_IMPL_H_ */
