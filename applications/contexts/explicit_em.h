/*
 * explicit_em_impl.h
 *
 *  Created on: 2013年12月29日
 *      Author: salmon
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
#include "../../src/utilities/log.h"

// Field expression
#include "../../src/fetl/field.h"
#include "../../src/fetl/ntuple.h"
#include "../../src/fetl/primitives.h"
#include "../../src/mesh/field_convert.h"

// Data IO
#include "../../src/fetl/save_field.h"
#include "../../src/io/data_stream.h"

// Modeling
#include "../../src/modeling/material.h"
#include "../../src/utilities/geqdsk.h"

// Field solver
#include "../../src/modeling/constraint.h"
#include "../solver/electromagnetic/solver.h"

// Particle
#include "../../src/particle/particle_factory.h"

namespace simpla
{
template<typename TM>
struct ExplicitEMContext
{
public:

	typedef TM mesh_type;

	DEFINE_FIELDS (TM)

	typedef ExplicitEMContext<mesh_type> this_type;

	ExplicitEMContext();

	~ExplicitEMContext();

	template<typename TDict> void Load(TDict const & dict);

	template<typename OS>
	void Save(OS & os) const;

	void NextTimeStep();

	void DumpData(std::string const & path = "") const;

	double CheckCourantDt() const;

public:

	mesh_type mesh;

	std::string description;

	Material<mesh_type> material_;

	bool isCompactStored_;

	Form<EDGE> E, dE;
	Form<FACE> B, dB;
	Form<VERTEX> phi; // electrostatic potential

	Form<EDGE> J;     // current density
	Form<EDGE> J0;     //background current density J0+Curl(B(t=0))=0
	Form<VERTEX> rho; // charge density

	typedef decltype(E) TE;
	typedef decltype(B) TB;
	typedef decltype(J) TJ;

	std::function<void(Real, TE const &, TB const &, TE*)> CalculatedE;

	std::function<void(Real, TE const &, TB const &, TB*)> CalculatedB;

	void ApplyConstraintToE(TE* pE)
	{
		if (constraintToE_.size() > 0)
		{
			LOGGER << "Apply Constraint to E";
			for (auto const & foo : constraintToE_)
			{
				foo(pE);
			}
		}
	}
	void ApplyConstraintToB(TB* pB)
	{
		if (constraintToB_.size() > 0)
		{
			LOGGER << "Apply Constraint to B";
			for (auto const & foo : constraintToB_)
			{
				foo(pB);
			}
		}
	}
	void ApplyConstraintToJ(TJ* pJ)
	{
		if (constraintToJ_.size() > 0)
		{
			LOGGER << "Apply Constraint to J";
			for (auto const & foo : constraintToJ_)
			{
				foo(pJ);
			}
		}
	}

private:

	std::list<std::function<void(TE*)> > constraintToE_;

	std::list<std::function<void(TB*)> > constraintToB_;

	std::list<std::function<void(TJ*)> > constraintToJ_;

	typedef ParticleWrap<TE, TB, TJ> ParticleType;

	std::map<std::string, ParticleType> particles_;

}
;

template<typename TM>
ExplicitEMContext<TM>::ExplicitEMContext()
		: isCompactStored_(true), material_(mesh),

		E(mesh), B(mesh), J(mesh), J0(mesh), dE(mesh), dB(mesh), rho(mesh), phi(mesh)
{
}

template<typename TM>
ExplicitEMContext<TM>::~ExplicitEMContext()
{
}
template<typename TM> template<typename TDict>
void ExplicitEMContext<TM>::Load(TDict const & dict)
{

	LOGGER << "Load ExplicitEMContext ";

	description = dict["Description"].template as<std::string>();

	LOGGER << "Description=\"" << description << "\" \n";

	mesh.Load(dict["Grid"]);

	material_.Update();

	Form<VERTEX> ne0(mesh);
	Form<VERTEX> Te0(mesh);
	Form<VERTEX> Ti0(mesh);

	if (dict["GFile"])
	{

		GEqdsk geqdsk(dict["GFile"].template as<std::string>());

		nTuple<3, Real> xmin, xmax;

		xmin[0] = geqdsk.GetMin()[0];
		xmin[1] = geqdsk.GetMin()[1];
		xmin[2] = 0;
		xmax[0] = geqdsk.GetMax()[0];
		xmax[1] = geqdsk.GetMax()[1];
		xmax[2] = 0;

		mesh.SetExtent(xmin, xmax);

		mesh.Update();

		material_.Add("Plasma", geqdsk.Boundary());
		material_.Add("Vacuum", geqdsk.Limiter());
		material_.Update();

		geqdsk.Save(std::cout);

		B.Clear();

		for (auto s : mesh.GetRange(FACE))
		{
			auto x = mesh.CoordinatesToCartesian(mesh.GetCoordinates(s));
			B[s] = mesh.template Sample<FACE>(Int2Type<FACE>(), s, geqdsk.B(x[0], x[1]));

		}

		ne0.Clear();
		Te0.Clear();
		Ti0.Clear();

		material_.template SelectCell<VERTEX>([&](typename mesh_type::index_type s )
		{
			auto x=mesh.CoordinatesToCartesian( mesh.GetCoordinates(s));
			auto p=geqdsk.psi(x[0],x[1]);

			ne0[s] = geqdsk.Profile("ne",p);
			Te0[s] = geqdsk.Profile("Te",p);
			Ti0[s] = geqdsk.Profile("Ti",p);

		}, "Plasma");

		J0 = Curl(B) / mesh.constants()["permeability of free space"];

		description = description + "\n GEqdsk ID:" + geqdsk.Description();

		LOGGER << Dump(ne0, "ne", false);
		LOGGER << Dump(Te0, "Te", false);
		LOGGER << Dump(Ti0, "Ti", false);
	}

	{
		auto dt = mesh.CheckCourantDt();
		if (dt < mesh.GetDt())
		{
			CHECK(dt);
			mesh.SetDt(dt);
		}
	}

	LOGGER << "Grid = { \n" << mesh << "\n}";

	LOGGER << "Material = { \n" << material_ << "\n}";

	if (dict["InitValue"])
	{
		auto init_value = dict["InitValue"];

		if (E.empty())
			LOG_CMD(LoadField(init_value["E"], &E));

		if (B.empty())
			LOG_CMD(LoadField(init_value["B"], &B));

		if (J.empty())
			LOG_CMD(LoadField(init_value["J"], &J));

		if (ne0.empty())
			LOG_CMD(LoadField(init_value["ne"], &ne0));

		if (Te0.empty())
			LOG_CMD(LoadField(init_value["Te"], &Te0));

		if (Ti0.empty())
			LOG_CMD(LoadField(init_value["Ti"], &Ti0));

	}

	if (E.empty())
		E.Clear();

	if (B.empty())
		B.Clear();

	if (J.empty())
		J.Clear();

	if (dB.empty())
		dB.Clear();

	if (dE.empty())
		dE.Clear();

	if (J0.empty())
		J0.Clear();

	LOGGER << "Load Particles";
	for (auto const &opt : dict["Particles"])
	{
		ParticleWrap<TE, TB, TJ> p;

		bool flag = false;

		if (opt.second["IsElectron"].template as<bool>(false))
		{
			flag = CreateParticle<Mesh, TE, TB, TJ>(mesh, opt.second["Type"].template as<std::string>(), &p, opt.second,
			        ne0, Te0);
		}
		else
		{
			flag = CreateParticle<Mesh, TE, TB, TJ>(mesh, opt.second["Type"].template as<std::string>(), &p, opt.second,
			        ne0, Ti0);
		}

		if (flag)
		{
			particles_.emplace(std::make_pair(opt.first.template as<std::string>(), p));
		}
	}

	LOGGER << "Load Constraints";
	for (auto const & item : dict["Constraints"])
	{

		auto dof = item.second["DOF"].template as<std::string>();

		LOGGER << "Add constraint to " << dof;

		if (dof == "E")
		{
			constraintToE_.push_back(CreateConstraint<TE>(material_, item.second));
		}
		else if (dof == "B")
		{
			constraintToB_.push_back(CreateConstraint<TB>(material_, item.second));
		}
		else if (dof == "J")
		{
			constraintToJ_.push_back(CreateConstraint<TJ>(material_, item.second));
		}
		else
		{
			//TODO Add particles constraints
			UNIMPLEMENT2("Unknown Constraints!!");
			continue;
		}

	}

	CreateEMSolver(dict["FieldSolver"], mesh, &CalculatedE, &CalculatedB, ne0, Te0, Ti0);

}

template<typename TM>
template<typename OS>
void ExplicitEMContext<TM>::Save(OS & os) const
{

	os

	<< "InitValue={" << "\n"

	<< "	E = " << Dump(E, "E", false) << ",\n"

	<< "	B = " << Dump(B, "B", false) << ",\n"

	<< "	J = " << Dump(J, "J", false) << ",\n"

	<< "}" << "\n"

	;
}
template<typename OS, typename TM>
OS &operator<<(OS & os, ExplicitEMContext<TM> const& self)
{
	self.Save(os);
	return os;
}
template<typename TM>
void ExplicitEMContext<TM>::NextTimeStep()
{
	Real dt = mesh.GetDt();

	mesh.NextTimeStep();

	DEFINE_PHYSICAL_CONST(mesh.constants());

	VERBOSE

	<< "Simulation Time = "

	<< (mesh.GetTime() / mesh.constants()["s"]) << "[s]"

	<< " dt = " << (dt / mesh.constants()["s"]) << "[s]";

	//************************************************************
	// Compute Cycle Begin
	//************************************************************

	LOG_CMD(dE = -(J + J0) / epsilon0 * dt);

	// dE = Curl(B)*dt
	CalculatedE(dt, E, B, &dE);

	// E(t=0  -> 1/2  )
	LOG_CMD(E += dE * 0.5);

	ApplyConstraintToE(&E);

	for (auto &p : particles_)
	{
		p.second.NextTimeStep(dt, E, B);	// particle(t=0 -> 1)
	}

	//  E(t=1/2  -> 1)
	LOG_CMD(E += dE * 0.5);

	ApplyConstraintToE(&E);

	dB.Clear();

	CalculatedB(dt, E, B, &dB);

	//  B(t=1/2 -> 1)
	LOG_CMD(B += dB * 0.5);

	ApplyConstraintToB(&B);

	J.Clear();

	ApplyConstraintToJ(&J);

	for (auto &p : particles_)
	{
		// B(t=0) E(t=0) particle(t=0) Jext(t=0)
		p.second.Scatter(&J, E, B);
	}

	// B(t=0 -> 1/2)
	LOG_CMD(B += dB * 0.5);

	ApplyConstraintToB(&B);

	//************************************************************
	// Compute Cycle End
	//************************************************************

}
template<typename TM>
void ExplicitEMContext<TM>::DumpData(std::string const & path) const
{
	GLOBAL_DATA_STREAM.OpenGroup(path);

	LOGGER << DUMP(E);
	LOGGER << DUMP(B);
	LOGGER << DUMP(J);

}
}
// namespace simpla

#endif /* EXPLICIT_EM_IMPL_H_ */
