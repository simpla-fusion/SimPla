/*
 * cold_fluid.h
 *
 *  Created on: 2013年11月13日
 *      Author: salmon
 */

#ifndef COLD_FLUID_H_
#define COLD_FLUID_H_

#include "../../../src/fetl/fetl.h"
#include "../../../src/engine/fieldsolver.h"
#include "../../../src/utilities/load_field.h"
#include "../../../src/utilities/log.h"
#include "../../../src/utilities/pretty_stream.h"
namespace simpla
{

template<typename TM>
class ColdFluidEM
{
public:
	DEFINE_FIELDS(TM)

	typedef Mesh mesh_type;

	typedef ColdFluidEM<mesh_type> this_type;

private:

	struct Species
	{
		Real m;
		Real Z;
		Form<0> n;
		VectorForm<0> J;

		Species(Real pm, Real pZ, mesh_type const &mesh)
				: m(pm), Z(pZ), n(mesh), J(mesh)
		{
		}
		~Species()
		{
		}

	};
	std::map<std::string, std::shared_ptr<Species>> sp_list_;
public:

	mesh_type const & mesh;

	ColdFluidEM(mesh_type const & pmesh)
			: mesh(pmesh)
	{
	}

	~ColdFluidEM()
	{
	}

	inline bool IsEmpty()
	{
		return sp_list_.empty();
	}

	void Deserialize(LuaObject const&cfg);
	std::ostream & Serialize(std::ostream & os) const;
	template<typename TJ, typename TE, typename TB> inline
	void NextTimeStep(double dt, TJ const &J, TE *E, TB *B);
	void DumpData() const;

}
;
template<typename TM>
template<typename TJ, typename TE, typename TB> inline
void ColdFluidEM<TM>::NextTimeStep(Real dt, TJ const &J, TE *E, TB *B)
{
	if (sp_list_.empty())
	{
		return;
	}

	LOGGER << "Cold Fluid Push E,M";

	const double mu0 = mesh.constants["permeability of free space"];
	const double epsilon0 = mesh.constants["permittivity of free space"];
	const double speed_of_light = mesh.constants["speed of light"];
	const double proton_mass = mesh.constants["proton mass"];
	const double elementary_charge = mesh.constants["elementary charge"];

	VectorForm<0> K_(mesh);
	VectorForm<0> K(mesh);

	Form<0> a(mesh);
	Form<0> b(mesh);
	Form<0> c(mesh);

	Form<0> BB(mesh);

	BB = Dot(*B, *B);

	VectorForm<0> Ev(mesh), Bv(mesh), dEvdt(mesh);
	Ev.Init();
	Bv.Init();

	MapTo(*E, &Ev);
	MapTo(*B, &Bv);

	for (auto &v : sp_list_)
	{

		auto & ns = v.second->n;
		auto & Js = v.second->J;
		auto ms = v.second->m * proton_mass;
		auto Zs = v.second->Z * elementary_charge;

		Form<0> as(mesh);

		as = 2.0 * ms / (dt * Zs);

		a += ns * Zs / as;
		b += ns * Zs / (BB + as * as);
		c += ns * Zs / ((BB + as * as) * as);

		K_ = /* 2.0 * nu * Js*/
		-2.0 * Cross(Js, Bv) - (Ev * ns) * (2.0 * Zs);

		K -= Js + 0.5 * (

		K_ / as

		+ Cross(K_, Bv) / (BB + as * as)

		+ Cross(Cross(K_, Bv), Bv) / (as * (BB + as * as))

		);
	}

	a = a * (0.5 * dt) / epsilon0 - 1.0;
	b = b * (0.5 * dt) / epsilon0;
	c = c * (0.5 * dt) / epsilon0;

	K /= epsilon0;

	dEvdt = K / a + Cross(K, Bv) * b / ((c * BB - a) * (c * BB - a) + b * b * BB)
	        + Cross(Cross(K, Bv), Bv) * (-c * c * BB + c * a - b * b)
	                / (a * ((c * BB - a) * (c * BB - a) + b * b * BB));

	for (auto &v : sp_list_)
	{
		auto & ns = v.second->n;
		auto & Js = v.second->J;
		auto ms = v.second->m * proton_mass;
		auto Zs = v.second->Z * elementary_charge;

		Form<0> as(mesh);

		as = 2.0 * ms / (dt * Zs);

		K_ = // 2.0*nu*(Js)
		        -2.0 * Cross(Js, Bv) - (2.0 * Ev + dEvdt * dt) * ns * Zs;
		Js +=

		K_ / as

		+ Cross(K_, Bv) / (BB + as * as)

		+ Cross(Cross(K_, Bv), Bv) / (as * (BB + as * as));
	}

	MapTo(Ev, E);
	MapTo(Bv, B);

	//		J -=  dEvdt;
	LOGGER << "Cold Fluid Push Done";

}

template<typename TM>
inline void ColdFluidEM<TM>::Deserialize(LuaObject const&cfg)
{
	if (cfg.IsNull())
	{
		WARNING << "No configure information!";
		return;
	}

	for (auto const & p : cfg)
	{

		std::string key;

		if (!p.first.is_number())
		{
			key = p.first.as<std::string>();
		}
		else
		{
			p.second.GetValue("Name", &key);
		}

		std::shared_ptr<Species> sp(
		        new Species(p.second["m"].template as<Real>(1.0), p.second["Z"].template as<Real>(1.0), mesh));

		sp->n.Init();
		sp->J.Init();

		if (!LoadField(p.second["n"], &(sp->n)))
		{
			WARNING << "[" << key << "] plasma density is not initialized";
		}

		if (!LoadField(p.second["J"], &(sp->J)))
		{
			WARNING << "[" << key << "] plasma current is not initialized";
		}

		sp_list_.emplace(std::make_pair(key, sp));

	}
	LOGGER << " Load Cold Fluid [Done]!";
}

template<typename TM>
void ColdFluidEM<TM>::DumpData() const
{
	GLOBAL_DATA_STREAM.OpenGroup("/DumpData");

	for (auto const & p : sp_list_)
	{
		LOGGER << "Dump " << p.first + ".n" << " to "
		        << Data(p.second->n.data(), p.first + ".n", p.second->n.GetShape(), true);

		LOGGER << "Dump " << p.first + ".J" << " to "
		        << Data(p.second->n.data(), p.first + ".J", p.second->n.GetShape(), true);
	}
}

template<typename TM>
std::ostream & ColdFluidEM<TM>::Serialize(std::ostream & os) const
{
	os << "-- Cold Fluid -------------------\n";
	os << "  ColdFluid={\n";

	for (auto const & p : sp_list_)
	{
		os << "\t" << p.first

		<< " = { " << " m =" << p.second->m << "," << " Z =" << p.second->Z << ",\n"

		<< "\t n0 = " << Data(p.second->n.data(), p.first + ".n", p.second->n.GetShape()) << "\n"

		<< "\t J0 = " << Data(p.second->J.data(), p.first + ".J", p.second->J.GetShape()) << "\n"

		<< "\t},\n";
	}
	os << "}\n";

	return os;
}

template<typename TM>
inline std::ostream & operator<<(std::ostream & os, ColdFluidEM<TM> const &self)
{
	return self.Serialize(os);
}

}  // namespace simpla

#endif /* COLD_FLUID_H_ */
