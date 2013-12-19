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
	void NextTimeStep(Real dt, TE const &E, TB const &B, TJ *J);

	void DumpData() const;

}
;
template<typename TM>
template<typename TJ, typename TE, typename TB> inline
void ColdFluidEM<TM>::NextTimeStep(Real dt, TE const &E, TB const &B, TJ *J)
{
	if (sp_list_.empty())
	{
		return;
	}

	LOGGER << "Push Cold Fluid" << START;

	const double mu0 = mesh.constants["permeability of free space"];
	const double epsilon0 = mesh.constants["permittivity of free space"];
	const double speed_of_light = mesh.constants["speed of light"];
	const double proton_mass = mesh.constants["proton mass"];
	const double elementary_charge = mesh.constants["elementary charge"];

	VectorForm<0> K(mesh);

	Form<0> a(mesh);
	Form<0> b(mesh);
	Form<0> c(mesh);

	Form<0> BB(mesh);

	BB = Dot(B, B);

	VectorForm<0> Ev(mesh), Bv(mesh), dEvdt(mesh);
	Ev.Init();
	Bv.Init();

	MapTo(E, &Ev);
	MapTo(B, &Bv);

	a.Fill(0);
	b.Fill(0);
	c.Fill(0);
	K.Fill(0);

	dEvdt = Ev / dt;

	for (auto &v : sp_list_)
	{

		auto & ns = v.second->n;
		auto & Js = v.second->J;
		auto ms = v.second->m * proton_mass;
		auto Zs = v.second->Z * elementary_charge;

		Real as = 2.0 * ms / (dt * Zs);

		CHECK(as);

		a += ns * Zs / as;
		b += ns * Zs / (BB + as * as);
		c += ns * Zs / ((BB + as * as) * as);

		VectorForm<0> K_(mesh);

		K_ = -2.0 * Cross(Js, Bv) - (Ev * ns) * (2.0 * Zs);

		K -= Js

		+ 0.5 * (K_ / as

		+ Cross(K_, Bv) / (BB + as * as)

		+ Cross(Cross(K_, Bv), Bv) / (as * (BB + as * as))

		);

	}
//	LOGGER << DUMP(BB);
//	LOGGER << DUMP(Ev);
//	LOGGER << DUMP(Bv);
//
//	LOGGER << DUMP(a);
//	LOGGER << DUMP(b);
//	LOGGER << DUMP(c);

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

		Real as = 2.0 * ms / (dt * Zs);

		VectorForm<0> K_(mesh);

		K_ = -2.0 * Cross(Js, Bv) - (2.0 * Ev + dEvdt * dt) * ns * Zs;

		Js +=

		K_ / as

		+ Cross(K_, Bv) / (BB + as * as)

		+ Cross(Cross(K_, Bv), Bv) / (as * (BB + as * as));
	}

	Form<1> E_(mesh);

	MapTo(dEvdt, &E_);

	*J += E_ * epsilon0;

	LOGGER << "Push Cold Fluid." << DONE;

}

template<typename TM>
inline void ColdFluidEM<TM>::Deserialize(LuaObject const&cfg)
{
	if (cfg.empty())
		return;

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
			sp->n.Fill(0);

		if (!LoadField(p.second["J"], &(sp->J)))
			sp->J.Fill(0);

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
		LOGGER << "Dump " << "n_" + p.first << " to "
		        << Data(p.second->n.data(), "n_" + p.first, p.second->n.GetShape(), true);

		LOGGER << "Dump " << "J_" + p.first << " to "
		        << Data(p.second->J.data(), "J_" + p.first, p.second->J.GetShape(), true);
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

		<< "\t n0 = " << Data(p.second->n.data(), "n_" + p.first, p.second->n.GetShape()) << "\n"

		<< "\t J0 = " << Data(p.second->J.data(), "J_" + p.first, p.second->J.GetShape()) << "\n"

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
