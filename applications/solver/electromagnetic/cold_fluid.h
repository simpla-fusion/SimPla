/*
 * cold_fluid.h
 *
 *  Created on: 2013年11月13日
 *      Author: salmon
 */

#ifndef COLD_FLUID_H_
#define COLD_FLUID_H_

#include "../../../src/fetl/fetl.h"
#include "../../../src/utilities/load_field.h"
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

		Species(Real pm, Real pZ, mesh_type const &mesh) :
				m(pm), Z(pZ), n(mesh), J(mesh)
		{
		}
		~Species()
		{
		}

	};
	std::map<std::string, std::shared_ptr<Species>> sp_list_;
public:

	mesh_type const & mesh;

	template<typename U>
	friend std::ostream & operator<<(std::ostream & os,
			ColdFluidEM<U> const &self);

	ColdFluidEM(mesh_type const & pmesh) :
			mesh(pmesh)
	{
	}

	template<typename TConfig>
	ColdFluidEM(mesh_type const & pmesh, TConfig const & cfg) :
			mesh(pmesh)
	{
		Deserialize(cfg);
	}

	~ColdFluidEM()
	{
	}

	inline void Deserialize(LuaObject const&cfg)
	{
		if (cfg.IsNull())
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

			std::string engine = p.second.at("Engine").as<std::string>();

			if (engine == "ColdFluid")
			{

				std::shared_ptr<Species> sp(
						new Species(p.second["m"].template as<Real>(1.0),
								p.second["Z"].template as<Real>(1.0), mesh));

				sp_list_.emplace(std::make_pair(key, sp));

				LoadField(p.second["n"], &(sp->n));
				LoadField(p.second["J"], &(sp->J));
			}
		}

	}

	template<typename PT>
	inline void Serialize(PT &cfg) const
	{

	}

	inline void Eval(...)
	{
	}

	template<typename TJ, typename TE, typename TB> inline
	void Eval(Real dt, TJ const &J, TE *E, TB *B)
	{

		const double mu0 = mesh.constants["permeability of free space"];
		const double epsilon0 = mesh.constants["permittivity of free space"];
		const double speed_of_light = mesh.constants["speed of light"];
		const double proton_mass = mesh.constants["proton mass"];
		const double elementary_charge = mesh.constants["elementary charge"];

		*E += (Curl((*B) / mu0) - J) / epsilon0 * dt;
		*B -= Curl(*E) * dt;

		VectorForm<0> K_(mesh);
		VectorForm<0> K(mesh);

		Form<0> a(mesh, 0.0);
		Form<0> b(mesh, 0.0);
		Form<0> c(mesh, 0.0);

		Form<0> BB(mesh);

		BB = Dot(B, B);

		VectorForm<0> Ev(mesh), Bv(mesh), dEvdt(mesh);

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

		dEvdt = K / a
				+ Cross(K, Bv) * b / ((c * BB - a) * (c * BB - a) + b * b * BB)
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

		//		J -=  dEvdt;
	}

}
;

template<typename TM>
inline std::ostream & operator<<(std::ostream & os, ColdFluidEM<TM> const &self)
{
	os << "-- Cold Fluid -------------------" << std::endl;
	os << "Particles={" << std::endl;

	for (auto const & p : self.sp_list_)
	{
		os << "\t" << p.first

		<< " = { "

		<< "Type=\"ColdFluid\","

		<< " m =" << p.second->m << ","

		<< " Z =" << p.second->Z << ","

		<< " n0 = {},  J0 = {}"

		<< "},"

		<< std::endl;
	}
	os << "}" << std::endl;
	return os;

}
}  // namespace simpla

#endif /* COLD_FLUID_H_ */
