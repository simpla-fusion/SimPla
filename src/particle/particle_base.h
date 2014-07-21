/*
 * particle_base.h
 *
 *  created on: 2014-1-16
 *      Author: salmon
 */

#ifndef PARTICLE_BASE_H_
#define PARTICLE_BASE_H_

#include <iostream>
#include <string>
#include <typeindex>
#include <typeinfo>
#include "../utilities/properties.h"
#include "../utilities/any.h"
#include "../utilities/primitives.h"

namespace simpla
{
/***
 *  \brief interface to Particle
 */
class ParticleBase
{

public:

	ParticleBase()
	{
	}

	virtual ~ParticleBase()
	{
	}

	//interface

	virtual bool check_mesh_type(std::type_info const & t_info) const=0;
	virtual bool check_E_type(std::type_info const & t_info) const=0;
	virtual bool check_B_type(std::type_info const & t_info) const=0;

	virtual std::string save(std::string const & path) const=0;

	virtual std::ostream& print(std::ostream & os) const =0;

	virtual Real get_mass() const=0;

	virtual Real get_charge() const=0;

	virtual bool is_implicit() const = 0;

	virtual std::string get_type_as_string() const = 0;

	virtual void const * get_rho() const= 0;

	virtual void const * get_J() const= 0;

	virtual void next_timestep_zero_(void const * E, void const*B)
	{
	}

	virtual void next_timestep_half_(void const * E, void const*B)
	{
	}

	virtual void update_fields() =0;

	virtual void set_property_(std::string const & name, Any const&v)=0;

	virtual Any const & get_property_(std::string const &name) const=0;

	template<typename T> void set_property(std::string const & name, T const&v)
	{
		set_property_(name, Any(v));
	}

	template<typename T> T get_property(std::string const & name) const
	{
		return get_property_(name).template as<T>();
	}

	template<typename T>
	T const & rho() const
	{
		return *reinterpret_cast<T const*>(get_rho());
	}
	template<typename T>
	T const & J() const
	{
		return *reinterpret_cast<T const*>(get_J());
	}

	template<typename TE, typename TB>
	void next_timestep_zero(TE const & E, TB const &B)
	{
		next_timestep_zero_(&E, &B);
	}
	template<typename TE, typename TB>
	void next_timestep_half(TE const & E, TB const &B)
	{
		next_timestep_half_(&E, &B);
	}

};
//template<typename TP>
//struct ParticleWrap: public ParticleBase
//{
//	typedef TP particle_type;
//	typedef typename particle_type::mesh_type mesh_type;
//	typedef typename mesh_type::scalar_type scalar_type;
//	typedef ParticleBase base_type;
//
//	ParticleWrap(std::shared_ptr<particle_type> p)
//			: self_(p), dummy_J(p->mesh), dummy_Jv(p->mesh)
//	{
//	}
//	~ParticleWrap()
//	{
//	}
//
//	template<typename ...Args>
//	static unsigned int Register(Factory<std::string, base_type, Args ...>* factory)
//	{
//		typename Factory<std::string, base_type, Args ...>::create_fun_callback call_back;
//
//		call_back = []( Args && ...args)
//		{
//			return std::dynamic_pointer_cast<base_type>(
//					std::shared_ptr<ParticleWrap<particle_type>>(
//							new ParticleWrap<particle_type>(particle_type::create( std::forward<Args>(args)...))));
//		};
//
//		return factory->Register(particle_type::get_type_as_string(), call_back);
//	}
//
//	Real get_mass() const
//	{
//		return self_->m;
//	}
//
//	Real get_charge() const
//	{
//		return self_->q;
//	}
//
//	bool is_implicit() const
//	{
//		return self_->is_implicit;
//	}
//	std::string get_type_as_string() const
//	{
//		return self_->get_type_as_string();
//	}
//
//	typename mesh_type:: template field<VERTEX, scalar_type> const& n() const
//	{
//		return self_->n;
//	}
//	typename mesh_type:: template field<EDGE, scalar_type> const& J() const
//	{
//		return J_(std::integral_constant<bool, particle_type::is_implicit>());
//	}
//	typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const&Jv() const
//	{
//		return Jv_(std::integral_constant<bool, particle_type::is_implicit>());
//	}
//
//	void next_timestep_zero(typename mesh_type:: template field<EDGE, scalar_type> const & E,
//	        typename mesh_type:: template field<FACE, scalar_type> const & B)
//	{
//		next_timestep_zero_(std::integral_constant<bool, particle_type::is_implicit>(), E, B);
//	}
//	void next_timestep_zero(typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & E,
//	        typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & B)
//	{
//		next_timestep_zero_(std::integral_constant<bool, particle_type::is_implicit>(), E, B);
//	}
//	void next_timestep_half(typename mesh_type:: template field<EDGE, scalar_type> const & E,
//	        typename mesh_type:: template field<FACE, scalar_type> const & B)
//	{
//		next_timestep_half_(std::integral_constant<bool, particle_type::is_implicit>(), E, B);
//	}
//	void next_timestep_half(typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & E,
//	        typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & B)
//	{
//		next_timestep_half_(std::integral_constant<bool, particle_type::is_implicit>(), E, B);
//	}
//
//	std::string save(std::string const & path, bool is_verbose = false) const
//	{
//		return self_->save(path, is_verbose);
//	}
//
//private:
//
////	typename std::enable_if<!particle_type::is_implicit, typename mesh_type:: template field < EDGE, scalar_type> const&>::type J_() const
////	{
////		return self_->J;
////	}
////
////	typename std::enable_if<particle_type::is_implicit, typename mesh_type:: template field < EDGE, scalar_type> const&>::type J_() const
////	{
////		return dummy_J;
////	}
////
////	typename std::enable_if<particle_type::is_implicit, typename mesh_type:: template field < VERTEX, nTuple<3, scalar_type>> const&>::type Jv_() const
////	{
////		return self_->J;
////	}
////
////	typename std::enable_if<!particle_type::is_implicit, typename mesh_type:: template field < VERTEX, nTuple<3, scalar_type>> const&>::type Jv_() const
////	{
////		return dummy_Jv;
////	}
//
//	typename mesh_type:: template field<EDGE, scalar_type> const& J_(std::integral_constant<bool, false>) const
//	{
//		return self_->J;
//	}
//
//	typename mesh_type:: template field<EDGE, scalar_type> const& J_(std::integral_constant<bool, true>) const
//	{
//		return dummy_J;
//	}
//
//	typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const& Jv_(
//	        std::integral_constant<bool, true>) const
//	{
//		return self_->J;
//	}
//
//	typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const& Jv_(
//	        std::integral_constant<bool, false>) const
//	{
//		return dummy_Jv;
//	}
//	void next_timestep_zero_(std::integral_constant<bool, true>,
//	        typename mesh_type:: template field<EDGE, scalar_type> const & E,
//	        typename mesh_type:: template field<FACE, scalar_type> const & B)
//	{
//	}
//	void next_timestep_zero_(std::integral_constant<bool, false>,
//	        typename mesh_type:: template field<EDGE, scalar_type> const & E,
//	        typename mesh_type:: template field<FACE, scalar_type> const & B)
//	{
//		self_->next_timestep_zero(E, B);
//	}
//	void next_timestep_zero_(std::integral_constant<bool, true>,
//	        typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & E,
//	        typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & B)
//	{
//		self_->next_timestep_zero(E, B);
//	}
//	void next_timestep_zero_(std::integral_constant<bool, false>,
//	        typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & E,
//	        typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & B)
//	{
//	}
//
//	void next_timestep_half_(std::integral_constant<bool, false>,
//	        typename mesh_type:: template field<EDGE, scalar_type> const & E,
//	        typename mesh_type:: template field<FACE, scalar_type> const & B)
//	{
//		self_->next_timestep_half(E, B);
//	}
//	void next_timestep_half_(std::integral_constant<bool, true>,
//	        typename mesh_type:: template field<EDGE, scalar_type> const & E,
//	        typename mesh_type:: template field<FACE, scalar_type> const & B)
//	{
//	}
//	void next_timestep_half_(std::integral_constant<bool, false>,
//	        typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & E,
//	        typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & B)
//	{
//	}
//	void next_timestep_half_(std::integral_constant<bool, true>,
//	        typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & E,
//	        typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & B)
//	{
//		self_->next_timestep_half(E, B);
//	}
//
//	std::shared_ptr<particle_type> self_;
//
//	typename mesh_type:: template field<EDGE, scalar_type> dummy_J;
//	typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> dummy_Jv;
//
//};

}
// namespace simpla

#endif /* PARTICLE_BASE_H_ */
