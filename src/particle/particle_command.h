/*
 * particle_command.h
 *
 *  Created on: 2014年4月21日
 *      Author: salmon
 */

#ifndef PARTICLE_COMMAND_H_
#define PARTICLE_COMMAND_H_

#include <fetl/primitives.h>
#include <model/material.h>
#include <particle/particle.h>
//#include <utilities/log.h>
#include <utilities/type_utilites.h>
#include <utilities/visitor.h>
#include <functional>
#include <initializer_list>
#include <list>
//#include <string>

namespace simpla
{

template<typename > class Command;
template<typename > class Particle;

template<typename Engine>
class Command<Particle<Engine> > : public VisitorBase
{
public:

	typedef Particle<Engine> particle_type;

	typedef Command<particle_type> this_type;

	typedef typename particle_type::mesh_type mesh_type;

	typedef typename mesh_type::iterator iterator;

	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef std::list<iterator> define_domain_type;

	static constexpr unsigned int IForm = particle_type::IForm;

	typedef typename particle_type::value_type value_type;

	mesh_type const & mesh;

private:

	define_domain_type def_domain_;
public:
	std::function<
			field_value_type(Real, coordinates_type, field_value_type const &)> op_;

	template<typename TDict, typename TModel, typename ...Others>
	Command(TDict const & dict, TModel const &model, Others const & ...);

	template<typename ...Others>
	static std::function<void()> Create(field_type* f, Others const & ...);

	template<typename ... Others>
	std::function<void(field_type*)> Create(Others const & ...);

	void Visit(field_type * f) const
	{
		// NOTE this is a danger opertaion , no type check

		for (auto s : def_domain_)
		{
			auto x = mesh.GetCoordinates(s);

			(*f)[s] = mesh.Sample(Int2Type<IForm>(), s,
					op_(mesh.GetTime(), x, (*f)(x)));
		}
	}

private:
	void Visit_(void * pf) const
	{
		Visit(reinterpret_cast<field_type*>(pf));
	}
}

template<typename TP, typename TDict>
void CreateParticleConstraint(Model<typename TP::mesh_type> const & model_,
		TDict const & dict, TP * p)
{

	CreateParticleConstriant<this_type>(dict["Constriants"], &commands_);

	return std::dynamic_pointer_cast<VisitorBase>(
			std::shared_ptr<ParticleBoundary<typename TM::mesh_type>>(
					new ParticleBoundary<typename TM::mesh_type>(material.mesh,
							dict)));
}

} // namespace simpla

#endif /* PARTICLE_COMMAND_H_ */
