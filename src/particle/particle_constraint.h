/*
 * particle_constraint.h
 *
 *  Created on: 2014年4月21日
 *      Author: salmon
 */

#ifndef PARTICLE_CONSTRAINT_H_
#define PARTICLE_CONSTRAINT_H_
#include "../utilities/visitor.h"

namespace simpla
{
template<typename > class Constraint;
template<typename > class Particle;
template<typename TEngine>
class Constraint<Particle<TEngine>> : public VisitorBase
{
public:

	static constexpr unsigned int IForm = Particle<TEngine>::IForm;

	typedef Particle<TEngine> field_type;

	typedef typename Particle<TEngine>::mesh_type mesh_type;

	typedef typename mesh_type::index_type index_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef typename Particle<TEngine>::value_type value_type;

	mesh_type const & mesh;

private:

	std::list<index_type> def_domain_;
public:
	std::function<void(coordinates_type *x, nTuple<3, Real> *v)> op_;

	Constraint(mesh_type const & m)
			: mesh(m)
	{
	}

	~Constraint()
	{
	}

	std::list<index_type> const &GetDefDomain() const
	{
		return def_domain_;
	}
	std::list<index_type> & GetDefDomain()
	{
		return def_domain_;
	}

	void Visit(void * pp) const
	{
		Particle<TEngine> & p = *reinterpret_cast<Particle<TEngine>*>(pp);

		for (auto s : def_domain_)
		{
			coordinates_type x;
			nTuple<3, Real> v;
			for (auto & point : p[s])
			{
				p.PullBack(point, &x, &v);
				op_(s.second, &x, &v);
				p.PushForward(x, v, &point);
			}
		}
	}

}
;

template<typename TM, typename TDict>
std::shared_ptr<VisitorBase> CreateParticleConstraint(TM const & mesh, TDict const & dict)
{
	std::shared_ptr<VisitorBase> res;

	return std::move(res);
}
//template<typename Engine>
//void Particle<Engine>::Boundary(Surface<mesh_type> const& surface, std::string const & type_str)
//{
//	if (type_str == "Absorb")
//	{
//		for (auto const &cell : surface)
//		{
//			pool_.splice(pool_.begin(), data_.at(mesh.Hash(cell.first)));
//		}
//
//	}
//	else if (type_str == "Reflect")
//	{
//
//		for (auto const &cell : surface)
//		{
//			auto const & plane = cell.second;
//
//			nTuple<3, Real> x, v;
//
//			for (auto & p : data_.at(mesh.Hash(cell.first)))
//			{
//				engine_type::PullBack(p, &x, &v);
//				Relection(plane, &x, &v);
//				engine_type::PushForward(x, v, &p);
//			}
//		}
//		for (auto const &cell : surface)
//		{
//			Resort(cell.first, &data_);
//		}
//	}
//	else
//	{
//		UNIMPLEMENT2("Unknown particle boundary type [" + type_str + "]!");
//	}
//}

//
//template<typename TM, typename TDict>
//std::function<void(std::string const &, std::shared_ptr<ParticleBase<TM>>)> CreateParticleConstraint(
//        Material<TM> const & material, TDict const & dict)
//{
//	std::function<void(std::string const &, std::shared_ptr<ParticleBase<TM>>)> res =
//	        [](std::string const &, std::shared_ptr<ParticleBase<TM>>)
//	        {
//		        WARNING<<"Nothing to do!";
//	        };
//
//	typedef TM mesh_type;
//
//	mesh_type const & mesh = material.mesh;
//
//	typedef typename mesh_type::index_type index_type;
//
//	typedef typename mesh_type::coordinates_type coordinates_type;
//
//	std::shared_ptr<Surface<TM>> surface(new Surface<TM>(mesh));
//
//	auto particle_name = dict["Name"].template as<std::string>("All");
//
//	auto op_str = dict["Type"].template as<std::string>("Absorb");
//
//	CreateSurface(surface.get(), dict["Select"]);
//
//	return [=](std::string const & name, std::shared_ptr<ParticleBase<TM>> p)
//	{
//		if(particle_name=="All"|| particle_name==name)
//		{
//			LOGGER<< "Particle " << particle_name<<" is  "<< op_str<<"ed on the boundary";
//			p->Boundary(*surface,op_str);
//		}
//	};;
//}

}// namespace simpla

#endif /* PARTICLE_CONSTRAINT_H_ */
