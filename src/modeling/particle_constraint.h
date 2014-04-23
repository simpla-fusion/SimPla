/*
 * particle_constraint.h
 *
 *  Created on: 2014年4月21日
 *      Author: salmon
 */

#ifndef PARTICLE_CONSTRAINT_H_
#define PARTICLE_CONSTRAINT_H_
#include "material.h"
#include "surface.h"
namespace simpla
{

template<typename TEngine>
class Constraint<Particle<TEngine>>
{
public:

	static constexpr unsigned int IForm = Particle<TEngine>::IForm;

	typedef Particle<TEngine> field_type;

	typedef typename Particle<TEngine>::mesh_type mesh_type;

	typedef typename Particle<TEngine>::value_type value_type;

	typedef typename mesh_type::index_type index_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	mesh_type const & mesh;

private:

	std::list<index_type> def_domain_;
public:
	std::function<value_type(Real, coordinates_type, value_type const &)> op_;

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

	void Apply(field_type * f)
	{
		for (auto s : def_domain_)
		{
			auto x = mesh.GetCoordinates(s);
			(*f)[s] = mesh.Sample(Int2Type<IForm>(), s, op_(mesh.GetTime(), x, (*f)(x)));
		}
	}

}
;

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

template<typename TEngine, typename TDict>
std::function<void(Particle<TEngine> *)> CreateParticleConstraint(
        Material<typename Particle<TEngine>::mesh_type> const & material, TDict const & dict)
{
	std::function<void(Particle<TEngine> *)> res;

	return std::move(res);
}
//template<typename TM, typename TDict>
//void CreateSurface(Surface<TM> *surf, TDict const &dict)
//{
//	if (dict["Width"].is_number())
//	{
//		CreateSurface(surf, dict["Width"].template as<Real>());
//	}
//	else
//	{
//		WARNING << "Unknown configuation!";
//	}
//}
//template<typename TM>
//void CreateSurface(Surface<TM> *surf, Real width)
//{
//	TM const & mesh = surf->mesh;
//
//	typedef typename Surface<TM>::plane_type plane_type;
//
//	auto extent = mesh.GetExtent();
//	auto dims = mesh.GetDimensions();
//	auto xmin = extent.first;
//	auto xmax = extent.second;
//	auto d = mesh.GetDx();
//	nTuple<3, Real> x0 = { 0, 0, 0 };
//	nTuple<3, Real> x1 = { d[0], 0, 0 };
//	nTuple<3, Real> x2 = { 0, d[1], 0 };
//	nTuple<3, Real> x3 = { 0, 0, d[2] };
//
//	for (auto s : mesh.GetRange(VERTEX))
//	{
//		auto x = mesh.GetCoordinates(s);
//
//		if (x[0] < xmin[0] + width)
//		{
//			surf->insert(s, plane_type( { x0, x1, x2 }));
//			continue;
//		}
//		else if (x[0] > xmax[0] - width)
//		{
//			surf->insert(s, plane_type( { x0, x2, x1 }));
//			continue;
//		}
//
//		if (x[1] < xmin[1] + width)
//		{
//			surf->insert(s, plane_type( { x0, x1, x2 }));
//			continue;
//		}
//		else if (x[1] > xmax[1] + width)
//		{
//			surf->insert(s, plane_type( { x0, x1, x2 }));
//			continue;
//		}
//
//		if (x[2] < xmin[2] + width)
//		{
//			surf->insert(s, plane_type( { x0, x1, x2 }));
//			continue;
//		}
//		else if (x[2] > xmax[2] - width)
//		{
//			surf->insert(s, plane_type( { x0, x1, x2 }));
//			continue;
//		}
//
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
