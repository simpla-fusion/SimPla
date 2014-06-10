/*
 * particle_boundary.h
 *
 *  Created on: 2014年4月24日
 *      Author: salmon
 */

#ifndef PARTICLE_BOUNDARY_H_
#define PARTICLE_BOUNDARY_H_

#include <map>

#include "../fetl/ntuple.h"
#include "../utilities/visitor.h"
#include "../model/geometric_algorithm.h"

namespace simpla
{
template<typename > class BoundaryCondition;
template<typename > class Particle;

template<typename Engine>
class BoundaryCondition<Particle<Engine> > : public VisitorBase
{
public:

	typedef Particle<Engine> particle_type;

	typedef BoundaryCondition<particle_type> this_type;

	typedef typename particle_type::mesh_type mesh_type;

	typedef typename mesh_type::iterator iterator;

	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef typename mesh_type::scalar_type scalar_type;

	typedef nTuple<3, coordinates_type> plane_type;

private:
	typename mesh_type::surface_type surface_;

	std::string op_str_;

	std::function<void(plane_type const&, coordinates_type *, nTuple<3, Real>*)> op_;
public:

	template<typename TDict, typename ...Others>
	BoundaryCondition(TDict const & dict, Others const &...);

	virtual ~BoundaryCondition();

	template<typename ...Others>
	static std::function<void()> Create(particle_type* f, Others const & ...);

//	template<typename ... Others>
//	static std::function<void(particle_type*)> Create(Others const & ...);

	void Visit(particle_type * p) const;

private:
	void Visit_(void * pp) const
	{
		Visit(reinterpret_cast<particle_type *>(pp));
	}

}
;

template<typename Engine>
template<typename TDict, typename ...Others>
BoundaryCondition<Particle<Engine> >::BoundaryCondition(TDict const & dict, Others const & ... others)
		: op_str_("")
{
//	mesh_type const & mesh = model.mesh;
//
//	CreateSurface(dict["Select"], model, &surface_);
//
//	if (dict["Operation"].is_string())
//	{
//		op_str_ = dict["Operation"].template as<std::string>();
//	}
//	else if (dict["Operation"].is_function())
//	{
//		auto obj = dict["Operation"];
//
//		op_str_ = "Custom";
//
//		dict["Operation"].as(&op_);
//
////			op_ = [obj](plane_type const& plane, scalar_type f, coordinates_type const& x , nTuple<3, Real> const &v)
////			{
////				obj()
////			}
//	}
}

template<typename Engine>
BoundaryCondition<Particle<Engine>>::~BoundaryCondition()
{
}

template<typename Engine>
template<typename ... Others>
std::function<void()> BoundaryCondition<Particle<Engine>>::Create(particle_type* f, Others const & ...others)
{

	return std::bind(&this_type::Visit,
	        std::shared_ptr<this_type>(new this_type(std::forward<Others const &>(others)...)), f);
}
//template<typename Engine>
//template<typename ... Others>
//std::function<void(Particle<Engine>*)> BoundaryCondition<Particle<Engine >>::Create(
//		Others const & ...others)
//{
//
//	return std::bind(&this_type::Visit,
//			std::shared_ptr<this_type>(
//					new this_type(std::forward<Others const &>(others)...)),
//			std::placeholders::_1);
//}

template<typename Engine>
void BoundaryCondition<Particle<Engine>>::Visit(particle_type * p) const
{

//	if (op_str_ == "Cycling")
//		return;
//
//	LOGGER << "Apply boundary constraint [" << op_str_ << "] to particles [" << p->GetTypeAsString() << "]";
//
//	for (auto const & cell : surface_)
//	{
//		auto const &plane = cell.second;
//
//		if (op_str_ == "Refelecting")
//		{
//			p->Modify(cell.first, [&plane](coordinates_type *x, nTuple<3, Real>*v)
//			{
//				Reflect(plane,x,v);
//			});
//		}
//		else if (op_str_ == "Absorbing")
//		{
//			p->Remove(cell.first,
//
//			[&](coordinates_type const & x, nTuple<3, Real> const & v)->bool
//			{
//				return Distance(plane, x )<0;
//			}
//
//			);
//		}
//		else if (op_str_ == "Custom")
//		{
//			UNIMPLEMENT;
//			return;
////				auto foo = [=](scalar_type f,coordinates_type const &x, nTuple<3, Real>const &v)
////				{	op_(f,x,v);};
////				p->Traversal(cell.first, foo);
//		}
//	}

}

//
//enum
//{
//	REFELECT, ABSORB
//};
//template<class Engine, typename TMediaTag>
//void Boundary(int flag, TMediaTag const &tag, typename TMediaTag::tag_type in, typename TMediaTag::tag_type out,
//		Particle<Engine> * self, Particle<Engine> * other = nullptr)
//{
//	DEFINE_FIELDS(typename Engine::mesh_type);
//
//	if (other == nullptr)
//		other = self;
//
//	tag.SelectBoundaryCell(Int2Type<0>(),
//
//	[self,other](iterator src)
//	{
//
//		auto & cell = (*self)[src];
//
//		auto pt = cell.begin();
//
//		while (pt != cell.end())
//		{
//			auto p = pt;
//			++pt;
//
//			iterator dest=src;
//			if (flag == REFELECT)
//			{
//				coordinates_type x;
//
//				nTuple<3,Real> v;
//
//				Engine::InvertTrans(p,&x,&v);
//
//				dest=self->mesh.Refelect(src,&x,&v);
//
//				Engine::Trans(x,v,&p);
//			}
//
//			if (dest != src)
//			{
//				other->data_[dest].splice(other->data_[dest].begin(), cell, p);
//			}
//			else
//			{
//				cell.erase(p);
//			}
//
//		}
//
//	}, in, out, TMediaTag::ON_BOUNDARY);
//
//}

}// namespace simpla

#endif /* PARTICLE_BOUNDARY_H_ */
