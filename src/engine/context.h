/*
 * Domain.h
 *
 *  Created on: 2012-10-9
 *      Author: salmon
 */

#ifndef DOMAIN_H_
#define DOMAIN_H_

#include "include/simpla_defs.h"
#include "fetl/fetl.h"
namespace simpla
{

template<typename TM>
class Context
{
public:
	typedef TM mesh_type;

	typedef Context<TM> this_type;

	template<int IFORM> using Form = Field<Geometry<TM,IFORM>,Real >;
	template<int IFORM> using VecForm = Field<Geometry<TM,IFORM>,nTuple<3,Real> >;
	template<int IFORM> using TensorForm = Field<Geometry<TM,IFORM>,nTuple<3,nTuple<3,Real> > >;

	std::map<size_t,
			std::function<
					void(Object const &, Form<1> const &, Form<2> const&, Real)>> method_dispatch_push_;

	mesh_type mesh_;

	Context();

	virtual ~Context();

	template<typename TOBJ> std::shared_ptr<TOBJ> CreateObject();

	template<typename TOBJ>
	std::shared_ptr<TOBJ> GetObject(std::string const & name = "");

private:

	Context & operator=(this_type const &);

	template<typename T>
	void Push(Object const & obj, Form<1> const &E, Form<2> const&B, Real dt)
	{
		method_dispatch_push_[obj.GetTypeIndex()](obj, E, B, dt);
	}

	template<typename T>
	void RegisterParticle()
	{
		method_dispatch_push_[typeid(T).hash_code()] =
				[](Object const & obj, Form<1> const &E, Form<2> const&B, Real dt)
				{
					obj.as<T>()->Push(E,B,dt);
				}

	}

}
;

}
// namespace simpla
#endif /* DOMAIN_H_ */
