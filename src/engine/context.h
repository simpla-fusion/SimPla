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

	DEFINE_FIELDS(TM)

	mesh_type mesh;

	Form<1> E;
	Form<1> J;
	Form<2> B;

	std::function<void(Form<1>&, Form<2>&, Form<1> const &, Real)> field_solver;

	template<typename TConfig>
	Context<TM>::Context(TConfig const & config);

	~Context();

private:

}
;

template<typename TM>
template<typename TConfig>
Context<TM>::Context(TConfig const & cfg) :
		E(mesh), J(mesh), B(mesh)
{
	mesh.Deserialize(cfg.GetChild("Grid"));
}

template<typename TM>
Context<TM>::~Context()
{
}

}
// namespace simpla
#endif /* DOMAIN_H_ */
