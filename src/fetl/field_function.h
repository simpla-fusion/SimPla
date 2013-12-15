/*
 * field_function.h
 *
 *  Created on: 2013年12月13日
 *      Author: salmon
 */

#ifndef FIELD_FUNCTION_H_
#define FIELD_FUNCTION_H_

#include <vector>
#include "primitives.h"
#include "field.h"
#include "../mesh/mesh_algorithm.h"

namespace simpla
{

template<typename TF, typename TFUN>
class FieldFunction
{
public:
	typedef TF field_type;
	static const int IForm = field_type::IForm;
	typedef typename TF::mesh_type mesh_type;
	typedef typename mesh_type::index_type index_type;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename field_type::field_value_type field_value_type;

	typedef TFUN fun_type;
private:

	std::vector<index_type> def_domain_;
	fun_type fun_;

public:

	FieldFunction()
	{
	}

	template<typename F>
	explicit FieldFunction(F const & f) :
			fun_( f )
	{
	}

	FieldFunction(fun_type const & fun, std::vector<index_type>const& idx) :
			def_domain_(idx), fun_(fun)
	{
	}
	FieldFunction(fun_type const & fun, std::vector<index_type>&& idx) :
			def_domain_(idx), fun_(fun)
	{
	}

	bool IsDefined() const
	{
		return def_domain_.empty();
	}

	inline void SetDefineDomain(mesh_type const & mesh,
			std::vector<coordinates_type>const& polyline)
	{
		SelectPointsInRegion(

		[&](index_type const & s, coordinates_type const &)
		{
			mesh.TraversalSubComponent(IForm,s, [&](index_type s)
					{	def_domain_.push_back(s);}
			);
		}, mesh, polyline);

	}
	inline std::vector<index_type> const & GetDefineDomain() const
	{
		return def_domain_;
	}

	inline void SetFunction(fun_type const &fun)
	{
		fun_ = fun;
	}

	inline fun_type const & GetFunction(fun_type const &fun) const
	{
		return fun_;
	}

	~FieldFunction()
	{
	}

	template<typename ...Args>
	void operator()(field_type* f, Args const & ...args)
	{

		for (auto const & s : def_domain_)
		{
			coordinates_type x = f->mesh.GetCoordinates(1, s);

			(*f)[s] = f->mesh.template GetWeightOnElement<IForm>(
					TypeCast<field_value_type>(
							fun_(x[0], x[1], x[2],
									std::forward<Args const&>(args)...)), s);
		}

	}

};

}  // namespace simpla

#endif /* FIELD_FUNCTION_H_ */
