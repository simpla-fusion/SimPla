/**
 * @file mesh.h
 *
 * @date 2015-2-9
 * @author salmon
 */

#ifndef CORE_MESH_MESH_H_
#define CORE_MESH_MESH_H_

#include <iostream>
#include <memory>


namespace simpla
{

template<typename ...> class Domain;

template<typename ...> class Field;

template<typename ...> class Mesh;

namespace policy
{
template<typename ...> class calculate;

template<typename ...> class interpolator;
}

template<typename TGeometry, typename CalculateTag, typename InterpolatorTag, typename ...Others>
class Mesh<TGeometry, CalculateTag, InterpolatorTag, Others ...>
		: public TGeometry, public policy::calculate<TGeometry, CalculateTag>,
				public policy::interpolator<TGeometry, InterpolatorTag>, public Others ...
{
public:

	typedef Mesh<TGeometry, CalculateTag, InterpolatorTag, Others ...> this_type;

	typedef TGeometry geometry_type;

	typedef policy::calculate<TGeometry, CalculateTag> calculate_policy;

	typedef policy::interpolator<TGeometry, InterpolatorTag> interpolator_policy;

	Mesh()
	{
	}

	virtual ~Mesh()
	{
	}

	Mesh(this_type const &other) :
			geometry_type(other)
	{
	}

	void swap(const this_type &other)
	{
		geometry_type::swap(other);
	}

	geometry_type const &geometry() const
	{
		return *this;
	}

	this_type &operator=(const this_type &other)
	{
		this_type(other).swap(*this);
		return *this;
	}


	template<typename OS>
	OS &print(OS &os) const
	{
		os << "Mesh<>" << std::endl;
		return os;

	}

	static std::string get_type_as_string()
	{
		return "Mesh< >";
	}


	template<typename ...Args>
	auto calculate(Args &&...args) const
	DECL_RET_TYPE((calculate_policy::eval(*this, std::forward<Args>(args)...)))


	template<int I, typename ...Args>
	auto sample(Args &&...args) const
	DECL_RET_TYPE((interpolator_policy::template sample<I>(*this, std::forward<Args>(args) ...)))


	template<typename ...Args>
	auto gather(Args &&...args) const
	DECL_RET_TYPE((interpolator_policy::gather(*this, std::forward<Args>(args)...)))


	template<typename ...Args>
	void scatter(Args &&...args) const
	{
		interpolator_policy::scatter(*this, std::forward<Args>(args)...);
	}

//! @}

}; //class Mesh


}//namespace simpla

#endif /* CORE_MESH_MESH_H_ */
