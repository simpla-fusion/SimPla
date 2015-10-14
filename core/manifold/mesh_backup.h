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


template<typename TopologyPolicy, typename GeometryPolicy, typename CalculatePolicy, typename TimePolicy>
class Mesh<TopologyPolicy, GeometryPolicy, CalculatePolicy, TimePolicy>
		: public TopologyPolicy, public GeometryPolicy, public CalculatePolicy, public TimePolicy
{
public:

	typedef Mesh<TopologyPolicy, GeometryPolicy, CalculatePolicy, TimePolicy> this_type;

	static constexpr size_t ndims = 3;


//***************************************************************************************************

	Mesh()
	{
	}

	~Mesh()
	{
	}

	Mesh(this_type const &other) :
			TopologyPolicy(other), GeometryPolicy(other), CalculatePolicy(other), TimePolicy(other)
	{
	}

	void swap(this_type &other)
	{
		TopologyPolicy::swap(other);
		GeometryPolicy::swap(other);
		CalculatePolicy::swap(other);
		TimePolicy::swap(other);
	}

	this_type &operator=(const this_type &other)
	{
		this_type(other).swap(*this);
		return *this;
	}


	template<typename TDict>
	void load(TDict const &dict)
	{
		TopologyPolicy::load(dict);
		GeometryPolicy::load(dict);
		CalculatePolicy::load(dict);
		TimePolicy::load(dict);

		INFORM << "Mesh.load" << std::endl;
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


	template<int I, typename ...Args>
	auto sample(Args &&...args) const
	DECL_RET_TYPE((CalculatePolicy::template sample<I>(*this, std::forward<Args>(args) ...)))


	template<typename TF, typename ...Args>
	auto gather(TF const &field, Args &&...args) const
	DECL_RET_TYPE((CalculatePolicy::gather(*this, field, std::forward<Args>(args)...)))


	template<typename ...Args>
	auto calculate(Args &&...args) const
	DECL_RET_TYPE((CalculatePolicy::calculate(*this, std::forward<Args>(args)...)))

	template<typename ...Args>
	void scatter(Args &&...args) const
	{
		CalculatePolicy::scatter(*this, std::forward<Args>(args)...);
	}

	//! @}

}; //class Mesh


}//namespace simpla

#endif /* CORE_MESH_MESH_H_ */
