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

template<typename ...> class _Field;


template<typename ...Policies>
class Mesh : public Policies ...
{
public:

	typedef Mesh<Policies ...> this_type;

	static constexpr size_t ndims = 3;


//***************************************************************************************************

	Mesh()
	{
	}

	~Mesh()
	{
	}

	Mesh(this_type const &other)
	{
	}

	void swap(this_type &other)
	{
	}

	this_type &operator=(const this_type &other)
	{
		this_type(other).swap(*this);
		return *this;
	}


	template<typename TDict>
	void load(TDict const &dict)
	{

		INFORM << "Mesh.load" << std::endl;

//		dimensions(dict["Dimensions"].as(index_tuple({10, 10, 10})));
//
//		extents(dict["Box"].template as<std::tuple<point_type, point_type> >());
//
//		dt(dict["dt"].template as<Real>(1.0));
	}

	template<typename OS>
	OS &print(OS &os) const
	{
		os << " This is a mock mesh " << std::endl;
		return os;

	}


	static std::string get_type_as_string()
	{
		return "Mesh< >";
	}


	template<int I, typename ...Args>
	auto sample(Args &&...args) const
	DECL_RET_TYPE((this->template proxy_sample<I>(*this, std::forward<Args>(args) ...)))


	template<typename TF, typename ...Args>
	auto gather(TF const &field, Args &&...args) const
	DECL_RET_TYPE((this->proxy_gather(*this, field, std::forward<Args>(args)...)))


	template<typename ...Args>
	auto calculate(Args &&...args) const
	DECL_RET_TYPE((this->proxy_calculate(*this, std::forward<Args>(args)...)))

	template<typename ...Args>
	void scatter(Args &&...args) const
	{
		this->proxy_scatter(*this, std::forward<Args>(args)...);
	}

	//! @}

}; //class Mesh


}//namespace simpla

#endif /* CORE_MESH_MESH_H_ */
