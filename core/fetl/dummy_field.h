/*
 * dummy_field.h
 *
 *  created on: 2014-6-29
 *      Author: salmon
 */

#ifndef DUMMY_FIELD_H_
#define DUMMY_FIELD_H_
#include "../utilities/log.h"
namespace simpla
{

template<typename TMesh,  unsigned int  IFORM, typename TV>
class DummyField
{
public:
	static constexpr   unsigned int   IForm = IFORM;

	typedef TMesh mesh_type;

	typedef TV value_type;

	typedef DummyField<mesh_type, IForm, value_type> this_type;

	static const  unsigned int  NDIMS = mesh_type::NDIMS;

	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef typename mesh_type::index_type index_type;

	typedef typename mesh_type::iterator mesh_iterator;

	typedef typename std::conditional<(IForm == VERTEX || IForm == VOLUME),  //
	        value_type, nTuple<NDIMS, value_type> >::type field_value_type;

	typedef std::map<index_type, value_type> container_type;

	container_type data_;

	mesh_type const &mesh;

	value_type default_value_;

	DummyField(mesh_type const &pmesh, value_type d = value_type()) :
			mesh(pmesh), default_value_(d)
	{
	}

	DummyField(this_type const & rhs) :
			mesh(rhs.mesh), default_value_(rhs.default_value_)
	{
	}

	/// Move Construct copy mesh, and move data,
	DummyField(this_type &&rhs) :
			mesh(rhs.mesh), data_(rhs.data_), default_value_(rhs.default_value_)
	{
	}

	~DummyField()
	{
	}

	void print() const
	{
		for (auto const & v : data_)
		{
			CHECK_BIT(v.first)<<v.second;
		}
	}

	template<typename TVistor>
	void Accept(TVistor const & visitor)
	{
		visitor.Visit(this);
	}

	void swap(this_type & rhs)
	{
		ASSERT(mesh == rhs.mesh);
		std::swap(data_, rhs.data_);
	}

	void init()
	{
	}

	template<typename ...Args>
	int get_dataset_shape(Args &&...others) const
	{
		return mesh.get_dataset_shape(IForm, std::forward<Args>(others)...);
	}

	container_type & data()
	{
		return data_;
	}

	const container_type & data() const
	{
		return data_;
	}
	size_t size() const
	{
		return mesh.get_num_of_elements(IForm);
	}
	bool empty() const
	{
		return data_ == nullptr;
	}

	void lock()
	{
		UNIMPLEMENTED;
	}
	void unlock()
	{
		UNIMPLEMENTED;
	}

	value_type & at(index_type s)
	{
		return get(s);
	}
	value_type const & at(index_type s) const
	{
		return get(s);
	}
	value_type & operator[](index_type s)
	{
		return get(s);
	}
	value_type & operator[](index_type s) const
	{
		return get(s);
	}
	inline value_type & get(index_type s)
	{
		value_type res;
		auto it = data_.find(s);
		if (it == data_.end())
		{
			data_[s] = default_value_;
		}

		return data_[s];
	}

	inline value_type const & get(index_type s) const
	{
		value_type res;
		auto it = data_.find(s);
		if (it == data_.end())
		{
			return default_value_;
		}
		else
		{
			return res = it->second;
		}
	}

	auto select() DECL_RET_TYPE((make_mapped_range( *this, mesh.select(IForm ))))
	auto select() const DECL_RET_TYPE((make_mapped_range( *this, mesh.select(IForm ))))

	template<typename ... Args>
	auto select(Args &&... args)
	DECL_RET_TYPE((make_mapped_range( *this, mesh.select(IForm,std::forward<Args>(args)...))))
	template<typename ... Args>
	auto select(Args &&... args) const
	DECL_RET_TYPE((make_mapped_range( *this, mesh.select(IForm,std::forward<Args>(args)...))))

	auto begin() DECL_RET_TYPE(simpla::begin(this->select()))
	auto begin() const DECL_RET_TYPE(simpla::begin(this->select()))
	auto end() DECL_RET_TYPE(simpla::end(this->select()))
	auto end() const DECL_RET_TYPE(simpla::end(this->select()))

	template<typename TD>
	void fill(TD default_value)
	{
		UNIMPLEMENTED;
	}

	void Clear()
	{
		data_.clear();
	}

	this_type & operator =(value_type rhs)
	{
		UNIMPLEMENTED;
		return (*this);
	}
	this_type & operator =(this_type const & rhs)
	{
		UNIMPLEMENTED;

		return (*this);
	}
	template<typename TR>
	this_type & operator =(TR const & rhs)
	{
		UNIMPLEMENTED;
		return (*this);
	}

	inline field_value_type operator()(coordinates_type const &x) const
	{
		return mesh.Gather(std::integral_constant<unsigned int ,IForm>(),*this,x);
	}

	template<typename TZ>
	inline void add(coordinates_type const &x,TZ const & z)
	{
		return mesh.Scatter(std::integral_constant<unsigned int ,IForm>(),this,z);
	}

};

}
  // namespace simpla

#endif /* DUMMY_FIELD_H_ */
