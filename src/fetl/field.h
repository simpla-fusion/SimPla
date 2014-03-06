/*
 * field.h
 *
 *  Created on: 2013-7-19
 *      Author: salmon
 */

#ifndef FIELD_H_
#define FIELD_H_

#include "primitives.h"
#include "../utilities/log.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <utility>
#include <mutex>
namespace simpla
{
template<typename TG, int IFORM, typename TValue> struct Field;

/***
 *
 * @brief Field
 *
 * @ingroup Field Expression
 *
 */

template<typename TM, int IFORM, typename TValue>
struct Field
{
	std::mutex write_lock_;
public:

	typedef TM mesh_type;

	enum
	{
		IForm = IFORM
	};

	typedef TValue value_type;

	typedef Field<mesh_type, IForm, value_type> this_type;

	static const int NDIMS = mesh_type::NDIMS;

	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef typename mesh_type::index_type index_type;

	typedef typename Geometry<mesh_type, IForm>::template field_value_type<value_type> field_value_type;

	typedef std::shared_ptr<value_type> container_type;
private:
	container_type data_;
public:

	mesh_type const &mesh;

	Field(mesh_type const &pmesh)
			: mesh(pmesh), data_(nullptr)
	{
	}

	Field(mesh_type const &pmesh, value_type d_value)
			: mesh(pmesh), data_(nullptr)
	{
		*this = d_value;
	}
	/**
	 *  Copy/clone Construct only copy mesh reference, but do not copy/move data, which is designed to
	 *  initializie stl containers, such as std::vector
	 *    \code
	 *       Field<...> default_value(mesh);
	 *       std::vector<Field<...> > v(4,default_value);
	 *    \endcode
	 *  the element in v have same mesh reference.
	 *
	 * @param rhs
	 */

	Field(this_type const & rhs)
			: mesh(rhs.mesh), data_(nullptr)
	{
	}

	/// Move Construct copy mesh, and move data,
	Field(this_type &&rhs)
			: mesh(rhs.mesh), data_(rhs.data_)
	{
	}

	~Field()
	{
	}

	void swap(this_type & rhs)
	{
		ASSERT(mesh==rhs.mesh);

		std::swap(data_, rhs.data_);
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
		return (data_ == nullptr) ? 0 : mesh.GetNumOfElements(IForm);
	}
	bool empty() const
	{
		return size() <= 0;
	}

	template<typename TC>
	struct iterator_
	{
		TC data_;

		typename mesh_type::template iterator<IForm> it_;

		typedef decltype(*data_) value_type;

		typedef iterator_<TC> this_type;

		iterator_(TC d, typename mesh_type::template iterator<IForm> s)
				: data_(d), it_(s)
		{

		}
		~iterator_()
		{
		}

		value_type & operator*()
		{
			return *(data_.get() + it_.Hash());
		}
		value_type const& operator*() const
		{
			return *(data_.get() + it_.Hash());
		}
		this_type & operator++()
		{
			++it_;
			return *this;
		}

		bool operator!=(this_type const &rhs) const
		{
			return it_ != rhs.it_ && data_ != rhs.data_;
		}
	};

	typedef iterator_<container_type> iterator;

	typedef iterator_<const container_type> const_iterator;

	iterator begin()
	{
		return iterator_<container_type>(data_, mesh.template begin<IForm>());
	}

	iterator end()
	{
		return iterator_<container_type>(data_, mesh.template end<IForm>());
	}

	const_iterator begin() const
	{
		return iterator_<const container_type>(data_, mesh.template begin<IForm>());
	}

	const_iterator end() const
	{
		return iterator_<const container_type>(data_, mesh.template end<IForm>());
	}

	inline value_type & operator[](index_type s)
	{
		return get(s);
	}

	inline value_type const & operator[](index_type s) const
	{
		return get(s);
	}

	inline value_type & at(index_type s)
	{
		return get(s);
	}

	inline value_type const & at(index_type s) const
	{
		return get(s);
	}

	inline value_type & get(index_type s)
	{
		return *(data_.get() + mesh.template Hash<IForm>(s));
	}

	inline value_type const & get(index_type s) const
	{
		return *(data_.get() + mesh.template Hash<IForm>(s));
	}
	void Init()
	{
		Update();
	}
	void Update()
	{
		if (data_ == nullptr)
		{
			data_ = mesh.template MakeContainer<IForm, value_type>();
		}

	}

	template<typename TD>
	void Fill(TD default_value)
	{
		Init();
		for (auto it = this->begin(), ie = this->end(); it != ie; ++it)
		{
			*it = default_value;
		}
	}

	void Clear()
	{
		Init();

		std::fill_n(reinterpret_cast<char*>(data_.get()), size() * sizeof(value_type), 0);

	}

	template<typename TR> inline this_type &
	operator =(Field<mesh_type, IForm, TR> const & rhs)
	{
		Init();
		mesh.template Traversal<IForm>([&](index_type s)
		{
			this->get(s)=rhs.get(s);
		});
		return (*this);
	}

	template<typename TR> inline this_type &
	operator =(TR const & rhs)
	{
		Init();
		mesh.template Traversal<IForm>([&](index_type s)
		{
			this->get(s)=rhs;
		});
		return (*this);
	}

#define DECL_SELF_ASSIGN( _OP_ )                                                                   \
		template<typename TR> inline this_type &                                                   \
		operator _OP_##= (TR const & rhs)                                                          \
		{	 *this = *this _OP_ rhs;                                                      \
			return (*this) ;                                                                        \
		}                                                                                          \


	DECL_SELF_ASSIGN(+ )

DECL_SELF_ASSIGN	(- )

	DECL_SELF_ASSIGN(* )

	DECL_SELF_ASSIGN(/ )
#undef DECL_SELF_ASSIGN

	inline field_value_type mean(coordinates_type const &x) const
	{
		return Gather(x);
	}

	inline field_value_type operator()(coordinates_type const &x) const
	{
		return Gather(x);
	}

	inline field_value_type operator()(index_type s, Real const *pcoords) const
	{
		return Gather(s, pcoords);
	}

	inline field_value_type Gather(coordinates_type const &x) const
	{

		coordinates_type pcoords;

		index_type s = mesh.SearchCell(x, &pcoords[0]);

		return Gather(s, &pcoords[0]);

	}

	inline field_value_type Gather(index_type const & s, Real const *pcoords) const
	{

		field_value_type res;

		index_type num = mesh.GetAffectedPoints(Int2Type<IForm>(), s);

		std::vector<index_type> points(num);

		std::vector<value_type> cache(num);

		mesh.GetAffectedPoints(Int2Type<IForm>(), s, &points[0]);

		for (int i = 0; i < num; ++i)
		{
			cache[i] = mesh.get_value(data_, points[i]);
		}

		res *= 0;

		mesh.Gather(Int2Type<IForm>(), pcoords, &cache[0], &res);

		return res;

	}

	template<typename TV>
	inline void Collect(TV const & v, coordinates_type const &x)
	{
		coordinates_type pcoords;

		index_type s = mesh.SearchCell(x, &pcoords[0]);

		Collect(v, s, &pcoords[0]);

	}
	template<typename TV>
	inline void Collect(TV const & v, index_type const & s, Real * pcoords, int affected_region = 2)
	{

		index_type num = mesh.GetAffectedPoints(Int2Type<IForm>(), s);

		if (num == 0)
		{
			CHECK(s );
			return;
		}

		index_type *points = new index_type[num];
		value_type *cache = new value_type[num];

		mesh.GetAffectedPoints(Int2Type<IForm>(), s, &points[0]);

		value_type zero_value;

		zero_value *= 0;

		for (int i = 0; i < num; ++i)
		{
			cache[i] = zero_value;
		}

		field_value_type vv;
		vv = v;

		mesh.Scatter(Int2Type<IForm>(), pcoords, vv, cache);

		Collect(num, points, cache);

		delete[] points;
		delete[] cache;
	}

	inline void Collect(index_type num, index_type const * points, value_type const * cache)
	{
		if (num == 0)
		WARNING << "Cache is empty!";

		write_lock_.lock();
		for (int i = 0; i < num; ++i)
		{
			mesh.get_value(data_, points[i]) += cache[i];
		}
		write_lock_.unlock();
	}
}
;

}
// namespace simpla

#endif /* FIELD_H_ */
