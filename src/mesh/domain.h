/*
 * domain.h
 *
 *  Created on: 2013年12月13日
 *      Author: salmon
 */

#ifndef DOMAIN_H_
#define DOMAIN_H_

#include <vector>

#include "../fetl/primitives.h"

namespace simpla
{
template<typename TM>
class Domain
{

public:
	static constexpr int MAX_NUM_OF_MEIDA_TYPE = 64;
	typedef TM mesh_type;
	mesh_type const &mesh;
	typedef std::bitset<MAX_NUM_OF_MEIDA_TYPE> tag_type;
	typedef typename mesh_type::index_type index_type;
	typedef typename mesh_type::coordinates_type coordinates_type;
private:
	typename mesh_type::template Container<tag_type> tags_[mesh_type::NUM_OF_COMPONENT_TYPE];

public:

	Domain(mesh_type const & m)
			: mesh(m), tags_(mesh.MakeContainer<0, tag_type>(0))
	{
	}
	~Domain()
	{
	}

	/**
	 * Set media tag on vertics
	 * @param tag media tag is  set to 1<<tag
	 * @param args args are trans-forward to
	 *      SelectVerticsInRegion(<lambda function>,*this,args)
	 */
	template<typename ...Args>
	void SetMediaTagOnVertics(unsigned int tag_pos, Args const & ... args)
	{
		if (tags_[0].empty())
			tags_[0].resize(mesh.GetNumOfElements(0), 0);

		SelectVerticsInRegion(

		[&](index_type const &s,coordinates_type const &x)
		{
			tags_[0][s].set(tag_pos);

		}, *this, this, std::forward<Args>(args)...);
	}

	/**
	 *  Update media tag on edge ,face and cell, base on media tag on vertics
	 */
	void UpdateMediaTags()
	{
		_UpdateMediaTags<1>();
		_UpdateMediaTags<2>();
		_UpdateMediaTags<3>();
	}
	enum
	{
		IN_INTERFACE, OUT_INTERFACE, THROUGH_INTERFACE
	};
	template<int IFORM, typename TFUN>
	void ForEachElementOnInterface(TFUN const &fun, int type = THROUGH_INTERFACE)
	{

	}

private:
	template<int I>
	void _UpdateMediaTags()
	{
		if (tags_[I].empty())
			tags_[I].resize(mesh.GetNumOfElements(I), 0);

		mesh.TraversalIndex(I,

		[&](int m, index_type const & s )
		{
			index_type v[mesh_type::MAX_NUM_VERTEX_PER_CEL];

			int n=GetVerticesOfCell<I>(v,m,s);
			tag_type flag = 0;
			for(int i=0;i<n;++i)
			{
				flag!=tags_[0][v[i]];
			}
			tags_[I][mesh.GetComponentIndex<I>(m,s)]=flag;

		}, mesh_type::DO_PARALLEL);
	}

	template<typename TFUN>
	void _ForEachElementOnInterface(Int2Type<0>, std::function<void(index_type)> const &fun, unsigned int in_tag,
	        unsigned int out_tag, int flag)
	{
		mesh.TraversalIndex(0,

		[&](index_type s)
		{
			if(tags_[3][meshs].count()>1)
			{
				fun(mesh.GetComponentIndex(0,m,s));
			}
		},

		mesh_type::DO_PARALLEL)

	}

};

}  // namespace simpla

#endif /* DOMAIN_H_ */
