/*
 * media_tag.h
 *
 *  Created on: 2013年12月15日
 *      Author: salmon
 */

#ifndef MEDIA_TAG_H_
#define MEDIA_TAG_H_

#include <algorithm>
#include <bitset>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../fetl/primitives.h"
#include "../utilities/log.h"
#include "../utilities/pretty_stream.h"
#include "pointinpolygen.h"
#include "select.h"
namespace simpla
{

template<typename TM>
class MediaTag
{

public:
	static constexpr int MAX_NUM_OF_MEIDA_TYPE = 64;
	typedef TM mesh_type;

	typedef std::bitset<MAX_NUM_OF_MEIDA_TYPE> tag_type;
	typedef typename mesh_type::index_type index_type;
	typedef typename mesh_type::coordinates_type coordinates_type;
private:

	const tag_type null_tag;

	mesh_type const &mesh;
	std::vector<tag_type> tags_[mesh_type::NUM_OF_COMPONENT_TYPE];
	std::map<std::string, tag_type> register_tag_;
	unsigned int max_tag_;
public:

	enum
	{
		NONE = 0, VACUUM = 1, PLASMA, CORE, BOUNDARY, PLATEAU, LIMTER,
		// @NOTE: add tags for different physical area or media
		CUSTOM = 20
	};

	MediaTag(mesh_type const & m)
			: null_tag(1 << NONE), mesh(m), max_tag_(CUSTOM + 1)
	{
		register_tag_.emplace("NONE", null_tag);

		register_tag_.emplace("Vacuum", tag_type(1 << VACUUM));
		register_tag_.emplace("Plasma", tag_type(1 << PLASMA));
		register_tag_.emplace("Core", tag_type(1 << CORE));
		register_tag_.emplace("Boundary", tag_type(1 << BOUNDARY));
		register_tag_.emplace("Plateau", tag_type(1 << PLATEAU));
		register_tag_.emplace("Limter", tag_type(1 << LIMTER));

	}
	~MediaTag()
	{
	}

	bool empty() const
	{
		return tags_[VERTEX].empty();
	}

	operator bool() const
	{
		return tags_[VERTEX].empty();
	}

	tag_type RegisterTag(std::string const & name)
	{
		tag_type res;
		if (register_tag_.find(name) != register_tag_.end())
		{
			res = register_tag_[name];
		}
		else if (max_tag_ < MAX_NUM_OF_MEIDA_TYPE)
		{
			res.set(max_tag_);
			++max_tag_;
		}
		else
		{
			ERROR << "Too much media Type";
		}
		return res;
	}

	unsigned int GetNumMediaType() const
	{
		return max_tag_;
	}
	tag_type GetTagFromNumber(unsigned int tag_pos) const
	{
		tag_type res;
		res.set(tag_pos);
		return std::move(res);
	}
	tag_type GetTagFromString(std::string const &name) const
	{
		return std::move(register_tag_.at(name));
	}
	tag_type GetTagFromString(std::string const &name)
	{
		return std::move(RegisterTag(name));
	}
	tag_type operator[](std::string const &name)
	{
		auto it = register_tag_.find(name);
		if (it != register_tag_.end())
		{
			RegisterTag(name);
		}

		return std::move(register_tag_.at(name));
	}

	void ClearAll()
	{
		for (auto &v : tags_[0])
		{
			v.reset();
		}

		Update();
	}

	template<typename TCfg>
	void Load(TCfg const & cfg)
	{
		if (cfg)
		{
			for (auto const & p : cfg)
			{
				Modify(p.second);
			}
		}

	}
	std::ostream & Save(std::ostream &os) const
	{

		os << "{ \n" << "\t -- register media type\n";

		for (auto const& p : register_tag_)
		{
			os << std::setw(10) << p.first << " = 0x" << std::hex << p.second.to_ulong() << std::dec << ", \n";
		}

//		<< Data(&tmp[0][0], "tag0", mesh.GetShape(0)) << ","
//
//		<< Data(&tmp[1][0], "tag1", mesh.GetShape(1)) << ","
//
//		<< Data(&tmp[2][0], "tag2", mesh.GetShape(2)) << ","
//
//		<< Data(&tmp[3][0], "tag3", mesh.GetShape(3)) << ",";

		os << " }\n"

		;
		return os;
	}

	void Init(int I = VERTEX)
	{
		if (tags_[I].empty())
		{
			tags_[I].resize(mesh.GetNumOfElements(I), null_tag);
		}
	}

	template<typename TCmd>
	void Modify(TCmd const& cmd)
	{
		std::string op = "";
		std::string type = "";

		cmd["Op"].template as<std::string>(&op);
		cmd["Type"].template as<std::string>(&type);

		if (type == "")
		{
			WARNING << "Illegal input! [ undefine type ]";
			return;
		}

		auto select = cmd["Select"];
		if (select.empty())
		{
			std::vector<coordinates_type> region;

			cmd["Region"].as(&region);

			if (op == "Set")
			{
				Set(type, region);
			}
			else if (op == "Remove")
			{
				Remove(type, region);
			}
			else if (op == "Add")
			{
				Add(type, region);
			}
		}
		else
		{
			if (op == "Set")
			{
				Set(type, select);
			}
			else if (op == "Remove")
			{
				Remove(type, select);
			}
			else if (op == "Add")
			{
				Add(type, select);
			}
		}

		LOGGER << op << " media " << type << DONE;
	}

	template<typename ...Args> inline
	void Set(std::string media_tag, Args const & ... args)
	{
		Set(GetTagFromString(media_tag), std::forward<Args const &>(args)...);
	}
	template<typename ...Args> inline
	void Set(unsigned int media_tag, Args const & ... args)
	{
		Set(GetTagFromNumber(media_tag), std::forward<Args const &>(args)...);
	}

	template<typename ...Args> inline
	void Add(std::string media_tag, Args const & ... args)
	{
		Add(GetTagFromString(media_tag), std::forward<Args const &>(args)...);
	}
	template<typename ...Args> inline
	void Add(unsigned int media_tag, Args const & ... args)
	{
		Add(GetTagFromNumber(media_tag), std::forward<Args const &>(args)...);
	}

	template<typename ...Args> inline
	void Remove(std::string media_tag, Args const & ... args)
	{
		Set(GetTagFromString(media_tag), std::forward<Args const &>(args)...);
	}
	template<typename ...Args> inline
	void Remove(unsigned int media_tag, Args const & ... args)
	{
		Set(GetTagFromNumber(media_tag), std::forward<Args const &>(args)...);
	}

	/**
	 * Set media tag on vertics
	 * @param tag media tag is  set to 1<<tag
	 * @param args args are trans-forward to
	 *      SelectVerticsInRegion(<lambda function>,mesh,args)
	 */
	template<typename ...Args>
	void Set(tag_type media_tag, Args const & ... args)
	{
		_ForEachVertics([&]( tag_type &v)
		{	v=media_tag;},

		std::forward<Args const &>(args)...);
	}

	template<typename ...Args>
	void Add(tag_type media_tag, Args const & ... args)
	{

		_ForEachVertics([&]( tag_type &v)
		{	v|=media_tag;},

		std::forward<Args const &>(args)...);
	}

	template<typename ...Args>
	void Remove(tag_type media_tag, Args const & ... args)
	{

		_ForEachVertics([&]( tag_type &v)
		{	v^=media_tag;},

		std::forward<Args const &>(args)...);
	}

	/**
	 *  Update media tag on edge ,face and cell, base on media tag on vertics
	 */
	void Update()
	{
		_UpdateTags<EDGE>();
		_UpdateTags<FACE>();
		_UpdateTags<VOLUME>();
	}

	/**
	 *  Choice elements that most close to and out of the interface,
	 *  No element cross interface.
	 * @param
	 * @param fun
	 * @param in_tag
	 * @param out_tag
	 * @param flag
	 */
	template<int IFORM, typename TEleList>
	void SelectBoundary(tag_type in, tag_type out, TEleList *ele_list) const;

	template<int IFORM, typename TEleList>
	void SelectElements(tag_type tag, TEleList *eles) const;

	template<int IFORM, typename TDict, typename TEleList>
	void Select(TDict const & dict, TEleList *eles) const;

private:

	/**
	 * Set media tag on vertics
	 * @param tag media tag is  set to 1<<tag
	 * @param args args are trans-forward to
	 *      SelectVerticsInRegion(<lambda function>,mesh,args)
	 */
	template<typename ...Args>
	void _ForEachVertics(std::function<void(tag_type&)> fun, Args const & ... args)
	{
		Init();

		SelectFromMesh<VERTEX>(mesh, [&]( index_type const &s,coordinates_type const & )
		{	fun( tags_[VERTEX][s]);}, std::forward<Args const&>(args)...);
	}

	void _ForEaceVericsInRegion(std::function<void(bool, index_type const &)> const & op,
	        std::vector<coordinates_type> const & points);

	void _ForEaceVericsInRegion(std::function<void(bool, index_type const &)> const & op,
	        std::vector<nTuple<2, Real>> const & points, unsigned int Z = 2);

	void _ForEaceVericsInRegion(std::function<void(bool, index_type const &)> const & op,
	        std::function<bool(index_type, coordinates_type const &)> const & select);

	void _ForEaceVericsInRegion(std::function<void(bool, index_type const &)> const & op,
	        std::function<bool(index_type)> const & select);

	void _ForEaceVericsInRegion(std::function<void(bool, index_type const &)> const & op,
	        std::function<bool(coordinates_type const &)> const & select);

	void _ForEaceVericsInRegion(std::function<void(bool, index_type const &)> const & op, LuaObject const & select);

	template<int I>
	void _UpdateTags()
	{
		Init(I);

		mesh.ParallelTraversal(I,

		[&](index_type const & s )
		{
			index_type v[mesh_type::MAX_NUM_VERTEX_PER_CEL];

			int n=mesh.template GetNeighbourCell(Int2Type<I>(),Int2Type<VERTEX>(),v,s);
			tag_type flag = null_tag;
			for(int i=0;i<n;++i)
			{
				flag|=tags_[VERTEX][v[i]];
			}
			tags_[I][s]=flag;

		});
	}
};
template<typename TM>
inline std::ostream & operator<<(std::ostream & os, MediaTag<TM> const &self)
{
	return self.Save(os);
}

template<typename TM> template<int IFORM, typename TEleList>
void MediaTag<TM>::SelectBoundary(tag_type in, tag_type out, TEleList *ele_list) const
{

	// Good
	//  +----------#----------+
	//  |          #          |
	//  |    A     #-> B   C  |
	//  |          #          |
	//  +----------#----------+
	//
	//  +--------------------+
	//  |         ^          |
	//  |       B |     C    |
	//  |     ########       |
	//  |     #      #       |
	//  |     #  A   #       |
	//  |     #      #       |
	//  |     ########       |
	//  +--------------------+
	//
	//             +----------+
	//             |      C   |
	//  +----------######     |
	//  |          | A  #     |
	//  |    A     | &  #  B  |
	//  |          | B  #->   |
	//  +----------######     |
	//             |          |
	//             +----------+
	//
	//     	       +----------+
	//       C     |          |
	//  +----------#----+     |
	//  |          # A  |     |
	//  |    B   <-# &  |  A  |
	//  |          # B  |     |
	//  +----------#----+     |
	//             |          |
	//             +----------+
	//
	//
	// 	 Bad
	//
	//  +--------------------+
	//  |                    |
	//  |        A           |
	//  |     ########       |
	//  |     #      #       |
	//  |     #->B C #       |
	//  |     #      #       |
	//  |     ########       |
	//  +--------------------+
	//
	// 	            +----------+
	//              |          |
	//   +-------+  |          |
	//   |       |  |          |
	//   |   B   |  |    A     |
	//   |       |  |          |
	//   +-------+  |          |
	//              |          |
	//              +----------+

	mesh.SerialTraversal(IFORM,

	[&]( index_type const&s , coordinates_type const &x)
	{

		if((this->tags_[IFORM].at(s)&in).none() && (this->tags_[IFORM].at(s)&out).any() )
		{
			index_type neighbours[mesh_type::MAX_NUM_NEIGHBOUR_ELEMENT];

			int num=this->mesh.GetNeighbourCell(Int2Type<IFORM>(),Int2Type<VOLUME>(),neighbours,s);

			for(int i=0;i<num;++i)
			{

				if(((this->tags_[VOLUME].at(neighbours[i])&in) ).any())
				{
					ele_list->emplace( s,x );

					break;
				}
			}
		}

	});

}

template<typename TM>
template<int IFORM, typename TEleList>
void MediaTag<TM>::SelectElements(tag_type tag, TEleList *eles) const
{

	auto const & tags = tags_[VOLUME];
	mesh.SerialTraversal(IFORM,

	[&]( index_type const&s , coordinates_type const &x)
	{
		if(((this->tags_[IFORM].at(s)&tag) ).any())
		{
			eles->emplace( s,x );
		}
	});
}

template<typename TM>
template<int IFORM, typename TDict, typename TEleList>
void MediaTag<TM>::Select(TDict const & dict, TEleList *eles) const
{

	if (dict["Type"])
	{
		auto type = dict["Type"].template as<std::string>("");

		if (type == "Boundary")
		{
			auto tag = GetTagFromString(dict["Tag"].template as<std::string>());
			SelectBoundary<IFORM>(tag, null_tag, eles);

		}
		else if (type == "Interface")
		{
			auto in = GetTagFromString(dict["In"].template as<std::string>());
			auto out = GetTagFromString(dict["Out"].template as<std::string>());
			SelectBoundary<IFORM>(in, out, eles);
		}
		else if (type == "Element")
		{
			auto tag = GetTagFromString(dict["Tag"].template as<std::string>());
			SelectElements<IFORM>(tag, eles);
		}
	}

}

/**
 *
 * @param mesh mesh
 * @param points  define region
 *          if points.size() == 1 ,select Nearest Point
 *     else if points.size() == 2 ,select in the rectangle with  diagonal points[0] ~ points[1]
 *     else if points.size() >= 3 && Z<3
 *                    select points in a polyline on the Z-plane whose vertex are points
 *     else if points.size() >= 4 && Z>=3
 *                    select points in a closed surface
 *                    UNIMPLEMENTED
 *     else   illegal input
 *
 * @param fun
 * @param Z  Z==0    polyline on yz-plane
 *           Z==1    polyline on zx-plane,
 *           Z==2    polyline on xy-plane
 *           Z>=3
 */
template<typename TM>
void MediaTag<TM>::_ForEaceVericsInRegion(std::function<void(bool, index_type const &)> const & op,
        std::vector<coordinates_type> const & points)
{
	Init();
	if (points.size() == 1)
	{
		index_type idx = mesh.GetNearestVertex(points[0]);

		_ForEaceVericsInRegion(op,

		[idx](index_type s)->bool
		{
			return (s==idx);

		});

	}
	else if (points.size() == 2) //select points in a rectangle with diagonal  (x0,y0,z0)~(x1,y1,z1）,
	{
		coordinates_type v0 = points[0];
		coordinates_type v1 = points[1];

		_ForEaceVericsInRegion(op,

		[v0,v1](index_type s, coordinates_type x )->bool
		{
			return (((v0[0]-x[0])*(x[0]-v1[0]))>=0)&&
			(((v0[1]-x[1])*(x[1]-v1[1]))>=0)&&
			(((v0[2]-x[2])*(x[2]-v1[2]))>=0);

		});
	}
	else if (points.size() >= 4)
	{
		UNIMPLEMENT << " select points in a closed surface";
	}
	else
	{
		ERROR << "Illegal input";
	}

}
template<typename TM>
void MediaTag<TM>::_ForEaceVericsInRegion(std::function<void(bool, index_type const &)> const & op,
        std::vector<nTuple<2, Real>> const & points, unsigned int Z)
{
	Init();
	if (Z < 3 && points.size() > 2) //select points in polyline
	{

		PointInPolygen checkPointsInPolygen(points);

		_ForEaceVericsInRegion(op, [&](index_type s, coordinates_type x )->bool
		{	return checkPointsInPolygen(x[(Z+1)%3],x[(Z+2)%3]);});

	}
	else
	{
		ERROR << "Illegal input";
	}

}

template<typename TM>
void MediaTag<TM>::_ForEaceVericsInRegion(std::function<void(bool, index_type const &)> const & op,
        std::function<bool(index_type, coordinates_type const &)> const & select)
{
	Init();
	typedef TM mesh_type;
	mesh.Traversal(VERTEX,

	[&](typename mesh_type::index_type const&s ,
			typename mesh_type::coordinates_type const &x)
	{
		op(select(s,x), s);
	});

}

template<typename TM>
void MediaTag<TM>::_ForEaceVericsInRegion(std::function<void(bool, index_type const &)> const & op,
        std::function<bool(index_type)> const & select)
{
	Init();
	mesh.Traversal(VERTEX,

	[&](typename mesh_type::index_type const&s ,
			typename mesh_type::coordinates_type const &x)
	{
		op(select(s), s);
	});

}

template<typename TM>
void MediaTag<TM>::_ForEaceVericsInRegion(std::function<void(bool, index_type const &)> const & op,
        std::function<bool(coordinates_type const &)> const & select)
{
	Init();
	typedef TM mesh_type;
	mesh.Traversal(VERTEX,

	[&](typename mesh_type::index_type const&s ,
			typename mesh_type::coordinates_type const &x)
	{
		op(select(x), s);
	});

}

template<typename TM>
void MediaTag<TM>::_ForEaceVericsInRegion(std::function<void(bool, index_type const &)> const & op,
        LuaObject const & select)
{
	Init();
	typedef TM mesh_type;
	mesh.SerialTraversal(VERTEX,

	[&](typename mesh_type::index_type const&s ,
			typename mesh_type::coordinates_type const &x)
	{
		op(select(x[0],x[1],x[2]).template as<bool>(), s);
	});

}

}
// namespace simpla

#endif /* MEDIA_TAG_H_ */
