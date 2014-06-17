/*
 * model.h
 *
 *  Created on: 2013年12月15日
 *      Author: salmon
 */

#ifndef MODEL_H_
#define MODEL_H_

#include <algorithm>
#include <bitset>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../utilities/primitives.h"
#include "../utilities/log.h"
#include "../utilities/pretty_stream.h"
#include "../utilities/utilities.h"
#include "../utilities/range.h"

#include "pointinpolygen.h"
namespace std
{
template<typename TI> struct iterator_traits;
}
namespace simpla
{

template<typename TM>
class Model
{

public:
	static constexpr int MAX_NUM_OF_MEIDA_TYPE = std::numeric_limits<unsigned long>::digits;
	typedef TM mesh_type;
	static constexpr unsigned int NDIMS = mesh_type::NDIMS;
	typedef std::bitset<MAX_NUM_OF_MEIDA_TYPE> material_type;
	typedef typename mesh_type::iterator iterator;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::compact_index_type compact_index_type;

	const material_type null_material;

	std::map<compact_index_type, material_type> material_;
	std::map<std::string, material_type> registered_material_;

	unsigned int max_material_;
public:

	enum
	{
		NONE = 0, VACUUM = 1, PLASMA, CORE, BOUNDARY, LIMITER,
		// @NOTE: add materials for different physical area or media
		CUSTOM = 20
	};

	mesh_type const &mesh;

	Model(mesh_type const & m)
			: null_material(), mesh(m), max_material_(CUSTOM + 1)
	{
		registered_material_.emplace("NONE", null_material);

		registered_material_.emplace("Vacuum", material_type(1 << VACUUM));
		registered_material_.emplace("Plasma", material_type(1 << PLASMA));
		registered_material_.emplace("Core", material_type(1 << CORE));
		registered_material_.emplace("Boundary", material_type(1 << BOUNDARY));
		registered_material_.emplace("Limiter", material_type(1 << LIMITER));

	}
	~Model()
	{
	}

	bool empty() const
	{
		return material_.empty();
	}

	operator bool() const
	{
		return material_.empty();
	}

	material_type RegisterMaterial(std::string const & name)
	{
		material_type res;
		if (registered_material_.find(name) != registered_material_.end())
		{
			res = registered_material_[name];
		}
		else if (max_material_ < MAX_NUM_OF_MEIDA_TYPE)
		{
			res.set(max_material_);
			++max_material_;
		}
		else
		{
			RUNTIME_ERROR("Too much media Type");
		}
		return res;
	}

	unsigned int GetNumMaterialType() const
	{
		return max_material_;
	}

	material_type GetMaterial(material_type const & m) const
	{
		return m;
	}

	material_type GetMaterial(unsigned int material_pos) const
	{
		material_type res;
		res.set(material_pos);
		return std::move(res);
	}
	material_type GetMaterial(std::string const &name) const
	{

		material_type res;

		try
		{
			res = registered_material_.at(name);

		} catch (...)
		{
			RUNTIME_ERROR("Unknown material name : " + name);
		}
		return std::move(res);
	}

	material_type get(compact_index_type s) const;

	material_type operator[](compact_index_type s) const
	{
		return get(s);
	}

	void Clear()
	{
		material_.clear();
	}

	template<typename TDict>
	void Load(TDict const & dict)
	{
		UNIMPLEMENT;
	}
	std::string Save(std::string const & path, bool is_verbose = false) const
	{
		UNIMPLEMENT;
		return "UNIMPLEMENT!";
	}
	template<typename OS>
	OS & Print(OS &os) const
	{
		UNIMPLEMENT;
		return os;
	}

	template<typename TR> using filter_pred_fun_type=std::function<bool(typename TR::iterator::value_type)>;

	template<typename TR> using filter_range_type=
	Range<FilterIterator<std::function<bool(typename TR::iterator::value_type)> , typename TR::iterator>>;

	typedef filter_range_type<typename mesh_type::range> filter_mesh_range;

	template<typename TDict>
	void Modify(TDict const& dict)
	{

		std::function<material_type(material_type const &)> fun;

		auto range = SelectByConfig(dict["Select"]);

		auto material = GetMaterial(dict["Value"].template as<std::string>(""));

		std::string op = dict["Op"].template as<std::string>("");

		if (op == "Set")
		{
			Set(range, material);
		}
		else if (op == "Remove")
		{
			Remove(range, material);
		}
		else if (op == "Add")
		{
			Add(range, material);
		}

		LOGGER << op << " material " << DONE;

	}

	template<typename TR>
	void Modify(TR const & r, std::function<material_type(material_type const &)> const &fun)
	{
		for (auto s : r)
		{
			material_[s] = fun(material_[s]);
			if (material_[s] == null_material)
				material_.erase(s);
		}
	}

	template<typename TR>
	void Erase(TR const & r)
	{
		for (auto s : r)
		{
			material_.erase(s);
		}
	}

	template<typename TR, typename M>
	void Set(TR const & r, M const& material)
	{
		auto t = GetMaterial(material);
		Modify(r, [t](material_type const & m)->material_type
		{	return m|t;});
	}

	template<typename TR, typename M>
	void Unset(TR const & r, M const& material)
	{
		auto t = GetMaterial(material);
		Modify(r, [t](material_type const & m)->material_type
		{	return m&(~t);});
	}

	template<typename TDict>
	filter_mesh_range SelectByConfig(TDict const & dict) const;

	template<typename TR, typename TDict>
	filter_range_type<TR> SelectByConfig(TR const & r, TDict const & dict) const;
	template<typename ...Args>
	filter_mesh_range SelectByConfig(int iform, Args const &...args) const
	{
		return SelectByConfig(mesh.Select(iform), std::forward<Args const &>(args)...);
	}

	template<typename TR, typename ...Args>
	filter_range_type<TR> SelectByMaterial(TR const& r, Args const &...args) const;

	template<typename ...Args>
	filter_mesh_range SelectByMaterial(int iform, Args const &...args) const
	{
		return SelectByMaterial(mesh.Select(iform), std::forward<Args const &>(args)...);
	}

	template<typename TR, typename T1, typename T2>
	filter_range_type<TR> SelectInterface(TR const &range, T1 const & in, T2 const & out) const;

	template<typename ...Args>
	filter_mesh_range SelectInterface(int iform, Args const &...args) const
	{
		return SelectInterface(mesh.Select(iform), std::forward<Args const &>(args)...);
	}

	template<typename TR>
	filter_range_type<TR> SelectByPoints(TR const& range, std::vector<coordinates_type> const & points, unsigned int Z =
	        2) const;

	template<typename TR>
	filter_range_type<TR> SelectByPoints(TR const& range, nTuple<3, Real> x) const;

	template<typename TR>
	filter_range_type<TR> SelectByPoints(TR const& range, coordinates_type v0, coordinates_type v1) const;

	template<typename TR>
	filter_range_type<TR> SelectByPoints(TR const& range, PointInPolygen checkPointsInPolygen) const;

	template<typename ...Args>
	filter_mesh_range SelectByPoints(int iform, Args const &...args) const
	{
		return SelectByPoints(mesh.Select(iform), std::forward<Args const &>(args)...);
	}

	typename mesh_type::range Select(unsigned int iform) const
	{
		return mesh.Select(iform);
	}

//private:
//
//	template<int IFORM>
//	void _UpdateMaterials()
//	{
//		LOGGER << "Update Material " << IFORM;
//
////		for (auto s : mesh.Select(IFORM))
////		{
////			typename iterator::value_type v[mesh_type::MAX_NUM_VERTEX_PER_CEL];
////
////			int n = mesh.template GetAdjacentCells(Int2Type<IFORM>(), Int2Type<VERTEX>(), s, v);
////
////			material_type flag = null_material;
////			for (int i = 0; i < n; ++i)
////			{
////				flag |= get(v[i]);
////			}
////			if (flag != null_material)
////				material_[s] = flag;
////
////		}
//
//		for (auto const & p : material_)
//		{
//			if (mesh.IForm(p.first) != VERTEX || p.second == null_material)
//				continue;
//			compact_index_type v[mesh_type::MAX_NUM_VERTEX_PER_CEL];
//			int n = mesh.template GetAdjacentCells(Int2Type<VERTEX>(), Int2Type<IFORM>(), p.first, v);
//
//			for (int i = 0; i < n; ++i)
//			{
//				material_[v[i]] |= p.second;
//			}
//		}
//
//	}
}
;
template<typename TM>
inline std::ostream & operator<<(std::ostream & os, Model<TM> const &self)
{
	self.Print(os);
	return os;
}

template<typename TM>
template<typename TR, typename TDict>
typename Model<TM>::template filter_range_type<TR> Model<TM>::SelectByConfig(TR const & range, TDict const & dict) const
{
	filter_range_type<TR> res;

	if (!dict["Type"])
		PARSER_ERROR("Modle.Select: 'Type' is not define ")

	auto type = dict["Type"].template as<std::string>("");

	if (type == "Boundary")
	{
		res = SelectInterface(range, dict["In"].template as<std::string>(), "NONE");

	}
	else if (type == "Interface")
	{
		res = SelectInterface(range, dict["In"].template as<std::string>(), dict["Out"].template as<std::string>());
	}

	else if (type == "Range" && dict["Points"].is_table())
	{
		std::vector<coordinates_type> points;

		dict["Points"].as(&points);

		res = SelectByPoints(range, points);

	}
	else if (!dict["Material"])
	{
		res = SelectByMaterial(range, dict["Material"].template as<std::string>());
	}
	else if (dict.is_function())
	{
		filter_pred_fun_type<TR> pred = [dict,this]( compact_index_type s )->bool
		{
			return (dict( this->mesh.GetCoordinates( s)).template as<bool>());
		};

		res = make_filter_range(pred, range);

	}
	return res;

}
template<typename TM>
template<typename TR, typename T1, typename T2>
typename Model<TM>::template filter_range_type<TR> Model<TM>::SelectInterface(TR const & range, T1 const & pin,
        T2 const & pout) const
{
	material_type in = GetMaterial(pin);
	material_type out = GetMaterial(pout);

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

	filter_pred_fun_type<TR> pred =

	[=]( compact_index_type s )->bool
	{
		auto iform = this->mesh.IForm(s);

		auto self=this->get(s);

		if (( self & in).none() && ( (self & out).any() || self == out ))
		{
			compact_index_type neighbours[mesh_type::MAX_NUM_NEIGHBOUR_ELEMENT];

			int num=0;
			switch(iform)
			{	case VERTEX:
				num= this->mesh.GetAdjacentCells(Int2Type<VERTEX>(), Int2Type<VOLUME>(), s, neighbours);
				break;
				case EDGE:
				num= this->mesh.GetAdjacentCells(Int2Type<EDGE>(), Int2Type<VOLUME>(), s, neighbours);
				break;
				case FACE:
				num= this->mesh.GetAdjacentCells(Int2Type<FACE>(), Int2Type<VOLUME>(), s, neighbours);
				break;
				case VOLUME:
				num= this->mesh.GetAdjacentCells(Int2Type<VOLUME>(), Int2Type<VOLUME>(), s, neighbours);
				break;
			}
			for (int i = 0; i < num; ++i)
			{
				if (((this->get(neighbours[i]) & in)).any())
				{
					return true;
				}
			}
		}

		return false;
	};

	return make_filter_range(pred, range);

}
template<typename TM>
typename Model<TM>::material_type Model<TM>::get(compact_index_type s) const
{

	material_type res = null_material;

	if (this->mesh.IForm(s) == VERTEX)
	{
		auto it = material_.find(s);
		if (it != material_.end())
		{
			res = it->second;
		}
	}
	else
	{
		compact_index_type neighbours[mesh_type::MAX_NUM_NEIGHBOUR_ELEMENT];

		int num = this->mesh.GetVertices(s, neighbours);

		for (int i = 0; i < num; ++i)
		{
			res |= this->get(neighbours[i]);
		}
	}
	return res;
}

template<typename TM> template<typename TR, typename ...Args>
typename Model<TM>::template filter_range_type<TR> Model<TM>::SelectByMaterial(TR const & range,
        Args const & ... args) const
{
	auto material = GetMaterial(std::forward<Args const&>(args)...);

	filter_pred_fun_type<TR> pred = [material,this]( compact_index_type s )->bool
	{
		return (this->get(s) & material).any();
	};
	return make_filter_range(pred, range);

}

template<typename TM> template<typename TR>
typename Model<TM>::template filter_range_type<TR> Model<TM>::SelectByPoints(TR const& range, nTuple<3, Real> x) const
{
	auto dest = mesh.CoordinatesGlobalToLocal(&x);

	filter_pred_fun_type<TR> pred = [dest,this](compact_index_type s )->bool
	{
		return this->mesh.GetCellIndex(s)==dest;
	};

	return make_filter_range(pred, range);
}

template<typename TM> template<typename TR>
typename Model<TM>::template filter_range_type<TR> Model<TM>::SelectByPoints(TR const& range, coordinates_type v0,
        coordinates_type v1) const
{
	filter_pred_fun_type<TR> pred =
	        [v0,v1,this]( compact_index_type s )->bool
	        {

		        auto x = this->mesh.GetCoordinates(s);
		        return ((((v0[0] - x[0]) * (x[0] - v1[0])) >= 0) && (((v0[1] - x[1]) * (x[1] - v1[1])) >= 0)
				        && (((v0[2] - x[2]) * (x[2] - v1[2])) >= 0));
	        };
	return make_filter_range(pred, range);
}

template<typename TM> template<typename TR>
typename Model<TM>::template filter_range_type<TR> Model<TM>::SelectByPoints(TR const& range,
        PointInPolygen checkPointsInPolygen) const
{
	filter_pred_fun_type<TR> pred = [ checkPointsInPolygen,this](compact_index_type s )->bool
	{	return (checkPointsInPolygen(this->mesh.GetCoordinates(s) ));};

	return make_filter_range(pred, range);

}
/**
 *
 * @param mesh mesh
 * @param points  define Range
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
template<typename TM> template<typename TR>
typename Model<TM>::template filter_range_type<TR> Model<TM>::SelectByPoints(TR const& range,
        std::vector<coordinates_type> const & points, unsigned int Z) const
{
	filter_range_type<TR> res;

	if (points.size() == 1)
	{
		res = SelectByPoints(range, points[0]);
	}
	else if (points.size() == 2) //select points in a rectangle with diagonal  (x0,y0,z0)~(x1,y1,z1）,
	{
		res = SelectByPoints(range, points[0], points[1]);
	}
	else if (Z < 3 && points.size() > 2) //select points in polyline
	{
		res = SelectByPoints(range, PointInPolygen(points, Z));
	}
	else
	{
		PARSER_ERROR("too less points " + ToString(points.size()));
	}
	return res;
}

}
// namespace simpla

#endif /* MODEL_H_ */
