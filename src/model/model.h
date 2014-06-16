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
	static constexpr int MAX_NUM_OF_MEIDA_TYPE = 64;
	typedef TM mesh_type;

	typedef std::bitset<MAX_NUM_OF_MEIDA_TYPE> material_type;
	typedef typename mesh_type::iterator iterator;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::compact_index_type compact_index_type;

	const material_type null_material;

	std::map<compact_index_type, material_type> material_;
	std::map<std::string, material_type> registered_material_;

	unsigned int max_material_;
	bool isChanged_;
public:

	enum
	{
		NONE = 0, VACUUM = 1, PLASMA, CORE, BOUNDARY, LIMTER,
		// @NOTE: add materials for different physical area or media
		CUSTOM = 20
	};

	mesh_type const &mesh;

	Model(mesh_type const & m)
			: null_material(), mesh(m), max_material_(CUSTOM + 1), isChanged_(true)
	{
		registered_material_.emplace("NONE", null_material);

		registered_material_.emplace("Vacuum", material_type(1 << VACUUM));
		registered_material_.emplace("Plasma", material_type(1 << PLASMA));
		registered_material_.emplace("Core", material_type(1 << CORE));
		registered_material_.emplace("Boundary", material_type(1 << BOUNDARY));
		registered_material_.emplace("Limter", material_type(1 << LIMTER));

	}
	~Model()
	{
	}

	bool empty() const
	{
		return material_[VERTEX].empty();
	}

	operator bool() const
	{
		return material_[VERTEX].empty();
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

	material_type get(compact_index_type s) const
	{
		material_type res = null_material;
		auto it = material_.find(s);
		if (it != material_.end())
		{
			res = it->second;
		}
		return res;
	}

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
		//		os << "{ \n" << "\t -- register media type\n";
		//
		//		for (auto const& p : register_material_)
		//		{
		//			os << std::setw(10) << p.first << " = 0x" << std::hex << p.second.to_ulong() << std::dec << ", \n";
		//		}
		//
		//		os << " }\n"
		//
		//		;
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

		isChanged_ = true;
	}

	template<typename TR>
	void Modify(TR const & r, std::function<material_type()> const &fun)
	{
		for (auto s : r)
		{
			material_[s] = fun();
			if (material_[s] == null_material)
				material_.erase(s);
		}
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
	template<typename TI, typename ...Args> inline
	void Set(TI material, Args const & ... args)
	{
		Set(Select(std::forward<Args const &>(args)...), GetMaterial(material));
	}

	template<typename TI, typename ...Args> inline
	void Add(TI material, Args const & ... args)
	{
		Add(Select(std::forward<Args const &>(args)...), GetMaterial(material));
	}

	template<typename TI, typename ...Args> inline
	void Remove(TI material, Args const & ... args)
	{
		Remove(Select(std::forward<Args const &>(args)...), GetMaterial(material));
	}

	template<typename TR>
	void Set(TR const & r, material_type material)
	{
		for (auto s : r)
		{
			material_[s] = material;
		}
	}

	template<typename TR>
	void Add(TR const & r, material_type material)
	{
		for (auto s : r)
		{
			material_[s] |= material;
		}

	}

	template<typename TR>
	void Remove(TR const & r, material_type material)
	{
		for (auto s : r)
		{
			material_[s] &= ~material
		}
	}

	/**
	 *  Update material on edge ,face and cell, base on material on vertics
	 */
	void Update()
	{

		if (isChanged_)
		{
			_UpdateMaterials<EDGE>();
			_UpdateMaterials<FACE>();
			_UpdateMaterials<VOLUME>();
			isChanged_ = false;
		}
	}
	bool IsChanged() const
	{
		return isChanged_;
	}

	template<typename TDict>
	filter_mesh_range SelectByConfig(TDict const & dict) const;

	template<typename TR, typename TDict>
	filter_range_type<TR> SelectByConfig(TR const & range, TDict const & dict) const;

	template<typename ...Args>
	filter_mesh_range SelectByMaterialName(int iform, Args const &...args) const
	{
		return SelectByName(mesh.Select(iform), std::forward<Args const &>(args)...);
	}

	template<typename TR, typename TS>
	filter_range_type<TR> SelectByMaterialName(TR const &range, TS const & m) const
	{
		return Select(range, GetMaterial(m));
	}

	template<typename ...Args>
	filter_mesh_range SelectInterface(int iform, Args const &...args) const
	{
		return SelectInterface(mesh.Select(iform), std::forward<Args const &>(args)...);
	}

	template<typename TR> filter_range_type<TR>
	SelectInterface(TR const &range, material_type in, material_type out) const;

	template<typename TR, typename T1, typename T2> auto SelectInterface(TR const &range, T1 const & in,
	        T2 const & out) const
	        DECL_RET_TYPE( SelectInterface(range, GetMaterial(in), GetMaterial(out)) )

	template<typename TR> filter_range_type<TR>
	Select(TR const &range, material_type) const;

	template<typename ...Args>
	filter_range_type<typename mesh_type::range> Select(int iform, Args const &...args) const
	{
		return Select(mesh.Select(iform), std::forward<Args const &>(args)...);
	}

	template<typename TR, int N, typename ...Others>
	filter_range_type<TR> Select(TR const& range, std::vector<nTuple<N, Real>, Others...> const & points,
	        unsigned int Z) const;

	template<typename TR>
	filter_range_type<TR> Select(TR const& range, nTuple<3, Real> x) const;

	template<typename TR>
	filter_range_type<TR> Select(TR const& range, coordinates_type v0, coordinates_type v1) const;

	template<typename TR>
	filter_range_type<TR> Select(TR const& range, PointInPolygen checkPointsInPolygen) const;
private:

	/**
	 * Set material on vertics
	 * @param material is  set to 1<<material
	 * @param args args are trans-forward to
	 *      SelectVerticsInRange(<lambda function>,mesh,args)
	 */
	template<typename ...Args>
	void _ForEachVertics(std::function<void(material_type&)> fun, Args const & ... args)
	{

		isChanged_ = true;

		for (auto s : Select(mesh.Select(VERTEX), std::forward<Args const&>(args)...))
		{
			fun(material_[s]);

			if (material_[s] == null_material)
				material_.erase(s);
		}
	}

	template<typename TR>
	void _ForEach(TR const & r, std::function<void(material_type&)> fun)
	{

		isChanged_ = true;

		for (auto s : r)
		{
			fun(material_[s]);

			if (material_[s] == null_material)
				material_.erase(s);
		}
	}

	template<int IFORM>
	void _UpdateMaterials()
	{
		LOGGER << "Update Material " << IFORM;

//		for (auto s : mesh.Select(IFORM))
//		{
//			typename iterator::value_type v[mesh_type::MAX_NUM_VERTEX_PER_CEL];
//
//			int n = mesh.template GetAdjacentCells(Int2Type<IFORM>(), Int2Type<VERTEX>(), s, v);
//
//			material_type flag = null_material;
//			for (int i = 0; i < n; ++i)
//			{
//				flag |= get(v[i]);
//			}
//			if (flag != null_material)
//				material_[s] = flag;
//
//		}

		for (auto const & p : material_)
		{
			if (mesh.IForm(p.first) != VERTEX || p.second == null_material)
				continue;
			compact_index_type v[mesh_type::MAX_NUM_VERTEX_PER_CEL];
			int n = mesh.template GetAdjacentCells(Int2Type<VERTEX>(), Int2Type<IFORM>(), p.first, v);

			for (int i = 0; i < n; ++i)
			{
				material_[v[i]] |= p.second;
			}
		}

	}
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

		res = Select(range, points);

	}
	else if (!dict["Material"])
	{
		res = SelectByName(range, dict["Material"].template as<std::string>());
	}
	else if (dict.is_function())
	{
		filter_pred_fun_type<TR> pred = [dict,this]( typename TR::iterator::value_type const & s )->bool
		{
			return (dict( this->mesh.GetCoordinates( s)).template as<bool>());
		};

		res = make_filter_range(pred, range);

	}
	return res;

}
template<typename TM>
template<typename TR>
typename Model<TM>::template filter_range_type<TR> Model<TM>::SelectInterface(TR const & range, material_type in,
        material_type out) const
{
	if (IsChanged())
	{
		LOGIC_ERROR("need update!!");
	}

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

	[=]( typename TR::iterator::value_type s )->bool
	{
		auto iform = mesh.IForm(s);

		if ((this->get(s) & in).none() && (this->get(s) & out).any())
		{
			typename TR::iterator::value_type neighbours[mesh_type::MAX_NUM_NEIGHBOUR_ELEMENT];

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

template<typename TM> template<typename TR>
typename Model<TM>::template filter_range_type<TR> Model<TM>::Select(TR const & range, material_type material) const
{
	filter_pred_fun_type<TR> pred = [material,this]( typename TR::iterator::value_type s )->bool
	{
		return (((this->get(s) & material)).any());
	};
	return make_filter_range(pred, range);

}

template<typename TM> template<typename TR>
typename Model<TM>::template filter_range_type<TR> Model<TM>::Select(TR const& range, nTuple<3, Real> x) const
{
	auto dest = mesh.CoordinatesGlobalToLocal(&x);

	filter_pred_fun_type<TR> pred = [dest,this](typename TM::iterator::value_type const &s )->bool
	{
		return this->mesh.GetCellIndex(s)==dest;
	};

	return make_filter_range(pred, range);
}

template<typename TM> template<typename TR>
typename Model<TM>::template filter_range_type<TR> Model<TM>::Select(TR const& range, typename TM::coordinates_type v0,
        typename TM::coordinates_type v1) const
{
	filter_pred_fun_type<TR> pred =
	        [v0,v1,this]( typename iterator::value_type const &s )->bool
	        {
		        auto x = this->mesh.GetCoordinates(s);
		        return ((((v0[0] - x[0]) * (x[0] - v1[0])) >= 0) && (((v0[1] - x[1]) * (x[1] - v1[1])) >= 0)
				        && (((v0[2] - x[2]) * (x[2] - v1[2])) >= 0));
	        };
	return make_filter_range(pred, range);
}

template<typename TM> template<typename TR>
typename Model<TM>::template filter_range_type<TR> Model<TM>::Select(TR const& range,
        PointInPolygen checkPointsInPolygen) const
{
	filter_pred_fun_type<TR> pred = [ checkPointsInPolygen,this](typename iterator::value_type const &s )->bool
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
template<typename TM> template<typename TR, int N, typename ...Others>
typename Model<TM>::template filter_range_type<TR> Model<TM>::Select(TR const& range,
        std::vector<nTuple<N, Real>, Others...> const & points, unsigned int Z) const
{
	CHECK(points.size());

	Range<FilterIterator<std::function<bool(typename TR::iterator::value_type const &)>, typename TR::iterator>> res;

	if (points.size() == 1)
	{

		typename TM::coordinates_type x = { 0, 0, 0 };

		for (int i = 0; i < N; ++i)
		{
			x[(i + Z + 1) % 3] = points[0][i];
		}
		res = Select(range, x);
	}
	else if (points.size() == 2) //select points in a rectangle with diagonal  (x0,y0,z0)~(x1,y1,z1）,
	{
		typename TM::coordinates_type v0 = { 0, 0, 0 };
		typename TM::coordinates_type v1 = { 0, 0, 0 };
		for (int i = 0; i < N; ++i)
		{
			v0[(i + Z + 1) % 3] = points[0][i];
			v1[(i + Z + 1) % 3] = points[1][i];
		}
		CHECK(v0);
		CHECK(v1);
		res = Select(range, v0, v1);
	}
	else if (Z < 3 && points.size() > 2) //select points in polyline
	{
		return Select(range, PointInPolygen(points, Z));
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
