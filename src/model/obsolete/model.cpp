/**
 * @file model.cpp
 * @author salmon
 * @date 2015-11-06.
 */
#include "model.h"

namespace simpla
{

static constexpr typename Model::tage_type Model::null_material;


}





//typename model::size_t model::get(id_type s) const
//{
//
//	size_t res = null_material;
//
//	if (this->geometry_type::IForm(s) == VERTEX)
//	{
//		auto it = m_data_.find(s);
//		if (it != m_data_.end())
//		{
//			res = it->second;
//		}
//	}
//	else
//	{
//		id_type neighbours[geometry_type::MAX_NUM_NEIGHBOUR_ELEMENT];
//
//		int num = this->geometry_type::get_vertices(s, neighbours);
//
//		for (int i = 0; i < num; ++i)
//		{
//			res |= this->get(neighbours[i]);
//		}
//	}
//	return std::move(res);
//}

//template<typename TM>
//template<typename TR, typename TDict>
//FilterRange<TR> Model<TM>::select_by_config(TR const& range,
//		TDict const& dict) const
//{
//	if (!dict)
//	{
//		pred_fun_type pred = [=]( id_type const & s )->bool
//		{
//			return true;
//		};
//
//		return std::move(FilterRange < TR > (range, std::move(pred)));
//	}
//	else if (dict.is_function())
//	{
//		pred_fun_type pred =
//				[=]( id_type const & s )->bool
//				{
//					return (dict( this->geometry_type::get_coordinates( s)).template as<bool>());
//				};
//
//		return std::move(FilterRange < TR > (range, std::move(pred)));
//	}
//	else if (dict["Material"])
//	{
//		return std::move(
//				SelectByMaterial(range,
//						dict["Material"].template as<std::string>()));
//	}
//	else if (dict["Type"])
//	{
//		auto type = dict["Type"].template as < std::string > ("");
//
//		if (type == "NGP")
//		{
//			return std::move(
//					SelectByNGP(range,
//							dict["Points"].template as<coordinates_type>()));
//
//		}
//		else if (type == "Boundary")
//		{
//			return std::move(
//					SelectInterface(range,
//							dict["In"].template as < std::string > ("NONE"),
//							"NONE"));
//
//		}
//		else if (type == "Interface")
//		{
//			return std::move(
//					SelectInterface(range,
//							dict["In"].template as < std::string > ("NONE"),
//							dict["Out"].template as < std::string > ("NONE")));
//		}
//		else if (type == "Range" && dict["Points"].is_table())
//		{
//			std::vector < coordinates_type > points;
//
//			dict["Points"].as(&points);
//
//			if (points.size() == 2) //select points in a rectangle with diagonal  (x0,y0,z0)~(x1,y1,z1）,
//			{
//				return std::move(SelectByRectangle(range, points[0], points[1]));
//			}
//			else if (points.size() > 2) //select points in polylines
//			{
//				return std::move(
//						SelectByPolylines(range,
//								PointInPolygon(points,
//										dict["Z-Axis"].template as<int>(2))));
//			}
//			else
//			{
//				PARSER_ERROR(
//						"Number of points  [" + value_to_string(points.size())
//								+ "]<2");
//			}
//
//		}
//
//	}
//
//	else
//	{
//		PARSER_ERROR("Unknown 'Select' options");
//	}
//	return FilterRange < TR > (range, [=](id_type const &)
//	{	return true;});
//}
//
//template<typename TM>
//template<typename TR, typename T1, typename T2>
//FilterRange<TR> Model<TM>::SelectInterface(TR const& range, T1 pin,
//		T2 pout) const
//{
//	/** \note
//	 * Good
//	 *  +----------#----------+
//	 *  |          #          |
//	 *  |    A     #-> B   C  |
//	 *  |          #          |
//	 *  +----------#----------+
//	 *
//	 *  +--------------------+
//	 *  |         ^          |
//	 *  |       B |     C    |
//	 *  |     ########       |
//	 *  |     #      #       |
//	 *  |     #  A   #       |
//	 *  |     #      #       |
//	 *  |     ########       |
//	 *  +--------------------+
//	 *
//	 *             +----------+
//	 *             |      C   |
//	 *  +----------######     |
//	 *  |          | A  #     |
//	 *  |    A     | &  #  B  |
//	 *  |          | B  #->   |
//	 *  +----------######     |
//	 *             |          |
//	 *             +----------+
//	 *
//	 *     	       +----------+
//	 *       C     |          |
//	 *  +----------#----+     |
//	 *  |          # A  |     |
//	 *  |    B   <-# &  |  A  |
//	 *  |          # B  |     |
//	 *  +----------#----+     |
//	 *             |          |
//	 *             +----------+
//	 *
//	 *
//	 * 	 Bad
//	 *
//	 *  +--------------------+
//	 *  |                    |
//	 *  |        A           |
//	 *  |     ########       |
//	 *  |     #      #       |
//	 *  |     #->B C #       |
//	 *  |     #      #       |
//	 *  |     ########       |
//	 *  +--------------------+
//	 *
//	 * 	            +----------+
//	 *              |          |
//	 *   +-------+  |          |
//	 *   |       |  |          |
//	 *   |   B   |  |    A     |
//	 *   |       |  |          |
//	 *   +-------+  |          |
//	 *              |          |
//	 *              +----------+
//	 */
//	size_t in = get_material(pin);
//	size_t out = get_material(pout);
//
//	if (in == out)
//		out = null_material;
//
//	pred_fun_type pred =
//
//			[=]( id_type const & s )->bool
//			{
//
//				size_t res;
//
//				auto iform = this->geometry_type::IForm(s);
//
//				auto self=this->get(s);
//
//				if (( self & in).none() && ( (self & out).any() || (out == null_material) ))
//				{
//					id_type neighbours[geometry_type::MAX_NUM_NEIGHBOUR_ELEMENT];
//
//					int num=0;
//					switch(iform)
//					{	case VERTEX:
//						num= this->geometry_type::get_adjacent_cells(std::integral_constant<size_t ,VERTEX>(), std::integral_constant<size_t ,VOLUME>(), s, neighbours);
//						break;
//						case EDGE:
//						num= this->geometry_type::get_adjacent_cells(std::integral_constant<size_t ,EDGE>(), std::integral_constant<size_t ,VOLUME>(), s, neighbours);
//						break;
//						case FACE:
//						num= this->geometry_type::get_adjacent_cells(std::integral_constant<size_t ,FACE>(), std::integral_constant<size_t ,VOLUME>(), s, neighbours);
//						break;
//						case VOLUME:
//						num= this->geometry_type::get_adjacent_cells(std::integral_constant<size_t ,VOLUME>(), std::integral_constant<size_t ,VOLUME>(), s, neighbours);
//						break;
//					}
//
//					for (int i = 0; i < num; ++i)
//					{
//						res |=this->get(neighbours[i]);
//					}
//				}
//
//				return (res & in).any();
//			};
//
//	return std::move(FilterRange < TR > (range, (pred)));
//
//}
//
//template<typename TM> template<typename TR, typename ...Args>
//FilterRange<TR> Model<TM>::SelectByMaterial(TR const& range,
//		Args && ... args) const
//{
//	auto material = get_material(std::forward<Args>(args)...);
//
//	if (material != null_material)
//	{
//		pred_fun_type pred = [=]( id_type const & s )->bool
//		{
//			return (this->get(s) & material).any();
//		};
//		return std::move(FilterRange < TR > (range, std::move(pred)));
//	}
//
//	else
//	{
//		pred_fun_type pred = [=]( id_type const & s )->bool
//		{
//			return (this->get(s) == null_material);
//		};
//		return std::move(FilterRange < TR > (range, std::move(pred)));
//	}
//
//}
