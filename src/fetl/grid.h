/*
 * grid.h
 *
 *  Created on: 2012-10-7
 *      Author: salmon
 */

#ifndef GRID_H_
#define GRID_H_
#include "include/simpla_defs.h"
namespace simpla
{

class BaseGrid
{
public:
	BaseGrid()
	{
	}
	virtual ~BaseGrid()
	{
	}
	virtual std::string Summary() const =0;

	static TR1::shared_ptr<BaseGrid> GridFactory(ptree const & pt);
	static std::map<std::string,
			TR1::function<TR1::shared_ptr<BaseGrid>(ptree const & pt)> > callback_;
};

}  // namespace simpla

#endif /* GRID_H_ */
