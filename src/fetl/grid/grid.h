/*
 * grid.h
 *
 *  Created on: 2012-10-7
 *      Author: salmon
 */

#ifndef GRID_H_
#define GRID_H_
#include "include/simpla_defs.h"
#include "primitives/properties.h"
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

};

TR1::shared_ptr<BaseGrid> make_grid(ptree const & pt);
}  // namespace simpla

#endif /* GRID_H_ */
