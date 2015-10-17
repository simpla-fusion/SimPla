/**
 * @file distributed_comm.h
 * @author salmon
 * @date 2015-10-17.
 */

#ifndef SIMPLA_DISTRIBUTED_COMM_H
#define SIMPLA_DISTRIBUTED_COMM_H

#include "../dataset/datatype.h"

namespace simpla
{
namespace parallel
{

struct link_node
{
	int dest_id;

	int tag;

	DataSet data;
};

}//namespace distributed_data

}//namespace simpla

#endif //SIMPLA_DISTRIBUTED_COMM_H
