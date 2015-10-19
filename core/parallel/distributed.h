/**
 * @file distributed.h
 * @author salmon
 * @date 2015-10-18.
 */

#ifndef SIMPLA_DISTRIBUTED_H
#define SIMPLA_DISTRIBUTED_H

#include "distributed_object.h"

namespace simpla
{
template<typename ...> class Distributed;

template<typename TBase>
class Distributed<TBase> : public DataSet, public DistributedObject
{
private:
	typedef TBase base_type;
	typedef Distributed<base_type> this_type;
public:

	Distributed(MPIComm &comm) : DistributedObject(comm) { };

	Distributed(const this_type &other) : DistributedObject(other), base_type(other) { }

	virtual ~Distributed();

	virtual void swap(this_type &other)
	{
		DistributedObject::swap(other);

		base_type::swap(other);
	};

	virtual void deploy();

private:

};

}//namespace simpla
#endif //SIMPLA_DISTRIBUTED_H
