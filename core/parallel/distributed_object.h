/**
 * @file distributed_object.h
 * @author salmon
 * @date 2015-10-17.
 */

#ifndef SIMPLA_DISTRIBUTED_OBJECT_H
#define SIMPLA_DISTRIBUTED_OBJECT_H

#include <bits/unique_ptr.h>
#include "mpi_comm.h"
#include "mpi_aux_functions.h"
#include "mpi_update.h"
#include "../dataset/dataset.h"

namespace simpla
{
struct DistributedObject
{
	DistributedObject(MPIComm &);

	DistributedObject(DistributedObject const &);

	virtual ~DistributedObject();

	void swap(DistributedObject &);

	virtual void sync();

	virtual void wait();

	virtual void deploy() = 0;

	virtual bool is_ready() const;

	bool is_distributed() const;

	void clear_links();

	void add_link(bool is_send, int const coord_offset[], int size,
			DataType const &d_type, std::shared_ptr<void> *p);

	void add_link(bool is_send, int const coord_offset[], DataSpace space,
			DataType const &d_type, std::shared_ptr<void> *p);

	template<typename ...Args>
	void add_link_send(Args &&...args) { add_link(true, std::forward<Args>(args)...); };

	template<typename ...Args>
	void add_link_recv(Args &&...args) { add_link(false, std::forward<Args>(args)...); };


private:

	struct pimpl_s;
	std::unique_ptr<pimpl_s> pimpl_;

};

}//namespace simpla
#endif //SIMPLA_DISTRIBUTED_OBJECT_H
