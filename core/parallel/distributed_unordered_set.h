/**
 * @file distributed_unordered_set.h
 * @author salmon
 * @date 2015-10-15.
 */

#ifndef SIMPLA_DISTRIBUTED_UNORDERED_SET_H
#define SIMPLA_DISTRIBUTED_UNORDERED_SET_H

#include "mpi_comm.h"
#include "mpi_aux_functions.h"
#include "mpi_update.h"


namespace simpla
{


//namespace parallel
struct DistributedObject
{
	//! Default constructor
	DistributedObject();

	//! destroy.
	virtual ~DistributedObject();

	DistributedObject(const DistributedObject &);

	int object_id() const { return m_object_id_; }

	virtual std::string get_type_as_string() const = 0;

	virtual DataSet dataset() const = 0;

	virtual bool empty() const = 0;

	virtual bool is_valid() const = 0;

	virtual void deploy() = 0;


	void deploy(std::vector<dist_sync_connection> const &ghost_shape);

	virtual void sync();

	virtual void async();

	virtual void wait() const;

	virtual bool is_ready() const;


protected:

	std::vector<mpi_send_recv_s> m_send_recv_list_;
	std::vector<mpi_send_recv_buffer_s> m_send_recv_buffer_;
	std::vector<MPI_Request> m_mpi_requests_;
	int m_object_id_;

};
}//namespace simpla
#endif //SIMPLA_DISTRIBUTED_UNORDERED_SET_H
