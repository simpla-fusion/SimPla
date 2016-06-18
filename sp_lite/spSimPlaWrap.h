/*
 * spIO.h
 *
 *  Created on: 2016年6月16日
 *      Author: salmon
 */

#ifndef SPIO_H_
#define SPIO_H_

struct spDataType_s;
typedef struct spDataType_s spDataType;
void spCreateDataType(spDataType**);
void spDestroyDataType(spDataType**);
bool spDataType_is_valid(spDataType const*);
void spDataType_extent(spDataType *, int rank, size_t const *d);
void spDataType_push_back(spDataType *, spDataType const *, char const name[]);

struct spDataSpace_s;
typedef struct spDataSpace_s spDataSpace;
void spCreateDataSpace_simple(spDataSpace**, int ndims, size_type const * dims);
void spCreateDataSpace_unordered(spDataSpace**, size_type num);
void spDestroyDataSpace(spDataSpace**);
void spDataSpace_select_hyperslab(spDataSpace*, ptrdiff_t const*offset, size_type const *count);

struct spDataSet_s;
typedef struct spDataSet_s spDataSet;
void spCreateDataSet(spDataSet **, void * d, spDataType const * dtype, spDataSpace const * mspace,
		spDataSpace const *fspace);
void spDestroyDataSet(spDataSet *);

struct spDistributedObject_s;
typedef struct spDistributedObject_s spDistributedObject;
void spCreateDistributedObject(spDistributedObject**);
void spDestroyDistributedObject(spDistributedObject**);
void spDistributedObject_nonblocking_sync(spDistributedObject*);
void spDistributedObject_wait(spDistributedObject*);
void spDistributedObject_add_send_link(spDistributedObject*, size_t id, const ptrdiff_t offset[3], const spDataSet *);
void spDistributedObject_add_recv_link(spDistributedObject*, size_t id, const ptrdiff_t offset[3], spDataSet *);
bool spDistributedObject_is_ready(spDistributedObject const*);

void hdf5_write_field(char const url[], spDataSet const *d, int flag);
void hdf5_write_field(char const url[], void *d, int ndims, size_type const * dims, ptrdiff_t const*offset,
		size_type const *count, int flag);
void hdf5_write_particle(char const url[], void *d, size_type ele_size_in_byte, size_type ndims, int flag);

#endif /* SPIO_H_ */
