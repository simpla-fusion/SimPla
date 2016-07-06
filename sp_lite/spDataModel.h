/*
 * spDataModel.h
 *
 *  Created on: 2016年7月6日
 *      Author: salmon
 */

#ifndef SPDATAMODEL_H_
#define SPDATAMODEL_H_
#define MAX_NUMBER_OF_DIMS 20

enum
{
	SP_TYPE_float, SP_TYPE_double, SP_TYPE_int, SP_TYPE_long, SP_TYPE_OPAQUE
};
#define SP_TYPE_Real SP_TYPE_float
typedef struct spDataSpace_s
{
	int ndims;
	size_type dimensions[MAX_NUMBER_OF_DIMS];
	size_type start[MAX_NUMBER_OF_DIMS];
	size_type stride[MAX_NUMBER_OF_DIMS];
	size_type count[MAX_NUMBER_OF_DIMS];
	size_type block[MAX_NUMBER_OF_DIMS];

} spDataSpace;
typedef struct spDataSet_s
{
	void * data;
	int data_type;
	size_t ele_size_in_byte;
	spDataSpace m_space;
	spDataSpace f_space;
} spDataSet;

#endif /* SPDATAMODEL_H_ */
