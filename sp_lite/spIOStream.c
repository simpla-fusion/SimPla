/*
 * spFileIO.c
 *
 *  Created on: 2016年7月6日
 *      Author: salmon
 */

#include <stdlib.h>
#include <string.h>
#include <hdf5.h>
#include <hdf5_hl.h>

#ifndef NO_MPI
#include <mpi.h>
#endif

#include "spIOStream.h"
#include "spDataModel.h"

#define H5_ERROR(_FUN_) if((_FUN_)<0){H5Eprint(H5E_DEFAULT, stderr); \
fprintf(stderr,"\e[1;32m HDF5 Error: %s \e[1;37m \n",__STRING(_FUN_) );}

#define MAX_STRING_LENGTH 2048
struct spIOStream_s
{
	int flag;
	hid_t file_id;
	hid_t group_id;
	char file_name[MAX_STRING_LENGTH];
	char group_path[MAX_STRING_LENGTH];
};
int parse_url(char const * url, char * file_name, char *grp_name, char * ds_name, char *attr_name)
{
	char * pos = strchr(url, ':');

	if (pos == NULL) // file name =""
	{
		if (file_name != NULL)
			file_name[0] = '\0';
	}
	else
	{
		if (file_name != NULL)
		{
			memcpy(file_name, url, pos - url);
			file_name[pos - url] = '\0';
		}
		url = pos + 1;
	}

	pos = strrchr(url, '/');

	if (pos == NULL) //  group name=""
	{
		if (grp_name != NULL)
			grp_name[0] = '\0';
	}
	else
	{
		if (grp_name != NULL)
		{
			memcpy(grp_name, url, pos - url);
			grp_name[pos - url] = '\0';
		}
		url = pos + 1;
	}

	pos = strchr(url, '.');

	if (pos == NULL) //  group name=""
	{
		if (ds_name != NULL)
		{

			strcpy(ds_name, url);
			ds_name[pos - url] = '\0';
		}
		if (attr_name != NULL)
		{
			attr_name[0] = '\0';
		}
	}
	else
	{
		if (ds_name != NULL)
		{
			memcpy(ds_name, url, pos - url);
			ds_name[pos - url] = '\0';
		}
		if (attr_name != NULL)
		{
			strcpy(attr_name, pos);
		}
	}

	return SP_SUCCESS;
}
int spIOStreamGroupOpen(struct spIOStream_s *os, char const * group_path);
int spIOStreamGroupClose(struct spIOStream_s *os);
int spIOStreamFileOpen(struct spIOStream_s * os, char const * file_name, int flag);
int spIOStreamFileClose(struct spIOStream_s *os);
int spIOStreamOpen(struct spIOStream_s *os, char const * url, int flag);
int spIOStreamClose(struct spIOStream_s *os);

int spIOStreamFileOpen(struct spIOStream_s * os, char const * file_name, int flag)
{
	spIOStreamFileClose(os);

	if (file_name != NULL)
	{
		strcpy(os->file_name, file_name);
	}

	if (os->file_id < 0)
	{

#ifndef NO_MPI
		hid_t plist_id;

		H5_ERROR(plist_id = H5Pcreate(H5P_FILE_ACCESS));

		H5_ERROR(H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL));

		H5_ERROR(os->file_id = H5Fcreate(os->file_name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id));

		H5_ERROR(H5Pclose(plist_id));
#else
		H5_ERROR(os->file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
#endif
	}

	return SP_SUCCESS;
}
int spIOStreamFileClose(struct spIOStream_s *os)
{

	if (os->file_id >= 0)
	{
		spIOStreamGroupClose(os);
//		flush();
		H5_ERROR(H5Fclose(os->file_id));
		os->file_id = -1;
		os->file_name[0] = '\0';

	}
	return SP_SUCCESS;
}

int spIOStreamGroupOpen(struct spIOStream_s *os, char const * group_path)
{
	if (group_path != NULL && group_path[0] != '\0' && strcmp(os->group_path, group_path) == 0)
	{
		return SP_SUCCESS;
	}
	else if (group_path[0] == '/')
	{
		strcpy(os->group_path, group_path);
	}
	else
	{
		strcat(os->group_path, group_path);
	}
	if (os->group_id >= 0)
	{
		H5_ERROR(H5Gclose(os->group_id));
	}

	if (H5Lexists(os->file_id, os->group_path, H5P_DEFAULT) != 0)
	{
		H5_ERROR(os->group_id = H5Gopen(os->file_id, os->group_path, H5P_DEFAULT));
	}
	else
	{
		H5_ERROR(os->group_id = H5Gcreate(os->file_id, os->group_path, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
	}
	return SP_SUCCESS;
}
int spIOStreamGroupClose(struct spIOStream_s *os)
{
	if (os->group_id >= 0)
	{
		H5_ERROR(H5Gclose(os->group_id));
		os->group_id = -1;
		os->group_path[0] = '/';
		os->group_path[1] = '\0';
	}
	return SP_SUCCESS;
}
int spIOStreamCreate(struct spIOStream_s ** os)
{
	if ((*os) == 0x0)
	{
		*os = (struct spIOStream_s*) (malloc(sizeof(struct spIOStream_s)));
		(*os)->file_id = -1;
		(*os)->group_id = -1;
		(*os)->file_name[0] = '\0';
		(*os)->group_path[0] = '/';
		(*os)->group_path[1] = '\0';
	}
	return SP_SUCCESS;
}

int spIOStreamDestroy(struct spIOStream_s** os)
{
	if ((*os) != 0x0)
	{
		spIOStreamClose(*os);
		free((void*) (*os));
		*os = 0x0;
	}
	return SP_SUCCESS;
}

int spIOStreamOpen(struct spIOStream_s * os, char const * url, int flag)
{
	char file_name[MAX_STRING_LENGTH];
	char group_path[MAX_STRING_LENGTH];

	parse_url(url, file_name, group_path, NULL, NULL);

	spIOStreamFileOpen(os, file_name, flag);
	spIOStreamGroupOpen(os, group_path);

	return SP_SUCCESS;
}
int spIOStreamClose(struct spIOStream_s *os)
{
	spIOStreamFileClose(os);
	return SP_SUCCESS;
}

hid_t convert_data_space_sp_to_h5(spDataSpace const *ds, size_t flag)
{

	hsize_t dims[MAX_NUMBER_OF_DIMS];
	hsize_t start[MAX_NUMBER_OF_DIMS];
	hsize_t stride[MAX_NUMBER_OF_DIMS];
	hsize_t count[MAX_NUMBER_OF_DIMS];
	hsize_t block[MAX_NUMBER_OF_DIMS];
	hsize_t max_dims[MAX_NUMBER_OF_DIMS];

	int ndims = ds->ndims;

	for (int i = 0; i < ds->ndims; ++i)
	{
		dims[i] = ds->dimensions[i];
		start[i] = ds->start[i];
		stride[i] = ds->stride[i];
		count[i] = ds->count[i];
		block[i] = ds->block[i];
		max_dims[i] = ds->dimensions[i];
	}

	if ((flag & SP_FILE_RECORD) != 0UL)
	{
		dims[ndims] = 1;
		start[ndims] = 0;
		count[ndims] = 1;
		stride[ndims] = 1;
		block[ndims] = 1;
		++ndims;
	}

	if ((flag & SP_FILE_APPEND) != 0UL)
	{
		max_dims[ndims - 1] = H5S_UNLIMITED;
	}
	else if ((flag & SP_FILE_RECORD) != 0UL)
	{
		max_dims[ndims - 1] = H5S_UNLIMITED;
	}
	hid_t res = H5Screate_simple(ndims, &dims[0], &max_dims[0]);

	H5_ERROR(H5Sselect_hyperslab(res, H5S_SELECT_SET, &start[0], &stride[0], &count[0], &block[0]));

	return res;

}
int spIOStreamWrite(struct spIOStream_s* os, char const * dsname, int flag, spDataSet const *ds)
{

	int is_existed = 0;

	hid_t d_type = H5T_FLOAT;
	switch (ds->data_type)
	{
	case SP_TYPE_float:
	case SP_TYPE_double:
	case SP_TYPE_int:
	case SP_TYPE_long:
	case SP_TYPE_OPAQUE:
		break;
	default:
		break;

	}

	hid_t m_space = convert_data_space_sp_to_h5(&(ds->m_space), SP_FILE_NEW);

	hid_t f_space = convert_data_space_sp_to_h5(&(ds->f_space), flag);

	hid_t dset;

	if (!is_existed)
	{

		hid_t dcpl_id = H5P_DEFAULT;

		if ((flag & (SP_FILE_APPEND | SP_FILE_RECORD)) != 0)
		{
			hsize_t current_dims[4];

			int f_ndims = H5Sget_simple_extent_ndims(f_space);

			H5_ERROR(H5Sget_simple_extent_dims(f_space, &current_dims[0], NULL));

			H5_ERROR(dcpl_id = H5Pcreate(H5P_DATASET_CREATE));

			H5_ERROR(H5Pset_chunk(dcpl_id, f_ndims, &current_dims[0]));
		}

		H5_ERROR(dset = H5Dcreate(os->group_id, dsname , d_type, f_space, H5P_DEFAULT, dcpl_id, H5P_DEFAULT));

		if (dcpl_id != H5P_DEFAULT)
		{
			H5_ERROR(H5Pclose(dcpl_id));
		}

		H5_ERROR(H5Fflush(os->group_id, H5F_SCOPE_GLOBAL));
	}
	else
	{

		H5_ERROR(dset = H5Dopen(os->group_id, dsname, H5P_DEFAULT));

		hsize_t current_dimensions[4];

		hid_t current_f_space;

		H5_ERROR(current_f_space = H5Dget_space(dset));

		int current_ndims = H5Sget_simple_extent_dims(current_f_space, &current_dimensions[0], NULL);

		H5_ERROR(H5Sclose(current_f_space));

		hsize_t new_f_dimensions[4];
		hsize_t new_f_max_dimensions[4];
		hsize_t new_f_offset[4];
		hsize_t new_f_end[4];

		int new_f_ndims = H5Sget_simple_extent_dims(f_space, &new_f_dimensions[0], &new_f_max_dimensions[0]);

		H5_ERROR(H5Sget_select_bounds(f_space, &new_f_offset[0], &new_f_end[0]));

		hssize_t new_f_offset2[4];

		for (int i = 0; i < 4; ++i)
		{
			new_f_offset2[i] = 0;
		}

		if ((flag & SP_FILE_APPEND) != 0)
		{

			new_f_dimensions[new_f_ndims - 1] += current_dimensions[new_f_ndims - 1];

			new_f_offset2[new_f_ndims - 1] += current_dimensions[new_f_ndims - 1];

		}
		else if ((flag & SP_FILE_RECORD) != 0)
		{
			new_f_dimensions[new_f_ndims - 1] += current_dimensions[new_f_ndims - 1];

			new_f_offset2[new_f_ndims - 1] = current_dimensions[new_f_ndims - 1];

		}

		H5_ERROR(H5Dset_extent(dset, &new_f_dimensions[0]));

		H5_ERROR(H5Sset_extent_simple(f_space, new_f_ndims, &new_f_dimensions[0], &new_f_max_dimensions[0]));

		H5_ERROR(H5Soffset_simple(f_space, &new_f_offset2[0]));

	}

	return SP_SUCCESS;
}

int spIOStreamRead(struct spIOStream_s*os, char const * name, spDataSet ** data)
{
	return SP_SUCCESS;
}
