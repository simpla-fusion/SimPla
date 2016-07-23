//
// Created by salmon on 16-7-23.
//

#include "spIO.h"

void spIOWriteSimple(spIOStream *os,
                     const char *url,
                     int d_type,
                     void *d,
                     int ndims,
                     size_type const *dims,
                     size_type const *start,
                     size_type const *stride,
                     size_type const *count,
                     size_type const *block,
                     int flag)
{

    spIOStreamWriteSimple(os, url,
                          d_type,
                          d,
                          ndims,
                          dims,
                          start,
                          stride,
                          count,
                          block,
                          flag
    );
}
int
spIOWriteH5(spIOStream *os, char const *,
            hid_t d_type,
            void *buffer,
            hid_t m_space,
            hid_t f_space,
            SP_FILE_APPEND)
{
    hsize_t m_dims[2] = {(hsize_t) sp->m_max_num_of_pages_, (hsize_t) sp->m_num_of_entities_in_page_};
    hsize_t m_max_dims[2] = {H5S_UNLIMITED, (hsize_t) sp->m_num_of_entities_in_page_};

    hid_t m_space = H5Screate_simple(ndims, m_dims, m_max_dims);

    hsize_t m_start[2] = {0, 0};
    hsize_t m_count[2] = {1, sp->m_num_of_entities_in_page_};

    m_dims[0] = (hsize_t) num_of_pages;

    hid_t f_space = H5Screate_simple(ndims, m_dims, m_max_dims);

    for (int i = 0; i < num_of_pages; ++i)
    {
        m_start[0] = page_disps[i];
        H5Sselect_hyperslab(m_space, H5S_SELECT_SET, &m_start[0], NULL, &m_count[0], NULL);
    }

};