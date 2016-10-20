/**
 * @file DataStream.h
 *
 *  created on: 2013-12-11
 *      Author: salmon
 *
 */

#ifndef DATA_STREAM_
#define DATA_STREAM_

#include <stddef.h>
#include <cstdbool>
#include <string>
#include <tuple>
#include "DataSet.h"
#include "Any.h"
#include "Properties.h"
#include "IOStream.h"

namespace simpla { namespace toolbox
{

/*
 * @brief m_data stream , should be a singleton
 */

class HDF5Stream : public IOStream
{
public:
    SP_OBJECT_HEAD(HDF5Stream, IOStream);

    HDF5Stream();

    virtual ~HDF5Stream();

    virtual std::string ext_name() const { return "h5"; }

    /**
     *	  change the working path (file/group) of m_data stream ,
     *
     * @param url_hint  <filename>:<group name>/<data_model name>
     * @param id SP_APPEND|SP_RECORD ...
     * @return  if data_model exists ,return <true,data_model name>
     *         else return ,return <false,data_model name>
     *         if <data_model name>=="" return <false,"">
     */
    std::tuple<bool, std::string> open(std::string const &url, int flag = 0UL);

    void open_group(std::string const &path);

    void open_file(std::string const &path, bool is_append = false);

    /**
    *  close m_data set,group and file
    */
    void close();

    void close_group();

    void close_file();

    void flush();

    std::string absolute_path(std::string const &url) const;

    /**
     * @return true if m_data stream is initialized.
     */

    bool is_valid() const;

    bool is_opened() const;

    /**
     * write data_model to url
     * @param url             data_model name or path
     * @param ds		  	   m_data set
     * @param id             id to define the operation
     * @return
     */

    virtual std::string write(std::string const &url, DataSet const &ds, int flag = 0UL);

    void push_buffer(std::string const &url, DataSet const &ds);

    std::string write_buffer(std::string const &url, bool is_forced_flush = false);

    /**
     * 	read m_data set from url
     * @param url
     * @param ds
     * @param id
     * @return
     */
    virtual std::string read(std::string const &url, DataSet *ds, int flag = 0UL);


    /**
     *
     * @param url  <file name>:/<group path>/<obj name>.<Attribute>
     * @param v
     */
    void set_attribute(std::string const &url, Properties const &v);

    Properties get_attribute(std::string const &url) const;

//    Properties get_all_attribute(std::string const &url) const;

    void delete_attribute(std::string const &url);

private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;

};


/** @} */
}}// namespace simpla
#define GLOBAL_DATA_STREAM  ::simpla::SingletonHolder<::simpla::io::HDF5Stream>::instance()

#endif /* DATA_STREAM_ */
