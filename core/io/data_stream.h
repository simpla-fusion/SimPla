/*
 * data_stream.h
 *
 *  created on: 2013-12-11
 *      Author: salmon
 *
 */

#ifndef DATA_STREAM_
#define DATA_STREAM_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../data_structure/data_set.h"
#include "../utilities/ntuple.h"
#include "../utilities/singleton_holder.h"
#include "../utilities/any.h"

#include "io.h"
namespace simpla
{
/** \defgroup  DataIO Data input/output system
 *  @{
 *   \defgroup  HDF5  HDF5 interface
 *   \defgroup  XDMF   XDMF interface
 *   \defgroup  NetCDF  NetCDF interface
 *     \brief UNIMPLEMENTED notfix
 *  @}
 *  */

/**
 * \ingroup DataIO
 * @class DataStream
 * \brief data stream , should be a singleton
 */
class DataStream
{
public:
	Properties properties;

	DataStream();

	~DataStream();

	void init(int argc = 0, char** argv = nullptr);

	/**
	 *	  change the workiong path (file/group) of datastream ,
	 *
	 * @param url_hint  <filename>:<group name>/<dataset name>
	 * @param flag SP_APPEND|SP_RECORD ...
	 * @return  if dataset exists ,return <true,dataset name>
	 *         else return ,return <false,dataset name>
	 *         if <dataset name>=="" return <false,"">
	 */
	std::tuple<bool, std::string> cd(std::string const & url,
			size_t flag = 0UL);

	/**
	 * @return current working path file/group
	 */
	std::string pwd() const;

	/**
	 *  close dataset,group and file
	 */
	void close();

	/**
	 * @return true if datastream is initialized.
	 */

	bool is_valid() const;

	/**
	 * write dataset to url
	 * @param url             dataset name or path
	 * @param ds		  	   data set
	 * @param flag             flag to define the operation
	 * @return
	 */

	std::string write(std::string const &url, DataSet const & ds, size_t flag =
			0UL);

	/**
	 * 	read dataset from url
	 * @param url
	 * @param ds
	 * @param flag
	 * @return
	 */
	std::string read(std::string const &url, DataSet *ds, size_t flag = 0UL);

	/**
	 *
	 * @param url  <file name>:/<group path>/<obj name>.<attribute>
	 * @param d_type
	 * @param v
	 */
	void set_attribute(std::string const &url, DataType const & d_type,
			void const * buff);

	void set_attribute(std::string const &url, Any const & v);

	Any get_attribute(std::string const &url) const;

	void delete_attribute(std::string const &url);

	void set_attribute(std::string const &url, char const str[])
	{
		set_attribute(url, std::string(str));
	}

	template<typename T>
	void set_attribute(std::string const & url, T const&v)
	{
		set_attribute(url, make_datatype<T>(), &v);
	}

private:
	struct pimpl_s;
	pimpl_s * pimpl_;

}
;

//! Global data stream entry
#define GLOBAL_DATA_STREAM  SingletonHolder<DataStream> ::instance()

}
// namespace simpla

#endif /* DATA_STREAM_ */
