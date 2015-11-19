/**
 * @file data_stream.h
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

#include "../gtl/any.h"
#include "../gtl/properties.h"

namespace simpla
{
struct DataSet;
} /* namespace simpla */

namespace simpla
{
namespace io
{

/**
 * @ingroup io
 * @{
 * \brief data stream , should be a singleton
 */

enum
{
    SP_NEW = 1UL << 1,
    SP_APPEND = 1UL << 2,
    SP_CACHE = (1UL << 3),
    SP_RECORD = (1UL << 4)
};

class DataStream
{
public:

    DataStream();

    ~DataStream();

    void init(int argc = 0, char **argv = nullptr);

    static std::string help_message();

    /**
     *  close dataset,group and file
     */
    void close();


    /**
     *	  change the working path (file/group) of datastream ,
     *
     * @param url_hint  <filename>:<group name>/<dataset name>
     * @param flag SP_APPEND|SP_RECORD ...
     * @return  if dataset exists ,return <true,dataset name>
     *         else return ,return <false,dataset name>
     *         if <dataset name>=="" return <false,"">
     */
    std::tuple<bool, std::string> cd(std::string const &url, size_t flag = 0UL);

    /**
     * @return current working path file/group
     */
    std::string pwd() const;


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

    std::string write(std::string const &url, DataSet const &ds, size_t flag =
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
     * @param v
     */
    void set_attribute(std::string const &url, any const &v);

    void set_attribute(std::string const &url, char const str[])
    {
        set_attribute(url, any(std::string(str)));
    }

    template<typename T>
    void set_attribute(std::string const &url, T const &v)
    {
        set_attribute(url, any(v));
    }

    void set_attribute(std::string const &url, Properties const &);

    any get_attribute(std::string const &url) const;

    Properties get_all_attribute(std::string const &url) const;

    void delete_attribute(std::string const &url);

private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> pimpl_;

};


/** @} */
}//namespace io

#define GLOBAL_DATA_STREAM  SingletonHolder<io::DataStream> ::instance()

}// namespace simpla

#endif /* DATA_STREAM_ */
