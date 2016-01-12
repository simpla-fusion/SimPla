/**
 * @file IOStream.h
 * @author salmon
 * @date 2015-12-20.
 */

#ifndef SIMPLA_IOSTREAM_H
#define SIMPLA_IOSTREAM_H

#include <string>
#include <tuple>
#include "../gtl/Properties.h"
#include "../data_model/DataSet.h"

namespace simpla { namespace io
{
/**
 * @ingroup io
 */
enum
{
    SP_NEW = 1UL << 1,
    SP_APPEND = 1UL << 2,
    SP_BUFFER = (1UL << 3),
    SP_RECORD = (1UL << 4)
};

class IOStream : public base::Object
{
public:

    SP_OBJECT_HEAD(IOStream, base::Object);

    HAS_PROPERTIES;

    DEFINE_PROPERTIES(std::string, current_file_name);

    DEFINE_PROPERTIES(std::string, current_group_name);

    IOStream();

    virtual ~IOStream();

    void init(int argc, char **argv);


    virtual void set_attribute(std::string const &url, Properties const &v) = 0;

    void set_attribute(std::string const &url, char const str[])
    {
        set_attribute(url, Properties(std::string(str)));
    }

    template<typename T>
    void set_attribute(std::string const &url, T const &v)
    {
        set_attribute(url, Properties(v));
    }

    std::tuple<std::string, std::string, std::string, std::string>
            parser_url(std::string const &url_hint) const;

    std::string pwd() const;

    std::string auto_increase_file_name(std::string filename, std::string const &ext_str = ".h5") const;


};

}}//namespace simpla { namespace io

#endif //SIMPLA_IOSTREAM_H
