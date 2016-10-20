/**
 * @file IOStream.h
 * @author salmon
 * @date 2015-12-20.
 */

#ifndef SIMPLA_IOSTREAM_H
#define SIMPLA_IOSTREAM_H

#include <string>
#include <tuple>
#include "Properties.h"
#include "DataSet.h"


namespace simpla { namespace toolbox
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

class IOStream : public toolbox::Object
{
public:

    SP_OBJECT_HEAD(IOStream, toolbox::Object);

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

    virtual std::string ext_name() const = 0;

    std::tuple<std::string, std::string, std::string, std::string>
    parser_url(std::string const &url_hint) const;

    std::string pwd() const;

    std::string auto_increase_file_name(std::string filename, std::string const &ext_str = ".h5") const;

    virtual void close() = 0;

    virtual std::tuple<bool, std::string> open(std::string const &url, int flag = 0UL) = 0;

    virtual std::string write(std::string const &url, toolbox::DataSet const &, int flag = 0UL) = 0;

    virtual std::string read(std::string const &url, toolbox::DataSet *ds, int flag = 0UL) = 0;


//    template<typename TV>
//    void write(get_mesh::MeshAtlas const &,
//               std::map<mesh::MeshBlockId, std::shared_ptr<TV> const &, get_mesh::MeshBlockId id= 0)
//    {
//        UNIMPLEMENTED;
//    }
//
//    template<typename TV>
//    void read(get_mesh::MeshAtlas const &,
//              std::map<mesh::MeshBlockId, std::shared_ptr<TV> *, get_mesh::MeshBlockId id= 0)
//    {
//        UNIMPLEMENTED;
//    }
};
}}//namespace simpla { namespace io

#endif //SIMPLA_IOSTREAM_H
