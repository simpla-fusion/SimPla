/**
 * @file IOStream.cpp
 * @author salmon
 * @date 2015-12-20.
 */

#include "IOStream.h"
#include "../parallel/MPIComm.h"
#include "../parallel/MPIAuxFunctions.h"

namespace simpla { namespace io
{


void IOStream::init(int argc, char **argv)
{

//    bool show_help = false;
//
//    parse_cmd_line(
//            argc, argv,
//
//            [&, this](std::string const &opt, std::string const &value) -> int
//            {
//                if (opt == "o" || opt == "prefix")
//                {
//
//                    std::string f_name, g_name;
//                    std::tie(f_name, g_name, std::ignore, std::ignore)
//                            = IOStream::parser_url(value);
//
//                    IOStream::current_file_name(f_name);
//
//                }
//                else if (opt == "h" || opt == "help")
//                {
//                    show_help = true;
//                    return TERMINATE;
//                }
//                return CONTINUE;
//            }
//
//    );

//    current_group_name("/");


}

std::string IOStream::pwd() const { return (current_file_name() + ":" + current_group_name()); }

/**
 *
 * @param url =<local path>/<obj name>.<Attribute>
 * @return
 */
std::tuple<std::string, std::string, std::string, std::string>
IOStream::parser_url(std::string const &url_hint) const
{
    std::string file_name(current_file_name());
    std::string grp_name(current_group_name());
    std::string obj_name(""), attribute("");

    if (file_name == "") { file_name = "untitled." + ext_name(); }
    if (grp_name == "") { grp_name = "/"; }

    std::string url = url_hint;

    auto it = url.find(':');

    if (it != std::string::npos)
    {
        file_name = url.substr(0, it);
        url = url.substr(it + 1);
    }

    it = url.rfind('/');

    if (it != std::string::npos)
    {
        grp_name = url.substr(0, it + 1);
        url = url.substr(it + 1);
    }

    it = url.rfind('.');

    if (it != std::string::npos)
    {
        attribute = url.substr(it + 1);
        obj_name = url.substr(0, it);
    }
    else
    {
        obj_name = url;
    }

    return std::make_tuple(file_name, grp_name, obj_name, attribute);

}

IOStream::IOStream()
{
    current_file_name("");
    current_group_name("");
}

IOStream::~IOStream()
{

}

inline bool CheckFileExists(std::string const &name)
{
    if (FILE *file = fopen(name.c_str(), "r"))
    {
        fclose(file);
        return true;
    }
    else { return false; }
}

std::string IOStream::auto_increase_file_name(std::string filename, std::string const &ext_str) const
{
    if ( /* !is_append && fixme need do sth for append*/ GLOBAL_COMM.process_num() == 0)
    {
        std::string prefix = filename;

        if (filename.size() > ext_str.size() && filename.substr(filename.size() - ext_str.size()) == ext_str)
        {
            prefix = filename.substr(0, filename.size() - ext_str.size());
        }

/// @todo auto mkdir directory

        filename = prefix +

                   AutoIncrease(

                           [&](std::string const &suffix) -> bool
                           {
                               std::string f = (prefix + suffix);
                               return
                                       f == ""
                                       || *(f.rbegin()) == '/'
                                       || (CheckFileExists(f + ext_str));
                           }

                   ) + ".h5";

    }

    parallel::bcast_string(&filename);

    return filename;
}
}}