/**
 * @file  parse_command_line.cpp
 *
 *  Created on: 2014-11-21
 *      Author: salmon
 */

#include "parse_command_line.h"

#include <stddef.h>
#include <algorithm>

namespace simpla {
namespace gtl {

std::string trim(std::string const &s)
{
    std::string value = s;

    size_t first = value.find_first_not_of(' ');
    size_t last = value.find_last_not_of(' ');
    if (last != first)
    {
        value = value.substr(first, (last - first + 1));
    }

    return std::move(value);
}

void parse_cmd_line(int argc, char **argv,
                    std::function<int(std::string const &, std::string const &)> const &options)
{
    if (argc <= 1 || argv == nullptr)
    {
        return;
    }

    std::string opt = "";
    std::string value = "";

    for (int i = 0; i < argc; ++i)
    {
        char *str = argv[i];

        if (str[0] == '-'
            && ((str[1] < '0' || str[1] > '9') && (str[1] != '.'))) // is configure flag
        {
            if (opt != "" || value != "")
            {
                if (options(opt, trim(value)) == TERMINATE)
                {
                    return;
                }
                opt = "";
                value = "";
            }

            if (str[1] == '-') // is long configure flag
            {
                opt = str + 2;
            }
            else // is short configure flag
            {
                opt = str[1];
                if (str[2] != '\0')
                {
                    value = str + 2;
                }
            }

        }
        else
        {
            value += " ";
            value += str;
        }
    }

    if (opt != "" || value != "")
    {
        options(opt, trim(value));
    }

}

std::tuple<bool, std::string> find_option_from_cmd_line(int argc, char **argv,
                                                        std::string const &key)
{
    std::string res("");
    bool is_found = false;

    parse_cmd_line(argc, argv,

                   [&](std::string const &opt, std::string const &value) -> int
                       {
                       if (opt == "V")
                       {
                           res = value;
                           is_found = true;
                           return TERMINATE;
                       }
                       else
                       {
                           return CONTINUE;
                       }
                       }

    );
    return std::make_tuple(is_found, res);
}

}
}//  namespace simpla::gtl
