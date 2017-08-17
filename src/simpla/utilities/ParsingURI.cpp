//
// Created by salmon on 17-8-17.
//
#include "ParsingURI.h"
#include <regex>
#include <tuple>
#include "Log.h"

namespace simpla {
static std::regex uri_regex(R"(^(([^:\/?#]+):)?(\/\/([^\/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?)");
static std::regex file_extension_regex(R"(^(.*)(\.([[:alnum:]]+))$)");

std::tuple<std::string, std::string, std::string, std::string, std::string> ParsingURI(std::string const &uri) {
    std::string scheme;
    std::string path = uri;
    std::smatch uri_match_result;
    std::string authority;
    std::string query;
    std::string fragment;
    if (!uri.empty()) {
        if (std::regex_match(uri, uri_match_result, uri_regex)) {
            //        for (size_type i = 0, ie = uri_match_result.size(); i < ie; ++i) {
            //            std::cout << i << "\t:" << uri_match_result.str(i) << std::endl;
            //        }
            scheme = (uri_match_result.str(2) == "file") ? "" : uri_match_result.str(2);
            authority = uri_match_result.str(4);
            path = uri_match_result.str(5);
            query = uri_match_result.str(7);
            fragment = uri_match_result.str(9);
        }

        if (scheme.empty()) {
            if (std::regex_match(path, uri_match_result, file_extension_regex)) { scheme = uri_match_result.str(3); }
        }
        if (scheme.empty()) {
            RUNTIME_ERROR << "illegal URI: [" << uri_match_result.str(0) << "|" << uri_match_result.str(1) << "|"
                          << uri_match_result.str(2) << "|" << uri_match_result.str(3) << "]" << std::endl;
        }
    }
    return std::make_tuple(scheme, authority, path, query, fragment);
};

}  // namespace simpla