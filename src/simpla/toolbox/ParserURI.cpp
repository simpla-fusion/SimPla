//
// Created by salmon on 17-3-11.
//
#include "ParserURI.h"
#include <simpla/toolbox/Log.h>
#include <regex>
namespace simpla {
namespace toolbox {

/**
 *  @quota https://tools.ietf.org/html/rfc3986#page-50
 *
 *     ^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?
 *      12            3  4          5       6  7        8 9
 *
 *  The numbers in the second line above are only to assist readability;
 *  they indicate the reference points for each subexpression (i.e., each
 *  paired parenthesis).  We refer to the value matched for subexpression
 *  <n> as $<n>.  For example, matching the above expression to
 *
 *     http://www.ics.uci.edu/pub/ietf/uri/#Related
 *
 *  results in the following subexpression matches:
 *
 *     $1 = http:
 *     $2 = http
 *     $3 = //www.ics.uci.edu
 *     $4 = www.ics.uci.edu
 *     $5 = /pub/ietf/uri/
 *     $6 = <undefined>
 *     $7 = <undefined>
 *     $8 = #Related
 *     $9 = Related
 *
 *  where <undefined> indicates that the component is not present, as is
 *  the case for the query component in the above example.  Therefore, we
 *  can determine the value of the five components as
 *
 *     scheme    = $2
 *     authority = $4
 *     path      = $5
 *     query     = $7
 *     fragment  = $9
 *
 *  Going in the opposite direction, we can recreate a URI reference from
 *  its components by using the algorithm of Section 5.3.
 */

/** <scheme , authority ,path ,query,fragment> */
std::tuple<std::string, std::string, std::string, std::string, std::string> ParseURI(std::string const &uri) {
    static std::regex url_regex(
        R"(^(([^:\/?#]+):)?(//([^\/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?)", std::regex::extended);
    std::smatch url_match_result;

    if (!std::regex_match(uri, url_match_result, url_regex)) {
        RUNTIME_ERROR << " illegal uri! [" << uri << "]" << std::endl;
    }
    return std::make_tuple(url_match_result[2].str(), url_match_result[4].str(), url_match_result[5].str(),
                           url_match_result[7].str(), url_match_result[9].str());
};
}  // namespace toolbox{
}  // namespace simpla{