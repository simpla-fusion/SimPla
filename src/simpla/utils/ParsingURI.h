//
// Created by salmon on 17-8-17.
//

#ifndef SIMPLA_PARSINGURI_H
#define SIMPLA_PARSINGURI_H

#include <string>
#include <utility>
namespace simpla {

/**
*   https://tools.ietf.org/html/rfc3986#page-50
*
*   Appendix B.  Parsing a URI Reference with a Regular Expression
*
*     As the "first-match-wins" algorithm is identical to the "greedy"
*     disambiguation method used by POSIX regular expressions, it is
*     natural and commonplace to use a regular expression for parsing the
*     potential five components of a URI reference.
*
*     The following line is the regular expression for breaking-down a
*     well-formed URI reference into its components.
*
*   ^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?
*    12            3  4          5       6  7        8 9
*
*     The numbers in the second line above are only to assist readability;
*     they indicate the reference points for each subexpression (i.e., each
*     paired parenthesis).  We refer to the value matched for subexpression
*     <n> as $<n>.  For example, matching the above expression to
*
*        http://www.ics.uci.edu/pub/ietf/uri/#Related
*
*     results in the following subexpression matches:
*
*        $1 = http:
*        $2 = http
*        $3 = //www.ics.uci.edu
*        $4 = www.ics.uci.edu
*        $5 = /pub/ietf/uri/
*        $6 = <undefined>
*        $7 = <undefined>
*        $8 = #Related
*        $9 = Related
*
*     where <undefined> indicates that the component is not present, as is
*     the case for the query component in the above example.  Therefore, we
*     can determine the value of the five components as
*
*        scheme    = $2
*        authority = $4
*        path      = $5
*        query     = $7
*        fragment  = $9
*
*     Going in the opposite direction, we can recreate a URI reference from
*     its components by using the algorithm of Section 5.3.
*
**/
std::tuple<std::string, std::string, std::string, std::string, std::string> ParsingURI(std::string const &uri);

}  // namespace simpla
#endif  // SIMPLA_PARSINGURI_H
