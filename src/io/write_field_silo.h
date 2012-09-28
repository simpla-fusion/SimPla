/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * WriteSilo.h
 *
 *  Created on: 2011-1-27
 *      Author: salmon
 */

#ifndef SRC_IO_WRITE_SILO_H_
#define SRC_IO_WRITE_SILO_H_

#define BOOST_FILESYSTEM_VERSION 3
#include <boost/filesystem.hpp>
#include <string>
#include <list>
#include "engine/context.h"
namespace IO {
class WriteSilo {
    Context & ctx_;

    H5::Group grp_;

    boost::filesystem::path path_;

    std::list<std::string> records_;

  public:

    WriteSilo(Context & ctx, const std::string & path,
        std::list<std::string> const & rec_list);
    ~WriteSilo();
    void run();
};
}  // namespace IO
#endif  // SRC_IO_WRITE_SILO_H_
