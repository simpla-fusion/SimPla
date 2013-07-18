/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * write_silo.h
 *
 *  Created on: 2011-3-30
 *      Author: salmon
 */

#ifndef WRITE_SILO_H_
#define WRITE_SILO_H_
#include <string>
#include <list>

class Context;

namespace IO {
class WriteSilo {
	Context * ctx_;
	SizeType step_;
	DBfile * silo_;

	std::list<std::string> records_;

public:
	WriteSilo(Context * ctx, SizeType step, const std::string & path,
			std::list<std::string> const & rec_list);

	~WriteSilo();

	void wirteField();
	void wirteParticle();
};
} // namespace IO

#endif /* WRITE_SILO_H_ */
