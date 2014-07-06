/*
 * data_wrapper.h
 *
 *  Created on: 2014-5-7
 *      Author: salmon
 */

#ifndef DATA_WRAPPER_H_
#define DATA_WRAPPER_H_

namespace simpla
{

template<typename T>
class DataWrapper
{
	void Load() const;
	void Save() const;
	void Sync() const;
};
}  // namespace simpla

#endif /* DATA_WRAPPER_H_ */
