/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * $Id$
 * */
#ifndef INCLUDE_ERRCHECK_H_
#define INCLUDE_ERRCHECK_H_
#include <mkl_vsl.h>
#include <stdlib.h>
#include <string>
#include "include/log.h"

#define CheckVslError(_NUM_) { int errcode=(_NUM_);\
        if (errcode < 0) {ERROR(vslErrorStr((errcode)));}}

std::string vslErrorStr(int num) {
  std::string res;
  switch (num) {
  case VSL_ERROR_FEATURE_NOT_IMPLEMENTED: {
    res = ((bf("Error: this feature not "
      "implemented yet (code %d).\n") % num).str());
    break;
  }
  case VSL_ERROR_UNKNOWN: {
    res = ((bf("Error: unknown error (code %d).\n") % num).str());
    break;
  }
  case VSL_ERROR_BADARGS: {
    res = ((bf("Error: bad arguments (code %d).\n") % num).str());
    break;
  }
  case VSL_ERROR_MEM_FAILURE: {
    res = ((bf("Error: memory failure. "
      "Memory allocation problem maybe (code %d).\n") % num).str());
    break;
  }
  case VSL_ERROR_NULL_PTR: {
    res = ((bf("Error: null pointer (code %d).\n") % num).str());
    break;
  }
  case VSL_ERROR_INVALID_BRNG_INDEX: {
    res = ((bf("Error: invalid BRNG index (code %d).\n") % num).str());
    break;
  }
  case VSL_ERROR_LEAPFROG_UNSUPPORTED: {
    res = ((bf("Error: leapfrog initialization is unsupported (code %d).\n")
        % num).str());
    break;
  }
  case VSL_ERROR_SKIPAHEAD_UNSUPPORTED: {
    res = ((bf("Error: skipahead initialization is unsupported (code %d).\n")
        % num).str());
    break;
  }
  case VSL_ERROR_BRNGS_INCOMPATIBLE: {
    res
        = ((bf("Error: BRNGs are not compatible for the operation (code %d).\n")
            % num).str());
    break;
  }
  case VSL_ERROR_BAD_STREAM: {
    res = ((bf("Error: random stream is invalid (code %d).\n") % num).str());
    break;
  }
  case VSL_ERROR_BRNG_TABLE_FULL: {
    res = ((bf("Error: table of"
      " registered BRNGs is full (code %d).\n") % num).str());
    break;
  }
  case VSL_ERROR_BAD_STREAM_STATE_SIZE: {
    res = ((bf("Error: value in StreamStateSize field is bad (code %d).\n")
        % num).str());
    break;
  }
  case VSL_ERROR_BAD_WORD_SIZE: {
    res = ((bf("Error: value in "
      "WordSize field is bad (code %d).\n") % num).str());
    break;
  }
  case VSL_ERROR_BAD_NSEEDS: {
    res = ((bf("Error: value in"
      " NSeeds field is bad (code %d).\n") % num).str());
    break;
  }
  case VSL_ERROR_BAD_NBITS: {
    res = ((bf("Error: value in"
      " NBits field is bad (code %d).\n") % num).str());
    break;
  }
  case VSL_ERROR_BAD_UPDATE: {
    res = ((bf("Error: number of updated"
      " entries in buffer is invalid (code %d).\n") % num).str());
    break;
  }
  case VSL_ERROR_NO_NUMBERS: {
    res = ((bf("Error: zero number of updated entries in buffer (code %d).\n")
        % num).str());
    break;
  }
  case VSL_ERROR_INVALID_ABSTRACT_STREAM: {
    res = ((bf("Error: abstract"
      " random stream is invalid (code %d).\n") % num).str());
    break;
  }
  case VSL_ERROR_FILE_CLOSE: {
    res = ((bf("Error: can`t close file (code %d).\n") % num).str());
    break;
  }
  case VSL_ERROR_FILE_OPEN: {
    res = ((bf("Error: can`t open file (code %d).\n") % num).str());
    break;
  }
  case VSL_ERROR_FILE_WRITE: {
    res = ((bf("Error: can`t write to file (code %d).\n") % num).str());
    break;
  }
  case VSL_ERROR_FILE_READ: {
    res = ((bf("Error: can`t read from file (code %d).\n") % num).str());
    break;
  }
  case VSL_ERROR_BAD_FILE_FORMAT: {
    res = ((bf("Error: file format is unknown (code %d).\n") % num).str());
    break;
  }
  case VSL_ERROR_UNSUPPORTED_FILE_VER: {
    res = ((bf("Error: unsupported file version (code %d).\n") % num).str());
    break;
  }
  }

  return res;
}
#endif  // INCLUDE_ERRCHECK_H_
