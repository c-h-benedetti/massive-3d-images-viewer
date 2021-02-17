#ifndef COMPARE_FILES_HPP_INCLUDED
#define COMPARE_FILES_HPP_INCLUDED

#include <stdint.h>

uint64_t compare_files(const char* f1, const char* f2, uint64_t max, bool& same);

#endif //COMPARE_FILES_HPP_INCLUDED
