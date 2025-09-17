#ifndef UTILS_H_
#define UTILS_H_

#include <memory>

std::unique_ptr<char[]> base64_encode(const unsigned char *data,size_t input_length,size_t *output_length);


#endif /* UTILS_H_ */
