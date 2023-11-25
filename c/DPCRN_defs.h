
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_experimental.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

#include <vector>
#include <string>
#include <cmath>

#include <fcntl.h>
#include <getopt.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>


#define BLOCK_LEN		(400)
#define FFT_OUT_SIZE    (BLOCK_LEN / 2 + 1)
#define CACHE_NUM      (14)

#define MODEL_NAME "dpcrn_rt.tflite"
#define PI 3.141592653589793238

#define SAMEPLERATE  (16000)

#define BLOCK_SHIFT  (200)

struct trg_engine {
    float mic_buffer[BLOCK_LEN] = { 0 };
    float out_buffer[BLOCK_LEN] = { 0 };

    std::vector<std::vector<float>> cache_buffer;

    TfLiteTensor* input_details[CACHE_NUM+2];
    const TfLiteTensor* output_details[CACHE_NUM+2]
    TfLiteInterpreter* interpreter;
    TfLiteModel* model;
};


#endif 



