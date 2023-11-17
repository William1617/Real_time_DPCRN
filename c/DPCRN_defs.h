
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
#define STATE_SIZE      (50*128)
#define RNN_CACHE_SIZE (5*50*128)

#define MODEL_NAME "dpcrn_rt.tflite"
#define PI 3.141592653589793238

#define SAMEPLERATE  (16000)

#define BLOCK_SHIFT  (200)

struct trg_engine {
    float mic_buffer[BLOCK_LEN] = { 0 };
    float out_buffer[BLOCK_LEN] = { 0 };

    float real_buffer[FFT_OUT_SIZE*11] = {0};
    float imag_buffer[FFT_OUT_SIZE*11] = {0};
    float rnn_cache[RNN_CACHE_SIZE] = {0};
    float states_h1[STATE_SIZE] = { 0 };
    float states_c1[STATE_SIZE] = { 0 };
    float states_h2[STATE_SIZE] = { 0 };
    float states_c2[STATE_SIZE] = { 0 };

    TfLiteTensor* input_details_a[7];
    const TfLiteTensor* output_details_a[7]
    TfLiteInterpreter* interpreter;
    TfLiteModel* model;
};





#endif 



