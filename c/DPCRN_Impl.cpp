
 
#include "DTLN_defs.h"
#include "AudioFile.h"


void ExportWAV(
        const std::string & Filename, 
		const std::vector<float>& Data, 
		unsigned SampleRate) {
    AudioFile<float>::AudioBuffer Buffer;
	Buffer.resize(1);

	Buffer[0] = Data;
	size_t BufSz = Data.size();

	AudioFile<float> File;
	File.setAudioBuffer(Buffer);
	File.setAudioBufferSize(1, (int)BufSz);
	File.setNumSamplesPerChannel((int)BufSz);
	File.setNumChannels(1);
	File.setBitDepth(16);
	File.setSampleRate(SAMEPLERATE);
	File.save(Filename, AudioFileFormat::Wave);		
}

void DPRCN() {


    trg_engine* m_pEngine;

    m_pEngine = new trg_engine;

	// load model
	m_pEngine->model = TfLiteModelCreateFromFile(MODEL_NAME);

    // Build the interpreter
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options, 1);

    // Create the interpreter.
    m_pEngine->interpreter = TfLiteInterpreterCreate(m_pEngine->model_a, options);
    if (m_pEngine->interpreter == nullptr) {
        printf("Failed to create interpreter\n");
        return ;
    }


    // Allocate tensor buffers.
    if (TfLiteInterpreterAllocateTensors(m_pEngine->interpreter) != kTfLiteOk) {
        printf("Failed to allocate tensors a!\n");
        return;
    }
    //Set input and output
    for(int =0;i<6;i++){
        m_pEngine->input_details[i] = TfLiteInterpreterGetInputTensor(m_pEngine->interpreter, i);
        m_pEngine->output_details[i] = TfLiteInterpreterGetOutputTensor(m_pEngine->interpreter, 0);

    }
    //Cal dft directly as Block_len is not the power of 2
    float cos_f[BLOCK_LEN] ={0};
    float sin_f[BLOCK_LEN] ={0};
    for (int i=0;i<BLOCK_LEN;i++){
        cos_f[i]=cosf(2*PI*i/BLOCK_LEN);
        sin_f[i]=-sinf(2*PI*i/BLOCK_LEN);

    }

    std::vector<float>  testdata; //vector used to store enhanced data in a wav file
    AudioFile<float> inputmicfile;
    std::string micfile="./wav/test.wav";
    inputmicfile.load(micfile);
    int audiolen=inputmicfile.getNumSamplesPerChannel();
    int process_num=audiolen/BLOCK_SHIFT-2;
    //for BLOCK_LEN input samples,do process_num infer
    for(int i=0;i<process_num;i++)
    {
        memmove(m_pEngine->mic_buffer, m_pEngine->mic_buffer + BLOCK_SHIFT, (BLOCK_LEN - BLOCK_SHIFT) * sizeof(float));
       
        for(int n=0;n<BLOCK_SHIFT;n++){
                m_pEngine->mic_buffer[n+BLOCK_LEN-BLOCK_SHIFT]=inputmicfile.samples[0][n+i*BLOCK_SHIFT];} 
        DTLNAECInfer(m_pEngine,cos_f,sin_f);
        for(int j=0;j<BLOCK_SHIFT;j++){
            testdata.push_back(m_pEngine->out_buffer[j]);    //for one forward process save first BLOCK_SHIFT model output samples
        }
    }
    ExportWAV("aectest.wav",testaecdata,SAMEPLERATE);


 }
 
void DPCRNInfer(trg_engine* m_pEngine, float* cos_f, float* sin_f) {

	float in_real[FFT_OUT_SIZE] = { 0 };
    float in_imag[FFT_OUT_SIZE] = { 0 };
    float estimated_block[BLOCK_LEN];
    
	for (int k = 0; k < FFT_OUT_SIZE; k++){
        float sum_real=0;
        float sum_img=0;
        for (int i=0;i<Block_len;i++){
            int coef_id=(k*i)%Block_len;
            sum_real +=m_pEngine->mic_buffer[i]*cos_f[coef_id];
            sum_img +=_pEngine->mic_buffer[i]*sin_f[coef_id];
        }
        in_real[k] =sum_real;
        in_imag[k] =sum_img;
	}
    memmove(m_pEngine->real_buffer, m_pEngine->real_buffer + FFT_OUT_SIZE, 10*FFT_OUT_SIZE * sizeof(float));
    memcpy(m_pEngine->real_buffer+10*FFT_OUT_SIZE,in_real,FFT_OUT_SIZE*sizeof(float));

    memmove(m_pEngine->imag_buffer, m_pEngine->imag_buffer + FFT_OUT_SIZE, 10*FFT_OUT_SIZE * sizeof(float));
    memcpy(m_pEngine->imag_buffer+10*FFT_OUT_SIZE,in_imag,FFT_OUT_SIZE*sizeof(float));
    //the data input of first model is the magnitude of input wav data
    TfLiteTensorCopyFromBuffer(m_pEngine->input_details_a[0], m_pEngine->real_buffer, 11*FFT_OUT_SIZE * sizeof(float));
    TfLiteTensorCopyFromBuffer(m_pEngine->input_details_a[1], m_pEngine->imag_buffer, 11*FFT_OUT_SIZE * sizeof(float));
    TfLiteTensorCopyFromBuffer(m_pEngine->input_details_a[2], m_pEngine->states_h1, STATE_SIZE * sizeof(float));
    TfLiteTensorCopyFromBuffer(m_pEngine->input_details_a[3], m_pEngine->states_c1, STATE_SIZE * sizeof(float));
    TfLiteTensorCopyFromBuffer(m_pEngine->input_details_a[4], m_pEngine->states_h2, STATE_SIZE * sizeof(float));
    TfLiteTensorCopyFromBuffer(m_pEngine->input_details_a[5], m_pEngine->states_c2, STATE_SIZE * sizeof(float));

    if (TfLiteInterpreterInvoke(m_pEngine->interpreter_a) != kTfLiteOk) {
        printf("Error invoking detection model\n");
    }

    float out_real[FFT_OUT_SIZE];
    float out_img[FFT_OUT_SIZE];
    TfLiteTensorCopyToBuffer(m_pEngine->output_details_a[0], out_real, FFT_OUT_SIZE * sizeof(float));
    TfLiteTensorCopyToBuffer(m_pEngine->output_details_a[1], out_img, FFT_OUT_SIZE * sizeof(float));
    //the putput state of current block will become the input state of next block
    TfLiteTensorCopyToBuffer(m_pEngine->output_details_a[2], m_pEngine->states_h1, STATE_SIZE * sizeof(float));
    TfLiteTensorCopyToBuffer(m_pEngine->output_details_a[3], m_pEngine->states_c1, STATE_SIZE * sizeof(float));
    TfLiteTensorCopyToBuffer(m_pEngine->output_details_a[4], m_pEngine->states_h2, STATE_SIZE * sizeof(float));
    TfLiteTensorCopyToBuffer(m_pEngine->output_details_a[5], m_pEngine->states_c2, STATE_SIZE * sizeof(float));

    //ifft
    float out_block[BLOCK_LEN];
	
    for (int k=0;k<BLOCK_LEN;k++){
        float buffer_sum=0;
        for(int i=0;i<BLOCK_LEN;i++){
            int coef_id=(k*i)%Block_len;
            if(i<FFT_OUT_SIZE){
                buffer_sum +=out_real[i]*cos_f[coef_id] +out_img[i]*sin_f[coef_id];
            }else{
                buffer_sum +=out_real[i]*cos_f[coef_id] -out_img[i]*sin_f[coef_id];
            }

        }
        out_block[k] =buffer_sum/Block_len;
    }

    //apply overlap_add
    memmove(m_pEngine->out_buffer, m_pEngine->out_buffer + BLOCK_SHIFT, (BLOCK_LEN - BLOCK_SHIFT) * sizeof(float));
    memset(m_pEngine->out_buffer + (BLOCK_LEN - BLOCK_SHIFT), 0, BLOCK_SHIFT * sizeof(float));
    for (int i = 0; i < BLOCK_LEN; i++)
        m_pEngine->out_buffer[i] += out_block[i];

}
 



