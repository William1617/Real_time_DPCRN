
 
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
    m_pEngine->cache_buffer.resize(CACHE_NUM);
    int cacache_size[CACHE_NUM]={203*2,101*32,52*32,52*32,52*64,50*256,50*128,50*64,50*64,100*64,50*128,50*128,50*128,50*128};
    for (int i=0;i<CACHE_NUM;i++){
        m_pEngine->cache_buffer[i].resize(cacache_size[i],0.0);
    }

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
    for(int =0;i<CACHE_NUM+2;i++){
        m_pEngine->input_details[i] = TfLiteInterpreterGetInputTensor(m_pEngine->interpreter, i);
        m_pEngine->output_details[i] = TfLiteInterpreterGetOutputTensor(m_pEngine->interpreter, i);

    }
    //Cal dft directly as Block_len is not the power of 2
    float cos_f[BLOCK_LEN] ={0};
    float sin_f[BLOCK_LEN] ={0};
    float windows[BLOCK_LEN]={0};
    for (int i=0;i<BLOCK_LEN;i++){
        cos_f[i]=cosf(2*PI*i/BLOCK_LEN);
        sin_f[i]=-sinf(2*PI*i/BLOCK_LEN);
        windows[i]=sinf((0.5+i)/BLOCK_LEN*PI);

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
        DPCRNInfer(m_pEngine,cos_f,sin_f,windows);
        for(int j=0;j<BLOCK_SHIFT;j++){
            testdata.push_back(m_pEngine->out_buffer[j]);    //for one forward process save first BLOCK_SHIFT model output samples
        }
    }
    ExportWAV("dpcrntest.wav",testdata,SAMEPLERATE);
    TfLiteInterpreterDelete(m_pEngine->interpreter);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(m_pEngine->model);
    delete m_pEngine;


 }
 
void DPCRNInfer(trg_engine* m_pEngine, float* cos_f, float* sin_f,float* windows) {

	float in_real[FFT_OUT_SIZE] = { 0 };
    float in_imag[FFT_OUT_SIZE] = { 0 };
    float estimated_block[BLOCK_LEN];
    
	for (int k = 0; k < FFT_OUT_SIZE; k++){
        float sum_real=0;
        float sum_img=0;
        for (int i=0;i<Block_len;i++){
            int coef_id=(k*i)%Block_len;
            sum_real +=m_pEngine->mic_buffer[i]*cos_f[coef_id]*windows[i];
            sum_img +=_pEngine->mic_buffer[i]*sin_f[coef_id]*windows[i];
        }
        in_real[k] =sum_real;
        in_imag[k] =sum_img;
	}

    int cache_size;
   
    TfLiteTensorCopyFromBuffer(m_pEngine->input_details[0], in_real, FFT_OUT_SIZE * sizeof(float));
    TfLiteTensorCopyFromBuffer(m_pEngine->input_details[1], in_imag, FFT_OUT_SIZE * sizeof(float));
    for(int i=0;i<CACHE_NUM;i++){
        cache_size= m_pEngine->cache_buffer[i].size();
        TfLiteTensorCopyFromBuffer(m_pEngine->input_details[i+2], m_pEngine->cache_buffer[i].data(), cacache_size * sizeof(float));

    }
    
    if (TfLiteInterpreterInvoke(m_pEngine->interpreter) != kTfLiteOk) {
        printf("Error invoking detection model\n");
    }

    float out_real[FFT_OUT_SIZE];
    float out_img[FFT_OUT_SIZE];
    TfLiteTensorCopyToBuffer(m_pEngine->output_details[0], out_real, FFT_OUT_SIZE * sizeof(float));
    TfLiteTensorCopyToBuffer(m_pEngine->output_details[1], out_img, FFT_OUT_SIZE * sizeof(float));
    
    for (int i=0;i<CACHE_NUM;i++){
        cache_size= m_pEngine->cache_buffer[i].size();
        TfLiteTensorCopyToBuffer(m_pEngine->output_details[i+2], m_pEngine->cache_buffer[i].data(), cache_size * sizeof(float));

    }
   

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
        m_pEngine->out_buffer[i] += out_block[i]*windows[i];

}
 



