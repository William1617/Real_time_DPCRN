import onnxruntime
import soundfile as sf
import numpy as np


block_len=400
block_shift=200
pi=3.142592653589793238462643383279
dpcrn=onnxruntime.InferenceSession('./dpcrn_rt.onnx')
model_input_name= [inp.name for inp in dpcrn.get_inputs()]
model_input = {
            inp.name: np.zeros(
                [dim if isinstance(dim, int) else 1 for dim in inp.shape],
                dtype=np.float32)
            for inp in dpcrn.get_inputs()}
windows=np.sin(np.arange(.5,block_len-.5+1)/block_len*np.pi)

dpcrn_2=onnxruntime.InferenceSession('./dpcrn_rt.onnx')

audio,sr=sf.read('./test2.wav')
frame_num  =len(audio)//block_shift -2

input_frame=np.zeros((11,400))


new_audio=np.zeros((len(audio)))


for idx in range(frame_num):
    input_frame[:-1,:] =input_frame[1:,:]
    input_frame[-1] = audio[block_shift*idx:block_shift*idx+block_len]

    in_frame=windows*input_frame

    input_complex=np.fft.rfft(in_frame)
    input_real=np.real(input_complex)
    input_img = np.imag(input_complex)
    #print(input_real[5:8,10])
    input_real =np.reshape(input_real,(1,11,201,1)).astype('float32')
    input_img = np.reshape(input_img,(1,11,201,1)).astype('float32')
    
    model_input[model_input_name[0]] = input_real
    model_input[model_input_name[1]] = input_img

    
    model_output = dpcrn.run(None, model_input)
    enhance_frame_real=model_output[0]
    enhance_frame_img=model_output[1]
    model_input[model_input_name[2]] = model_output[2] 
    model_input[model_input_name[3]] = model_output[3] 
    model_input[model_input_name[4]] = model_output[4] 
    model_input[model_input_name[5]] = model_output[5] 
    model_input[model_input_name[6]] = model_output[6] 

    enhance_frame_real=np.squeeze(enhance_frame_real)
   
    enhance_frame_img=np.squeeze(enhance_frame_img)
    enhance_frame_complex=enhance_frame_real +1j*enhance_frame_img
    enhance_frame=np.fft.irfft(enhance_frame_complex)
    enhance_frame *=windows

    new_audio[idx*block_shift:idx*block_shift+block_len] +=enhance_frame
sf.write('./test2out.wav',new_audio,16000)

