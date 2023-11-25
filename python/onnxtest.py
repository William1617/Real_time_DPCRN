import onnxruntime
import soundfile as sf
import numpy as np
import time


block_len=400
block_shift=200
decode_framenum=20
pi=3.142592653589793238462643383279
dpcrn=onnxruntime.InferenceSession('./dpcrn_rt.onnx')
model_input_name= [inp.name for inp in dpcrn.get_inputs()]
model_input = {
            inp.name: np.zeros(
                [dim if isinstance(dim, int) else 1 for dim in inp.shape],
                dtype=np.float32)
            for inp in dpcrn.get_inputs()}
windows=np.sin(np.arange(.5,block_len-.5+1)/block_len*np.pi)


audio,sr=sf.read('./test2.wav')
frame_num  =(len(audio)//block_shift -2)
print(frame_num)

input_frame=np.zeros((1,400))


new_audio=np.zeros((len(audio)))

start_time=time.time()
for idx in range(frame_num):
    
    input_frame[0] = audio[idx*block_shift:idx*block_shift+block_len]

    in_frame=windows*input_frame

    input_complex=np.fft.rfft(in_frame)
    input_real=np.real(input_complex)
    input_img = np.imag(input_complex)
    input_real =np.reshape(input_real,(1,1,201,1)).astype('float32')
    input_img = np.reshape(input_img,(1,1,201,1)).astype('float32')
    
    
    model_input[model_input_name[0]] = input_real
    model_input[model_input_name[1]] = input_img

    
    model_output = dpcrn.run(None, model_input)
    enhance_frame_real=model_output[0]
    enhance_frame_img=model_output[1]
    
    for idx3 in range(2,16):
        model_input[model_input_name[idx3]] = model_output[idx3] 


    enhance_frame_real=np.squeeze(enhance_frame_real)
    enhance_frame_img=np.squeeze(enhance_frame_img)
    enhance_frame_complex=enhance_frame_real +1j*enhance_frame_img
    enhance_frame=np.fft.irfft(enhance_frame_complex)
    enhance_frame *=windows
    
    
    new_audio[idx*block_shift:idx*block_shift+block_len] +=enhance_frame
sf.write('./test1out.wav',new_audio,16000)
audio_time=len(audio)/16000
print((time.time()-start_time)/audio_time)

