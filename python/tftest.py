
import soundfile as sf
import tflite_runtime.interpreter as tflite

import numpy as np
import time

block_len=400
block_shift=200
fft_num=block_len//2+1

interpreter = tflite.Interpreter('./dpcrn_rt.tflite')
interpreter.allocate_tensors()

audio,sr=sf.read('./test2.wav')
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

conv_caches=[]
conv_caches.append(np.zeros((1,1,203,2)).astype('float32'))
conv_caches.append(np.zeros((1,1,101,32)).astype('float32'))
conv_caches.append(np.zeros((1,1,52,32)).astype('float32'))
conv_caches.append(np.zeros((1,1,52,32)).astype('float32'))
conv_caches.append(np.zeros((1,1,52,64)).astype('float32'))

deconv_caches=[]
deconv_caches.append(np.zeros((1,1,50,256)).astype('float32'))
deconv_caches.append(np.zeros((1,1,50,128)).astype('float32'))
deconv_caches.append(np.zeros((1,1,50,64)).astype('float32'))
deconv_caches.append(np.zeros((1,1,50,64)).astype('float32'))
deconv_caches.append(np.zeros((1,1,100,64)).astype('float32'))

state_caches=[]
for k in range(4):
    state_caches.append(np.zeros((1,50,128)).astype('float32'))

in_real=np.zeros(fft_num)
in_imag=np.zeros(fft_num)

frame_num=len(audio)//block_shift -2
windows=np.sin(np.arange(.5,block_len-.5+1)/block_len*np.pi)
out_audio=np.zeros(len(audio))
start_time=time.time()
for idx in range(frame_num):
    audip_frame=audio[idx*block_shift:idx*block_shift+block_len]
    frame_fft=np.fft.rfft(audip_frame*windows)
    frame_real=np.real(frame_fft)
    frame_img=np.imag(frame_fft)

    

    interpreter.set_tensor(input_details[0]['index'] ,np.reshape(frame_real,(1,1,201,1)).astype('float32'))
    interpreter.set_tensor(input_details[1]['index'] ,np.reshape(frame_img,(1,1,201,1)).astype('float32'))
    
    for id2 in range(2,16):
        if(id2<7):
            interpreter.set_tensor(input_details[id2]['index'] ,conv_caches[id2-2])
        elif(id2<12):
            interpreter.set_tensor(input_details[id2]['index'] ,deconv_caches[id2-7])
        else:
            interpreter.set_tensor(input_details[id2]['index'] ,state_caches[id2-12])



    interpreter.invoke()
    out_real=interpreter.get_tensor(output_details[0]['index'])
    out_imag=interpreter.get_tensor(output_details[1]['index'])

    for id3 in range(2,16):
        if(id3<7):
            conv_caches[id3-2]=interpreter.get_tensor(output_details[id3]['index'])
        elif(id3<12):
            deconv_caches[id3-7] =interpreter.get_tensor(output_details[id3]['index'])
        else:
            state_caches[id3-12]=interpreter.get_tensor(output_details[id3]['index'])


    enhance_frame=np.fft.irfft(np.squeeze(out_real)+1j*np.squeeze(out_imag))

    enhance_frame *=windows

    out_audio[idx*block_shift:idx*block_shift+block_len] +=enhance_frame
sf.write('./test3out.wav',out_audio,16000)
audio_time=len(audio)/16000
print((time.time()-start_time)/audio_time)