
import soundfile as sf
import tflite_runtime.interpreter as tflite

import numpy as np

block_len=400
block_shift=200
fft_num=block_len//2+1

interpreter = tflite.Interpreter('./dpcrn_rt.tflite')
interpreter.allocate_tensors()

audio,sr=sf.read('./test2.wav')
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
in_h1=np.zeros((1,50,128)).astype('float32')
in_c1=np.zeros((1,50,128)).astype('float32')

in_h2=np.zeros((1,50,128)).astype('float32')
in_c2=np.zeros((1,50,128)).astype('float32')
rnn_cache=np.zeros((1,5,50,128)).astype('float32')

in_real=np.zeros(11*fft_num)
in_imag=np.zeros(11*fft_num)

frame_num=len(audio)//block_shift -2
windows=np.sin(np.arange(.5,block_len-.5+1)/block_len*np.pi)
out_audio=np.zeros(len(audio))

for idx in range(frame_num):
    audip_frame=audio[idx*block_shift:idx*block_shift+block_len]
    frame_fft=np.fft.rfft(audip_frame*windows)
    frame_real=np.real(frame_fft)
    frame_img=np.imag(frame_fft)

    in_real[:10*fft_num] = in_real[fft_num:]
    in_imag[:10*fft_num] = in_imag[fft_num:]
    in_real[-fft_num:]=frame_real
    in_imag[-fft_num:]=frame_img

    interpreter.set_tensor(input_details[0]['index'] ,np.reshape(in_real,(1,11,201,1)).astype('float32'))
    interpreter.set_tensor(input_details[1]['index'] ,np.reshape(in_imag,(1,11,201,1)).astype('float32'))

    interpreter.set_tensor(input_details[2]['index'] ,rnn_cache)
    interpreter.set_tensor(input_details[3]['index'] ,in_h1)
    interpreter.set_tensor(input_details[4]['index'] ,in_c1)
    interpreter.set_tensor(input_details[5]['index'] ,in_h2)
    interpreter.set_tensor(input_details[6]['index'] ,in_c2)

    interpreter.invoke()
    out_real=interpreter.get_tensor(output_details[0]['index'])
    out_imag=interpreter.get_tensor(output_details[1]['index'])
    rnn_cache=interpreter.get_tensor(output_details[2]['index'])
    in_h1=interpreter.get_tensor(output_details[3]['index'])
    in_c1=interpreter.get_tensor(output_details[4]['index'])
    in_h2=interpreter.get_tensor(output_details[5]['index'])
    in_c2=interpreter.get_tensor(output_details[6]['index'])

    enhance_frame=np.fft.irfft(np.squeeze(out_real)+1j*np.squeeze(out_imag))

    enhance_frame *=windows

    out_audio[idx*block_shift:idx*block_shift+block_len] +=enhance_frame
sf.write('./test2out.wav',out_audio,16000)