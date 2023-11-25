import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Input, Conv2D, BatchNormalization, Conv2DTranspose, Concatenate, LayerNormalization, PReLU,ZeroPadding2D,Activation

import tf2onnx

#from modules import DprnnBlock
from stream_module import DprnnBlock

class DPCRN_model():
    def __init__(self):
        self.model=None
        self.numDP = 2
        self.numUnits=128
    
    def mk_mask(self,x):
        '''
        Method for complex ratio mask and add helper layer used with a Lambda layer.
        '''
        [noisy_real,noisy_imag,mask] = x
        noisy_real = noisy_real[:,:,:,0]
        noisy_imag = noisy_imag[:,:,:,0]
        
        mask_real = mask[:,:,:,0]
        mask_imag = mask[:,:,:,1]
        
        enh_real = noisy_real * mask_real - noisy_imag * mask_imag
        enh_imag = noisy_real * mask_imag + noisy_imag * mask_real
        
        return [enh_real,enh_imag]


    def export_real_time_model(self,save_name,quant=False,is_tflite=True,name = 'model0'):
        real_dat = Input(batch_shape=(1,1,201,1))
        img_dat = Input(batch_shape=(1,1,201,1))
        
        conv_cache1=Input(batch_shape=(1,1,203,2))
        conv_cache2=Input(batch_shape=(1,1,101,32))
        conv_cache3=Input(batch_shape=(1,1,52,32))
        conv_cache4=Input(batch_shape=(1,1,52,32))
        conv_cache5=Input(batch_shape=(1,1,52,64))

        deconv_cache1=Input(batch_shape=(1,1,50,256))
        deconv_cache2=Input(batch_shape=(1,1,50,128))
        deconv_cache3=Input(batch_shape=(1,1,50,64))
        deconv_cache4=Input(batch_shape=(1,1,50,64))
        deconv_cache5=Input(batch_shape=(1,1,100,64))
        
        in_h1 = Input(batch_shape=(1,50,128))
        in_c1 = Input(batch_shape=(1,50,128))

        in_h2 = Input(batch_shape=(1,50,128))
        in_c2 = Input(batch_shape=(1,50,128))
        
        input_complex_spec = Concatenate(axis = -1)([real_dat,img_dat])

    
        input_complex_spec = LayerNormalization(axis = [-1,-2], name = 'input_norm')(input_complex_spec)
        input_complex_spec_pad = ZeroPadding2D(padding=((0,0),(0,2)))(input_complex_spec)
        conv_outcache1=input_complex_spec_pad
        input_complex_spec_pad=Concatenate(axis=1)([conv_cache1,input_complex_spec_pad])
        conv_1 = Conv2D(32, (2,5),(1,2),name = name+'_conv_1',padding = 'valid')(input_complex_spec_pad)
        bn_1 = BatchNormalization(name = name+'_bn_1')(conv_1)
        out_1 = PReLU(shared_axes=[1,2])(bn_1)

        out_1pad=ZeroPadding2D(padding=((0,0),(0,1)))(out_1)
        conv_outcache2=out_1pad
        out_1pad=Concatenate(axis=1)([conv_cache2,out_1pad])
        conv_2 = Conv2D(32, (2,3),(1,2),name = name+'_conv_2',padding = 'valid')(out_1pad)
        bn_2 = BatchNormalization(name = name+'_bn_2')(conv_2)
        out_2 = PReLU(shared_axes=[1,2])(bn_2)

        # causal padding [1,0],[1,1]
        out_2pad=ZeroPadding2D(padding=((0,0),(1,1)))(out_2)
        conv_outcache3=out_2pad
        out_2pad=Concatenate(axis=1)([conv_cache3,out_2pad])
       
        conv_3 = Conv2D(32, (2,3),(1,1),name = name+'_conv_3',padding = 'valid')(out_2pad)
        bn_3 = BatchNormalization(name = name+'_bn_3')(conv_3)
        out_3 = PReLU(shared_axes=[1,2])(bn_3)

        # causal padding [1,0],[1,1]
        out_3pad=ZeroPadding2D(padding=((0,0),(1,1)))(out_3)
        conv_outcache4=out_3pad
        out_3pad=Concatenate(axis=1)([conv_cache4,out_3pad])
        conv_4 = Conv2D(64, (2,3),(1,1),name = name+'_conv_4',padding = 'valid')(out_3pad)
        bn_4 = BatchNormalization(name = name+'_bn_4')(conv_4)
        out_4 = PReLU(shared_axes=[1,2])(bn_4)

        # causal padding [1,0],[1,1]
        out_4pad=ZeroPadding2D(padding=((0,0),(1,1)))(out_4)
        conv_outcache5=out_4pad
        out_4pad=Concatenate(axis=1)([conv_cache5,out_4pad])
        conv_5 = Conv2D(128, (2,3),(1,1),name = name+'_conv_5',padding = 'valid')(out_4pad)
        bn_5 = BatchNormalization(name = name+'_bn_5')(conv_5)
        out_5 = PReLU(shared_axes=[1,2])(bn_5)

        dp_in = out_5[:,-1:,:,:]
            
        dp_in,out_h1,out_c1 = DprnnBlock(numUnits = self.numUnits, batch_size =1, L = 1,width = 50,channel = 128, causal=True)(dp_in,in_h1,in_c1)
        out_h1=tf.expand_dims(out_h1,axis=0)
        out_c1=tf.expand_dims(out_c1,axis=0)
        dp_in,out_h2,out_c2 = DprnnBlock(numUnits = self.numUnits, batch_size =1, L = 1,width = 50,channel = 128, causal=True)(dp_in,in_h2,in_c2)
        out_h2=tf.expand_dims(out_h2,axis=0)
        out_c2=tf.expand_dims(out_c2,axis=0)
        
        dp_out = dp_in
       

        skipcon_1 = Concatenate(axis = -1)([out_5,dp_out])
        deconv_outcache1=skipcon_1
        skipcon_1=Concatenate(axis=1)([deconv_cache1,skipcon_1])
       
        deconv_1 = Conv2DTranspose(64,(2,3),(1,1),name = name+'_dconv_1',padding = 'same')(skipcon_1)
        dbn_1 = BatchNormalization(name = name+'_dbn_1')(deconv_1)
        dout_1 = PReLU(shared_axes=[1,2])(dbn_1)

        skipcon_2 = Concatenate(axis = -1)([out_4,dout_1[:,1:,:,:]])
        deconv_outcache2=skipcon_2
        skipcon_2=Concatenate(axis=1)([deconv_cache2,skipcon_2])
        
        deconv_2 = Conv2DTranspose(32,(2,3),(1,1),name = name+'_dconv_2',padding = 'same')(skipcon_2)
        dbn_2 = BatchNormalization(name = name+'_dbn_2')(deconv_2)
        dout_2 = PReLU(shared_axes=[1,2])(dbn_2)
        
        skipcon_3 = Concatenate(axis = -1)([out_3,dout_2[:,1:,:,:]])
        deconv_outcache3=skipcon_3
        skipcon_3=Concatenate(axis=1)([deconv_cache3,skipcon_3])
      
        deconv_3 = Conv2DTranspose(32,(2,3),(1,1),name = name+'_dconv_3',padding = 'same')(skipcon_3)
        dbn_3 = BatchNormalization(name = name+'_dbn_3')(deconv_3)
        dout_3 = PReLU(shared_axes=[1,2])(dbn_3)
        
        skipcon_4 = Concatenate(axis = -1)([out_2,dout_3[:,1:,:,:]])
        deconv_outcache4=skipcon_4
        skipcon_4=Concatenate(axis=1)([deconv_cache4,skipcon_4])

        deconv_4 = Conv2DTranspose(32,(2,3),(1,2),name = name+'_dconv_4',padding = 'same')(skipcon_4)
        dbn_4 = BatchNormalization(name = name+'_dbn_4')(deconv_4)
        dout_4 = PReLU(shared_axes=[1,2])(dbn_4)
        
        skipcon_5 = Concatenate(axis = -1)([out_1,dout_4[:,1:,:,:]])
        deconv_outcache5=skipcon_5
        skipcon_5=Concatenate(axis=1)([deconv_cache5,skipcon_5])
        
        deconv_5 = Conv2DTranspose(2,(2,5),(1,2),name = name+'_dconv_5',padding = 'valid')(skipcon_5)
        deconv_5 = deconv_5[:,-2:-1,:-2]

        #output_mask = Activation('tanh')(dbn_5)
        output_mask = deconv_5

        enh_spec = Lambda(self.mk_mask)([real_dat,img_dat,output_mask])
       # bt=[bn_1,bn_2,bn_3,bn_4,bn_5,dbn_1,dbn_2,dbn_3,dbn_4]

        enh_real, enh_imag = enh_spec[0],enh_spec[1]

        self.model = Model(inputs=[real_dat,img_dat,conv_cache1,conv_cache2,conv_cache3,conv_cache4,conv_cache5,deconv_cache1,deconv_cache2,deconv_cache3,deconv_cache4,deconv_cache5
                                   ,in_h1,in_c1,in_h2,in_c2],outputs=[enh_real,enh_imag,conv_outcache1,conv_outcache2,conv_outcache3,conv_outcache4,conv_outcache5,deconv_outcache1,
                                    deconv_outcache2,deconv_outcache3,deconv_outcache4,deconv_outcache5,out_h1,out_c1,out_h2,out_c2])
        self.model.summary()
        self.model.load_weights('./pretrained.h5')
        if(is_tflite):
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            if quant:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            with tf.io.gfile.GFile(save_name+'.tflite', 'wb') as f:
                f.write(tflite_model)
        else:
            tf2onnx.convert.from_keras(self.model,output_path=save_name+'.onnx',opset=12)
       
dpcrn=DPCRN_model()
dpcrn.export_real_time_model('dpcrn_rt',is_tflite=True,quant=False)