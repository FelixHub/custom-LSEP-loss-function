
import tensorflow as tf

import tensorflow.keras.backend as K

# i found some backend fonctions on the keras backend API, some were only available 
# through tensorflow such as einsum wich wasn't easily replicable.


def customLoss(weights_list,num_classes) :
    
    def lsep_loss(y_true,y_pred):
        
        # here's a little trick to get the number of sample of the batch
        # i couldn't find easy way to do that online (.shape() of all kinds didn't work)
        batchsize = tf.math.floordiv(K.sum(K.exp(y_true - y_true)),num_classes)
        
        # reshape labels and predictions to get 2d matrix of 1d samples
        y_t = K.reshape(y_true,(batchsize,num_classes))
        y_p = K.reshape(y_pred,(batchsize,num_classes))
        
        M_unit = tf.ones((batchsize, num_classes)) 
        
        # here's the class weights 
        M1 = ( M_unit - y_t ) * K.reshape ( K.tile(weights_list,[batchsize]) , (batchsize, num_classes) )
       
        # Einstein notations allow to compute easily the functions components
        # without complex block-matrix multiplication
        M_pairwise = tf.einsum('ij,ik->ijk', M1, y_t) #shape(2,12,12)
        M_large = tf.einsum('ij,ik->ijk', M_unit , y_p) #shape(2,12,12)

        
        M_diff = K.exp(K.permute_dimensions (M_large, (0,2,1)) - M_large) #shape(2,12,12)


        M = M_pairwise*M_diff #shape(2,12,12)

        return (K.mean(K.log(1 + K.sum(K.sum(M,2),1))))
    
    return(lsep_loss)
    
