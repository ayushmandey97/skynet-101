
#******* NORMAL CONVOLUTION *******
def single_step(a_slice_prev, W, b):
    #Performs convolution between once slice of image and a filter
    s = np.multiply(a_slice_prev, W)
    Z = np.sum(s)
    Z = Z + float(b)

    return Z

def simple_conv(A_prev, W, b):

    '''
    For 4-3 and 5-3 convolutions mentioned in the assignment problem.

    A_prev contains the output from the previous layer, or the input image
    W - weights, B - bias, hparameters - dictionary containing padding and stride
    
    '''
    
    # Get the dimensions from the previous layer/input
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Get filter dimensions
    (f, f, n_C_prev, n_C) = W.shape
    
    # Computing the dimensions of the CONV output volume using the formula
    n_H = n_H_prev - f + 1
    n_W = n_W_prev - f + 1
    
    # Initialize the output volume Z with zeros
    Z = np.zeros((m, n_H, n_W, n_C))
    
    for i in range(m):
        a_prev = A_prev[i]                     
        
        for h in range(n_H):                           
            for w in range(n_W):                       
                    
                # Defining the corner points for the convolution operation
                vert_start = h
                vert_end = vert_start + f
                horiz_start = w
                horiz_end = horiz_start + f

                a_slice_prev = a_prev[vert_start:vert_end, horiz_start:horiz_end, :]

                #Convolving the entire slice with the filter
                Z[i, h, w, c] = single_step(a_slice_prev, W[:,:,:,:], b[:,:,:,:])
                                        

    # Check if shape is right
    assert(Z.shape == (m, n_H, n_W, n_C))
    
    return Z

#******** 1x1 CONVOLUTION *********
def oneXone_conv(A_prev, W, b):
    '''
        Performing a 1 X 1 convolution on the previous layer output or input image

    '''

    # Get the dimensions from the previous layer/input
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # W here is just an array of filter elements with size as the number of filters
    n_C = W.shape[0]
    
    # The feature map is of the same shape as that of the input
    n_H = n_H_prev
    n_W = n_W_prev
    
    # Initialize the output volume Z with zeros
    Z = np.zeros((m, n_H, n_W, n_C))
    
    for i in range(m):
        for h in range(n_H):                           
            for w in range(n_W):                       
                for c in range(n_C):
                    # Performing element-wise multiplication between the input and filter element                  
                    Z[i, h, w, c] = A_prev[i, h, w, c] * W[c]
                                        

    # Check if shape is right
    assert(Z.shape == (m, n_H, n_W, n_C))
    
    return Z


#********** SEPARABLE CONV ********
def depth_separable_conv(A_prev, W, b, W_depth):
    
    # Get the dimensions from the previous layer/input
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Get filter dimensions
    (f, f, n_C_prev, n_C) = W.shape
    
    # Computing the dimensions of the CONV output volume using the formula
    n_H = n_H_prev - f + 1
    n_W = n_W_prev - f + 1
    
    # Initialize the output volume Z with zeros
    Z = np.zeros((m, n_H, n_W, n_C))
    
    for i in range(m):
        a_prev = A_prev[i]                     
        
        for h in range(n_H):                           
            for w in range(n_W):                       
                for c in range(n_C):                   
                    
                    # Defining the corner points for the convolution operation
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f

                    a_slice_prev = a_prev[vert_start:vert_end, horiz_start:horiz_end, :]

                    #Convolving the slice with the filter using a function defined previously
                    Z[i, h, w, c] = single_step(a_slice_prev, W[:,:,:,c], b[:,:,:,c])



    # Get the dimensions from the previous layer/input
    (m, n_H_prev, n_W_prev, n_C_prev) = Z.shape
    
    # W here is just an array of filter elements with size as the number of filters
    n_C = W_depth.shape[0]
    
    # The feature map is of the same shape as that of the input
    n_H = n_H_prev
    n_W = n_W_prev
    
    # Initialize the output volume Z with zeros
    Z = np.zeros((m, n_H, n_W, n_C))
    
    for i in range(m):
        for h in range(n_H):                           
            for w in range(n_W):                       
                for c in range(n_C):
                    # Performing element-wise multiplication between the input and filter element                  
                    Z[i, h, w, c] = A_prev[i, h, w, c] * W[c]
                                        
    # Check if shape is right
    assert(Z.shape == (m, n_H, n_W, n_C))
    
    return Z


#******** DILATED CONVOLUTION *********
def dilated_conv(A_prev, W, b, dilation_factor):

    '''
    For dilated convolutions; the dilation factor tells how "spread-out" the receptive view of the feature map is
    
    '''

    # Get the dimensions from the previous layer/input
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Get filter dimensions
    (f, f , n_C) = W.shape
    

    #getting the reference points
    ch, cw = (int(f/2), int(f/2))
    

    #Expanding the filter by spreading the original filter based of dilation factor
    new_W = np.zeros(f + 2 * (dilation_factor-1), f + 2 * (dilation_factor-1), n_C)
    for h in range(f):
        for w in range(f):
            h = (h-ch)*dilation_factor
            w = (w-cw)*dilation_factor

            new_W[new_W.shape[0] - f + h , new_W.shape[1] - f - w, :] = W[w, h, :]

    # Resultant image is going to be the same
    n_H = n_H_prev - f + 1
    n_W = n_W_prev - f + 1

    
    # Initialize the output volume Z with zeros
    Z = np.zeros((m, n_H, n_W, n_C))
    
    for i in range(m):
        a_prev = A_prev[i]                     
        
        for h in range(n_H):                           
            for w in range(n_W):                       
                for c in range(n_C):                   
                    
                    # Defining the corner points for the convolution operation
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f

                    a_slice_prev = a_prev[vert_start:vert_end, horiz_start:horiz_end, :]

                    #Convolving the slice with the filter using a function defined previously
                    Z[i, h, w, c] = single_step(a_slice_prev, new_W[:,:,:,c], b[:,:,:,c])
                                        

    # Check if shape is right
    assert(Z.shape == (m, n_H, n_W, n_C))
    
    return Z


#DEPTH CONV
def depth_conv(A_prev, W, b):
    '''
        Separates out the different channels and runs the filters concurrently on them

    '''
    
    # Get the dimensions from the previous layer/input
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Get filter dimensions
    (f, f, n_C_prev, n_C) = W.shape
    
    # Computing the dimensions of the CONV output volume using the formula
    n_H = n_H_prev - f + 1
    n_W = n_W_prev - f + 1
    
    # Initialize the output volume Z with zeros
    Z = np.zeros((m, n_H, n_W, n_C))
    
    for i in range(m):
        a_prev = A_prev[i]                     
        
        for h in range(n_H):                           
            for w in range(n_W):                       
                for c in range(n_C):                   
                    
                    # Defining the corner points for the convolution operation
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f

                    a_slice_prev = a_prev[vert_start:vert_end, horiz_start:horiz_end, :]

                    #Convolving the slice with the filter using a function defined previously
                    Z[i, h, w, c] = single_step(a_slice_prev, W[:,:,:,c], b[:,:,:,c])


    # Check if shape is right
    assert(Z.shape == (m, n_H, n_W, n_C))
    
return Z


#*********** GROUPED**************
def grouped_convolution(y, nb_channels, _strides):
    '''
        In a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        and convolutions are separately performed within each group
    
    '''
    # when `cardinality` == 1 this is just a standard convolution
    if cardinality == 1:
        return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
    
    channels = nb_channels / cardinality

    
    groups = []
    for j in range(cardinality):
        group = layers.Lambda(lambda z: z[:, :, :, j*channels : j*channels + channels])(y)
        groups.append(layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))
        
    # the grouped convolutional layer concatenates them as the outputs of the layer
    y = layers.concatenate(groups)

    return y


#************ DECONVOLUTIONS ************
def zero_pad(X, pad):
    """
        Pad with zeros all images of the dataset X.

    """
    X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), 'constant', constant_values = (0) )
    
    return X_pad


def deconv(A_prev, W, b, pad):
    """
    Implements the deconvolution operation after padding a feature map
    
    """
 
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    (f, f, n_C_prev, n_C) = W.shape
    
    #Calculating final size considering padding
    n_H = int(n_H_prev - f + 2 * pad) + 1
    n_W = int(n_W_prev - f + 2 * pad) + 1
    
    Z = np.zeros((m, n_H, n_W, n_C))
    
    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev, pad)
    
    for i in range(m):                                 
        a_prev_pad = A_prev_pad[i]                     
        for h in range(n_H):                           
            for w in range(n_W):                       
                for c in range(n_C):                  
                    
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f

                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:,:,:,c], b[:,:,:,c])
                                        

    assert(Z.shape == (m, n_H, n_W, n_C))

    return Z
