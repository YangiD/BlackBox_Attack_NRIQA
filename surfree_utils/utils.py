def atleast_kdim(x, ndim): #(a,1),a*b
    shape = x.shape + (1,) * (ndim - len(x.shape))
    return x.reshape(shape)