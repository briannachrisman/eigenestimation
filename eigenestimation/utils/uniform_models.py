def ZeroOutput(outputs):
    return 0*outputs

def UniformProbs(outputs):
    return (0*outputs+1).softmax()