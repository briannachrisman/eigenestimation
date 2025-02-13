def ZeroOutput(outputs):
    return 0*outputs

def SameOutput(outputs):
    return outputs.detach()

def MeanOutput(outputs):
    return outputs.mean(dim=0, keepdim=True).detach()


def UniformProbs(outputs):
    return (0*outputs+1).softmax()