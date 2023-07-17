import GPUtil
import os
import tensorflow as tf

#Calls to this function MUST be placed in the __main__ method or imports,
# or env variables will be ignored
#Calling tf things will screw up jax (e.g. making it ignore VISIBLE_CUDA_DEVICES)
#So only deal with jax and tf at the same time when you have greater control over when each is used
#i.e., multiprocessing, launchpad, etc.

def SetGPU(gpuNum=-2, setTF = False):
    if setTF:
        SetGPUJAXAndTF(gpuNum)
    else:
        SetGPUJAXOnly(gpuNum)

def SetGPUJAXAndTF(gpuNum):
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    print("__________________________________________________________")
    GPUtil.showUtilization()

    tfGPUs = tf.config.list_physical_devices('GPU')
    tfCPUs = tf.config.list_physical_devices('CPU')
    for gpu in tfGPUs:
        tf.config.experimental.set_memory_growth(gpu, True)

    if gpuNum == -2:
        gpuNum = GPUtil.getFirstAvailable(order='load',
                                          maxLoad=0.90, maxMemory=0.8,
                                          attempts=240, interval=300)
        print("Using GPU: {}".format(gpuNum[0]))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuNum[0])
        tfGPUNum = 0 if gpuNum[0] == 0 else 1
        tf.config.set_visible_devices([tfGPUs[tfGPUNum], tfCPUs[0]])
    elif gpuNum == -1:
        print("Using CPU")
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

        tf.config.set_visible_devices(tfCPUs[0])
    elif gpuNum == -3:
        print("Using all GPUs")

    else:
        print("Using GPU: {}".format(gpuNum))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuNum)
        tfGPUNum = 0 if gpuNum == 0 else 1
        devList = [tfGPUs[tfGPUNum], tfCPUs[0]]
        tf.config.set_visible_devices(devList)
    print("__________________________________________________________")

def SetGPUJAXOnly(gpuNum):
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    print("__________________________________________________________")
    GPUtil.showUtilization()

    if gpuNum == -2:
        gpuNum = GPUtil.getFirstAvailable(order='load',
                                          maxLoad=0.90, maxMemory=0.8,
                                          attempts=240, interval=300)
        print("Using GPU: {}".format(gpuNum[0]))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuNum[0])
        # tfGPUNum = 0 if gpuNum[0] == 1 else 1
        # tf.config.set_visible_devices([tfGPUs[tfGPUNum], tfCPUs[0]])
    elif gpuNum == -1:
        print("Using CPU")
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    else:
        print("Using GPU: {}".format(gpuNum))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuNum)

    print("__________________________________________________________")



