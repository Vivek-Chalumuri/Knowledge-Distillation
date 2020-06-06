"""
For obtaining the logists from teacher network.
For hinton's Knowledge distillation method
"""
import torch
from utils import progressbar
from utils import evalation
import torch.backends.cudnn as cudnn
from models.vgg import Vgg
from utils.load_data import get_train_valid_cifar10_dataloader
from utils.load_data import get_test_cifar10_dataloader

def get_model_outputs(model, dataloader, device='cuda'):
    """
    """
    model.eval()
    ret = []
    with torch.no_grad():
        for inputs, _ in progressbar(dataloader, prefix="Evaluating Logist"):
            inputs = inputs.to(device)
            if device == 'cuda':
                inputs = inputs.half()
            outputs = model(inputs)
            ret.append(outputs)
    return ret

if __name__ == "__main__":
    batch_size = 100
    #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        cudnn.benchmark = True
    #
    # load the teacher
    teacher = Vgg('VGG16', batch_norm=True)
    chkpt = torch.load("vgg16bn_teacher.tar")
    teacher.load_state_dict(chkpt['state_dict'])
    teacher.to(device)
    if device == 'cuda':
        teacher = teacher.half()

    #
    trainloader, validloader = get_train_valid_cifar10_dataloader('../../data', batch_size=batch_size)
    testloader = get_test_cifar10_dataloader('../../data', batch_size)

    # check if a good teacher
    score = evalation(validloader, teacher, device)
    print("Evaluation score of validation set = ", score)

    score = evalation(testloader, teacher, device)
    print("Evaluation score of test set = ", score)

    logits = get_model_outputs(teacher, trainloader)
    data = {
        'batch_size': 100,
        'logits': logits
    }
    torch.save(data, 'teacher_logits_{}.tar'.format(batch_size))