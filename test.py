import torch
from torchvision import datasets, transforms
from model import MNISTNet

def print_model_info(model):
    # Print total parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameter Count: {total_params}")

    # Check for Batch Normalization, Dropout, and Fully Connected Layers or GAP
    has_batch_norm = any(isinstance(layer, torch.nn.BatchNorm2d) for layer in model.modules())
    has_dropout = any(isinstance(layer, torch.nn.Dropout) for layer in model.modules())
    has_fc_or_gap = any(isinstance(layer, (torch.nn.Linear, torch.nn.AdaptiveAvgPool2d)) for layer in model.modules())

    print(f"Use of Batch Normalization: {'Yes' if has_batch_norm else 'No'}")
    print(f"Use of DropOut: {'Yes' if has_dropout else 'No'}")
    print(f"Use of Fully Connected Layer or GAP: {'Yes' if has_fc_or_gap else 'No'}")

def test_model():
    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTNet().to(device)
    model.load_state_dict(torch.load('mnist_model.pth'))
    model.eval()

    # Print model information
    print_model_info(model)

    # Prepare test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128)

    # Test the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

if __name__ == '__main__':
    test_model() 