from typing import Dict, Optional, Tuple

import torch

__all__ = ["MTLModel", "MTLModelWithCELoss", "MTLLeNet", "MTLResNet18"]


class MTLModel(torch.nn.Module):
    """Base MTL classification model.

    Args:
        n_tasks (int): Number of tasks.
        n_classes_per_task (int): Number of classes per task.
        input_shape (Tuple[int, int, int]): Shape of each input image.
    """
    def __init__(self, n_tasks: int, n_classes_per_task: int, input_shape: Tuple[int, int, int]) -> None:
        super(MTLModel, self).__init__()
        self.n_tasks = n_tasks
        self.n_classes_per_task = n_classes_per_task
        self.input_shape = input_shape
    
    @property
    def input_channels(self) -> int:
        return self.input_shape[0]
  
    @property
    def input_height(self) -> int:
        return self.input_shape[1]

    @property
    def input_width(self) -> int:
        return self.input_shape[2]

    def forward(self, images: torch.FloatTensor, task_id: Optional[int] = None) -> torch.FloatTensor:
        """Perform a forward pass on a batch of images.

        Args:
            images (torch.FloatTensor): [batch x channels x height x width] image tensor.
            task_id (Optional[int]): Optionally specify the desired task id.

        Returns:
            (torch.FloatTensor): [batch x num_tasks x num_classes_per_task] logits if
                `task_id` is None. Otherwise, [batch x num_classes_per_task] logits instead.
        """
        raise NotImplementedError()

    def get_shared_parameters(self) -> Dict[str, torch.nn.Parameter]:
        """Return shared network parameters."""
        raise NotImplementedError()

    def get_task_parameters(self, task_id: int) -> Dict[str, torch.nn.Parameter]:
        """Return task-specific network parameters."""
        raise NotImplementedError()


class MTLModelWithCELoss(torch.nn.Module):
    """Wrap an MTL model with cross-entropy loss.

    Args:
        model (MTLModel): Multi-task classification model to wrap.
    """
    def __init__(self, model: MTLModel) -> None:
        super(MTLModelWithCELoss, self).__init__()
        self.model = model
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        images: torch.FloatTensor,
        labels: torch.LongTensor,
        task_id: Optional[int] = None
    ) -> torch.FloatTensor:
        """Compute classification loss on a batch on images.

        Args:
            images (torch.FloatTensor): [batch x channels x height x width] image tensor.
            labels (torch.LongTensor): [batch x n_tasks] labels tensor.
            task_id (Optional[int]): Optionally specify the desired task id.

        Returns:
            (torch.FloatTensor): [num_tasks] losses if `task_id` is None.
                Otherwise, [1] loss tensor for the specified `task_id` instead.
        """
        if task_id is not None:
            return self.ce_loss(self.model(images, task_id), labels[:, task_id])

        task_loss = []
        task_logits = self.model(images)
        for task_id in range(self.model.n_tasks):
            task_loss.append(self.ce_loss(task_logits[:, task_id], labels[:, task_id]))
        return torch.stack(task_loss, dim=0)


class MTLLeNet(MTLModel):
    """MTL LeNet model.

    Args:
        n_tasks (int): Number of tasks.
        n_classes_per_task (int): Number of classes per task.
        input_shape (Tuple[int, int, int]): Shape of each input image.
    """
    def __init__(self, n_tasks: int, n_classes_per_task: int, input_shape: Tuple[int, int, int]) -> None:
        super(MTLLeNet, self).__init__(n_tasks, n_classes_per_task, input_shape)
        output_width = (((self.input_width - 8) // 2) - 4) // 2
        output_height = (((self.input_height - 8) // 2) - 4) // 2
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(self.input_channels, 10, 9, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(10, 20, 5, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Flatten(),
            torch.nn.Linear(20 * output_height * output_width, 50)
        )

        for task_id in range(self.n_tasks):
            setattr(self, f"task_{task_id}", torch.nn.Linear(50, self.n_classes_per_task))

    def forward(self, images: torch.FloatTensor, task_id: Optional[int] = None) -> torch.FloatTensor:
        x = self.net(images)
        if task_id is not None:
            return getattr(self, f"task_{task_id}")(x)

        logits = []
        for task_id in range(self.n_tasks):
            logits.append(getattr(self, f"task_{task_id}")(x))
        return torch.stack(logits, dim=1)

    def get_shared_parameters(self) -> Dict[str, torch.nn.Parameter]:
        return self.net.named_parameters()

    def get_task_parameters(self, task_id: int) -> Dict[str, torch.nn.Parameter]:
        return getattr(self, f"task_{task_id}").named_parameters()


class MTLResNet18(torch.nn.Module):
    """MTL ResNet18 model.

    Args:
        n_tasks (int): Number of tasks.
        n_classes_per_task (int): Number of classes per task.
        input_shape (Tuple[int, int, int]): Shape of each input image.
    """

    def __init__(self, n_tasks: int, n_classes_per_task: int, input_shape: Tuple[int, int, int]) -> None:
        super(MTLResNet18, self).__init__(n_tasks, n_classes_per_task, input_shape)
        self.net = torchvision.models.resnet18()

        # replace number of input channels
        self.net.conv1 = torch.nn.Conv2d(
            self.input_channels,
            self.net.conv1.out_channels,
            kernel_size=self.net.conv1.kernel_size,
            stride=self.net.conv1.stride,
            padding=self.net.conv1.padding,
            bias=self.net.conv1.bias,
        )

        # skip final FC layer
        num_hidden_channels = self.net.fc.in_features
        self.net.fc = torch.nn.Sequential()

        # per-task predictors
        for task_id in range(self.n_tasks):
            setattr(self, f"task_{task_id}", torch.nn.Linear(num_hidden_channels, self.n_classes_per_task))

    def forward(self, images: torch.FloatTensor, task_id: Optional[int] = None) -> torch.FloatTensor:
        x = self.net(images)
        if task_id is not None:
            return getattr(self, f"task_{task_id}")(x)

        logits = []
        for task_id in range(self.n_tasks):
            logits.append(getattr(self, f"task_{task_id}")(x))
        return torch.stack(logits, dim=1)

    def get_shared_parameters(self) -> Dict[str, torch.nn.Parameter]:
        return self.net.named_parameters()

    def get_task_parameters(self, task: int) -> Dict[str, torch.nn.Parameter]:
        return getattr(self, f"task_{task_id}").named_parameters()
