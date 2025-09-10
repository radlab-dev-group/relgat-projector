import torch


class MSELoss:
    @staticmethod
    def calculate(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Zwraca średni MSE dla par (a_i, b_i).
        """
        return torch.nn.functional.mse_loss(a, b)
