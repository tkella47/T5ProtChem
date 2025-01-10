from torchmetrics import Metric
from torchmetrics import R2Score
import torch


class CIndex(Metric):

    def __init__(self):
        super().__init__()
        self.add_state("numerator", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.tensor(0.), dist_reduce_fx="sum")


    def update(self, preds, target):
        Y_diff = target.view(-1, 1) - target.view(1, -1)
        P_diff = preds.view(-1, 1) - preds.view(1, -1)
        mask = Y_diff > 0
        best_case = P_diff > 0
        tie_case = P_diff == 0

        self.numerator += (best_case * mask).sum() + .5 * (tie_case * mask).sum()
        self.denominator += mask.sum()

    def compute(self):
        return self.numerator.float() / self.denominator.float()

class Boundary_Accuracy(Metric):
    def __init__(self,boundary=10.0):
        super().__init__()
        self.add_state("numerator", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.boundary = torch.tensor(boundary, dtype=torch.float32)

    def update(self, preds, target):
        in_range = torch.abs(preds - target) < self.boundary
        self.numerator += in_range.sum()
        self.denominator += len(preds)

    def compute(self):
        return self.numerator.float() / self.denominator.float()




class R2MPaper(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("y_true", default=torch.tensor([]), dist_reduce_fx="cat")
        self.add_state("y_pred", default=torch.tensor([]), dist_reduce_fx="cat")
        self.add_state("n_obs", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        self.y_true = torch.cat((self.y_true, y_true), dim=0)
        self.y_pred = torch.cat((self.y_pred, y_pred), dim=0)
        self.n_obs += y_true.numel()

    def compute(self):
        r02 = self.get_r02()
        r2 = self.get_r2()
        return r2 * (1 - torch.sqrt(torch.abs(r2 - r02)))

    def get_k(self):
        return torch.sum(self.y_true * self.y_pred) / torch.sum(self.y_pred ** 2)

    def get_r02(self):
        k = self.get_k()
        y_true_mean = torch.full_like(self.y_true, self.y_true.mean())
        numerator = torch.sum((self.y_true - (k*self.y_pred))**2)
        denominator = torch.sum((self.y_true - y_true_mean)**2)
        return 1 - (numerator / denominator)
    def get_r2(self):
        y_true_mean = torch.full_like(self.y_true, self.y_true.mean())
        y_pred_mean = torch.full_like(self.y_pred, self.y_pred.mean())
        mult = torch.abs(torch.sum((self.y_pred - y_pred_mean) * (self.y_true - y_true_mean)))**2
        y_true_sqr = sum((self.y_true - y_true_mean)**2)
        y_pred_sqr = sum((self.y_pred - y_pred_mean)**2)
        return mult / (y_true_sqr * y_pred_sqr)
class R2MDTA(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("y_true", default=torch.tensor([]), dist_reduce_fx="cat")
        self.add_state("y_pred", default=torch.tensor([]), dist_reduce_fx="cat")
        self.add_state("n_obs", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        self.y_true = torch.cat((self.y_true, y_true), dim=0)
        self.y_pred = torch.cat((self.y_pred, y_pred), dim=0)
        self.n_obs += y_true.numel()

    def compute(self):
        r02 = self.get_r02()
        r2 = self.get_r2()
        return r2* (1 - torch.sqrt(torch.abs((r2**2 - r02**2))))

    def get_k(self):
        return torch.sum(self.y_true * self.y_pred) / torch.sum(self.y_pred ** 2)

    def get_r02(self):
        k = self.get_k()
        y_true_mean = torch.full_like(self.y_true, self.y_true.mean())
        numerator = torch.sum((self.y_true - (k*self.y_pred))**2)
        denominator = torch.sum((self.y_true - y_true_mean)**2)
        return 1 - (numerator / denominator)
    def get_r2(self):
        y_true_mean = torch.full_like(self.y_true, self.y_true.mean())
        y_pred_mean = torch.full_like(self.y_pred, self.y_pred.mean())
        mult = torch.abs(torch.sum((self.y_pred - y_pred_mean) * (self.y_true - y_true_mean)))**2
        y_true_sqr = sum((self.y_true - y_true_mean)**2)
        y_pred_sqr = sum((self.y_pred - y_pred_mean)**2)
        return mult / (y_true_sqr * y_pred_sqr)

class PosAccuracy(Metric):
   
    def __init__(self):
        super().__init__()
        self.add_state("num_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_obs", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, protein_length, preds, labels):
        self.n_obs += labels.numel()
        preds = (preds * protein_length).int()
        labels = (labels * protein_length).int()
        self.num_correct = (preds == labels).sum()

    def compute(self):
        return self.num_correct / self.n_obs




