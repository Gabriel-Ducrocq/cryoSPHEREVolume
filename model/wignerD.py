import torch


class WignerD():
    def __init__(self, l_max, device):
        self.l_max = l_max
        self.device = device


    def so3_generators(self, l):
        """
        Computes the so3 generators
        :param l: integer
        :return:
        """
        X = su2_generators(l)
        Q = change_basis_real_to_complex(l)
        X = torch.conj(Q.T) @ X @ Q
        assert torch.all(torch.abs(torch.imag(X)) < 1e-5)
        return torch.real(X)

    def su2_generators(j) -> torch.Tensor:
        m = torch.arange(-j, j)
        raising = torch.diag(-torch.sqrt(j * (j + 1) - m * (m + 1)), diagonal=-1)

        m = torch.arange(-j + 1, j + 1)
        lowering = torch.diag(torch.sqrt(j * (j + 1) - m * (m - 1)), diagonal=1)

        m = torch.arange(-j, j + 1)
        return torch.stack([
            0.5 * (raising + lowering),  # x (usually)
            torch.diag(1j * m),  # z (usually)
            -0.5j * (raising - lowering),  # -y (usually)
        ], dim=0)