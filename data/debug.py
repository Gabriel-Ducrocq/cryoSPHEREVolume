import torch
import e3nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x1 = torch.randn(3, dtype=torch.float32)

# Generate Real Spherical Harmonics L=2
K1 = 0.5 * torch.sqrt(torch.tensor(15./torch.pi))
K2 = 0.25 * torch.sqrt(torch.tensor(5./torch.pi))
x,y,z = x1; r2 = torch.sum(x1**2)
x,y,z = z,x,y
Yx = torch.tensor([
    K1 * x*y/r2,
    K1 * y*z/r2,
    K2 * (3*z**2 - r2)/r2,
    K1 * x*z/r2,
    K1 * 0.5 * (x**2 - y**2)/r2]
    )

# Random unitary matrix
R,Res = torch.linalg.qr(torch.rand(3,3))
euler_angles = e3nn.o3.matrix_to_angles(R)

# Rotated coordinates SH
Rx1 = R @ x1
x,y,z = Rx1; r2 = torch.sum(Rx1**2)
x,y,z = z,x,y
YRx = torch.tensor([
    K1 * x*y/r2,
    K1 * y*z/r2,
    K2 * (3*z**2 - r2)/r2,
    K1 * x*z/r2,
    K1 * 0.5 *(x**2 - y**2)/r2
    ])

# Generate Wigner D matrix (This and e3nn.o3.wigner_D gives same results)
ir = e3nn.o3.Irreps("1x2e")
D = ir.D_from_matrix(R)
D = e3nn.o3.wigner_D(2, euler_angles[0], euler_angles[1], euler_angles[2])

# ∑𝑚′𝐷(𝑙)𝑚𝑚′(𝑔)𝑌(𝑙)𝑚′(𝑟̂ )
DYx = D @ Yx

print(DYx - YRx)