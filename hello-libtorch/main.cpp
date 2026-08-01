#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "=== LibTorch Hello World ===" << std::endl;
    std::cout << "PyTorch version: " << TORCH_VERSION_MAJOR << "."
              << TORCH_VERSION_MINOR << "." << TORCH_VERSION_PATCH << std::endl;

    // 创建一个随机张量
    torch::Tensor t = torch::rand({3, 3});
    std::cout << "\nRandom 3x3 tensor:\n" << t << std::endl;

    // 矩阵乘法
    torch::Tensor a = torch::ones({2, 3});
    torch::Tensor b = torch::ones({3, 2});
    torch::Tensor c = torch::mm(a, b);
    std::cout << "\nMatrix multiply (2x3) @ (3x2):\n" << c << std::endl;

    // 自动求导
    torch::Tensor x = torch::tensor({2.0f}, torch::requires_grad(true));
    torch::Tensor y = x * x * 3;  // y = 3x^2, dy/dx = 6x
    y.backward();
    std::cout << "\nAutograd: y = 3*x^2 at x=2, dy/dx = " << x.grad() << std::endl;

    std::cout << "\nDevice: " << (torch::cuda::is_available() ? "CUDA" : "CPU") << std::endl;
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
