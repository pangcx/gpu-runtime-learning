/**
 * 项目02：模板与泛型编程
 *
 * 学习目标：
 *   - 函数模板：写一次，适用多种类型
 *   - 类模板：泛型容器、泛型算法
 *   - 模板特化：对特定类型提供定制实现
 *   - 理解 PyTorch/CUDA 代码里大量模板的由来
 *
 * 编译运行：
 *   cd build && cmake .. -G Ninja && ninja && ./templates_demo
 */

#include <iostream>
#include <string>
#include <vector>
#include <type_traits>

// ─────────────────────────────────────────────
// Part 1: 函数模板
// ─────────────────────────────────────────────

// 没有模板时：为每种类型写一遍
int    add_int(int a, int b)       { return a + b; }
double add_double(double a, double b) { return a + b; }
// ...太麻烦了

// 用函数模板：写一次，编译器自动生成各类型版本
template <typename T>
T add(T a, T b) {
    return a + b;
}

// 支持不同类型相加，推导返回类型（C++14）
template <typename T, typename U>
auto add_mixed(T a, U b) -> decltype(a + b) {
    return a + b;
}

void demo_function_templates() {
    std::cout << "\n=== 函数模板 ===\n";

    std::cout << "add<int>(3, 4) = " << add<int>(3, 4) << "\n";
    std::cout << "add(3.14, 2.0) = " << add(3.14, 2.0) << "\n";      // 类型自动推导
    std::cout << "add(std::string) = " << add(std::string("hello"), std::string(" world")) << "\n";
    std::cout << "add_mixed(3, 2.5) = " << add_mixed(3, 2.5) << "\n"; // int + double
}

// ─────────────────────────────────────────────
// Part 2: 类模板
// ─────────────────────────────────────────────

// 自己实现一个简单的泛型栈（类似 std::stack）
template <typename T>
class Stack {
public:
    void push(const T& value) {
        data_.push_back(value);
    }

    void pop() {
        if (empty()) throw std::runtime_error("Stack is empty");
        data_.pop_back();
    }

    const T& top() const {
        if (empty()) throw std::runtime_error("Stack is empty");
        return data_.back();
    }

    bool empty() const { return data_.empty(); }
    size_t size() const { return data_.size(); }

private:
    std::vector<T> data_;
};

void demo_class_templates() {
    std::cout << "\n=== 类模板 ===\n";

    Stack<int> int_stack;
    int_stack.push(1);
    int_stack.push(2);
    int_stack.push(3);
    std::cout << "int 栈顶: " << int_stack.top() << ", 大小: " << int_stack.size() << "\n";
    int_stack.pop();
    std::cout << "pop 后栈顶: " << int_stack.top() << "\n";

    Stack<std::string> str_stack;
    str_stack.push("hello");
    str_stack.push("world");
    std::cout << "string 栈顶: " << str_stack.top() << "\n";
}

// ─────────────────────────────────────────────
// Part 3: 模板特化
// ─────────────────────────────────────────────

// 通用版本
template <typename T>
struct TypeInfo {
    static std::string name() { return "unknown"; }
};

// 对 int 特化
template <>
struct TypeInfo<int> {
    static std::string name() { return "int"; }
};

// 对 float 特化
template <>
struct TypeInfo<float> {
    static std::string name() { return "float"; }
};

// 对 double 特化
template <>
struct TypeInfo<double> {
    static std::string name() { return "double"; }
};

void demo_specialization() {
    std::cout << "\n=== 模板特化 ===\n";
    std::cout << "TypeInfo<int>    = " << TypeInfo<int>::name() << "\n";
    std::cout << "TypeInfo<float>  = " << TypeInfo<float>::name() << "\n";
    std::cout << "TypeInfo<double> = " << TypeInfo<double>::name() << "\n";
    std::cout << "TypeInfo<char>   = " << TypeInfo<char>::name() << "\n";  // unknown
}

// ─────────────────────────────────────────────
// Part 4: constexpr 与编译期计算
// ─────────────────────────────────────────────

// constexpr：在编译期就算好，运行时零开销
template <int N>
struct Factorial {
    static constexpr int value = N * Factorial<N-1>::value;
};

template <>
struct Factorial<0> {
    static constexpr int value = 1;
};

// 也可以用 constexpr 函数（C++14）
constexpr int factorial(int n) {
    return n <= 1 ? 1 : n * factorial(n - 1);
}

void demo_constexpr() {
    std::cout << "\n=== 编译期计算 ===\n";
    // 这两个在编译期就算好了，运行时直接用结果
    std::cout << "5! (模板元) = " << Factorial<5>::value << "\n";
    std::cout << "6! (constexpr) = " << factorial(6) << "\n";

    // std::is_same：编译期类型判断，PyTorch 源码里大量使用
    std::cout << "int == int: "    << std::is_same<int, int>::value << "\n";
    std::cout << "int == float: "  << std::is_same<int, float>::value << "\n";
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────
int main() {
    demo_function_templates();
    demo_class_templates();
    demo_specialization();
    demo_constexpr();

    std::cout << "\n所有 demo 完成！\n";
    return 0;
}

/*
 * ══════════════════════════════════════════
 * 📝 练习题
 * ══════════════════════════════════════════
 *
 * 1. 实现一个泛型 Pair<T, U> 类：
 *    - 存储两个不同类型的值
 *    - 实现 first() / second() 访问器
 *    - 实现 swap() 方法
 *    - 类比 std::pair
 *
 * 2. 实现一个 clamp 函数模板：
 *    - clamp(value, min, max)：将 value 限制在 [min, max] 范围内
 *    - 要求 T 类型支持比较运算符
 *    - 用 static_assert 加编译期检查
 *
 * 3. 实现一个简单的 Tensor<T> 类模板（为 CUDA 学习做铺垫）：
 *    - 内部用 vector<T> 存储数据
 *    - 支持 shape（行数、列数）
 *    - 实现 at(row, col) 访问
 *    - 实现矩阵加法 operator+
 *    - 分别用 Tensor<float> 和 Tensor<int> 测试
 */
