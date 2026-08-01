/**
 * 项目04：STL 容器与算法
 *
 * 学习目标：
 *   - 掌握核心容器：vector / map / unordered_map / set
 *   - 掌握常用算法：sort / find / transform / reduce
 *   - 理解迭代器模式
 *   - 熟悉 Lambda 表达式（现代 C++ 最常用的特性之一）
 *   - PyTorch 源码里大量用到这些容器和算法
 *
 * 编译运行：
 *   cd build && cmake .. -G Ninja && ninja && ./stl_demo
 */

#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <numeric>
#include <string>
#include <functional>

// ─────────────────────────────────────────────
// Part 1: vector —— 最常用的容器
// ─────────────────────────────────────────────
void demo_vector() {
    std::cout << "\n=== vector ===\n";

    std::vector<int> v = {5, 3, 1, 4, 2};

    // 遍历（range-for，现代 C++ 风格）
    std::cout << "原始: ";
    for (int x : v) std::cout << x << " ";
    std::cout << "\n";

    // 排序
    std::sort(v.begin(), v.end());
    std::cout << "排序后: ";
    for (int x : v) std::cout << x << " ";
    std::cout << "\n";

    // Lambda + transform（每个元素平方）
    std::vector<int> squared(v.size());
    std::transform(v.begin(), v.end(), squared.begin(),
                   [](int x) { return x * x; });  // Lambda 表达式
    std::cout << "平方: ";
    for (int x : squared) std::cout << x << " ";
    std::cout << "\n";

    // reduce（求和，等价 Python sum()）
    int sum = std::reduce(v.begin(), v.end(), 0);
    std::cout << "求和: " << sum << "\n";

    // find_if（找第一个大于 3 的元素）
    auto it = std::find_if(v.begin(), v.end(), [](int x) { return x > 3; });
    if (it != v.end()) std::cout << "第一个 >3 的元素: " << *it << "\n";
}

// ─────────────────────────────────────────────
// Part 2: map / unordered_map
// ─────────────────────────────────────────────
void demo_map() {
    std::cout << "\n=== map vs unordered_map ===\n";

    // map：有序（红黑树），O(log n) 查找
    std::map<std::string, int> word_count;
    std::vector<std::string> words = {"apple", "banana", "apple", "cherry", "banana", "apple"};
    for (const auto& w : words) ++word_count[w];

    std::cout << "词频统计（有序）:\n";
    for (const auto& [word, count] : word_count) {  // C++17 结构化绑定
        std::cout << "  " << word << ": " << count << "\n";
    }

    // unordered_map：无序（哈希表），O(1) 平均查找，更快
    std::unordered_map<std::string, int> fast_map;
    fast_map["layer_0"] = 64;
    fast_map["layer_1"] = 128;
    fast_map["layer_2"] = 256;

    std::cout << "按 key 快速查找: layer_1 的值 = " << fast_map["layer_1"] << "\n";
    std::cout << "layer_3 存在?: " << (fast_map.count("layer_3") ? "是" : "否") << "\n";
}

// ─────────────────────────────────────────────
// Part 3: Lambda 表达式深入
// ─────────────────────────────────────────────
void demo_lambda() {
    std::cout << "\n=== Lambda 表达式 ===\n";

    // 基本语法：[捕获列表](参数) -> 返回类型 { 函数体 }

    // 捕获外部变量
    int threshold = 3;
    auto is_above = [threshold](int x) { return x > threshold; };  // 按值捕获
    std::cout << "5 > 3? " << is_above(5) << "\n";

    // 按引用捕获（可以修改外部变量）
    int count = 0;
    auto counter = [&count]() { ++count; };
    counter(); counter(); counter();
    std::cout << "调用了 " << count << " 次\n";

    // 作为参数传递（高阶函数）
    std::vector<int> nums = {1, 5, 2, 8, 3, 9, 4};

    // 自定义排序：按绝对值大小降序
    std::sort(nums.begin(), nums.end(), [](int a, int b) { return a > b; });
    std::cout << "降序排列: ";
    for (int x : nums) std::cout << x << " ";
    std::cout << "\n";

    // filter 模拟（std::copy_if）
    std::vector<int> evens;
    std::copy_if(nums.begin(), nums.end(), std::back_inserter(evens),
                 [](int x) { return x % 2 == 0; });
    std::cout << "偶数: ";
    for (int x : evens) std::cout << x << " ";
    std::cout << "\n";
}

// ─────────────────────────────────────────────
// Part 4: 现代 C++ 特性速览
// ─────────────────────────────────────────────
void demo_modern_features() {
    std::cout << "\n=== 现代 C++ 特性 ===\n";

    // auto：类型推导
    auto x = 3.14;          // double
    auto s = std::string("hello");
    std::cout << "auto 推导: " << x << ", " << s << "\n";

    // 结构化绑定（C++17）
    std::pair<std::string, int> p = {"Alice", 30};
    auto [name, age] = p;
    std::cout << "结构化绑定: " << name << " 年龄 " << age << "\n";

    // if with initializer（C++17）
    std::map<std::string, int> m = {{"a", 1}, {"b", 2}};
    if (auto it = m.find("a"); it != m.end()) {
        std::cout << "找到 'a': " << it->second << "\n";
    }

    // std::optional（C++17）—— 比返回 -1 或 nullptr 更安全
    auto safe_divide = [](int a, int b) -> std::optional<double> {
        if (b == 0) return std::nullopt;
        return static_cast<double>(a) / b;
    };

    if (auto result = safe_divide(10, 3)) {
        std::cout << "10/3 = " << *result << "\n";
    }
    if (!safe_divide(10, 0)) {
        std::cout << "除以 0 返回 nullopt，安全！\n";
    }
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────
int main() {
    demo_vector();
    demo_map();
    demo_lambda();
    demo_modern_features();

    std::cout << "\n所有 demo 完成！\n";
    return 0;
}

/*
 * ══════════════════════════════════════════
 * 📝 练习题
 * ══════════════════════════════════════════
 *
 * 1. 实现 top-K 词频统计：
 *    - 给定一段文本（用空格分词）
 *    - 统计每个单词出现次数（unordered_map）
 *    - 输出出现次数最多的 K 个单词
 *    - 提示：把 map 转成 vector of pair，再 sort
 *
 * 2. 实现一个简单的计算图（为 PyTorch autograd 做铺垫）：
 *    - 用 map<string, float> 存储变量值
 *    - 用 vector<tuple<string,string,string,char>> 存储计算关系
 *      如 ("c", "a", "b", '+') 表示 c = a + b
 *    - 按拓扑顺序执行计算
 *
 * 3. 实现函数式风格的数据流水线：
 *    - 用 lambda + transform + filter + reduce
 *    - 完成：给定 vector<int>，过滤出偶数，每个平方，求和
 *    - 写成一行链式表达式
 */
