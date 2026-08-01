/**
 * 项目01：内存管理与智能指针
 *
 * 学习目标：
 *   - 理解栈 vs 堆内存的区别
 *   - 掌握 unique_ptr / shared_ptr / weak_ptr 的用法和原理
 *   - 理解 RAII 惯用法（资源获取即初始化）
 *   - 避免内存泄漏
 *
 * 编译运行：
 *   cd build && cmake .. -G Ninja && ninja && ./memory_demo
 */

#include <iostream>
#include <memory>
#include <string>
#include <vector>

// ─────────────────────────────────────────────
// Part 1: 裸指针 vs 智能指针
// ─────────────────────────────────────────────

// 模拟一个"有资源"的对象，构造/析构时打印日志
struct Resource {
    std::string name;

    explicit Resource(std::string n) : name(std::move(n)) {
        std::cout << "[构造] Resource: " << name << "\n";
    }

    ~Resource() {
        std::cout << "[析构] Resource: " << name << "\n";
    }

    void use() const {
        std::cout << "[使用] Resource: " << name << "\n";
    }
};

void demo_raw_pointer() {
    std::cout << "\n=== 裸指针（危险！）===\n";

    // 裸指针：必须手动 delete，容易忘记或异常时跳过
    Resource* r = new Resource("裸指针资源");
    r->use();
    delete r;  // 如果这里抛出异常，delete 就跑不到了 → 内存泄漏
}

void demo_unique_ptr() {
    std::cout << "\n=== unique_ptr（独占所有权）===\n";

    // unique_ptr：离开作用域时自动 delete，不能复制，只能 move
    auto r = std::make_unique<Resource>("unique资源");
    r->use();

    // 转移所有权
    auto r2 = std::move(r);
    // r 现在是 nullptr，r2 接管了所有权
    std::cout << "r 是否为空: " << (r == nullptr ? "是" : "否") << "\n";
    r2->use();

    // 离开函数时 r2 自动销毁，析构函数自动调用
}

void demo_shared_ptr() {
    std::cout << "\n=== shared_ptr（共享所有权）===\n";

    // shared_ptr：引用计数，最后一个持有者销毁时才 delete
    auto r1 = std::make_shared<Resource>("共享资源");
    std::cout << "引用计数: " << r1.use_count() << "\n";  // 1

    {
        auto r2 = r1;  // 复制，引用计数 +1
        std::cout << "引用计数: " << r1.use_count() << "\n";  // 2
        r2->use();
    }  // r2 离开作用域，引用计数 -1

    std::cout << "引用计数: " << r1.use_count() << "\n";  // 1
    // r1 离开函数，引用计数变 0，Resource 析构
}

// ─────────────────────────────────────────────
// Part 2: RAII 惯用法
// ─────────────────────────────────────────────
// RAII = Resource Acquisition Is Initialization
// 核心思想：把资源绑定到对象生命周期，构造时获取，析构时释放
// unique_ptr / shared_ptr 本身就是 RAII 的体现

// 自己实现一个简单的 RAII 文件句柄
struct FileHandle {
    FILE* fp = nullptr;

    explicit FileHandle(const char* path, const char* mode) {
        fp = fopen(path, mode);
        if (fp) std::cout << "[RAII] 文件已打开\n";
        else    std::cout << "[RAII] 文件打开失败\n";
    }

    ~FileHandle() {
        if (fp) {
            fclose(fp);
            std::cout << "[RAII] 文件已关闭（析构自动触发）\n";
        }
    }

    // 禁止复制，允许 move
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;
};

void demo_raii() {
    std::cout << "\n=== RAII 惯用法 ===\n";
    FileHandle f("/tmp/test_raii.txt", "w");
    // 无论函数是正常返回还是抛出异常，析构都会自动调用
}

// ─────────────────────────────────────────────
// Part 3: 常见内存问题演示（注释掉的是错误示范）
// ─────────────────────────────────────────────
void demo_memory_issues() {
    std::cout << "\n=== 常见内存问题（已注释，仅供学习）===\n";

    // ❌ 悬空指针（use-after-free）
    // int* p = new int(42);
    // delete p;
    // std::cout << *p;  // 未定义行为！

    // ❌ 双重释放
    // int* p = new int(42);
    // delete p;
    // delete p;  // 崩溃！

    // ❌ 内存泄漏
    // int* p = new int(42);
    // return;  // 没有 delete，内存泄漏

    // ✅ 用 unique_ptr 全部避免
    auto p = std::make_unique<int>(42);
    std::cout << "安全访问: " << *p << "\n";
    // 自动释放，不会泄漏，不会悬空，不会双重释放
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────
int main() {
    demo_raw_pointer();
    demo_unique_ptr();
    demo_shared_ptr();
    demo_raii();
    demo_memory_issues();

    std::cout << "\n所有 demo 完成！\n";
    return 0;
}

/*
 * ══════════════════════════════════════════
 * 📝 练习题（在此文件基础上修改完成）
 * ══════════════════════════════════════════
 *
 * 1. 实现一个 Matrix 类：
 *    - 构造时用 new[] 分配二维数组
 *    - 析构时 delete[] 释放
 *    - 实现拷贝构造函数（深拷贝）
 *    - 用 unique_ptr 改写，去掉手动 delete
 *
 * 2. 模拟循环引用问题：
 *    - 创建两个 shared_ptr，互相持有对方
 *    - 观察析构函数是否被调用（不会被调用 = 内存泄漏）
 *    - 用 weak_ptr 打破循环，再次观察
 *
 * 3. 用 shared_ptr 实现一个简单的对象池（Object Pool）：
 *    - 预先创建 N 个 Resource 对象放入 vector
 *    - 提供 acquire() 和 release() 接口
 *    - 观察引用计数的变化
 */
