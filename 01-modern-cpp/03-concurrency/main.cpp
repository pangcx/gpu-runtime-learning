/**
 * 项目03：多线程与并发
 *
 * 学习目标：
 *   - 创建和管理线程（std::thread）
 *   - 用 mutex 保护共享数据
 *   - 用 future/promise 做线程间通信
 *   - 理解数据竞争（data race）和死锁
 *   - 为 CUDA 的并行编程思维打基础
 *
 * 编译运行：
 *   cd build && cmake .. -G Ninja && ninja && ./concurrency_demo
 */

#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <future>
#include <vector>
#include <chrono>
#include <numeric>

// ─────────────────────────────────────────────
// Part 1: 基础线程创建
// ─────────────────────────────────────────────

void worker(int id, int count) {
    for (int i = 0; i < count; ++i) {
        std::cout << "线程 " << id << " 执行第 " << i+1 << " 次\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void demo_basic_thread() {
    std::cout << "\n=== 基础线程 ===\n";

    std::thread t1(worker, 1, 3);
    std::thread t2(worker, 2, 3);

    t1.join();  // 等待 t1 完成
    t2.join();  // 等待 t2 完成

    std::cout << "两个线程都完成了\n";
}

// ─────────────────────────────────────────────
// Part 2: 数据竞争 vs mutex 保护
// ─────────────────────────────────────────────

// ❌ 有数据竞争的版本（多线程同时写 counter，结果不确定）
int unsafe_counter = 0;
void unsafe_increment() {
    for (int i = 0; i < 10000; ++i) {
        ++unsafe_counter;  // 非原子操作！读-改-写三步，可能被打断
    }
}

// ✅ 用 mutex 保护的版本
int safe_counter = 0;
std::mutex counter_mutex;
void safe_increment() {
    for (int i = 0; i < 10000; ++i) {
        std::lock_guard<std::mutex> lock(counter_mutex);  // RAII 锁，离开作用域自动释放
        ++safe_counter;
    }
}

// ✅ 用 atomic 的版本（最简洁，适合简单计数）
std::atomic<int> atomic_counter = 0;
void atomic_increment() {
    for (int i = 0; i < 10000; ++i) {
        ++atomic_counter;  // 原子操作，线程安全
    }
}

void demo_data_race() {
    std::cout << "\n=== 数据竞争 vs 安全并发 ===\n";

    // 不安全版本
    std::thread t1(unsafe_increment);
    std::thread t2(unsafe_increment);
    t1.join(); t2.join();
    std::cout << "不安全计数（期望 20000）: " << unsafe_counter << " ← 可能不是 20000！\n";

    // mutex 保护版本
    std::thread t3(safe_increment);
    std::thread t4(safe_increment);
    t3.join(); t4.join();
    std::cout << "mutex 保护（期望 20000）: " << safe_counter << "\n";

    // atomic 版本
    std::thread t5(atomic_increment);
    std::thread t6(atomic_increment);
    t5.join(); t6.join();
    std::cout << "atomic（期望 20000）:     " << atomic_counter << "\n";
}

// ─────────────────────────────────────────────
// Part 3: future / async（线程间获取结果）
// ─────────────────────────────────────────────

// 模拟耗时的计算任务
long long heavy_computation(int n) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return std::accumulate(std::vector<long long>(n, 1LL).begin(),
                           std::vector<long long>(n, 1LL).end(), 0LL);
}

void demo_future() {
    std::cout << "\n=== future/async（异步任务）===\n";

    auto start = std::chrono::steady_clock::now();

    // 串行：依次等待
    auto r1_serial = heavy_computation(1000000);
    auto r2_serial = heavy_computation(1000000);
    auto serial_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();

    start = std::chrono::steady_clock::now();

    // 并行：同时跑两个任务
    auto f1 = std::async(std::launch::async, heavy_computation, 1000000);
    auto f2 = std::async(std::launch::async, heavy_computation, 1000000);
    auto r1 = f1.get();  // 等待并获取结果
    auto r2 = f2.get();
    auto parallel_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();

    std::cout << "串行耗时: " << serial_ms << " ms\n";
    std::cout << "并行耗时: " << parallel_ms << " ms\n";
    std::cout << "加速比: " << (double)serial_ms / parallel_ms << "x\n";
}

// ─────────────────────────────────────────────
// Part 4: 线程池雏形（简化版）
// ─────────────────────────────────────────────
// 真实的线程池更复杂，这里展示核心思想

void demo_parallel_for() {
    std::cout << "\n=== 并行处理数据 ===\n";

    const int N = 8;
    std::vector<int> data(N);
    for (int i = 0; i < N; ++i) data[i] = i;

    std::vector<int> results(N);
    std::vector<std::thread> threads;

    // 每个线程处理一个元素（模拟 GPU 每个线程处理一个像素的思想）
    for (int i = 0; i < N; ++i) {
        threads.emplace_back([i, &data, &results]() {
            results[i] = data[i] * data[i];  // 平方
        });
    }

    for (auto& t : threads) t.join();

    std::cout << "并行平方计算结果: ";
    for (int r : results) std::cout << r << " ";
    std::cout << "\n";
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────
int main() {
    demo_basic_thread();
    demo_data_race();
    demo_future();
    demo_parallel_for();

    std::cout << "\n所有 demo 完成！\n";
    return 0;
}

/*
 * ══════════════════════════════════════════
 * 📝 练习题
 * ══════════════════════════════════════════
 *
 * 1. 实现一个简单的线程安全队列（Thread-safe Queue）：
 *    - 用 mutex + condition_variable 实现
 *    - push() 入队，pop() 出队（队列为空时阻塞等待）
 *    - 用一个生产者线程和两个消费者线程测试
 *
 * 2. 并行求和：
 *    - 有一个很大的 vector<int>（比如 1000 万个元素）
 *    - 把它分成 N 段，每段用一个线程求和
 *    - 最后把 N 个部分和加起来
 *    - 对比单线程和多线程的耗时
 *
 * 3. 模拟 GPU 并行：
 *    - 有两个 float 数组 A[1024] 和 B[1024]
 *    - 用 1024 个线程并行计算 C[i] = A[i] + B[i]
 *    - 这个模式就是 CUDA kernel 的原型
 *    - 思考：为什么实际 GPU 可以用上万个线程而 CPU 只能用几十个？
 */
