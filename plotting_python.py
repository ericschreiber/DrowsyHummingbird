import matplotlib.pyplot as plt
import pandas as pd

# Values are profiled with NCU
# Note: Next time I'm smarter and save to csv and load from there, but for now I just copied the values by hand.

colors = ["#9D8352", "#AC4A8C", "#5BAD96"]


matrix_sizes =  [1024]*9 + [2048] *9 + [4096]*9 + [8192]*9
algorithms = ['3-Stage Buffering', 'PyTorch', 'CuTe tensorop_gemm Example']
# Own implementation
data = {
    'matrix_size': matrix_sizes,
    'execution_time_micros': [
        25.12,25.28,25.18,25.06,24.96,24.90,24.99,24.93,25.06
        ,115.3,115.5,115.4,115.3,115.9,115.6,115.7,115.4,115.6
        ,707,707,708,708,709,707,707,707,709
        ,5980,5720,5730,6000,5590,5730,6000,5760,5780
        ],
    'algorithm': ['3-Stage Buffering'] * 36
}
df = pd.DataFrame(data)

# PyTorch Implementation
data = {
    'matrix_size': matrix_sizes,
    'execution_time_micros': [
        23.68,23.74,23.74,23.65,23.84,23.58,23.71,23.68,23.68
        ,134.4,134.2,134.7,134.5,134.6,134.6,134.5,134.4,134.3
        ,629,630,629,629,629,628,629,630,630
        ,4620,4620,4630,4620,4630,4630,4630,4620,4620
    ],
    'algorithm': ['PyTorch'] * 36
}
df = pd.concat([df, pd.DataFrame(data)])

# tensorop_gemm Example
data = {
    'matrix_size': matrix_sizes,
    'execution_time_micros': [
        27.68,28.06,28.03,27.81,28.00,27.65,27.97,27.81,28.10
        ,121.1,120.9,121.1,120.9,120.8,120.9,120.9,120.9,121.0
        ,724,724,725,725,725,725,725,725,725
        ,5350,5350,5350,5350,5350,5350,5350,5360,5350
    ],
    'algorithm': ['CuTe tensorop_gemm Example'] * 36
}
df = pd.concat([df, pd.DataFrame(data)])


# Add the FLOPs column assuming a standard matrix multiplication FLOPs calculation
df['FLOPs'] = 2 * (df['matrix_size'] ** 3)
# Calculate TFLOP/s
df['TFLOP/s'] = df['FLOPs'] / (df['execution_time_micros'] * 1e-6) / 1e12

print("Stats:")
print(f"Avg TFLOP/s for 3-Stage Buffering in 2048x2048: {df[(df['algorithm'] == '3-Stage Buffering') & (df['matrix_size'] == 2048)]['TFLOP/s'].mean():.2f}")
print(f"Avg TFLOP/s for PyTorch in 2048x2048: {df[(df['algorithm'] == 'PyTorch') & (df['matrix_size'] == 2048)]['TFLOP/s'].mean():.2f}")
print(f"Avg TFLOP/s for CuTe tensorop_gemm Example in 2048x2048: {df[(df['algorithm'] == 'CuTe tensorop_gemm Example') & (df['matrix_size'] == 2048)]['TFLOP/s'].mean():.2f}")
print(f"Avg execution time for 3-Stage Buffering in 2048x2048: {df[(df['algorithm'] == '3-Stage Buffering') & (df['matrix_size'] == 2048)]['execution_time_micros'].mean():.2f} µs")
print(f"Avg execution time for PyTorch in 2048x2048: {df[(df['algorithm'] == 'PyTorch') & (df['matrix_size'] == 2048)]['execution_time_micros'].mean():.2f} µs")
print(f"Avg execution time for CuTe tensorop_gemm Example in 2048x2048: {df[(df['algorithm'] == 'CuTe tensorop_gemm Example') & (df['matrix_size'] == 2048)]['execution_time_micros'].mean():.2f} µs")

# Make avg and std columns for each algorithm and matrix size
df = df.groupby(['algorithm', 'matrix_size']).agg(
    avg_execution_time_micros=('execution_time_micros', 'mean'),
    std_execution_time_micros=('execution_time_micros', 'std'),
    avg_TFLOP_s=('TFLOP/s', 'mean'),
    std_TFLOP_s=('TFLOP/s', 'std')
).reset_index()
# Plotting
plt.figure(figsize=(10, 6))
for algorithm in algorithms:
    group = df[df['algorithm'] == algorithm]
    plt.plot(group['matrix_size'].to_numpy(), group['avg_TFLOP_s'].to_numpy(), marker='o', label=algorithm, 
             color=colors[algorithms.index(algorithm)], linewidth=2, markersize=8)
    plt.fill_between(group['matrix_size'].to_numpy(), 
                     group['avg_TFLOP_s'].to_numpy() - group['std_TFLOP_s'].to_numpy(),
                     group['avg_TFLOP_s'].to_numpy() + group['std_TFLOP_s'].to_numpy(),
                     alpha=0.2, color=colors[algorithms.index(algorithm)])
plt.xscale('log', base=2)
# Fixed x-axis ticks 
plt.xticks([1024, 2048, 4096, 8192], ['1024', '2048', '4096', '8192'])
plt.xlabel('Square Matrix Size', fontsize=14)
plt.ylabel('TFLOP/s', fontsize=14)
# plt.title('MM Kernel Performance', fontsize=16) 
plt.legend(fontsize=12)
# plt.grid(True)
plt.savefig('out/mm_performance_cute_dsl.png', dpi=300)