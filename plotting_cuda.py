import matplotlib.pyplot as plt
import pandas as pd

# Values are profiled with NCU
# Note: Next time I'm smarter and save to csv and load from there, but for now I just copied the values by hand.

colors = ["#AC4A8C", "#888888", "#5678B3", "#4B8DA1", "#9D8352", "#AC4A8C"]

matrix_sizes =  [1024]*9 + [2048] *9 + [4096]*9 + [8192]*9
algorithms = ['CUBLAS', 'Base Implementation', '2-Stage Buffering', 'L2 Cache Swizzling', '3-Stage Buffering']
# CUBLAS
data = {
    'matrix_size': matrix_sizes,
    'execution_time_micros': [
        23.62,23.62,23.55,23.55,23.65,23.55,23.58,23.49,23.52
        ,133.50,133.82,133.47,133.86,133.73,133.98,133.76,133.60,133.70
        ,623.46,622.75,623.20,623.01,623.07,622.98,623.01,622.72,622.78
        ,4580,4580,4570,4570,4570,4580,4570,4580,4570
        ],
    'algorithm': ['CUBLAS'] * 36
}
df = pd.DataFrame(data)

# Base Implementation
data = {
    'matrix_size': matrix_sizes,
    'execution_time_micros': [
        45.44,45.54,45.89,45.60,45.57,45.54,45.44,45.54,45.44
        ,131.10,132.64,132.70,133.38,131.33,132.45,130.59,131.23,132.74
        ,836.22,833.95,836.77,832.77,832.19,833.50,832.83,835.04,835.10
        ,6070,6070,6110,6090,6090,6070,6110,6070,6100
    ],
    'algorithm': ['Base Implementation'] * 36
}
df = pd.concat([df, pd.DataFrame(data)])

# 2-Stage Buffering
data = {
    'matrix_size': matrix_sizes,
    'execution_time_micros': [
        30.37,30.50,30.59,30.72,30.75,30.91,30.66,30.72,30.59
        ,120.70,120.83,120.99,120.48,121.34,121.18,120.90,120.03,119.94
        ,719.17,713.98,714.53,713.09,713.92,714.8,717.92,714.78,717.12
        ,5340,5380,5400,5410,5370,5380,5400,5380,5390
    ],
    'algorithm': ['2-Stage Buffering'] * 36
}
df = pd.concat([df, pd.DataFrame(data)])

# L2 Cache Swizzling
data = {
    'matrix_size': matrix_sizes,
    'execution_time_micros': [
        30.59,30.75,30.88,30.62,30.72,30.75,30.66,30.91,30.66
        ,119.42,120.22,119.36,119.97,119.17,119.30,120.45,119.52,120.32
        ,713.02,713.34,712.70,710.46,709.82,715.62,711.58,709.73,712.42
        ,5330,5340,5330,5340,5330,5350,5350,5360,5370
    ],
    'algorithm': ['L2 Cache Swizzling'] * 36
}
df = pd.concat([df, pd.DataFrame(data)])

# 3-Stage Buffering
data = {
    'matrix_size': matrix_sizes,
    'execution_time_micros': [
        25.02,24.99,24.93,24.96,25.02,25.31,24.90,25.06,24.93
        ,115.07,115.17,115.36,115.33,115.10,114.75,115.84,116.29,115.33
        ,684.10,682.59,682.27,683.33,683.78,682.53,683.74,682.62,684.61
        ,5470,5510,5480,5520,5450,5600,5440,5510,5530
    ],
    'algorithm': ['3-Stage Buffering'] * 36
}
df = pd.concat([df, pd.DataFrame(data)])

# Add the FLOPs column assuming a standard matrix multiplication FLOPs calculation
df['FLOPs'] = 2 * (df['matrix_size'] ** 3)
# Calculate TFLOP/s
df['TFLOP/s'] = df['FLOPs'] / (df['execution_time_micros'] * 1e-6) / 1e12

print("Stats:")
print(f"Avg TFLOP/s for CUBLAS in 2048x2048: {df[(df['algorithm'] == 'CUBLAS') & (df['matrix_size'] == 2048)]['TFLOP/s'].mean():.2f}")
print(f"Avg TFLOP/s for Base Implementation in 2048x2048: {df[(df['algorithm'] == 'Base Implementation') & (df['matrix_size'] == 2048)]['TFLOP/s'].mean():.2f}")
print(f"Avg TFLOP/s for 2-Stage Buffering in 2048x2048: {df[(df['algorithm'] == '2-Stage Buffering') & (df['matrix_size'] == 2048)]['TFLOP/s'].mean():.2f}")
print(f"Avg TFLOP/s for L2 Cache Swizzling in 2048x2048: {df[(df['algorithm'] == 'L2 Cache Swizzling') & (df['matrix_size'] == 2048)]['TFLOP/s'].mean():.2f}")
print(f"Avg TFLOP/s for 3-Stage Buffering in 2048x2048: {df[(df['algorithm'] == '3-Stage Buffering') & (df['matrix_size'] == 2048)]['TFLOP/s'].mean():.2f}")
print(f"Avg execution time for CUBLAS in 2048x2048: {df[(df['algorithm'] == 'CUBLAS') & (df['matrix_size'] == 2048)]['execution_time_micros'].mean():.2f} µs")
print(f"Avg execution time for Base Implementation in 2048x2048: {df[(df['algorithm'] == 'Base Implementation') & (df['matrix_size'] == 2048)]['execution_time_micros'].mean():.2f} µs")
print(f"Avg execution time for 2-Stage Buffering in 2048x2048: {df[(df['algorithm'] == '2-Stage Buffering') & (df['matrix_size'] == 2048)]['execution_time_micros'].mean():.2f} µs")
print(f"Avg execution time for L2 Cache Swizzling in 2048x2048: {df[(df['algorithm'] == 'L2 Cache Swizzling') & (df['matrix_size'] == 2048)]['execution_time_micros'].mean():.2f} µs")
print(f"Avg execution time for 3-Stage Buffering in 2048x2048: {df[(df['algorithm'] == '3-Stage Buffering') & (df['matrix_size'] == 2048)]['execution_time_micros'].mean():.2f} µs")
      
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
plt.savefig('out/mm_performance.png', dpi=300)