"""
Extended plotting script to benchmark PyTorch, CuTe DSL implementation, and tensorop_gemm example.
Benchmarks square matrix sizes: 1024, 2048, 4096, 8192
"""

# ********************************************************************
#
# WARNING: This script works but the measured numbers are not reliable. Please take the real measurements from the profiles.
#
# ********************************************************************

import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np

def run_benchmarks(matrix_sizes, num_iterations=100, num_runs=9):
    """
    Run benchmarks for all implementations across multiple matrix sizes.
    
    Args:
        matrix_sizes: List of square matrix sizes to benchmark
        num_iterations: Number of iterations per benchmark run
        num_runs: Number of runs to collect statistics
    
    Returns:
        DataFrame with benchmark results
    """
    results = []
    
    for size in matrix_sizes:
        print(f"\n{'='*60}")
        print(f"Benchmarking matrix size: {size}x{size}x{size}")
        print(f"{'='*60}")
        
        mnk = (size, size, size)
        
        for run in range(num_runs):
            print(f"\nRun {run + 1}/{num_runs}")
            
            # Benchmark PyTorch (PyTorch)
            print("\n--- PyTorch ---")
            try:
                elapsed_ms, tflops = benchmark_cublas(mnk, num_iters=num_iterations)
                results.append({
                    'matrix_size': size,
                    'algorithm': 'PyTorch',
                    'execution_time_micros': elapsed_ms * 1000,  # Convert ms to microseconds
                    'TFLOP/s': tflops,
                    'run': run
                })
            except Exception as e:
                print(f"PyTorch benchmark failed: {e}")
            
            # Benchmark tensorop_gemm example
            print("\n--- tensorop_gemm Example ---")
            try:
                elapsed_ms, tflops = benchmark_sgemm_example(mnk, num_iters=num_iterations)
                if elapsed_ms is not None:
                    results.append({
                        'matrix_size': size,
                        'algorithm': 'CuTe tensorop_gemm Example',
                        'execution_time_micros': elapsed_ms * 1000,  # Convert ms to microseconds
                        'TFLOP/s': tflops,
                        'run': run
                    })
            except Exception as e:
                print(f"tensorop_gemm benchmark failed: {e}")

             # Benchmark CuTe DSL implementation
            print("\n--- CuTe DSL Implementation ---")
            try:
                elapsed_ms, tflops = benchmark(mnk, num_iters=num_iterations)
                results.append({
                    'matrix_size': size,
                    'algorithm': '3-Stage Pipeline Grid Swizzle',
                    'execution_time_micros': elapsed_ms * 1000,  # Convert ms to microseconds
                    'TFLOP/s': tflops,
                    'run': run
                })
            except Exception as e:
                print(f"CuTe DSL benchmark failed: {e}")
            
            
            # Clear GPU cache between runs
            torch.cuda.empty_cache()
    
    return pd.DataFrame(results)


def plot_results(df, output_path='out/mm_performance_cute_comparison.png'):
    """
    Plot benchmark results with mean and standard deviation.
    
    Args:
        df: DataFrame with benchmark results
        output_path: Path to save the output plot
    """
    # Calculate statistics
    df_stats = df.groupby(['algorithm', 'matrix_size']).agg(
        avg_execution_time_micros=('execution_time_micros', 'mean'),
        std_execution_time_micros=('execution_time_micros', 'std'),
        avg_TFLOP_s=('TFLOP/s', 'mean'),
        std_TFLOP_s=('TFLOP/s', 'std')
    ).reset_index()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(df_stats.to_string(index=False))
    print("="*80)
    
    # Create the plot
    plt.figure(figsize=(10,6))
    
    algorithms = df_stats['algorithm'].unique()
    colors = ['#2ca02c', '#ff7f0e', '#1f77b4']
    markers = ['o', 's', '^']
    
    for idx, algorithm in enumerate(algorithms):
        group = df_stats[df_stats['algorithm'] == algorithm]
        plt.plot(
            group['matrix_size'].to_numpy(),
            group['avg_TFLOP_s'].to_numpy(),
            marker=markers[idx % len(markers)],
            label=algorithm,
            linewidth=2,
            markersize=8,
            color=colors[idx % len(colors)]
        )
        plt.fill_between(
            group['matrix_size'].to_numpy(),
            group['avg_TFLOP_s'].to_numpy() - group['std_TFLOP_s'].to_numpy(),
            group['avg_TFLOP_s'].to_numpy() + group['std_TFLOP_s'].to_numpy(),
            alpha=0.2,
            color=colors[idx % len(colors)]
        )
    
    plt.xscale('log', base=2)
    plt.xticks([1024, 2048, 4096, 8192], ['1024', '2048', '4096', '8192'])
    plt.xlabel('Square Matrix Size', fontsize=14)
    plt.ylabel('TFLOP/s', fontsize=14)
    plt.title('Matrix Multiplication Performance Comparison', fontsize=16)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Also save the raw data
    csv_path = output_path.replace('.png', '.csv')
    df_stats.to_csv(csv_path, index=False)
    print(f"Summary data saved to: {csv_path}")
    
    # Save raw data with all runs
    raw_csv_path = output_path.replace('.png', '_raw.csv')
    df.to_csv(raw_csv_path, index=False)
    print(f"Raw data saved to: {raw_csv_path}")


def main():
    """Main function to run benchmarks and create plots."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch, CuTe DSL, and tensorop_gemm implementations"
    )
    parser.add_argument(
        '--sizes',
        type=str,
        default='1024,2048,4096,8192',
        help='Comma-separated list of matrix sizes to benchmark (default: 1024,2048,4096,8192)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Number of iterations per benchmark run (default: 100)'
    )
    parser.add_argument(
        '--runs',
        type=int,
        default=5,
        help='Number of runs to collect statistics (default: 9)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='out/mm_performance_cute_dsl.png',
        help='Output path for the plot (default: out/mm_performance_cute_dsl.png)'
    )
    parser.add_argument(
        '--from-csv',
        type=str,
        default=None,
        help='Read data from a previously saved CSV file instead of running benchmarks'
    )
    
    args = parser.parse_args()
    
    # Check if reading from CSV
    if args.from_csv:
        print("="*60)
        print("READING DATA FROM CSV")
        print("="*60)
        print(f"CSV file: {args.from_csv}")
        print("="*60)
        
        # Read the raw data from CSV
        try:
            df = pd.read_csv(args.from_csv)
            print(f"\nSuccessfully loaded {len(df)} rows from CSV")
            print(f"Columns: {', '.join(df.columns)}")
            
            # Check if this is aggregated data or raw data
            if 'avg_execution_time_micros' in df.columns:
                # This is already aggregated data, use it directly for plotting
                print("\nDetected aggregated data format")
                df_stats = df
            else:
                # This is raw data, compute statistics
                print("\nDetected raw data format, computing statistics...")
                df_stats = df.groupby(['algorithm', 'matrix_size']).agg(
                    avg_execution_time_micros=('execution_time_micros', 'mean'),
                    std_execution_time_micros=('execution_time_micros', 'std'),
                    avg_TFLOP_s=('TFLOP/s', 'mean'),
                    std_TFLOP_s=('TFLOP/s', 'std')
                ).reset_index()
            
            # Print summary
            print("\n" + "="*80)
            print("DATA SUMMARY")
            print("="*80)
            print(df_stats.to_string(index=False))
            print("="*80)
            
            # Create the plot
            plt.figure(figsize=(10,6))
            
            algorithms = df_stats['algorithm'].unique()
            colors = ['#2ca02c', '#ff7f0e', '#1f77b4']
            markers = ['o', 's', '^']
            
            for idx, algorithm in enumerate(algorithms):
                group = df_stats[df_stats['algorithm'] == algorithm]
                plt.plot(
                    group['matrix_size'].to_numpy(),
                    group['avg_TFLOP_s'].to_numpy(),
                    marker=markers[idx % len(markers)],
                    label=algorithm,
                    # linewidth=2,
                    # markersize=8,
                    color=colors[idx % len(colors)]
                )
                # Only plot error bars if std data is available and not NaN
                if 'std_TFLOP_s' in df_stats.columns and not group['std_TFLOP_s'].isna().all():
                    plt.fill_between(
                        group['matrix_size'].to_numpy(),
                        group['avg_TFLOP_s'].to_numpy() - group['std_TFLOP_s'].to_numpy(),
                        group['avg_TFLOP_s'].to_numpy() + group['std_TFLOP_s'].to_numpy(),
                        alpha=0.2,
                        color=colors[idx % len(colors)]
                    )
            
            plt.xscale('log', base=2)
            plt.xticks([1024, 2048, 4096, 8192], ['1024', '2048', '4096', '8192'])
            plt.xlabel('Square Matrix Size', fontsize=14)
            plt.ylabel('TFLOP/s', fontsize=14)
            plt.title('Matrix Multiplication Performance Comparison', fontsize=16)
            plt.legend(fontsize=11, loc='best')
            # plt.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            
            # Save the plot
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            plt.savefig(args.output, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to: {args.output}")
            
            print("\n✓ Plot generation complete!")
            return
            
        except FileNotFoundError:
            print(f"\n✗ Error: CSV file '{args.from_csv}' not found!")
            return
        except Exception as e:
            print(f"\n✗ Error reading CSV file: {e}")
            return
    # else:
    #     # Add the path to the cute_mma_matmul_3stage-pip_grid_swizzle_dsl.py module
    #     sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'matmuls/clean'))

    #     # Import the benchmark functions from the DSL file
    #     from cute_mma_matmul_3stage_pip_grid_swizzle_dsl import benchmark, benchmark_cublas, benchmark_sgemm_example

    #     # Original benchmark execution path
    #     # Check CUDA availability
    #     if not torch.cuda.is_available():
    #         raise RuntimeError("CUDA is required to run benchmarks")
        
    #     # Parse matrix sizes
    #     matrix_sizes = [int(x.strip()) for x in args.sizes.split(',')]
        
    #     print("="*60)
    #     print("MATRIX MULTIPLICATION BENCHMARK")
    #     print("="*60)
    #     print(f"Matrix sizes: {matrix_sizes}")
    #     print(f"Iterations per run: {args.iterations}")
    #     print(f"Number of runs: {args.runs}")
    #     print(f"Device: {torch.cuda.get_device_name(0)}")
    #     print("="*60)
        
    #     # Run benchmarks
    #     df = run_benchmarks(matrix_sizes, num_iterations=args.iterations, num_runs=args.runs)
        
    #     # Plot results
    #     plot_results(df, output_path=args.output)
        
    #     print("\n✓ Benchmarking complete!")


if __name__ == '__main__':
    main()
