#!/bin/bash

CUDA_EXEC=./build/CUDA_KNN
PYTHON_EXEC="python sklearn_imp.py"

DISTANCES=("L1" "L2" "Cosine" "LINF")
KS=(1 5 10 20 43)
N_RUNS=5

echo "Distance,K,CUDA_avg_ms,Python_avg_ms,Speedup"

for dist in "${DISTANCES[@]}"; do
    for k in "${KS[@]}"; do
        cuda_total=0
        python_total=0
        valid_runs=0

        for ((i=1;i<=N_RUNS;i++)); do
            cuda_output=$($CUDA_EXEC $dist $k)
            python_output=$($PYTHON_EXEC $dist $k)

            cuda_time=$(echo "$cuda_output" | grep "CUDA time:" | awk '{print $3}' | sed 's/ms//')
            python_time=$(echo "$python_output" | grep "Python KNN time:" | awk '{print $4}' | sed 's/s//')

            if [[ -z "$cuda_time" || -z "$python_time" ]]; then
                echo "Warning: failed to parse times for $dist, k=$k, run $i"
                continue
            fi

            cuda_total=$(echo "$cuda_total + $cuda_time" | bc -l)
            # convert Python seconds to ms
            python_total=$(echo "$python_total + ($python_time*1000)" | bc -l)
            valid_runs=$((valid_runs+1))
        done

        if [[ $valid_runs -eq 0 ]]; then
            echo "$dist,$k,ERROR,ERROR,ERROR"
            continue
        fi

        cuda_avg=$(echo "$cuda_total / $valid_runs" | bc -l)
        python_avg=$(echo "$python_total / $valid_runs" | bc -l)
        speedup=$(echo "$python_avg / $cuda_avg" | bc -l)

        printf "%s,%d,%.2f,%.2f,%.2f\n" "$dist" "$k" "$cuda_avg" "$python_avg" "$speedup"
    done
done

