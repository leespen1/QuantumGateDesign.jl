#!/bin/bash

## Define parameter ranges
#X_values=(1 2 3)
#Y_values=(3 5 7)
#
## Loop over parameter combinations
#for X in "${X_values[@]}"; do
#    for Y in "${Y_values[@]}"; do
#        # Create a unique SLURM script for this combination
#        job_script="submit_X${X}_Y${Y}.sbatch"
#        sed "s/\${X}/${X}/g; s/\${Y}/${Y}/g" submit_template.sbatch > "$job_script"
#        
#        # Submit the job
#        sbatch "$job_script"
#        
#        # Optionally, clean up the generated script (uncomment if desired)
#        # rm "$job_script"
#    done
#done


MAX_TIME=24
MAX_ITER=5
USE_JUQBOX=false


ORDER_VALUES=(2 4 6)
TARGETERROR_VALUES=(1e-1 1e-2)
SEED_VALUES=(0 1 2)
for ORDER in 2 4 6 8 10 12; do
    for TARGET_ERROR in "1e-1" "1e-2"; do
        DIR="Order=${ORDER}_TargetError=${TARGET_ERROR}"
        mkdir $DIR

        for SEED in "${SEED_VALUES[@]}"; do

            sed "s/ORDER/${ORDER}/; s/TARGET_ERROR/${TARGET_ERROR}/; s/SEED/${SEED}/; s/MAX_TIME/${MAX_TIME}/; s/MAX_ITER/${MAX_ITER}/; s/USE_JUQBOX/${USE_JUQBOX}/" cnot3_optimization_meta.jl > "$DIR/cnot3_optimization_meta.jl"
            echo $ORDER $TARGET_ERROR

            # Put some sed statments in here
            # run the script (will need an sbatch here)
            #julia cnot3_optimization_meta.jl
        done
    done
done



