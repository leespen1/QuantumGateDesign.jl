#!/bin/bash
INPUT_SCRIPT=cnot3_optimization.jl
INPUT_SLURM=cnot3_optimization.sb


MAXTIME=24
MAXITER=10000
USEJUQBOX=false

ORDER_VALUES=(2 4)
TARGETERROR_VALUES=("1e-1" "1e-2")
SEED_VALUES=(0 1)
for ORDER in "${ORDER_VALUES[@]}"; do
    for TARGETERROR in "${TARGETERROR_VALUES[@]}"; do
        DIR="Order=${ORDER}_TargetError=${TARGETERROR}"
        mkdir $DIR

        for SEED in "${SEED_VALUES[@]}"; do
            # Edit the Julia script, place it in directory
            OUTPUT_SCRIPT="cnot3_optimization_order=${ORDER}_targeterror=${TARGETERROR}_seed=${SEED}_maxtime=${MAXTIME}_maxiter=${MAXITER}_usejuqbox=${USEJUQBOX}.jl"
            sed "s/ORDER/${ORDER}/; s/TARGETERROR/${TARGETERROR}/; s/SEED/${SEED}/; s/MAXTIME/${MAXTIME}/; s/MAXITER/${MAXITER}/; s/USEJUQBOX/${USEJUQBOX}/" $INPUT_SCRIPT > $DIR/$OUTPUT_SCRIPT

            # Edit the SLURM script, place it in directory
            OUTPUT_SLURM="cnot3_optimization_order=${ORDER}_targeterror=${TARGETERROR}_seed=${SEED}_maxtime=${MAXTIME}_maxiter=${MAXITER}_usejuqbox=${USEJUQBOX}.sb"
            sed "s/ORDER/${ORDER}/; s/TARGETERROR/${TARGETERROR}/; s/SEED/${SEED}/; s/MAXTIME/${MAXTIME}/; s/MAXITER/${MAXITER}/; s/USEJUQBOX/${USEJUQBOX}/; s/OUTPUT_SCRIPT/${OUTPUT_SCRIPT}/; s/DIR/$DIR/" $INPUT_SLURM > $DIR/$OUTPUT_SLURM
    
            # Run the SLURM script (which runs the Julia script) [doing bash instead of sbatch for now]
            sbatch $DIR/$OUTPUT_SLURM
        done
    done
done


# Do another loop for juqbox (which can only have order 2, and targeterror doesn't do anything right now)

USEJUQBOX=true
ORDER=2
TARGETERROR=0
DIR="Order=${ORDER}_TargetError=${TARGETERROR}"
mkdir $DIR

for SEED in "${SEED_VALUES[@]}"; do
    # Edit the Julia script, place it in directory
    OUTPUT_SCRIPT="cnot3_optimization_order=${ORDER}_targeterror=${TARGETERROR}_seed=${SEED}_maxtime=${MAXTIME}_maxiter=${MAXITER}_usejuqbox=${USEJUQBOX}.jl"
    sed "s/ORDER/${ORDER}/; s/TARGETERROR/${TARGETERROR}/; s/SEED/${SEED}/; s/MAXTIME/${MAXTIME}/; s/MAXITER/${MAXITER}/; s/USEJUQBOX/${USEJUQBOX}/" $INPUT_SCRIPT > $DIR/$OUTPUT_SCRIPT

    # Edit the SLURM script, place it in directory
    OUTPUT_SLURM="cnot3_optimization_order=${ORDER}_targeterror=${TARGETERROR}_seed=${SEED}_maxtime=${MAXTIME}_maxiter=${MAXITER}_usejuqbox=${USEJUQBOX}.sb"
    sed "s/ORDER/${ORDER}/; s/TARGETERROR/${TARGETERROR}/; s/SEED/${SEED}/; s/MAXTIME/${MAXTIME}/; s/MAXITER/${MAXITER}/; s/USEJUQBOX/${USEJUQBOX}/; s/OUTPUT_SCRIPT/${OUTPUT_SCRIPT}/; s/DIR/$DIR/" $INPUT_SLURM > $DIR/$OUTPUT_SLURM

    # Run the SLURM script (which runs the Julia script) [doing bash instead of sbatch for now]
    sbatch $DIR/$OUTPUT_SLURM
done


