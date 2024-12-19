#!/bin/bash
INPUT_SCRIPT=cnot3_optimization.jl
INPUT_SLURM=cnot3_optimization.sb


MAXTIME=24
MAXITER=10000
USEJUQBOX=false


### Check that package precompiles correctly!
echo "Checking that QuantumGateDesign precompiles and loads correctly ..."
# Define variables
PRECOMPILE_LOG_FILE="precompile_output.log"
EMAIL_ADDRESS="leespen1@msu.edu"
EMAIL_SUBJECT="SpencerHPCC | Julia Script Failure: QuantumGateDesign.jl"
PROJECTENV=/mnt/home/leespen1/Research/QuantumGateDesign.jl # Should this change? The director is different on compute nodes

# Run the Julia script and capture output (stdout and stderr)
srun julia --project=$PROJECTENV $JULIA_SCRIPT -e "import Pkg; Pkg.precompile(); using QuantumGateDesign" > $PRECOMPILE_LOG_FILE 2>&1
PRECOMPILE_EXIT_CODE=$?


# Check if the precompilation failed, send email if so
if [ $PRECOMPILE_EXIT_CODE -ne 0 ]; then
    # Email the output log
    mail -s "$EMAIL_SUBJECT" "$EMAIL_ADDRESS" < $PRECOMPILE_LOG_FILE

    printf "Precompilation failed!\nOutput was:\n\n"
    cat $PRECOMPILE_LOG_FILE
    printf "\n\nExiting early ..."
    exit 1 # End script prematurely if precompilation fails. No need to send 
fi

echo "Precompilation succesful! Continuing ..."

# I hope that using srun fails when sbatch fails. Otherwise I may need to do an additional check


#ORDER_VALUES=(2 4 6 8 10 12)
#TARGETERROR_VALUES=("1e-1" "1e-2" "1e-3" "1e-4" "1e-5")
#SEED_VALUES=(0 1 2 3 4 5 6 7 8 9)
ORDER_VALUES=(4 6)
TARGETERROR_VALUES=("1e-1")
SEED_VALUES=(0)
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
            ########################################sbatch $DIR/$OUTPUT_SLURM
        done
    done
done


## Do another loop for juqbox (which can only have order 2, and targeterror doesn't do anything right now)
#
#USEJUQBOX=true
#ORDER=2
#TARGETERROR=0
#DIR="Order=${ORDER}_TargetError=${TARGETERROR}"
#mkdir $DIR
#
#for SEED in "${SEED_VALUES[@]}"; do
#    # Edit the Julia script, place it in directory
#    OUTPUT_SCRIPT="cnot3_optimization_order=${ORDER}_targeterror=${TARGETERROR}_seed=${SEED}_maxtime=${MAXTIME}_maxiter=${MAXITER}_usejuqbox=${USEJUQBOX}.jl"
#    sed "s/ORDER/${ORDER}/; s/TARGETERROR/${TARGETERROR}/; s/SEED/${SEED}/; s/MAXTIME/${MAXTIME}/; s/MAXITER/${MAXITER}/; s/USEJUQBOX/${USEJUQBOX}/" $INPUT_SCRIPT > $DIR/$OUTPUT_SCRIPT
#
#    # Edit the SLURM script, place it in directory
#    OUTPUT_SLURM="cnot3_optimization_order=${ORDER}_targeterror=${TARGETERROR}_seed=${SEED}_maxtime=${MAXTIME}_maxiter=${MAXITER}_usejuqbox=${USEJUQBOX}.sb"
#    sed "s/ORDER/${ORDER}/; s/TARGETERROR/${TARGETERROR}/; s/SEED/${SEED}/; s/MAXTIME/${MAXTIME}/; s/MAXITER/${MAXITER}/; s/USEJUQBOX/${USEJUQBOX}/; s/OUTPUT_SCRIPT/${OUTPUT_SCRIPT}/; s/DIR/$DIR/" $INPUT_SLURM > $DIR/$OUTPUT_SLURM
#
#    # Run the SLURM script (which runs the Julia script) [doing bash instead of sbatch for now]
#    sbatch $DIR/$OUTPUT_SLURM
#done


