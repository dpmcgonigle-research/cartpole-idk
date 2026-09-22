#! /usr/bin/env bash

for TRAJ in $(grep  --color=never traj_ datasets/generated/200000-iter_nominal/100_episodes/generation_report.json | \grep -oE 'traj_[^"]+'); do 
    echo $TRAJ;
    sleep 2; 
    cartpole-replay datasets/generated/200000-iter_nominal/100_episodes --trajectory $TRAJ; 
done
