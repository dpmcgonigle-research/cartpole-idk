## Description
These commands were used to create the sanity check

## GENERATE
```bash
CKPT=artifacts/dqn/dqn_seed42_200K/checkpoints/step_000200000.pt

INIT_ARGS=(
  --initial-distribution uniform
  --initial-x-min -0.05
  --initial-x-max 0.05
  --initial-x-dot-min -0.05
  --initial-x-dot-max 0.05
  --initial-theta-min -0.05
  --initial-theta-max 0.05
  --initial-theta-dot-min -0.05
  --initial-theta-dot-max 0.05
)

cartpole-generate \
  --checkpoint "$CKPT" \
  --episodes 500 \
  --output datasets/generated/familiarity/fit-nominal-500 \
  --seed 10001 \
  "${INIT_ARGS[@]}"

cartpole-generate \
  --checkpoint "$CKPT" \
  --episodes 200 \
  --output datasets/generated/familiarity/reference-nominal-200 \
  --seed 10002 \
  "${INIT_ARGS[@]}"

cartpole-generate \
  --checkpoint "$CKPT" \
  --episodes 200 \
  --output datasets/generated/familiarity/calibration-nominal-200 \
  --seed 10003 \
  "${INIT_ARGS[@]}"

cartpole-generate \
  --checkpoint "$CKPT" \
  --episodes 200 \
  --output datasets/generated/familiarity/test-nominal-200 \
  --seed 10004 \
  "${INIT_ARGS[@]}"

for DELAY in 1 2 3; do
  cartpole-generate \
    --checkpoint "$CKPT" \
    --episodes 200 \
    --output datasets/generated/familiarity/delay-${DELAY}-200 \
    --seed $((20000 + DELAY)) \
    --perturbation action-delay \
    --onset-mean 100 \
    --onset-std 5 \
    --delay-steps "$DELAY" \
    "${INIT_ARGS[@]}"
done
```

## PREPARE
```bash
cartpole-prepare \
  datasets/generated/familiarity/fit-nominal-500 \
  --output datasets/prepared/familiarity/fit-nominal-500 \
  --seed 30001 \
  --episode-length 300 \
  --episode-start 0 \
  --success-threshold 500 \
  --success-buffer 0

cartpole-prepare \
  datasets/generated/familiarity/reference-nominal-200 \
  --output datasets/prepared/familiarity/reference-nominal-200 \
  --seed 30002 \
  --episode-length 300 \
  --episode-start 0 \
  --success-threshold 500 \
  --success-buffer 0

cartpole-prepare \
  datasets/generated/familiarity/calibration-nominal-200 \
  --output datasets/prepared/familiarity/calibration-nominal-200 \
  --seed 30003 \
  --episode-length 300 \
  --episode-start 0 \
  --success-threshold 500 \
  --success-buffer 0

cartpole-prepare \
  datasets/generated/familiarity/test-nominal-200 \
  --output datasets/prepared/familiarity/test-nominal-200 \
  --seed 30004 \
  --episode-length 300 \
  --episode-start 0 \
  --success-threshold 500 \
  --success-buffer 0

for DELAY in 1 2 3; do
  cartpole-prepare \
    datasets/generated/familiarity/delay-${DELAY}-200 \
    --output datasets/prepared/familiarity/delay-${DELAY}-200 \
    --seed $((31000 + DELAY)) \
    --episode-length 300 \
    --episode-start 0 \
    --success-threshold 500 \
    --success-buffer 0
done
```

## FIT
```bash
for REP in state transition state_action state_action_next_state; do
  for PSI in 8 16 32 64 128 256 512; do
    OUT="artifacts/fit/familiarity/psi-${PSI}_t-200_${REP}"

    rm -rf "$OUT"

    cartpole-fit \
      datasets/prepared/familiarity/fit-nominal-500 \
      --representation "$REP" \
      --psi "$PSI" \
      --t 200 \
      --random-state 20261014 \
      --output "$OUT"
  done
done

for T in 25 50 100 200 400; do
  for RS in 101 202 303 404 505; do
    cartpole-fit \
      datasets/prepared/familiarity/fit-nominal-500 \
      --representation state \
      --psi 64 \
      --t "$T" \
      --random-state "$RS" \
      --output "artifacts/fit/familiarity/t-sweep/state_psi-64_t-${T}_seed-${RS}"
  done
done
```

## EMBED
```bash
for REP in state transition state_action state_action_next_state; do
  for PSI in 8 16 32 64 128 256 512; do

    FIT="artifacts/fit/familiarity/psi-${PSI}_t-200_${REP}"
    OUTROOT="artifacts/embeddings/familiarity/psi-${PSI}_t-200_${REP}"

    for DATA in \
      reference-nominal-200 \
      calibration-nominal-200 \
      test-nominal-200
    do
      rm -rf "${OUTROOT}/${DATA}/whole"

      cartpole-embed \
        --fit "${FIT}" \
        --dataset "datasets/prepared/familiarity/${DATA}" \
        --mode whole \
        --output "${OUTROOT}/${DATA}/whole"
    done

    for DELAY in 1 2 3; do
      rm -rf "${OUTROOT}/delay-${DELAY}/whole"

      cartpole-embed \
        --fit "${FIT}" \
        --dataset "datasets/prepared/familiarity/delay-${DELAY}-200" \
        --mode whole \
        --output "${OUTROOT}/delay-${DELAY}/whole"
    done

  done
done
```
