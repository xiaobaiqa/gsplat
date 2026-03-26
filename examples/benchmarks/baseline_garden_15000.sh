SCENE="garden"
SCENE_DIR="data/360_v2"
RESULT_DIR="results/benchmark_15000/garden"

CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --disable-viewer \
    --data-dir ${SCENE_DIR}/${SCENE}/ \
    --data-factor 4 \
    --result-dir ${RESULT_DIR}/ \
    --batch-size 1 \
    --max-steps 15000 \
    --eval-steps 7000 15000 \
    --save-steps 7000 15000 \
    --test-every 8 \
    --init-type sfm \
    --init-num-pts 100000 \
    --init-extent 3.0 \
    --init-opa 0.1 \
    --init-scale 1.0 \
    --sh-degree 3 \
    --sh-degree-interval 1000 \
    --ssim-lambda 0.2 \
    --near-plane 0.01 \
    --far-plane 1e10 \
    --no-packed \
    --no-sparse-grad \
    --no-visible-adam \
    --no-antialiased \
    --no-random-bkgd \
    --means-lr 1.6e-4 \
    --scales-lr 5e-3 \
    --opacities-lr 5e-2 \
    --quats-lr 1e-3 \
    --sh0-lr 2.5e-3 \
    --shN-lr 1.25e-4 \
    --strategy.prune-opa 0.005 \
    --strategy.grow-grad2d 0.0002 \
    --strategy.grow-scale3d 0.01 \
    --strategy.grow-scale2d 0.05 \
    --strategy.prune-scale3d 0.1 \
    --strategy.prune-scale2d 0.15 \
    --strategy.refine-start-iter 500 \
    --strategy.refine-stop-iter 15000 \
    --strategy.refine-every 100 \
    --strategy.reset-every 3000 \
    --strategy.no-absgrad \
    --strategy.no-revised-opacity \
    --strategy.key-for-gradient means2d
