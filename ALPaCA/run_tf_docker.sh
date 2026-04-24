#!/bin/bash
# Run TensorFlow ALPaCA in a native Linux x64 Docker container to bypass macOS ARM threading bugs
docker run --rm -v $(pwd):/workspace -w /workspace -it --platform linux/amd64 continuumio/miniconda3 bash -c "
pip install tensorflow==2.15.0 numpy scipy matplotlib scikit-learn pyyaml && \
python run_tests.py --dataset dummy_test --backend tensorflow --train_iters 500 --n_context 10 --use_past_n_as_state --calibrate_sigma
"
