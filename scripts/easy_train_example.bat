python easy_train.py ^
    --training-dataset=c:/dev/nnue-pytorch/noob_master_leaf_static_d12_85M_0.binpack ^
    --validation-dataset=c:/dev/nnue-pytorch/d8_100000.binpack ^
    --num-workers=1 ^
    --threads=1 ^
    --gpus="0," ^
    --runs-per-gpu=1 ^
    --batch-size=1024 ^
    --max_epoch=10 ^
    --do-network-training=True ^
    --do-network-testing=True ^
    --tui=True ^
    --network-save-period=1 ^
    --random-fen-skipping=3 ^
    --fail-on-experiment-exists=False ^
    --build-engine-arch=x86-64-modern ^
    --build-threads=1 ^
    --epoch-size=1048500 ^
    --validation-size=4096 ^
    --network-testing-threads=2 ^
    --network-testing-explore-factor=1.5 ^
    --network-testing-book="https://github.com/official-stockfish/books/raw/master/UHO_XXL_%%2B0.90_%%2B1.19.epd.zip" ^
    --network-testing-nodes-per-move=1000 ^
    --network-testing-hash-mb=8 ^
    --network-testing-games-per-round=200 ^
    --engine-base=official-stockfish/Stockfish/master ^
    --engine-test=official-stockfish/Stockfish/master ^
    --nnue-pytorch=Sopel97/nnue-pytorch/easy_train ^
    --workspace-path=./easy_train_data ^
    --experiment-name=test ^
    --resume-training=True ^
    --features="HalfKAv2_hm%^" ^