mkdir -p .temp

if [ ! -f .temp/run_clm.py ]; then
    # https://stackoverflow.com/a/13735150
    curl https://raw.githubusercontent.com/huggingface/transformers/master/examples/pytorch/language-modeling/run_clm.py -O
    mv run_clm.py .temp/run_clm.py
fi

python .temp/run_clm.py
