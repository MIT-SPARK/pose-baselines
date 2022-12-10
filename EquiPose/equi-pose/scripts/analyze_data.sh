
# analyzing modelnet40_complete test data
python main.py \
eval=True \
dataset='modelnet40_complete' \
category='airplane' \
exp_num='0.813' \
training='complete_pcloud' \
writer=True \
analyze_data=True


# analyzing modelnet40_partial test data
python main.py \
eval=True \
dataset='modelnet40_partial' \
category='airplane' \
exp_num='0.913r' \
training='partial_pcloud' \
writer=True \
analyze_data=True

