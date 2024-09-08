loop=20
seed=919
profile_fore='.outputs/LIVE_result_DBCNN_'
profile_aft='_l'$loop'_sd'${seed}

record_name_fore='.outputs/record/LIVE_result_DBCNN_'
record_name_aft='_sd'${seed}'_l'

action='incr'
profile1=${profile_fore}${action}${profile_aft}
record_name1=${record_name_fore}${action}${record_name_aft}

action2='decr'
profile2=${profile_fore}${action2}${profile_aft}
record_name2=${record_name_fore}${action2}${record_name_aft}

start=5
CUDA_VISIBLE_DEVICES=2 nohup python attack_LIVE.py -s ${start} -m dbcnn -l ${loop} --seed ${seed} -incr \
-o ${profile1} \
> ${record_name1}${loop}_s${start}.txt 2>&1 &

start=6
CUDA_VISIBLE_DEVICES=1 nohup python attack_LIVE.py -s ${start} -m dbcnn -l ${loop} --seed ${seed} -incr \
-o ${profile1} \
> ${record_name1}${loop}_s${start}.txt 2>&1 &

start=7
CUDA_VISIBLE_DEVICES=2 nohup python attack_LIVE.py -s ${start} -m dbcnn -l ${loop} --seed ${seed} -incr \
-o ${profile1} \
> ${record_name1}${loop}_s${start}.txt 2>&1 &

start=8
CUDA_VISIBLE_DEVICES=2 nohup python attack_LIVE.py -s ${start} -m dbcnn -l ${loop} --seed ${seed} -incr \
-o ${profile1} \
> ${record_name1}${loop}_s${start}.txt 2>&1 &

start=9
CUDA_VISIBLE_DEVICES=1 nohup python attack_LIVE.py -s ${start} -m dbcnn -l ${loop} --seed ${seed} -incr \
-o ${profile1} \
> ${record_name1}${loop}_s${start}.txt 2>&1 &


start=5
CUDA_VISIBLE_DEVICES=1 nohup python attack_LIVE.py -s ${start} -m dbcnn -l ${loop} --seed ${seed} \
-o ${profile2} \
> ${record_name2}${loop}_s${start}.txt 2>&1 &

start=6
CUDA_VISIBLE_DEVICES=0 nohup python attack_LIVE.py -s ${start} -m dbcnn -l ${loop} --seed ${seed} \
-o ${profile2} \
> ${record_name2}${loop}_s${start}.txt 2>&1 &

start=7
CUDA_VISIBLE_DEVICES=0 nohup python attack_LIVE.py -s ${start} -m dbcnn -l ${loop} --seed ${seed} \
-o ${profile2} \
> ${record_name2}${loop}_s${start}.txt 2>&1 &

start=8
CUDA_VISIBLE_DEVICES=0 nohup python attack_LIVE.py -s ${start} -m dbcnn -l ${loop} --seed ${seed} \
-o ${profile2} \
> ${record_name2}${loop}_s${start}.txt 2>&1 &

start=9
CUDA_VISIBLE_DEVICES=1 nohup python attack_LIVE.py -s ${start} -m dbcnn -l ${loop} --seed ${seed} \
-o ${profile2} \
> ${record_name2}${loop}_s${start}.txt 2>&1 &

echo ${profile1}' + '${profile2}
echo $(date +%Y-%m-%d\ %H:%M:%S)

start=0
CUDA_VISIBLE_DEVICES=2 nohup python attack_LIVE.py -s ${start} -m dbcnn -l ${loop} --seed ${seed} -incr \
-o ${profile1} \
> ${record_name1}${loop}_s${start}.txt 2>&1 &

start=1
CUDA_VISIBLE_DEVICES=0 nohup python attack_LIVE.py -s ${start} -m dbcnn -l ${loop} --seed ${seed} -incr \
-o ${profile1} \
> ${record_name1}${loop}_s${start}.txt 2>&1 &

start=2
CUDA_VISIBLE_DEVICES=0 nohup python attack_LIVE.py -s ${start} -m dbcnn -l ${loop} --seed ${seed} -incr \
-o ${profile1} \
> ${record_name1}${loop}_s${start}.txt 2>&1 &

start=3
CUDA_VISIBLE_DEVICES=0 nohup python attack_LIVE.py -s ${start} -m dbcnn -l ${loop} --seed ${seed} -incr \
-o ${profile1} \
> ${record_name1}${loop}_s${start}.txt 2>&1 &

start=4
CUDA_VISIBLE_DEVICES=1 nohup python attack_LIVE.py -s ${start} -m dbcnn -l ${loop} --seed ${seed} -incr \
-o ${profile1} \
> ${record_name1}${loop}_s${start}.txt 2>&1 &


start=0
CUDA_VISIBLE_DEVICES=1 nohup python attack_LIVE.py -s ${start} -m dbcnn -l ${loop} --seed ${seed} \
-o ${profile2} \
> ${record_name2}${loop}_s${start}.txt 2>&1 &

start=1
CUDA_VISIBLE_DEVICES=1 nohup python attack_LIVE.py -s ${start} -m dbcnn -l ${loop} --seed ${seed} \
-o ${profile2} \
> ${record_name2}${loop}_s${start}.txt 2>&1 &

start=2
CUDA_VISIBLE_DEVICES=2 nohup python attack_LIVE.py -s ${start} -m dbcnn -l ${loop} --seed ${seed} \
-o ${profile2} \
> ${record_name2}${loop}_s${start}.txt 2>&1 &

start=3
CUDA_VISIBLE_DEVICES=2 nohup python attack_LIVE.py -s ${start} -m dbcnn -l ${loop} --seed ${seed} \
-o ${profile2} \
> ${record_name2}${loop}_s${start}.txt 2>&1 &

start=4
CUDA_VISIBLE_DEVICES=2 nohup python attack_LIVE.py -s ${start} -m dbcnn -l ${loop} --seed ${seed} \
-o ${profile2} \
> ${record_name2}${loop}_s${start}.txt 2>&1 &
