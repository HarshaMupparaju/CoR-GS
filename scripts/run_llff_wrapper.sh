scene_names=(
    # "fern"
    # "flower"
    # "fortress"
    # "horns"
    # "leaves"
    # "orchids"
    "room"
    # "trex"
)

n_views_list=(
    # 2
    3
    4
)

downsample_factor=4

for n_views in "${n_views_list[@]}"
do
    echo "Running LLFF with $n_views views"
    for scene_name in "${scene_names[@]}"
    do
        echo "Running LLFF on scene: $scene_name"
        bash scripts/run_llff.sh 0 "data/nerf_llff_data/$scene_name" "output/llff_$n_views/$scene_name" $n_views $downsample_factor
    done
done