scene_names=(
    "bicycle"
    # "bonsai"
    # "counter"
    # "garden"
    # "kitchen"
    # "room"
    "stump"
)

n_views_list=(
    12
    20
    # 36
)

for n_views in "${n_views_list[@]}"
do
    echo "Running LLFF with $n_views views"
    for scene_name in "${scene_names[@]}"
    do
        if [[ "$scene_name" == "bicycle" || "$scene_name" == "garden" || "$scene_name" == "stump" ]]; then
            downsample_factor=4
        else
            downsample_factor=2
        fi
        echo "Running LLFF on scene: $scene_name"
        bash scripts/run_360.sh 0 "data/mipnerf360/$scene_name" "output/mipnerf360_$n_views/$scene_name" $n_views $downsample_factor
    done
done