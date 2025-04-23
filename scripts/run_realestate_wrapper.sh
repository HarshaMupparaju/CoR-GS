scene_names=(
    "00000"
    "00001"
    "00003"
    "00004"
    "00006"
)

n_views_list=(
    2
    3
    4
)

downsample_factor=1

for n_views in "${n_views_list[@]}"
do
    echo "Running LLFF with $n_views views"
    for scene_name in "${scene_names[@]}"
    do
        echo "Running LLFF on scene: $scene_name"
        bash scripts/run_realestate.sh 0 "data/RealEstate10K/$scene_name" "output/realestate_$n_views/$scene_name" $n_views $downsample_factor
    done
done