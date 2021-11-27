# This is a script for doing things  with the data
using Images
using AWSS3
using AWS
using Metalhead
using Test
using ImageTransformations
#using genex
using JLD
using CUDA
using PyCall
#using BenchmarkTools

aws = global_aws_config(; region="us-east-1")

## base image 
raw_img = download_raw_img("age1048_Male_irmao-erisvaldo-d.jpg", aws)

#### Test scaling ********************

# Comparison function in python
py"""
from tensorflow.keras.applications.vgg19 import preprocess_input
"""
function py_process_input(image_array)
    image_array_cx = deepcopy(image_array)
    image_array_cx .= py"preprocess_input"(image_array_cx)
    return image_array_cx
end

myrand = rand(224, 224, 1, 3);
@test py_process_input(myrand) ≈ jimage_net_scale(myrand);
@test jimage_net_scale(zeros(1, 224, 224, 3)) ≈ py_process_input(zeros(1, 224, 224, 3))

## BOTTLENECK FUNCTION 
myt = tempname();
download("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d7/Kevin_Bacon_SDCC_2014.jpg/220px-Kevin_Bacon_SDCC_2014.jpg",  myt);
@test sum(capture_bottleneck(myt)) > 0
@test sum(capture_bottleneck(myt) |> gpu ) > 0

# Checking face segmentation - currently 2485 missing face segs due to squeese
tfile = tempname()
AWSS3.s3_get_file(aws, "brazil.face.locations", "face_locations.jld", tfile);
face_locations = load(tfile);

sum(isempty.(values(face_locations)))
emtpyfaces = findall(isempty.(values(face_locations)))
raw_img = download_raw_img(collect(keys(face_locations))[emtpyfaces][1], aws)

## Testing the mega function 
face_test_imgs = collect(keys(face_locations))[1:10]
face_test_dict = Dict(key=>value for (key,value) in face_locations if key ∈ face_test_imgs)
body, face, failed = generate_expression_features(face_test_dict, aws)

# trying to find the problematic keys in the data 
# "age53_Male_ceara-do-gas-phs-d.jpg", age32_Male_gleyson-barbosa-d.jpg", "age52_Male_joao-muniz-pmdb-d.jpg"
prob_keys = collect(keys(face_locs))[occursin.("papagaio", keys(face_locs))]

# Figuring out which images lack face locations 
#empty_faces = String[]

#for (i, img_key) in enumerate(keys(face_locations))
#    if isempty(face_locations[img_key])
#        push!(empty_faces, img_key)
#    else 
#        continue
#    end
#end

