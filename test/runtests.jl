# This is a script for doing things  with the data
using Images
using AWSS3
using AWS
using Metalhead
using Test
using ImageTransformations
using genex
using JLD
using CUDA
#using BenchmarkTools

aws = global_aws_config(; region="us-east-1")

## base image 
raw_img = download_raw_img("age1048_Male_irmao-erisvaldo-d.jpg", aws)

## expand_dims
img_proc = raw_img |>
    x -> imresize(x, 224, 224) |>
    x -> Float32.(x) |>
    x -> my_py_expand(x) |>
    x -> my_tf_preprocess(x)

@test typeof(img_proc) == Array{Float32,3}

# Resize the Array to fit (224, 224, 3, 1)
out = zeros(224, 224, 3, 1)

# fill all RGB dimensions of 'out' object with gray processed image 
[ out[:, :, x, 1] .= img_proc[1, :, :] for x in 1:3 ]

## how to optimize the processing of raw images 
CUDA.allowscalar(false)

ResMod = ResNet()
run_thru_resnet = function(img_array::Array{Float32,4}, Resnet)
    (Resnet.layers[1:20](img_array) |> Flux.gpu)[:, 1]
end


# preprocess 

## Get face locations_path
mybasepath = "/home/swojcik/github/masc_faces/julia"
face_locations = load("$mybasepath/face_locations.jld")

## You will need to run aws configure first to make this work
@test typeof(aws) == AWSConfig

# Check that you can get list of object from AWS
a = s3_list_objects(aws, "brazil.images")
@test typeof(popfirst!(a)["Key"]) == String

# See if you can download the image object
img_path = popfirst!(a)["Key"]
img_raw = download_raw_img(img_path, aws) 


#img_3d = rand(Float32, (224, 224, 3, 1));
#run_thru_resnet(img_3d, ResMod)
#@benchmark run_thru_resnet(rand(Float32, (224, 224, 3, 1)), ResMod)

#@benchmark testfunkrow(ResMod)

## THIS METHOD IS HALF THE TIME!! Julia prefers you access whole columns, not rows 
testfunkcol = function(nnmodel)
    for i in 1:100
        img_3d_sample = rand(Float32, (224, 224, 3, 1))
        local test_out = CUDA.zeros(Float32, 2048, 100)
        preds = run_thru_resnet(img_3d_sample, nnmodel)
        @inbounds test_out[:, i] .= preds
    end
    return test_out
end

@benchmark testfunkcol(ResMod)

## TESTING THE GENEX FUNCTIONS on the live data 
testdat = Dict(collect(face_locations)[1:200])

body, face, failed = generate_expression_features(testdat, ResNet(), aws)

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

