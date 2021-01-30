# This is a script for doing things  with the data
using Images
using AWSS3
using AWS
using Metalhead
using Test
using ImageMagick
using OffsetArrays
using Flux
using ProgressMeter
using genex
using JLD
using CUDA

mybasepath = "/home/swojcik/github/masc_faces/julia"
## Get face locations_path
face_locations = load("$mybasepath/face_locations.jld")

## You will need to run aws configure first to make this work
aws = global_aws_config(; region="us-east-1")
@test typeof(aws) == AWSConfig

# Check that you can get list of object from AWS
a = s3_list_objects(aws, "brazil.images")
@test typeof(popfirst!(a)["Key"]) == String

# See if you can download the image object
img_path = popfirst!(a)["Key"]
img_raw = download_raw_img(img_path, aws) 

## how to optimize the processing of raw images 
CUDA.allowscalar(false)

ResMod = ResNet()
run_thru_resnet = function(img_array::Array{Float32,4}, Resnet)
    (Resnet.layers[1:20](img_array) |> Flux.gpu)[:, 1]
end

#img_3d = rand(Float32, (224, 224, 3, 1));
#run_thru_resnet(img_3d, ResMod)
#@benchmark run_thru_resnet(rand(Float32, (224, 224, 3, 1)), ResMod)

#### Version 1: @inbounds: 133.489 ms - row first
testfunkrow = function(nnmodel)
    for i in 1:100
        img_3d_sample = rand(Float32, (224, 224, 3, 1))
        test_out = CUDA.zeros(Float32, 100,2048)
        preds = run_thru_resnet(img_3d_sample, nnmodel)
        @inbounds test_out[i, :] .= preds[:, 1]
    end
    return test_out
end

#@benchmark testfunkrow(ResMod)

## THIS METHOD IS HALF THE TIME!! Julia prefers you access whole columns, not rows 
testfunkcol = function(nnmodel)
    for i in 1:100
        img_3d_sample = rand(Float32, (224, 224, 3, 1))
        test_out = CUDA.zeros(Float32, 2048, 100)
        preds = run_thru_resnet(img_3d_sample, nnmodel)
        @inbounds test_out[:, i] .= preds
    end
    return test_out
end

@benchmark testfunkcol(ResMod)

## TESTING THE GENEX FUNCTIONS on the live data 
testdat = Dict(collect(face_locs)[1:20])

body, face, failed = generate_expression_features(testdat, ResNet(), aws)
