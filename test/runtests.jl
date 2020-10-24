# This is a script for doing things  with the data


using Images
using AWSCore
using AWSS3
using Metalhead
using Test
using ImageMagick
using OffsetArrays
using Flux
using ProgressMeter
using genex

## You will need to run aws configure first to make this work
aws = AWSCore.aws_config()
@test typeof(aws) == Dict{Symbol,Any}

# Check that you can get list of object from AWS
a = s3_list_objects(aws, "brazil.images")
@test typeof(popfirst!(a)["Key"]) == String

# See if you can get an iobuffer object 
img_raw = popfirst!(a)["Key"] |>  
    x -> s3_get(aws, "brazil.images", x) 
@test typeof(img_raw) == Array{UInt8,1}

# Test that the buffer worked    
img_buf = IOBuffer(img_raw)
@test typeof(img_buf) == Base.GenericIOBuffer{Array{UInt8,1}}

# Load the buffer object 
img_parsed = readblob(take!(img_buf))
@test typeof(img_parsed) == Array{Gray{Normed{UInt8,8}},2}

# Convert image to Float 
img_float = Float64.(img_parsed)
@test size(img_float) == (165, 120)

# Pad image to proper size 
img_pad = padarray(img_float, Fill(1,(29,52),(30,52)))
@test size(img_pad) == (224, 224)

# Create empty three-channel Array
img_3d = zeros(Float64, (224, 224, 3, 1))

# Fix the origins of the Offset Array 
img_orig = OffsetArray(img_pad, 1:224, 1:224)
@test axes(img_orig) == (Base.Slice(1:224), Base.Slice(1:224))

# Dump the array into the empty one
img_3d[:, :, 1:3, 1] .= img_orig

# Now, finally create the estimate
res = ResNet()
pred = res.layers[1:20](img_3d)
@test length(pred) == 2048

## Test the processing functions
aws = AWSCore.aws_config()
mybucket = s3_list_objects(aws, "brazil.images")

# Test the processing link function 
proc_img = process_aws_link(popfirst!(mybucket)["Key"])
@test size(proc_img) == (224, 224, 3, 1)

# Test the compression function 
out = compress_images(mybucket, ResNet(), test=true)
@test size(out)[1] == 24

####

