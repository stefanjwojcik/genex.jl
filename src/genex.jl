"""
NOTES:
Loading an image on Linux requires a few dependencies, including ImageMagic. In order to make this work 
on AWS, I had to run: 
`wget https://github.com/ImageMagick/ImageMagick/archive/7.0.10-11.tar.gz` (to download a slightly old version of ImageMagick - newer version failed with error)
then, 
cd ImageMagick..., then 
`./configure`, then 
`make` , then 
`sudo make install`, then 
`sudo ldconfig /usr/local/lib`
Finally, it then appears that you have to use a workaround on Linux machines
using ImageMagick
img = readblob(take!(buffer)) # in order to read from a real buffer 
"""
module genex

import Metalhead
import Flux
import Images
import AWS
import AWSS3
import OffsetArrays
import ImageMagick
# MUST: add CUDA@1.3.3
import CUDA
import ProgressMeter
import PyCall

using ImageTransformations
using PyCall

#aws = AWSCore.aws_config()

# NEW FUNCTIONS 

"""
download_raw_img(img_key::String)
takes an image link from AWS bucket, drops it into a buffer oject, converts to float32, then pads it for a perfect fit into the right
dimensions. Finally, it converts it into a 3d object for rbg types. 
"""
function download_raw_img(img_key::String, aws)
    img_processed = img_key |>
        x -> AWSS3.s3_get(aws, "brazil.images", x) |>
        x -> IOBuffer(x) |>
        x -> ImageMagick.readblob(take!(x)) 
    return img_processed
end

py"""
from tensorflow.keras.applications.vgg19 import preprocess_input

def tf_preprocess(x):
    return preprocess_input(x)
"""

py"""
import numpy as np

def my_expand(x):
    return np.expand_dims(x, axis=0)
"""

function my_process(raw_img)
    out = raw_img |>
    x -> imresize(x, 224, 224) |>
    x -> Float32.(x) |>
    x -> py"my_expand"(x) |>
    x -> py"tf_preprocess"(x) 
    return out[1, :, :, :, :]
end

# Newer mega-function - uses @inbounds and cuarrays to process the data 
function generate_expression_features(face_locations, resnet_model, aws)
    # Progress Meter just to see where we are in the process 
    p = ProgressMeter.Progress(length(face_locations)) # total of iterations, by 1
    ProgressMeter.next!(p)
    failed_cases = String[]

    # Creating empty arrays to store the results 
    face_features_out = CUDA.zeros(Float32, 2048, length(face_locations)) # create base 
    body_features_out = CUDA.zeros(Float32, 2048, length(face_locations))

    for (i, img_key) in enumerate(keys(face_locations))
        #printstyled(img_key*" \n", color=:green)
        try
            raw_img = download_raw_img(img_key, aws)
            if !isempty(face_locations[img_key])
                top, right, bottom, left = face_locations[img_key][1]
                face_seg_img = raw_img[top:bottom, left:right] #bug if locations include 0
                body_padded, face_padded = my_process(raw_img), my_process(face_seg_img)
                @inbounds face_features_out[:, i] .= (resnet_model.layers[1:20](face_padded) |> Flux.gpu)[:, 1]
                @inbounds body_features_out[:, i] .= (resnet_model.layers[1:20](body_padded) |> Flux.gpu)[:, 1]
            else 
                body_padded = my_process(raw_img)
                @inbounds body_features_out[:, i] .= (resnet_model.layers[1:20](body_padded) |> Flux.gpu)[:, 1]
                @inbounds face_features_out[:, i] .= body_features_out[:, i] # substitute face w/ body if empty 
            end
        catch
            @warn "$img_key failed"
            push!(failed_cases, img_key)
            @inbounds face_features_out[:, i] .= CUDA.zeros(2048, )
            @inbounds body_features_out[:, i] .= CUDA.zeros(2048, )
        end

        ProgressMeter.next!(p)
    end
    # Write out the files
    out = (body_features_out, face_features_out, failed_cases)
    printstyled("DONE \n", color=:blue)
    return out
end



export download_raw_img, 
       my_process, 
       generate_expression_features

end # module
