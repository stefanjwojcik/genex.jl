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

"""
download_raw_img(img_key::String)
takes an image link from AWS bucket, drops it into a buffer oject, converts to float32, then pads it for a perfect fit into the right
dimensions. Finally, it converts it into a 3d object for rbg types. 
"""
function download_raw_img(img_key::String, aws)
    img_processed = @pipe img_key |>
        x -> AWSS3.s3_get(aws, "brazil.images", x) |>
        x -> IOBuffer(x) |>
        x -> ImageMagick.readblob(take!(x)) 
    return img_processed
end


py"""
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
"""

function py_process_input(image_array)
    image_array_cx = deepcopy(image_array)
    image_array_cx .= py"preprocess_input"(image_array_cx)
    return image_array_cx
end

imagenet_means = mean(py_process_input(zeros(224, 224, 3)), dims=(1, 2));

function image_net_gen_scale(imagenet_means)
    #imagenet_means = [103.93899999996464, 116.77900000007705, 123.67999999995286]
    function pyscale(x::AbstractArray)
        dx = copy(x)
        # swap R and G channels like python does - only during channels_last 
        dx[:, :, :, 1], dx[:, :, :, 3] = dx[:, :, :, 3], dx[:, :, :, 1]
        dx[:, :, :, 1] .+= imagenet_means[1]
        dx[:, :, :, 2] .+= imagenet_means[2]
        dx[:, :, :, 3] .+= imagenet_means[3]
        #return cor(collect(Iterators.flatten(dx)),collect(Iterators.flatten(py_scaled_image)))
        return(dx)
    end
end

image_net_scale = image_net_gen_scale(imagenet_means)

function get_features_from_image(img, nmodel)
    j_image_load = @pipe imresize(load(img), 224, 224) |> 
                    x -> Float32.(channelview(x) ) |> 
                    x -> permutedims(x, [2,3,1]) |>
                    x -> reshape(x, 224, 224, 3, 1) |> 
                    x -> image_net_scale(x)
    proc_pic = @pipe j_image_load |> 
        x -> nmodel(x)
    return proc_pic
end

function generate_expression_features(face_locations, nmodel, aws)
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
                body_padded = get_features_from_image(raw_img, nmodel)
                face_padded = get_features_from_image(face_seg_img, nmodel)
                @inbounds face_features_out[:, i] .= face_padded[:, 1]
                @inbounds body_features_out[:, i] .= body_padded[:, 1]
            else 
                body_padded = get_features_from_image(raw_img, nmodel)
                @inbounds body_features_out[:, i] .= body_padded[:, 1]
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

#######################****************************************************

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


## PYTHON FUNCTIONS TO CALL 

function __init__()
    py"""
    from tensorflow.keras.applications.vgg19 import preprocess_input
    import numpy as np

    def tf_preprocess(x):
        return preprocess_input(x)

    def my_expand(x):
        return np.expand_dims(x, axis=0)

    """
end

my_py_expand(x) = py"my_expand"(x)
my_tf_preprocess(x) = py"tf_preprocess"(x)

function my_process(raw_img)
    exp_shape = zeros(Float32, 224, 224, 3, 1)
    out = raw_img |>
    x -> imresize(x, 224, 224) |>
    x -> Float32.(x) |>
    x -> my_py_expand(x) |>
    x -> my_tf_preprocess(x)
# fill all RGB dimensions of 'out' object with gray processed image 
    [ exp_shape[:, :, x, 1] .= out[1, :, :] for x in 1:3 ]
    return exp_shape
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
       my_py_expand, 
       my_tf_preprocess,
       my_process, 
       generate_expression_features

end # module
