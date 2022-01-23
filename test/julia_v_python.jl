

# LOADING JULIA LIBRARIES 
using PyCall
using Metalhead
using Flux
using ImageTransformations
using Images
using CUDA
using Pipe 
CUDA.allowscalar(false)

## LOADING PYTHON LIBRARIES
py"""
# Brazil processing by hand
import os
import re
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np 
from PIL import ImageEnhance, ImageDraw, Image as Pimage

"""

#SCALING TESTS 
# Comparison function in python - Python Breaks on Pop OS
#py"""
#from tensorflow.keras.applications.vgg19 import preprocess_input
#"""
#function py_process_input(image_array)
#    image_array_cx = deepcopy(image_array)
#    image_array_cx .= py"preprocess_input"(image_array_cx)
#    return image_array_cx
#end

#myrand = rand(224, 224, 1, 3);
#@test py_process_input(myrand) ≈ jimage_net_scale(myrand);
#@test jimage_net_scale(zeros(1, 224, 224, 3)) ≈ py_process_input(zeros(1, 224, 224, 3))


# Example image
example_image_path = "/home/swojcik/github/masc_faces/images/age1048_Male_irmao-erisvaldo-d.jpg"

# Core Python STUFF
py"""
_ = ResNet50(weights='imagenet')
nn_model = Model(inputs=_.input, outputs=_.get_layer('avg_pool').output)
"""

p_image_load = @pipe py"image.load_img"(example_image_path, target_size=(224, 224)) |> 
                x -> py"image.img_to_array"(x) |> 
                x -> py"np.expand_dims"(x, axis=0) |>
                x -> py"preprocess_input"(x) |> 
                x -> py"nn_model.predict"(x)

# to make the loaded python image equivalently viewable in Julia:

function image_net_scale(x::AbstractArray)
    imagenet_means = [103.939, 116.779, 123.68]
    x[:, :, :, 1] .-= imagenet_means[1]
    x[:, :, :, 2] .-= imagenet_means[2]
    x[:, :, :, 3] .-= imagenet_means[3]
    return x
end 

# Create and fill a peculiar array in the same way that python does it 
j_image_load = Array{Float32, 4}(undef, 1, 224, 224, 3) # python channels-last
j_image_load[1, :, :, 1:3] .= imresize(load(example_image_path), 224, 224) # resize the image 
j_image_load .*= 255 # multiply by 255 for proper pixel scale
j_image_load = image_net_scale(j_image_load) # scale with image net 
j_image_load -> py"nn_model.predict"(j_image_load)

    #  now process with resnet 
mod = ResNet50(pretrain=true)
j_features = mod.layers[2][1](j_image_load)[1, 1, :, 1]

####################

#############
j_image_normed = Array{Float32, 4}(undef, 1, 224, 224, 3)
j_image_normed[1, :, :, 1:3] .= py"image.img_to_array"(p_image_load)
                x -> py"image.img_to_array"(x) |> # python convert to numpy(also float32)
                x -> py"np.expand_dims"(x, axis=0) |> 
                x -> image_net_scale(x) 

j_image_normed = @pipe load(example_image_path) |> 
                x -> Float32.(x) |> # python convert to numpy(also float32)
                x -> fill(255, (1, 165, 120, 3))
                x -> reshape(x, (1, 165, 120, 3)) |> 
                x -> image_net_scale(x) 

##############

p_image_normed = @pipe p_image_load |> 
                x -> py"image.img_to_array"(x) |> # python convert to numpy(also float32)
                x -> py"np.expand_dims"(x, axis=0) |> 
                x -> image_net_scale(x) |> 
                x -> permutedims(x, [3, 1, 2]) |> 
                x -> Float64.(x) |> # need Float64 to view 
                x -> colorview(RGB, x/255) |> 
                x -> Float32.(Gray.(x))



# Core JULIA STUFF 

# Read in the raw image - JULIA 
j_image_load = load(example_image_path) 

j_image_expand = @pipe load("/home/swojcik/github/masc_faces/images/age1048_Male_irmao-erisvaldo-d.jpg") |> 
    x -> imresize(x, 224, 224)

j_image_process = @pipe load("/home/swojcik/github/masc_faces/images/age1048_Male_irmao-erisvaldo-d.jpg") |> 
    x -> imresize(x, 224, 224) |> 


j_image_bottleneck = 


# Read in the raw image - Python 

# CORE PYTHON FUNCTIONALITY 

py"""
def get_nn_model():
    _ = ResNet50(weights='imagenet')
    nn_model = Model(inputs=_.input, outputs=_.get_layer('avg_pool').output)
    return(nn_model)


def get_preprocess(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return(preprocess_input(x))
"""


out = py"get_preprocess(\"/home/swojcik/Desktop/masc_faces/images/age19_Female_ana-luisa-neves-d.jpg\")"

######## ** TEST OF CORE FUNCTIONALITY HERE  + WARM UP

## Modify original code with this function

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

py"tf_preprocess"(rand(224,224))

py"""
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
_ = ResNet50(weights='imagenet')
nn_model = Model(inputs=_.input, outputs=_.get_layer('avg_pool').output)


"""

function myprocess(raw_img)
    out = raw_img |>
    x -> imresize(x, 224, 224)
    #x -> Float32.(x) 
    return out
end

function my_process(raw_img)
    out = raw_img |>
    x -> imresize(x, 224, 224) |>
    x -> Float32.(x) |>
    x -> py"np.expand_dims(x, axis=0)" |>
    x -> py"tf_preprocess"(x) 
    return out[1, :, :, :, :]
end

img = @pipe load("/home/swojcik/Desktop/masc_faces/images/age19_Female_ana-luisa-neves-d.jpg") |>
    x -> my_process(x)

jout = py"tf_preprocess"(img)

