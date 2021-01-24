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

"""
Calculate proper padding size = in this case it's 224 
"""
function calculate_img_pad(raw_img_size; size_squared=224)
    diff = size_squared-raw_img_size
    split = Int(round((diff)/2)) #calculate padding size for x and y
    out1, out2 = split, diff-(split*2) + split # dealing with odd cases 
end

"""
Pad the image 
"""
function pad_it(raw_img)
    (x1, x2), (y1, y2) = calculate_img_pad.(size(raw_img))
    img_3d = zeros(Float32, (224, 224, 3, 1))
    out = raw_img |>
    x -> Float32.(x) |>
    x -> Images.padarray(x, Images.Fill(1,(x1,y1),(x2,y2))) |>
    x -> OffsetArrays.OffsetArray(x, 1:224, 1:224) |>
    x -> img_3d[:, :, 1:3, :] .= x
    return out
end


# This is a slightly older way to process the data
"""
process_aws_link(img_key::String)
takes an image link from AWS bucket, drops it into a buffer oject, converts to float32, then pads it for a perfect fit into the right
dimensions. Finally, it converts it into a 3d object for rbg types. 
"""
function process_aws_link(img_key::String, aws)
    img_3d = zeros(Float32, (224, 224, 3, 1)) # skeleton array
    img_processed = img_key |>
        x -> AWSS3.s3_get(aws, "brazil.images", x) |>
        x -> IOBuffer(x) |>
        x -> ImageMagick.readblob(take!(x)) |>
        x -> Float32.(x) |>
        x -> Images.padarray(x, Images.Fill(1,(29,52),(30,52))) |>
        x -> OffsetArrays.OffsetArray(x, 1:224, 1:224) |>
        x -> img_3d[:, :, 1:3, :] .= x
    return img_processed
end


function compress_images(bucket, model, aws; test=true)
    out = Array{Float32,2}[] # array to fill 
    prediction_array = zeros(Float32, (2048, 1)) #skeleton array to keep replacing
    nit = 0 # counter for stopping 
    max_iters = test ? 25 : 193_108 #ternary operator 
    p = ProgressMeter.Progress(max_iters) # total of iterations, by 1
    ProgressMeter.next!(p)
    img_links = [x["Key"] for x in bucket]
    for ik in img_links
        nit += 1
        ProgressMeter.next!(p)
        if (test & (nit >= max_iters))
            break
        end 
        preproc_img = process_aws_link(ik, aws)
        prediction_array .= model.layers[1:20](preproc_img) |> Flux.gpu
        push!(out, copy(prediction_array))
    end
    return out
end

export process_aws_link,
       compress_images, 
       download_raw_img

end # module
