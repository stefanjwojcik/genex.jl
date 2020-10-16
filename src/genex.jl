module genex

import Metalhead
import Flux
import Images
import AWSCore
import AWSS3
import OffsetArrays
import ImageMagick
# MUST: add CUDA@1.3.3
import CUDA

aws = AWSCore.aws_config()

# create mega_function
"""
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
function process_data(aws, max_iters)
    res = ResNet() # load the resnet model 
    out = Array{Float64,2}[] # array to fill 
    img_3d = zeros(Float64, (224, 224, 3, 1)) # skeleton array
    prediction_array = zeros(Float64, (2048, 1))
    count = 0 # counter for stopping 
    for img in s3_list_objects(aws, "brazil.images")
        img_processed = img["Key"] |>
            x -> s3_get(aws, "brazil.images", x) |>
            x -> IOBuffer(x) |>
            x -> readblob(take!(x)) |>
            x -> Float64.(x) |>
            x -> padarray(x, Fill(1,(29,52),(30,52))) |>
            x -> OffsetArray(x, 1:224, 1:224) |>
            x -> img_3d[:, :, 1:3, :] .= x
        count += 1
        pred .= res.layers[1:20](img_processed)
        push!(out, copy(pred))
        if count == max_iters
            break
        end
    end
    return out
end


end # module
