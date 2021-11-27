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

###########################
# dependencies
#################################

using Metalhead
using Flux
using Images
using AWS
using AWSS3
using OffsetArrays
using ImageMagick
# MUST: add CUDA@1.3.3
using CUDA
using ProgressMeter
using Pipe
#using PyCall


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

function jimage_net_scale(dx::AbstractArray)
    imagenet_means = [-103.93899999996464, -116.77900000007705, -123.67999999995286]
    #dx = copy(x)
    # swap R and G channels like python does - only during channels_last 
    dx[:, :, :, 1], dx[:, :, :, 3] = dx[:, :, :, 3], dx[:, :, :, 1]
    dx[:, :, :, 1] .+= imagenet_means[1]
    dx[:, :, :, 2] .+= imagenet_means[2]
    dx[:, :, :, 3] .+= imagenet_means[3]
    #return cor(collect(Iterators.flatten(dx)),collect(Iterators.flatten(py_scaled_image)))
    return(dx)
end

# function to flatten an array 
function cflat(x::AbstractArray)
    collect(Iterators.flatten(x))
end

function create_bottleneck_pipeline(neural_model)
    function capture_bottleneck(img)
        out = @pipe load(image_path) |> #
        x -> imresize(x, 224, 224) |> #
        x -> Float32.(channelview(x) * 255) |> #
        x -> permutedims(x, [2, 3, 1]) |> #
        x -> reshape(x, (1, 224, 224, 3) ) |> # Python style for comparison sake 
        x -> jimage_net_scale(x) |>
        x -> reshape(x, (224, 224, 3, 1)) |>
        x -> cflat(neural_model(x))
        return out
    end
end

nn_model = VGG19().layers[1:25];
capture_bottleneck = create_bottleneck_pipeline(nn_model);

# TODO: Fix this function, cut out the face segmentation for separate step 
function generate_expression_features(face_locations, aws)
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
                body_padded = capture_bottleneck(raw_img)
                face_padded = capture_bottleneck(face_seg_img)
                @inbounds face_features_out[:, i] .= face_padded[:, 1]
                @inbounds body_features_out[:, i] .= body_padded[:, 1]
            else 
                body_padded = capture_bottleneck(raw_img)
                @inbounds body_features_out[:, i] .= body_padded[:, 1]
                @inbounds face_features_out[:, i] .= body_features_out[:, i] # substitute face w/ body if empty 
            end
        catch
            #@warn "$img_key failed"
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
       jimage_net_scale,
       cflat,
       capture_bottleneck,
       generate_expression_features

end # module
