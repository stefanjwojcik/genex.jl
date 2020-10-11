
using Images
using AWSCore
using AWSS3
using Metalhead

aws = AWSCore.aws_config()

# get the keys for all objects 
get_objects() = [x["Key"] for x in s3_list_objects(aws, "brazil.images")]

#img_paths = get_objects()

# load a random Image
get_random_image() = load(IOBuffer(s3_get(aws, "brazil.images", img_paths[rand(1:length(img_paths))])))

# load a specific numbered image
get_num_image(filenum) = load(IOBuffer(s3_get(aws, "brazil.images", img_paths[filenum])))

# TODO
# - create a function that can auto-pad an image with white based on the required dimensions (in the case of resnet - 7x7)

# gray img to 3 channel
function myf(img)
    fimg = Float64.(img)
    padfig = padarray(fimg, Fill(1,(29,40),(30,50))) # pad excess with white 
    padfig = OffsetArray(padfig, OffsetArrays.Origin(1))
    padfig = padfig[2:end, :] # 225x224 -> 224x224
    d1, d2 = size(padfig)
    out = zeros(Float64, (d1, d2, 3, 1))
    out[:, :, 1:3, 1] .= padfig
end
# convert image to Float
res = ResNet()
hi = get_num_image(4);
fig = Float64.(hi);
this = zeros(165, 120, 3, 1)
this[:, :, 1, :] = fig

#fig = Gray.(hi)
fixex = Array{Float32,4}

figflat = reshape(fig, 165, 120, 1, 1)

x = rand(Float32, 224, 224, 3, 1)
im = Float64.(hi)
im3d = reshape([im; im; im], 165, 120, 3, 1)
reshape()

x[:, :, 2, 1] = x


# padding an array 
padarray(fig, 5)

x = rand(Float32, 224, 224, 3, 1)
pred = res.layers[1:20](x)


pred = res.layers[1:20](out)
