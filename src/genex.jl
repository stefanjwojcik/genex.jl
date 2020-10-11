module genex

import Metalhead
import Flux
import Images
import AWSCore
import AWSS3
import OffsetArrays

aws = AWSCore.aws_config()

# get the keys for all objects #img_paths = get_objects()
get_objects() = [x["Key"] for x in s3_list_objects(aws, "brazil.images")]

# load a random Image
get_random_image() = load(IOBuffer(s3_get(aws, "brazil.images", img_paths[rand(1:length(img_paths))])))

# load a specific numbered image
get_num_image(filenum) = load(IOBuffer(s3_get(aws, "brazil.images", img_paths[filenum])))

# iterate over every image 
# must be fault tolerant
# must not leave gaps 
function proces_all()
    processed_img_paths = []
    array_alloc = Array{Float64, 4}[]
    for x in s3_list_objects(aws, "brazil.images")
        push!(processed_img_paths, x["Key"])
        img_raw 
    end
end    

greet() = print("Hello World!")

fthis() = print("ffed")

end # module
