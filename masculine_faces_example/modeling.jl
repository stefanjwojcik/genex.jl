"""
So this script was originally designed to replace the current python implementation to process the candidate images. 
However, Python currently has much stronger image preprocessing capabilities. 

This script currently draws on the genex package I wrote to 
"""

using AWSS3
using AWS
using Metalhead
using Test
using genex
using JLD
using CUDA
CUDA.allowscalar(false)

# AWS and getting the list of images in the bucket 
aws = global_aws_config(; region="us-east-1")
mybucket = s3_list_objects(aws, "brazil.images")

# Get face locations - based on prior run in 'get_face_locations.jl'
face_locs = load("/home/swojcik/github/masc_faces/julia/face_locations.jld")

# Create test data 
testdat = Dict(collect(face_locs)[1:20])

######## ** TEST OF CORE FUNCTIONALITY HERE  + WARM UP
@time body, face, failed = generate_expression_features(testdat, ResNet(), aws);

# FINAL RUN ****** VERY LONG PROCESSING HERE 
body, face, failed = generate_expression_features(face_locs, ResNet(), aws)

# Save the data
JLD.save("face_feb1.jld", Dict("face"=>convert(Array, face)))
JLD.save("body_feb1.jld", Dict("body"=>convert(Array, body)))
JLD.save("failed_cases.jld", Dict("failed"=>failed))

# ****************************************************************************
## TESTS OF THE FAILED CASES - SOLVING Bug involving face segmentation where one value = 0
using JLD
using Flux
using Metalhead
using genex
using ProgressMeter
using AWSS3
using AWS
using ScikitLearn
using StatsBase
using Random

face_locs = load("/home/swojcik/github/masc_faces/julia/face_locations.jld")
failed_cases = JLD.load("/home/swojcik/github/masc_faces/julia/failed_cases.jld") 
face = JLD.load("/home/swojcik/github/masc_faces/julia/face_feb1.jld")["face"]
body = JLD.load("/home/swojcik/github/masc_faces/julia/body_feb1.jld")["body"]
aws = global_aws_config(; region="us-east-1")

#download_raw_img(failed_cases["failed"][1], aws)
#face_locs[failed_cases["failed"][1]]

# Create function to replace zeros in the face locations to 1
otherwise = function(existing_face_location::Tuple)
    return tuple([maximum([1, x]) for x in existing_face_location]...)
end

# fixing the failed cases
"""
This function updates the existing estimates for the cases that failed last time, so we now have no failed cases 
"""
fix_failed = function(face, body, face_locations, failed, aws)
    #look up the index of the failed cases 
    dict_index = Dict()
    all_face_keys = collect(keys(face_locations))
    p = ProgressMeter.Progress(length(failed["failed"])) # total of iterations, by 1
    ProgressMeter.next!(p)
    # fill this dict() object with the indexes where the failed cases are 
    [ dict_index[all_face_keys[x]] = x for x in 1:length(face_locations) if all_face_keys[x] ∈ failed["failed"]]
    for (key, value) in dict_index
        raw_img = download_raw_img(key, aws)
        # do face locations 
        fixed_face_locations = otherwise(face_locations[key][1])
        top, right, bottom, left = fixed_face_locations
        face_seg_img = raw_img[top:bottom, left:right] #bug if locations include 0
        body_padded, face_padded = my_process(raw_img), my_process(face_seg_img)
        # fill the face object 
        @inbounds face[:, value] .= (ResNet().layers[1:20](face_padded) |> Flux.gpu)[:, 1]
        @inbounds body[:, value] .= (ResNet().layers[1:20](body_padded) |> Flux.gpu)[:, 1]
        # Update the progress meter
        ProgressMeter.next!(p)
    end
    return(face, body)
end

## Run the function to fix the failed cases
## Sci-kit learn modeling 

using DataFrames, Pipe, MLJ, LossFunctions, StatsBase

fix_failed(face, body, face_locs, failed_cases, aws)
JLD.save("/home/swojcik/github/masc_faces/julia/face_complete.jld", Dict("face"=>convert(Array, face)))
JLD.save("/home/swojcik/github/masc_faces/julia/body_complete.jld", Dict("body"=>convert(Array, body)))
## Load the saved data 
face = JLD.load("/home/swojcik/github/masc_faces/julia/face_complete.jld")["face"]
body = JLD.load("/home/swojcik/github/masc_faces/julia/body_complete.jld")["body"]
face_locs = JLD.load("/home/swojcik/github/masc_faces/julia/face_locations.jld")

# FullX, FullY 
FullY = [occursin("_Female_", x) ? 1 : 0 for x in keys(face_locs)] #Getting gender 
y = @pipe categorical(FullY) |> DataFrames.recode(_, 0=>"Male",1=>"Female");
FullXface = convert(Array, face')
FullXbody = convert(Array, body')
# Get random samples 
fullX_co_face = coerce(DataFrame(FullXface), Count=>Continuous)
fullX_co_body = coerce(DataFrame(FullXbody), Count=>Continuous)

female_indexes = findall(x->x=="Female", y) # returns indexes 
male_indexes = findall(x->x=="Male", y)
trainrows = shuffle([StatsBase.sample(female_indexes, 2500, replace=false); 
                    StatsBase.sample(male_indexes, 2500, replace=false)])

# svm = LinearSVC(C=.01, loss='squared_hinge', penalty='l2', multi_class='ovr', random_state = 35552)
possible_models = models(matching(fullX_co_face, y))

possible_models = []
models() do model
    matching(model,fullX_co_face, y) && model.prediction_type == :probabilistic
end

svm = @load SVMLinearClassifier()
svm().C = .01
svm().random_state = 35552

# Load the model and set some parameters 
svtm = machine(svm(), fullX_co_face, y)
fit!(svtm, rows=trainrows)
yhat = MLJ.predict(svtm, rows=trainrows)
#accuracy(ScikitLearn.predict(svm, rows=trainrows), fullY[trainrows])

sum(ScikitLearn.predict(svm, FullXface[trainrows, :]) .== FullY[trainrows]) / length(FullY[trainrows])

predict(X_test)

svtm = machine(svm, fullXface, fullY)
ScikitLearn.fit!(svtm, rows=trainrows)
accuracy(predict(svtm, rows=testrows), fullY[testrows])

##########################3

using RDatasets: dataset

iris = dataset("datasets", "iris")

# ScikitLearn.jl expects arrays, but DataFrames can also be used - see
# the corresponding section of the manual
X = convert(Array, iris[:, [:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]])
y = convert(Array, iris[:, :Species])

@sk_import linear_model: LogisticRegression
model = LogisticRegression(fit_intercept=true)
ScikitLearn.fit!(model, X, y)

accuracy = sum(ScikitLearn.predict(model, X) .== y) / length(y)

## USING PYCALL

py"""
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV

pd.read_csv("~/Downloads/face_as.csv")
"""

py"""
svm = LinearSVC(C=.01, loss='squared_hinge', penalty='l2', multi_class='ovr', random_state = 35552)
		clf = CalibratedClassifierCV(svm) 
		clf.fit(features, np.array(gender_labels))
"""
py"sinpi"(1)