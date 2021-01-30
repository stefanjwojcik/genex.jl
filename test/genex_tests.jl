function gef(face_locations, resnet_model, aws)
    # Progress Meter just to see where we are in the process 
    p = ProgressMeter.Progress(length(face_locations)) # total of iterations, by 1
    ProgressMeter.next!(p)
    failed_cases = String[]

    # Creating empty arrays to store the results 
    face_features_out = zeros(Float32, 2048, length(face_locations)) # create base 
    body_features_out = zeros(Float32, 2048, length(face_locations))

    for (i, img_key) in enumerate(keys(face_locations))
        #printstyled(img_key*" \n", color=:green)
        try
            raw_img = download_raw_img(img_key, aws)
            if !isempty(face_locs[img_key])
                top, right, bottom, left = face_locs[img_key][1]
                face_seg_img = raw_img[top:bottom, left:right]
                body_padded, face_padded = pad_it(raw_img), pad_it(face_seg_img)
                @inbounds face_features_out[:, i] .= (resnet_model.layers[1:20](face_padded) |> Flux.gpu)[:, 1]
                @inbounds body_features_out[:, i] .= (resnet_model.layers[1:20](body_padded) |> Flux.gpu)[:, 1]
            else 
                body_padded = pad_it(raw_img)
                @inbounds body_features_out[:, i] .= (resnet_model.layers[1:20](body_padded) |> Flux.gpu)[:, 1]
                @inbounds face_features_out[:, i] .= body_features_out[i, :] # substitute face w/ body if empty 
            end
        catch
            @warn "$img_key failed"
            push!(failed_cases, img_key)
            @inbounds face_features_out[:, i] .= zeros(2048, )
            @inbounds body_features_out[:, i] .= zeros(2048, )
        end

        ProgressMeter.next!(p)
    end
    # Write out the files
    out = (body_features_out, face_features_out, failed_cases)
    printstyled("DONE \n", color=:blue)
    return out
end
