println("precompiling/loading: Plots.")
using Plots;
println("\tprecompiled/loaded: Plots.")
println("precompiling/loading: Flux.")
using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
println("\tprecompiled/loaded: Flux.")
println("precompiling/loading: Base.Iterators.")
using Base.Iterators: repeated
println("\tprecompiled/loaded: Base.Iterators.")
println("precompiling/loading: Images.")
using Images
println("\tprecompiled/loaded: Images.")
println("precompiling/loading: FileIO & ImageIO.")
using FileIO
using ImageIO
println("\tprecompiled/loaded: FileIO & ImageIO.")
println("precompiling/loading: MLDatasets.")
using MLDatasets.MNIST: convert2image
using MLDatasets.MNIST.Reader: readimages, readlabels
println("\tprecompiled/loaded: MLDatasets.")


function getCharImages()
    imgs = readimages("./emnist-letters-train-images-idx3-ubyte.gz")[:,:,1:120000];
    toRet = Vector();
    z = size(imgs)[3];
    for i = 1:z
        img = Gray.(convert2image(imgs[:,:,i]));
        img = fixDigit(img);
        push!(toRet, img);
    end

    return toRet;
end

function fixDigit(image::Matrix{<:Gray}) # for some reason, all of the images in this database are rotated around the diagonal
    img = Matrix{Gray}(undef, 28, 28);
    row = 1;
    for j = 1:28
        s = [];
        i = row;
        k = j;
        while i <= 28 && k >= 1
            append!(s,[image[i,k]])
            i += 1
            k -= 1
        end

        i = row 
        k = j
        while i <= 28 && k >= 1
            img[i,k] = pop!(s);
            i += 1
            k -= 1
        end
    end
    column = 28;
    for j = 1:28
        s = [];
        i = j;
        k = column

        while i <= 28 && k >= 1
            append!(s,[image[i,k]])
            i += 1
            k -= 1
        end

        i = j;
        k = column

        while i <= 28 && k >= 1
            img[i,k] = pop!(s);
            i += 1
            k -= 1
        end
    end

    return img;
end

key = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'];
println("Loading Digit Images.")
digit_imgs = MNIST.images();
digit_labels = MNIST.labels();
println("\tDigit Images Loaded.")

println("Loading Character Images. This will take time as there are over 120000 of them, and they all need to be mirrored along the diagonal.")
char_imgs = getCharImages();
char_labels = Int.(readlabels("./emnist-letters-train-labels-idx1-ubyte.gz")[1:120000]).+9;
println("\tCharacter Images Loaded.")

#  My stuff declared after the function

# Defining the model (a neural network)
println("Creating Neural Network.")
m = Chain(
    Dense(28*28, 128, relu),
    Dense(128, 64),
    Dense(64, 36),
    softmax)
println("\tNeural Netowrk linked.")

function train()
    # Load training data. 28x28 grayscale images of characters
    println("Compiling images:")
    imgs = Vector();
    append!(imgs, digit_imgs[100:60000]);
    append!(imgs, char_imgs[100:120000]);
    append!(imgs, char_imgs[100:120000]); # I want these to have more examples but can't combine the repeat functions
    X = hcat(imagestrip.(imgs)...)
    println("\tImages Compiled.")

    # Target output. What character each image represents.
    println("Compiling Labels:")
    labels = Vector();
    append!(labels, digit_labels[100:60000]);
    append!(labels, char_labels[100:120000]);
    append!(labels, char_labels[100:120000]); # I want these to have more examples but can't combine the repeat functions
    Y = onehotbatch(labels, 0:35);
    println("\tLabels Compiled.")
    
    loss(x, y) = crossentropy(m(x), y)
    println("Creating dataset. 200 of each image/key pair:")
    dataset = repeated((X, Y), 200)
    println("\tDataset Created.")
    opt = ADAM()

    evalcb = () -> @show(loss(X, Y))

    # Perform training
    println("Training:");
    Flux.train!(loss, params(m), dataset, opt, cb = throttle(evalcb, 2.5))


end

function invert(x::Gray) # inverts greyscale
    return Gray(1.0-x.val);
end

function imagestrip(image::Matrix{<:Gray})
    return Float32.(reshape(image, :));
end

function getOutput(image::Matrix{<:Gray})
    return toOutput(onecold(m(imagestrip(image))));
end

function getLabels(string::String)
    ret = Vector();
    for (index, value) in enumerate(string)
        label = findfirst(x -> x == value, key);
        if typeof(label) != Nothing
            push!(ret,  label - 1);
        end
    end
    return ret;
end

function toOutput(value::Int)
    return key[value];
end

function toGray(image)
    return Gray.(image);
end

function standardize(columns::Array)
    cols,  = size(columns);
    remaining = (28 - cols) / 2;
    standard = Matrix{Gray}(undef, 28, 28);
    current = 1;

    for i = 1:remaining
        standard[:,current] = zeros(Gray, 28, 1);
        current += 1;
    end
    for i = 1:cols
        standard[:,current] = columns[i];
        current += 1;
    end
    for i = 1:remaining
        standard[:,current] = zeros(Gray, 28, 1);
        current += 1;
    end
    while current <= 28
        standard[:,current] = zeros(Gray, 28, 1);
        current += 1;
    end
    return standard;
end

function hasText(image::Array{Gray{Float64},1})
    for i = 1:28
        if (image[i].val > .75)
            return true;
        end
    end
    return false;
end

function splitLine(image::Array)
    rows, cols = size(image);
    split = Vector();
    inLetter = false;

    letter = Vector();
    for i = 1:cols
        column = image[:,i];
        if (hasText(column))
            if (!inLetter)
                inLetter = true;
                letter = Vector();
            end
            push!(letter, column)
        else
            if(inLetter && length(letter) > 0)
                inLetter = false;
                push!(split, standardize(letter));
            end
        end
        
    end
    return split;
end

function loadImage(filename::String)
    imString = string(pwd(), "/lines/", filename);
    return invert.(toGray(load(imString)));
end

function getImageString(filename::String)
    return filename[1:end-4];
end

function getCharacterData()
    toRet = Vector{Tuple{Array{Gray{Float64},2},String}}();
    imString = pwd() * "/lines/";
    lines = cd(readdir, imString);
    for (index, value) in enumerate(lines)
        push!(toRet, (loadImage(value), getImageString(value)))
    end

    return(toRet);
end

function expandCharacterData()
    characterData = getCharacterData();
    imgs = Vector();
    labels = Vector();
    
    for dp in characterData
        split = splitLine(dp[1]);
        append!(imgs, split);
        append!(labels, getLabels(dp[2]));
    end

    return imgs, labels;
end

println("Loading My lines:")
my_imgs, my_labels = expandCharacterData();
println("\tMy lines loaded.")