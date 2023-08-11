
function allequal(collection)
    if isempty(collection)
        return true
    end
    c = first(collection)
    return all(collection .== c)
end
