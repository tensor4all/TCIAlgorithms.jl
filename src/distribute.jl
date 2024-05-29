using Distributed

struct TaskQueue{T,R}
    queue::Set{T}
    workers::Vector{Union{Future,Nothing}}
    results::Set{R}
end

function TaskQueue{T,R}() where {T,R}
    TaskQueue(Set{T}(), Union{Nothing,Future}[nothing for n in 1:nworkers()], Set{R}())
end

Base.isempty(obj::TaskQueue{T,R}) where {T,R} = isempty(obj.queue)
take!(obj::TaskQueue{T,R}) where {T,R} = pop!(obj.queue)
put!(obj::TaskQueue{T,R}, x) where {T,R} = push!(obj.queue, x)

availableworker(obj::TaskQueue{T,R}) where {T,R} = findfirst(isnothing, obj.workers)
readytofetch(obj::TaskQueue{T,R}) where {T,R} = findfirst(x->x !== nothing && isready(x), obj.workers)

function TaskQueue{T,R}(initialtasks::Set{T}) where {T,R}
    TaskQueue(initialtasks, Union{Nothing,Future}[nothing for n in 1:nworkers()], Set{R}())
end

# loop
function loop(queue::TaskQueue{T,R}, f::Function; sleepsec=1e-3) where {T,R}
    while true
        ireadytofetch = readytofetch(queue)
        if ireadytofetch !== nothing
            fetched = fetch(queue.workers[ireadytofetch])
            if fetched isa RemoteException
                err_msg = sprint(showerror, fetched.captured)
                error("$err_msg")
            end
            result, newtasks = fetched
            queue.workers[ireadytofetch] = nothing
            if result !== nothing
                push!(queue.results, result)
            end
            if newtasks !== nothing
                for task in newtasks
                    put!(queue, task)
                end
            end
        end
    
        if !isempty(queue)
            iworker = availableworker(queue)
            if iworker !== nothing
                t = take!(queue)
                queue.workers[iworker] = @spawnat iworker f(t)
            end
        end

        if isempty(queue) && all(isnothing, queue.workers)
            break
        end

        sleep(sleepsec)
    end
    return queue.results
end