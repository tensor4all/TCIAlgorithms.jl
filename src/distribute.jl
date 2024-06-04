using Distributed

struct TaskQueue{T,R}
    tasks::Vector{T}
    results::Set{R}
end

function TaskQueue{T,R}(initialtasks::Vector{T}) where {T,R}
    return TaskQueue(initialtasks, Set{R}())
end

function loop(obj::TaskQueue{T,R}, f::Function; verbosity=0) where {T,R}
    while length(obj.tasks) > 0
        if verbosity > 0
            println("Processing $(length(obj.tasks)) tasks...")
        end
        #results = @distributed (append!) for t in obj.tasks
        #[f(t)]
        #end
        results = [f(t) for t in obj.tasks]

        empty!(obj.tasks)
        for (r, newt) in results
            if r !== nothing
                push!(obj.results, r)
            end
            if newt !== nothing
                append!(obj.tasks, newt)
            end
        end
    end

    return obj.results
end
